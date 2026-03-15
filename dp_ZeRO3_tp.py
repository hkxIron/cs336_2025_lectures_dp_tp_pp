import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
import math
from typing import List

from dp_tp_pp import generate_sample_data, setup
from lecture_08_utils import get_init_params, int_divide, spawn_wrapper
from torch_util import get_device

"""
矩阵乘法的梯度推导：
Y=X@W
dL/dW = X.T @ dL/dY
dL/dX = dL/dY @ W.T

在 ZeRO-3 的基础上引入张量并行（Tensor Parallelism, TP），我们需要构建一个 **3D 并行（TP + DP/ZeRO）** 的微型架构。

在 8 张卡，`TP=2, DP=4` 的设定下，核心逻辑变化如下：
1. **通信组划分 (Process Groups)**：8张卡需要被划分为 4 个 TP 组（每组 2 张卡）和 2 个 DP 组（每组 4 张卡）。
2. **数据切分**：数据只在 DP 维度切分（切成 4 份），同一个 TP 组内的 2 张卡看到的是**完全相同**的输入数据。
3. **模型切分 (TP + ZeRO3)**：
   * **TP 切分**：通常将线性层分为 **Column Parallel**（列切分，切输出维度）和 **Row Parallel**（行切分，切输入维度）。
   * **ZeRO-3 切分**：在 TP 切分的基础上，进一步在 DP 组内将参数切成 4 份。因此每张卡最终只保存 `1 / (TP * DP) = 1/8` 的参数。
4. **前向/反向传播**：
   * **Column Parallel**：前向无 TP 通信；反向时对输入 `x` 的梯度需要在 TP 组内做 `All-Reduce SUM`。 
    Y = X @[W1, W2] = [Y1, Y2]
    即:
    Y1= X @ W1
    Y2= X @ W2

   * **Row Parallel**：前向时对输出结果需要在 TP 组内做 `All-Reduce SUM`；反向无 TP 通信。 
    Z = Y @ [ V1; 
              V2 ]
     = [Y1, Y2] @ [V1; 
                   V2]
     = Z1 + Z2 
     = Y1 @ V1 + Y2 @ V2 
    即：
    Z1 = Y1 @ V1
    Z2 = Y2 @ V2

    因为此处的Y为分块矩阵[ Y1, Y2 ]
    dL/dY = [dL/dY1, dL/dY2] 
          = [dL/dZ1 @ V1.T, dL/dZ2 @ V2.T ], 直接concat即可，无需通信
    dL/dV = [dL/dV1;
             dL/dV2]
          = [dL/dZ1 @ Y1.T; 
             dL/dZ2 @ Y2.T ], 直接concat即可，无需通信

    dL/dW = [dL/dW1, dL/dW2] 
          = [X.T @ dL/dY1, X.T @ dL/dY2]

    dL/dX = dL/dY1 @ W1.T + dL/dY2 @ W2.T , 此处需要All-Reduce SUM

### 核心改动解析：
1. **正交的通信组**：通过 `dist.new_group` 创建了 `tp_group` 和 `dp_group`。ZeRO-3 的 `All-Gather` 和 `Reduce-Scatter` 仅在 `dp_group` 内进行；而 TP 的 `All-Reduce` 仅在 `tp_group` 内进行。
2. **双重切分 (Double Sharding)**：
   * 以前的 ZeRO-3：参数大小为 `[num_dim / 8, num_dim]`。
   * 现在的 TP+ZeRO-3：以 Column Parallel 为例，参数先被 TP 切成 `[num_dim, num_dim / 2]`，再被 ZeRO-3 切成 `[num_dim / 4, num_dim / 2]`。单卡显存占用依然是 `1/8`，但计算被分摊到了 TP 组。
3. **Autograd 拆分**：为了符合 Megatron-LM 的标准做法，将 Linear 拆分成了 `ColumnLinear` 和 `RowLinear`。
   * `ColumnLinear`：前向直接算，反向时对 `grad_x` 做 TP All-Reduce。
   * `RowLinear`：前向算完后对 `out` 做 TP All-Reduce，反向直接算。
4. **优化器完全解耦**：你会发现**优化器更新的代码一行都没改**。因为无论是 TP 还是 ZeRO，最终落到每张卡上的都是一个普通的 Tensor 及其对应的 local grad，AdamW 只需要对这块 local tensor 进行逐元素更新即可。
   
下面是完整的代码实现。为了体现 TP 的特点，我将原来的单层 Linear 改为了经典的 **MLP 结构 (ColumnLinear -> GELU -> RowLinear)**：
"""

# ==========================================
# 自定义 Autograd Function: ZeRO-3 + TP (Column Parallel)
# ==========================================
class ZeRO3_TP_ColumnLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, local_param: Parameter, dp_group, tp_group, dp_size: int, in_dim: int, out_dim_per_tp: int):
        # 1. ZeRO-3: 在 DP 组内 All-Gather，恢复完整的 TP 分片参数
        # x: [local_batch_size, in_dim=num_dim]
        # local_param:   [in_dim / dp_size, out_dim_per_tp]
        # gather=>
        # full_tp_param: [in_dim, out_dim_per_tp]
        full_tp_param = torch.empty(in_dim, out_dim_per_tp, dtype=local_param.dtype, device=local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=local_param, group=dp_group)

        # 2. 计算前向 (Column Parallel 前向不需要 TP 通信)
        # x: [local_batch_size, in_dim]
        # full_tp_param: [in_dim, out_dim_per_tp]
        # =>
        # out: [local_batch_size, out_dim_per_tp]
        out = x @ full_tp_param

        # 3. 保存上下文
        ctx.save_for_backward(x) # save_for_backward(x)里的x无法自动计算梯度, 需要自己计算梯度
        ctx.local_param = local_param # local_param支自动计算梯度
        ctx.dp_group = dp_group
        ctx.tp_group = tp_group
        ctx.in_dim = in_dim
        ctx.out_dim_per_tp = out_dim_per_tp

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        矩阵乘法的梯度推导：
        Y=X@W
        dL/dW = X.T @ dL/dY
        dL/dX = dL/dY @ W.T

        =>
        Y = X @[W1, W2] = [Y1, Y2]
        即:
        Y1= X @ W1
        Y2= X @ W2
        
        W要拆开
        dL/dW = [dL/dW1, dL/dW2] 
              = [X.T @ dL/dY1, X.T @ dL/dY2]
        X要相加
        dL/dX = dL/dY1 @ W1.T + dL/dY2 @ W2.T , 此处需要All-Reduce SUM
        """
        x, = ctx.saved_tensors

        # 1. ZeRO-3: 再次 All-Gather 恢复 TP 分片参数
        # local_param:   [in_dim / dp_size, out_dim_per_tp]
        # gather =>
        # full_tp_param: [in_dim=num_dim, out_dim_per_tp]
        full_tp_param = torch.empty(ctx.in_dim, ctx.out_dim_per_tp, dtype=ctx.local_param.dtype, device=ctx.local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=ctx.local_param, group=ctx.dp_group)

        # 2. 计算对 x 的梯度
        # dL/dX = dL/dY @ W.T
        # grad_out: [local_batch_size, out_dim_per_tp]
        # full_tp_param: [in_dim=num_dim, out_dim_per_tp]
        # =>
        # grad_x_partial: [local_batch_size, in_dim=num_dim]
        grad_x_partial = grad_out @ full_tp_param.T

        # 3. 【TP 通信】Column Parallel 反向传播需要对 grad_x_partial 在 TP 组内求和 (All-Reduce)
        # grad_x_partial: [local_batch_size, in_dim=num_dim]
        # dL/dX = dL/dY1 @ W1.T + dL/dY2 @ W2.T
        dist.all_reduce(grad_x_partial, op=dist.ReduceOp.SUM, group=ctx.tp_group)

        # 4. 计算对参数的梯度
        # x: [local_batch_size, in_dim=num_dim]
        # grad_out: [local_batch_size, out_dim_per_tp]
        # grad_full_tp_param: [in_dim, out_dim_per_tp]
        grad_full_tp_param = x.T @ grad_out
        del full_tp_param # 阅后即焚

        # 5. ZeRO-3: 在 DP 组内 Reduce-Scatter 梯度
        # local_grad: [in_dim / dp_size, out_dim_per_tp]
        local_grad = torch.empty_like(ctx.local_param)
        # 注意：这里用 AVG 是为了在 DP 维度上平均不同 batch 的梯度, 即不同data所产生的不同梯度进行平均
        # grad_full_tp_param: [in_dim, out_dim_per_tp]
        # =>
        # local_grad: [in_dim / dp_size, out_dim_per_tp]
        dist.reduce_scatter_tensor(output=local_grad, input=grad_full_tp_param, op=dist.ReduceOp.AVG, group=ctx.dp_group)

        return grad_x_partial, local_grad, None, None, None, None, None

# ==========================================
# 自定义 Autograd Function: ZeRO-3 + TP (Row Parallel)
# ==========================================
class ZeRO3_TP_RowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, local_param: Parameter, dp_group, tp_group, dp_size: int, in_dim_per_tp: int, out_dim: int):
        """
        权重按行并行
        Z = Y @ [ V1; 
                  V2 ]
        = [Y1, Y2] @ [V1; 
                      V2]
        = Z1 + Z2 
        = Y1 @ V1 + Y2 @ V2 

        即：
        Z1 = Y1 @ V1
        Z2 = Y2 @ V2

        dL/dY = [dL/dY1, dL/dY2] 
            = [dL/dZ1 @ V1.T, dL/dZ2 @ V2.T ], 直接concat即可，无需通信
        dL/dV = [dL/dV1;
                dL/dV2]
            = [dL/dZ1 @ Y1.T; 
                dL/dZ2 @ Y2.T ], 直接concat即可，无需通信

        """

        # 1. ZeRO-3: 在 DP 组内 All-Gather
        # local_param: [in_dim_per_tp / dp_size, out_dim = num_dim]
        # gather=>
        # full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        full_tp_param = torch.empty(in_dim_per_tp, out_dim, dtype=local_param.dtype, device=local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=local_param, group=dp_group)

        # 2. 计算前向
        # x: [local_batch_size, in_dim_per_tp]
        # full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        # out_partial: [local_batch_size, out_dim = num_dim]
        out_partial = x @ full_tp_param

        # 3. 【TP 通信】Row Parallel 前向传播需要对结果在 TP 组内求和 (All-Reduce)
        # NOTE: 这里之所以要用 SUM 是因为 Row Parallel 的前向计算里每个TP里的前向值Z_i只是向量内积的一部分，所以需要把所有部分加起来
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM, group=tp_group)
        # out: [local_batch_size, out_dim = num_dim]
        out = out_partial

        # 4. 保存上下文
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.dp_group = dp_group
        ctx.tp_group = tp_group
        ctx.in_dim_per_tp = in_dim_per_tp
        ctx.out_dim = out_dim

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors

        # 1. ZeRO-3: 再次 All-Gather
        # full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        full_tp_param = torch.empty(ctx.in_dim_per_tp, ctx.out_dim, dtype=ctx.local_param.dtype, device=ctx.local_param.device)
        # local_param: [in_dim_per_tp / dp_size, out_dim = num_dim]
        # gather=>
        # full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=ctx.local_param, group=ctx.dp_group)

        # 2. 计算对 x 的梯度 (Row Parallel 反向不需要 TP 通信)
        # grad_out: [local_batch_size, out_dim = num_dim]
        # full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        # grad_x: [local_batch_size, in_dim_per_tp]
        grad_x = grad_out @ full_tp_param.T

        # 3. 计算对参数的梯度
        # x: [local_batch_size, in_dim_per_tp]
        # grad_out: [local_batch_size, out_dim = num_dim]
        # => grad_full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        grad_full_tp_param = x.T @ grad_out
        del full_tp_param # 阅后即焚

        # 4. ZeRO-3: 在 DP 组内 Reduce-Scatter 梯度
        # grad_full_tp_param: [in_dim_per_tp, out_dim = num_dim]
        # reduce_scatter=>
        # local_grad: [in_dim_per_tp / dp_size, out_dim = num_dim]
        local_grad = torch.empty_like(ctx.local_param)
        dist.reduce_scatter_tensor(output=local_grad, input=grad_full_tp_param, op=dist.ReduceOp.AVG, group=ctx.dp_group)

        return grad_x, local_grad, None, None, None, None, None


# ==========================================
# 主训练逻辑
# ==========================================
def manual_tp_dp_zero3_parallel(rank: int, world_size: int, tp_size:int, dp_size:int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    print(f"[Rank:{rank}] TP:{tp_size} DP:{dp_size} world_size:{world_size}")
    # 假设外部已经调用了 dist.init_process_group
    device = get_device(rank)

    # ------------------------------------------
    # 1. 初始化 TP 和 DP 通信组
    # ------------------------------------------
    #tp_size = 2
    #dp_size = 4
    assert world_size == tp_size * dp_size, "world_size 必须等于 tp_size * dp_size"

    # 划分 TP 组: [0,1], [2,3], [4,5], [6,7]
    tp_group_num = world_size // tp_size
    tp_group = None
    for i in range(tp_group_num):
        # NOTE: TP组通信量大，要求相邻的GPU被划分到同一个 TP 组
        # GPU: [0,1], [2,3], [4,5], [6,7], 相邻的GPU被划分到同一个 TP 组, 共4组TP组
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        if rank==0:
            print(f"TP Group[{i}]: {ranks}")

        group = dist.new_group(ranks)
        if rank in ranks: # 当前GPU是否属于当前的TP组
            tp_group = group
            # 当前rank在当前TP组中的rank index, 为连续值
            tp_rank_index = ranks.index(rank) # 获取当前rank在当前TP组中的rank

    # 划分 DP 组: [0,2,4,6], [1,3,5,7], 共2组DP组
    dp_group_num = world_size // dp_size
    dp_group = None
    for i in range(dp_group_num):
        ranks = list(range(i, world_size, dp_group_num))
        group = dist.new_group(ranks)
        if rank==0:
            print(f"DP Group[{i}]: {ranks}")
        if rank in ranks:
            dp_group = group
            # 当前rank在当前DP组中的rank index, 为连续值
            dp_rank_index = ranks.index(rank)

    if rank == 0:
        print(f"Initialized 3D Parallelism: TP={tp_size}, DP={dp_size}")

    # ------------------------------------------
    # 2. 数据切分 (仅在 DP 维度切分)
    # ------------------------------------------
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // dp_size
    # 同一个 TP 组内的卡 (dp_rank 相同) 拿到的是同一份数据
    local_data = data[dp_rank_index * local_batch_size : (dp_rank_index + 1) * local_batch_size].to(device)

    # ------------------------------------------
    # 3. 模型参数初始化 (TP 切分 + ZeRO-3 切分)
    # ------------------------------------------
    local_params_all_layers: List[Parameter] = []

    # 为了演示 TP，我们将每层定义为一个 MLP Block: ColumnLinear -> GELU -> RowLinear
    # 假设 num_layers 是 MLP Block 的数量, 由于MLP Block 有2层layer = ColumnLinear + GELU + RowLinear, 但GELU无参数, 所以num_layers需要是偶数
    num_mlp_layers = int_divide(num_layers, 2)
    for layer_idx in range(num_mlp_layers):
        # --- Layer 1: Column Parallel ---
        # 全局权重: [num_dim, num_dim]
        full_w1 = get_init_params(num_dim, num_dim, rank)

        # Column Parallel TP 按列切分 (切输出维度 dim=1)， 将每层的权重按切分成TP份, 每份参数大小为 [num_dim, num_dim / tp_size]
        tp_chunk_out = num_dim // tp_size
        # w1_tp: [num_dim, num_dim / tp_size]
        w1_tp = full_w1[:, tp_rank_index * tp_chunk_out : (tp_rank_index + 1) * tp_chunk_out]

        # ZeRO-3 DP按行切分 (在 DP 组内切输入维度 dim=0)
        dp_chunk_in = num_dim // dp_size
        # w1_local: [num_dim / dp_size, num_dim / tp_size]
        # NOTE: 对w1按行dp切分, 所以TP + DP之后，参数大小为原始的：1/(dp_size * tp_size)
        w1_local = w1_tp[dp_rank_index * dp_chunk_in : (dp_rank_index + 1) * dp_chunk_in, :].clone()
        local_params_all_layers.append(Parameter(w1_local))

        # --- Layer 2: Row Parallel ---
        # 全局权重: [num_dim, num_dim]
        full_w2 = get_init_params(num_dim, num_dim, rank)

        # Row Parallel TP 按行切分 (切输入维度 dim=0), 每份参数大小为: [num_dim / tp_size, num_dim]
        tp_chunk_in = num_dim // tp_size
        # w2_tp: [num_dim / tp_size, num_dim], 按行切分
        w2_tp = full_w2[tp_rank_index * tp_chunk_in : (tp_rank_index + 1) * tp_chunk_in, :]

        # ZeRO-3 DP再次按行切分 (在 DP 组内切输入维度 dim=0) 
        # NOTE: 对w2按行dp切分, 所以TP + DP之后，参数大小为原始的：1/(dp_size * tp_size)
        dp_chunk_in_row = tp_chunk_in // dp_size
        # w2_local: [num_dim / tp_size / dp_size, num_dim]
        w2_local = w2_tp[dp_rank_index * dp_chunk_in_row : (dp_rank_index + 1) * dp_chunk_in_row, :].clone()
        local_params_all_layers.append(Parameter(w2_local))

    # ------------------------------------------
    # 4. 优化器状态初始化
    # ------------------------------------------
    optim_states = {}
    lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 1e-2

    # ------------------------------------------
    # 5. 训练循环
    # ------------------------------------------
    for step in range(num_steps):
        # x: [local_batch_size, num_dim]
        x = local_data

        # 前向传播
        for i in range(num_mlp_layers):
            # 注意：每层放了2个参数w1,w2，所以要乘以2
            # w1_local: [num_dim / dp_size, num_dim / tp_size], 列切分
            # w2_local: [num_dim / tp_size / dp_size, num_dim]，行切分
            w1_local:Parameter = local_params_all_layers[2 * i]
            w2_local:Parameter = local_params_all_layers[2 * i + 1]

            # x: [local_batch_size, num_dim]
            # w1_local: [num_dim / dp_size, num_dim / tp_size], 列切分
            # Column Linear
            x = ZeRO3_TP_ColumnLinear.apply(
                x, w1_local, dp_group, tp_group, dp_size, 
                #in_dim=num_dim, out_dim_per_tp = num_dim // tp_size
                num_dim, num_dim // tp_size
            )
            # x: [local_batch_size, num_dim]
            x = F.gelu(x)

            # Row Linear
            # x: [local_batch_size, num_dim]
            # w2_local: [num_dim / tp_size / dp_size, num_dim]， 行切分
            x = ZeRO3_TP_RowLinear.apply(
                x, w2_local, dp_group, tp_group, dp_size, 
                #in_dim_per_tp=num_dim // tp_size, out_dim=num_dim
                num_dim // tp_size, num_dim
            )
            x = F.gelu(x)

        loss = x.square().mean()

        # 反向传播 (自动触发 Autograd 中的 Gather, ReduceScatter, AllReduce)
        loss.backward()

        # adamw优化器状态与参数更新 (完全无通信，纯本地更新)
        for layer_param_idx, local_param in enumerate(local_params_all_layers):
            if local_param.grad is None: 
                continue
            # local_grad: 即可能是w1_local, 也可能是w2_local
            # w1_local: [num_dim / dp_size, num_dim / tp_size]
            # w2_local: [num_dim / tp_size / dp_size, num_dim]
            local_grad = local_param.grad

            if layer_param_idx not in optim_states:
                optim_states[layer_param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_param),
                    'exp_avg_sq': torch.zeros_like(local_param)
                }

            state = optim_states[layer_param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            local_param.data.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            local_param.data.addcdiv_(exp_avg, denom, value=-step_size)

            local_param.grad = None # 清理梯度

        #if rank == 0 and step % 5 == 0:
        if rank==0:
            print(f"[ZeRO-3 + TP rank:{rank}] Step {step:2d}, Loss = {loss.item():.6f}")

    # 打印显存占用验证
    if rank == 0:
        # 注意：参数w的第0维按dp切分
        print(f"W1 (up_project)   local shape: {local_params_all_layers[0].shape} (Expected: [{num_dim//dp_size}, {num_dim//tp_size}])")
        print(f"W2 (down_project) local shape: {local_params_all_layers[1].shape} (Expected: [{(num_dim//tp_size)//dp_size}, {num_dim}])")

    torch.distributed.destroy_process_group()

if  __name__ == "__main__":
    data = generate_sample_data()
    #spawn_wrapper(manual_tp_dp_zero3_parallel, world_size=8, tp_size=2, dp_size=4, data=data, num_layers=4, num_steps=40)
    spawn_wrapper(manual_tp_dp_zero3_parallel, world_size=4, tp_size=2, dp_size=2, data=data, num_layers=4, num_steps=40)