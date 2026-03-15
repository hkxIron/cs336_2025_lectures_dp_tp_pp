import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
import math
from typing import List, Tuple

from dp_tp_pp import generate_sample_data, setup
from lecture_08_utils import get_init_params, int_divide, spawn_wrapper
from torch_util import get_device


"""
在现有的 ZeRO-3 (DP) + Tensor Parallelism (TP) 的基础上引入 **Pipeline Parallelism (PP)** 并实现 **1F1B (One Forward, One Backward)** 调度，我们需要完成以下几个核心步骤：

1. **3D 并行通信组划分**：将 `world_size` 划分为 `pp_size * dp_size * tp_size`，并为每个维度建立通信组。
2. **模型切分 (PP)**：将模型的 `num_layers` 按 `pp_size` 切分，每个 GPU 只初始化和保存属于自己 PP 阶段的层。
3. **数据微批次化 (Micro-batching)**：将本地的 batch 进一步切分为多个 `micro_batch`，以供流水线流动。
4. **P2P 通信**：使用 `dist.isend` 和 `dist.irecv` 在相邻的 PP 阶段之间传递前向的激活值 (Activations) 和反向的梯度 (Gradients)。
5. **1F1B 调度逻辑**：实现 Warmup（预热）、Steady（1F1B 稳态）和 Cooldown（冷却）三个阶段。

### 核心代码解析：

1. **3D 拓扑映射**：
   通过公式 `rank = (pp * dp_size + dp) * tp_size + tp` 建立 3D 坐标系。
   - **TP 组**：`PP` 和 `DP` 相同的卡组成。
   - **DP 组**：`PP` 和 `TP` 相同的卡组成。
   - **P2P 通信**：`DP` 和 `TP` 相同，但 `PP` 相邻的卡进行点对点通信（例如 Rank 0 发送给 Rank 4）。

2. **模型切分 (PP)**：
   `local_mlp_layers = int_divide(num_mlp_layers, pp_size)`。每个 GPU 只初始化属于自己 PP 阶段的层，极大地节省了显存。

3. **1F1B 调度 (One Forward, One Backward)**：
   - **Warmup**：前置 PP 阶段先执行若干次 Forward，填满流水线。
   - **Steady (1F1B)**：执行一次 Forward，紧接着执行一次 Backward。
   - **Cooldown**：排空流水线，执行剩余的 Backward。

4. **P2P 与 Autograd 的桥接**：
   - 在 `forward_microbatch` 中，接收到的 `x` 必须调用 `x.requires_grad_()`，这样 PyTorch 才能在本地构建计算图。
   - 在 `backward_microbatch` 中，调用 `out.backward(grad_out)` 会自动计算本地参数的梯度，**同时也会计算出 `x.grad`**。
   - 最后将 `x.grad` 通过 P2P 发送给上一个 PP 阶段，完美衔接了跨进程的计算图。

"""

# ==========================================
# 自定义 Autograd Function: ZeRO-3 + TP (保持不变)
# ==========================================
class ZeRO3_TP_ColumnLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, dp_group, tp_group, dp_size, in_dim, out_dim_per_tp):
        full_tp_param = torch.empty(in_dim, out_dim_per_tp, dtype=local_param.dtype, device=local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=local_param, group=dp_group)
        out = x @ full_tp_param
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.dp_group = dp_group
        ctx.tp_group = tp_group
        ctx.in_dim = in_dim
        ctx.out_dim_per_tp = out_dim_per_tp
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        full_tp_param = torch.empty(ctx.in_dim, ctx.out_dim_per_tp, dtype=ctx.local_param.dtype, device=ctx.local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=ctx.local_param, group=ctx.dp_group)
        grad_x_partial = grad_out @ full_tp_param.T
        dist.all_reduce(grad_x_partial, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        grad_full_tp_param = x.T @ grad_out
        del full_tp_param
        local_grad = torch.empty_like(ctx.local_param)
        dist.reduce_scatter_tensor(output=local_grad, input=grad_full_tp_param, op=dist.ReduceOp.AVG, group=ctx.dp_group)
        return grad_x_partial, local_grad, None, None, None, None, None

class ZeRO3_TP_RowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, dp_group, tp_group, dp_size, in_dim_per_tp, out_dim):
        full_tp_param = torch.empty(in_dim_per_tp, out_dim, dtype=local_param.dtype, device=local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=local_param, group=dp_group)
        out_partial = x @ full_tp_param
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM, group=tp_group)
        out = out_partial
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
        full_tp_param = torch.empty(ctx.in_dim_per_tp, ctx.out_dim, dtype=ctx.local_param.dtype, device=ctx.local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_tp_param, input_tensor=ctx.local_param, group=ctx.dp_group)
        grad_x = grad_out @ full_tp_param.T
        grad_full_tp_param = x.T @ grad_out
        del full_tp_param
        local_grad = torch.empty_like(ctx.local_param)
        dist.reduce_scatter_tensor(output=local_grad, input=grad_full_tp_param, op=dist.ReduceOp.AVG, group=ctx.dp_group)
        return grad_x, local_grad, None, None, None, None, None

# ==========================================
# P2P 通信辅助函数 (避免死锁)
# ==========================================
def p2p_send_sync(tensor, dst_rank):
    if dst_rank is not None:
        dist.send(tensor, dst_rank)

def p2p_recv_sync(tensor, src_rank):
    if src_rank is not None:
        dist.recv(tensor, src_rank)

def p2p_send_async(tensor, dst_rank):
    if dst_rank is not None:
        req = dist.isend(tensor, dst_rank)
        #req.wait()
        return req
    return None

def p2p_recv_async(tensor, src_rank):
    if src_rank is not None:
        req = dist.irecv(tensor, src_rank)
        return req
    return None


# ==========================================
# 主训练逻辑: 3D Parallelism (PP + DP + TP) + 1F1B
# ==========================================
def manual_tp_pp_dp_zero3_parallel(rank: int, world_size: int, kwargs: dict):
    tp_size = kwargs['tp_size']
    pp_size = kwargs['pp_size']
    dp_size = kwargs['dp_size']
    data = kwargs['data']
    num_layers = kwargs['num_layers']
    num_steps = kwargs['num_steps']
    num_microbatches = kwargs['num_microbatches']

    setup(rank, world_size, port="15624")
    device = get_device(rank)
    assert world_size == pp_size * tp_size * dp_size, "world_size 必须等于 pp_size * tp_size * dp_size"

    # ------------------------------------------
    # 1. 3D 拓扑与通信组初始化
    # 在本代码中，GPU 的分组逻辑在物理映射（Rank 的连续性）上，确实是按照 TP → DP → PP 的顺序由内向外划分的。
    # 我们可以把整个集群的 GPU 想象成一个三维数组 GPU_ARRAY[pp_size][dp_size][tp_size]
    # ------------------------------------------
    # 先TP分组,   TP=2: [0,1], [2,3], [4,5], [6,7] (相邻的 GPU 被划分到同一个 TP 组), GPU0与GPU1共享一个完整的weight
    # 再DP ZeRO3分组,   PP=2: [0,2], [1,3], [4,6], [5,7], 即 GPU0, GPU2共享一个完整的batch data
    # 最后PP分组,  DP=2: [0,4], [1,5], [2,6], [3,7], 即GPU0, GPU4共享一个完整的模型
    # GPU_ARRAY[pp_size][dp_size][tp_size]
    # 映射公式: rank = (pp_rank * dp_size + dp_rank) * tp_size + tp_rank
    """
    从物理 Rank 的连续性来看（假设 tp=2, dp=2, pp=2，共 8 张卡）：
    想象一下秒针/分针/时针的映射关系：
    秒针（TP Rank）：每秒走一步, 
    分针（DP 组）：每分钟走一步, 60s为一组
    时针（PP 组）：每小时,3600s为一组

    TP 组（最内层，步长为 1）：Rank 0 和 1 属于同一个 TP 组。它们在物理上是相邻的。
    DP 组（中间层，步长为 tp_size）：Rank 0 和 2 属于同一个 DP 组。它们在物理上隔了一个 TP 的距离。
    PP 组（最外层，步长为 tp_size * dp_size）：Rank 0 和 4 属于同一个 PP 组（即流水线的上下游）。它们在物理上距离最远。
    """
    tp_rank_index = rank % tp_size
    dp_rank_index = (rank // tp_size) % dp_size
    pp_rank_index = rank // (tp_size * dp_size)

    # 计算 PP 阶段的上下游 Rank (用于 P2P 通信)
    # 保证 TP 和 DP 的 index 不变，只改变 PP 的 index
    prev_pp_rank = rank - (tp_size * dp_size) if pp_rank_index > 0 else None
    next_pp_rank = rank + (tp_size * dp_size) if pp_rank_index < pp_size - 1 else None

    # 划分 TP 组 (同 PP, 同 DP)
    tp_group = None
    for pp in range(pp_size):
        pp_group=[]
        for dp in range(dp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for tp in range(tp_size)]
            if rank==0:
                print(f"TP Group[pp_{pp}][dp_{dp}]: {ranks}")
                pp_group.extend(ranks)
            group = dist.new_group(ranks)
            if rank in ranks: 
                tp_group = group
        if rank==0:
            print(f"PP Group[pp_{pp}]: {pp_group}")

    # 划分 DP 组 (同 PP, 同 TP)
    dp_group = None
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for dp in range(dp_size)]
            group = dist.new_group(ranks)
            if rank==0:
                print(f"DP Group[pp_{pp}][tp_{tp}]: {ranks}")
            if rank in ranks: 
                dp_group = group

    if rank == 0:
        print(f"Initialized 3D Parallelism: PP={pp_size}, DP={dp_size}, TP={tp_size}")

    # ------------------------------------------
    # 2. 数据切分 (DP 切分 + Micro-batch 切分)
    # ------------------------------------------
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // dp_size
    micro_batch_size = local_batch_size // num_microbatches # 这个是pp_size维度的microbatch

    # DP 切分
    local_data = data[dp_rank_index * local_batch_size : (dp_rank_index + 1) * local_batch_size].to(device)
    """
    Micro-batch 切分, 按第0维切分为多个 micro-batch
    NOTE: 这里是按第0维切分，所以每个microbatch的shape为[local_batch_size//micro_batch_size, num_dim]
    # 在batch维度将batch切分成chunks个micro_batches, 每个micro_batch的shape为(batch/chunks, hidden_dim)
    micro_batches = torch.chunk(batch, chunk_num, dim=0)
    """
    micro_batches:Tuple[torch.Tensor] = local_data.split(micro_batch_size)

    # ------------------------------------------
    # 3. 模型参数初始化 (PP 切分 + TP 切分 + ZeRO-3 切分)
    # ------------------------------------------
    local_params_all_layers: List[Parameter] = []
    # 为了演示 TP，我们将每层定义为一个 MLP Block: ColumnLinear -> GELU -> RowLinear
    # 假设 num_layers 是 MLP Block 的数量, 由于MLP Block 有2层layer = ColumnLinear + GELU + RowLinear, 但GELU无参数, 所以num_layers需要是偶数
    num_mlp_layers = int_divide(num_layers, 2) # mlp = up_project + down_project, 所以要除以2

    # PP 切分：当前 GPU 只负责一部分 Layer
    local_mlp_layers = int_divide(num_mlp_layers, pp_size)

    for layer_idx in range(local_mlp_layers):
        # --- Layer 1: Column Parallel ---
        # full_w1: [num_dim, num_dim]
        full_w1 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_out = num_dim // tp_size
        # w1_tp: [num_dim, num_dim / tp_size], TP并行按列切分
        w1_tp = full_w1[:, tp_rank_index * tp_chunk_out : (tp_rank_index + 1) * tp_chunk_out]
        dp_chunk_in = num_dim // dp_size
        # ZeRO-3 DP 按行切分 (在 DP 组内切输入维度 dim=0)
        # w1_local: [num_dim / dp_size, num_dim / tp_size]
        w1_local = w1_tp[dp_rank_index * dp_chunk_in : (dp_rank_index + 1) * dp_chunk_in, :].clone()
        local_params_all_layers.append(Parameter(w1_local))

        # --- Layer 2: Row Parallel ---
        # full_w2: [num_dim, num_dim]
        full_w2 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_in = num_dim // tp_size
        # w2_tp: [num_dim/tp_size, num_dim], TP并行按行切分
        w2_tp = full_w2[tp_rank_index * tp_chunk_in : (tp_rank_index + 1) * tp_chunk_in, :]
        dp_chunk_in_row = tp_chunk_in // dp_size
        # ZeRO-3 DP 按行切分 (在 DP 组内切输入维度 dim=0)
        # w2_local: [num_dim / tp_size / dp_size, num_dim]
        w2_local = w2_tp[dp_rank_index * dp_chunk_in_row : (dp_rank_index + 1) * dp_chunk_in_row, :].clone()
        local_params_all_layers.append(Parameter(w2_local))

    optim_states = {}
    lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 1e-2

    # ------------------------------------------
    # 4. 训练循环 (1F1B 调度)
    # ------------------------------------------
    for step in range(num_steps):
        # 用于保存前向的激活值，供反向传播使用
        saved_tensors = []
        step_loss = 0.0

        # 1F1B 阶段计算
        num_warmup = min(num_microbatches, pp_size - pp_rank_index - 1)
        num_1f1b = num_microbatches - num_warmup
        num_cooldown = num_warmup

        async_reqs = []

        def forward_microbatch(mb_idx):
            if pp_rank_index == 0:
                # 第一个 PP 阶段：从数据中读取
                x = micro_batches[mb_idx].clone().requires_grad_()
            else:
                # 其他 PP 阶段：从上一个阶段接收
                x = torch.empty(micro_batch_size, num_dim, device=device)
                p2p_recv_sync(x, prev_pp_rank)
                x.requires_grad_() # 必须设置，以连接计算图, 这样x才会有梯度

            out = x
            for i in range(local_mlp_layers):
                # 每层有w1,w2两个参数，所以要乘以2
                w1_local = local_params_all_layers[2 * i]
                w2_local = local_params_all_layers[2 * i + 1]

                out = ZeRO3_TP_ColumnLinear.apply(out, w1_local, dp_group, tp_group, dp_size, num_dim, num_dim // tp_size)
                out = F.gelu(out)
                out = ZeRO3_TP_RowLinear.apply(out, w2_local, dp_group, tp_group, dp_size, num_dim // tp_size, num_dim)
                out = F.gelu(out)

            if pp_rank_index == pp_size - 1:
                # 最后一个 PP 阶段：计算 Loss
                loss = out.square().mean()
                saved_tensors.append((x, out, loss))
                return loss.item()
            else:
                # 其他 PP 阶段：发送给下一个阶段
                p2p_send_sync(out, next_pp_rank)
                saved_tensors.append((x, out, None))
                return 0.0

        def backward_microbatch():
            # 将队列中的梯度逐个取出，进行反向传播
            x, out, loss = saved_tensors.pop(0)

            if pp_rank_index == pp_size - 1:
                # 最后一个 PP 阶段：直接从 loss 开始反向传播
                loss.backward()
            else:
                # 其他 PP 阶段：接收来自下一个阶段的梯度，并继续反向传播
                grad_out = torch.empty_like(out)
                p2p_recv_sync(grad_out, next_pp_rank)
                out.backward(grad_out)

            if pp_rank_index != 0:
                # NOTE:将输入 x 的梯度发送给上一个 PP 阶段, 只有此处使用异步计算, 不能全部使用同步通信，否则会产生死锁
                req= p2p_send_async(x.grad, prev_pp_rank)
                async_reqs.append(req)

        # --- 执行 1F1B 调度 ---
        # 1. Warmup 阶段 (只做 Forward)
        for i in range(num_warmup):
            step_loss += forward_microbatch(i)

        # 2. Steady 1F1B 阶段 (一前一后)
        for i in range(num_1f1b):
            step_loss += forward_microbatch(num_warmup + i)
            backward_microbatch()

        # 3. Cooldown 阶段 (只做 Backward)
        for i in range(num_cooldown):
            backward_microbatch()

        # 等待所有异步发送完成，防止内存泄漏
        for req in async_reqs:
            if req is not None: 
                req.wait()
        # ------------------------------------------
        # 5. 优化器更新 (梯度平均与 AdamW)
        # ------------------------------------------
        for layer_param_idx, local_layer_param in enumerate(local_params_all_layers):
            if local_layer_param.grad is None: 
                continue

            # 因为 loss 是 mean，且累加了 num_microbatches 次，需要求平均
            # w1_local: [num_dim / dp_size, num_dim / tp_size], 列切分
            # w2_local: [num_dim / tp_size / dp_size, num_dim]，行切分
            local_grad = local_layer_param.grad.div_(num_microbatches)

            if layer_param_idx not in optim_states:
                optim_states[layer_param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_layer_param), # optimizer state放在当前的GPU上
                    'exp_avg_sq': torch.zeros_like(local_layer_param)
                }

            state = optim_states[layer_param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            local_layer_param.data.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            local_layer_param.data.addcdiv_(exp_avg, denom, value=-step_size)

            local_layer_param.grad = None # 清理梯度

        if rank == world_size - 1: # 最后一个 rank 负责打印 loss
            avg_loss = step_loss / num_microbatches
            print(f"[3D(TP+PP+DP) Parallel Rank:{rank}] Step {step:2d}, Loss = {avg_loss:.8f}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    data = generate_sample_data(batch_size=256, num_dim=1024)

    # 3D 并行配置: PP=2, DP=2, TP=2 => world_size = 8
    # num_layers=4, num_microbatches=4
    kwargs = {
        'tp_size': 2,
        'pp_size': 2,
        'dp_size': 2,
        'data': data,
        'num_layers': 4,
        'num_steps': 40,
        'num_microbatches': 4
    }

    spawn_wrapper(
        manual_tp_pp_dp_zero3_parallel, 
        world_size=8, 
        kwargs=kwargs
    )