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
将 ZeRO-3 降级为 **ZeRO-1**，核心的区别在于**参数（Parameters）的存储与通信方式**：

1. **ZeRO-3 (原代码)**：参数、梯度、优化器状态全部在 DP 组内切分。前向和反向计算时，需要通过 `all_gather` 临时收集完整参数，计算完后立刻丢弃。
2. **ZeRO-1 (新代码)**：**参数在 DP 组内是完整的**（不切分），只有**梯度**和**优化器状态**在 DP 组内切分。
   - **前向/反向计算**：直接使用本地完整的参数计算，**不需要任何 DP 通信**（只需保留 TP 的通信）。
   - **梯度同步**：反向传播结束后，对梯度进行 `reduce_scatter`，每个 GPU 只拿到属于自己那一部分的梯度。
   - **优化器更新**：每个 GPU 只更新自己负责的那部分参数（利用切分后的优化器状态）。
   - **参数同步**：更新完成后，通过 `all_gather` 将更新后的参数块广播给 DP 组内的其他 GPU，拼成完整的最新参数，供下一步使用。

### 核心修改点解析：

1. **Autograd 简化**：
   在 ZeRO-3 中，前向和反向都需要 `all_gather` 收集参数。在 ZeRO-1 中，参数在显存中已经是完整的（相对于 DP 而言），因此 `TP_ColumnLinear` 和 `TP_RowLinear` 退化成了纯粹的 Tensor Parallelism 算子，**内部不再包含任何 DP 通信**。
2. **参数初始化**：
   去掉了 `dp_chunk_in` 的切分逻辑。每个 DP Rank 都会初始化并保留完整的 `w1_tp` 和 `w2_tp`。
3. **ZeRO-1 优化器核心逻辑**：
   - **Reduce-Scatter 梯度**：使用 `dist.reduce_scatter_tensor` 将完整的梯度 `flat_grad` 聚合求平均，并切分成 `chunk_size` 大小，存入 `local_grad_chunk`。
   - **局部更新**：优化器状态 `exp_avg` 和 `exp_avg_sq` 的大小只有完整参数的 `1/dp_size`。我们只对 `local_param_chunk` 进行 AdamW 更新。
   - **All-Gather 参数**：更新完毕后，使用 `dist.all_gather_into_tensor` 将各个 GPU 上更新好的 `local_param_chunk` 重新拼装回 `flat_param`，为下一步的前向传播做准备。

### 1. 上面的代码中的 GPU 划分方式如何？是按先 TP，再 DP，最后 PP 的顺序吗？

**是的，完全正确。** 
在上面的代码中，GPU 的物理 Rank 映射公式为：
`rank = pp_rank_index * (dp_size * tp_size) + dp_rank_index * tp_size + tp_rank_index`

这种映射方式意味着：
* **TP 是最内层（步长为 1）**：相邻的 Rank（如 Rank 0 和 Rank 1）属于同一个 TP 组。
* **DP 是中间层（步长为 `tp_size`）**：相隔 `tp_size` 的 Rank（如 Rank 0 和 Rank 2）属于同一个 DP 组。
* **PP 是最外层（步长为 `dp_size * tp_size`）**：相隔最远的 Rank（如 Rank 0 和 Rank 4）属于同一个 PP 组。

---

### 2. 这是最合理的 GPU 划分方式吗？

**是的，这是目前工业界（如 Megatron-LM, DeepSpeed）公认的最优、最合理的 3D 并行物理映射方式。**

我们可以从**通信量**和**硬件拓扑（NVLink vs PCIe/InfiniBand）**的角度来分析为什么这样划分最合理：

1. **TP (Tensor Parallelism) - 通信量极大，延迟要求极高**
   * **特点**：在每一次前向和反向传播中，每一层都需要进行 `All-Reduce` 通信。
   * **硬件要求**：必须放在同一个物理节点（Node）内，利用带宽高、延迟极低的 **NVLink** 进行通信。
   * **映射策略**：因此 TP 必须是最内层（连续的 Rank），保证它们大概率被分配在同一台机器的相邻 GPU 上。

2. **PP (Pipeline Parallelism) - 通信量极小，对带宽要求最低**
   * **特点**：只需要在相邻的流水线阶段之间进行点对点（P2P）通信，传递的是激活值（Activations）和梯度（Gradients），数据量仅为 `[batch_size, seq_len, hidden_dim]`。
   * **硬件要求**：非常适合跨节点（Cross-node）通信，即使是通过较慢的网卡（如普通的 InfiniBand 或以太网）也不会成为瓶颈。
   * **映射策略**：因此 PP 被放在最外层（步长最大），让流水线的上下游分布在不同的物理机器上。

3. **DP (Data Parallelism / ZeRO) - 通信量较大，但频率较低**
   * **特点**：在 ZeRO-1 中，需要在反向传播结束时进行 `Reduce-Scatter`，在优化器更新后进行 `All-Gather`。通信量等于整个模型的参数量。
   * **硬件要求**：带宽要求介于 TP 和 PP 之间。通常在超大规模集群中，DP 组会跨越多个节点。
   * **映射策略**：放在中间层。

**总结**：`TP (同节点) -> DP (跨节点) -> PP (跨节点)` 的顺序完美契合了现代 GPU 集群“节点内 NVLink 极快，节点间网卡较慢”的非对称网络拓扑。


下面是修改后的完整代码。我重写了 Autograd 函数（移除了 ZeRO-3 的通信），并重构了优化器更新逻辑以实现 ZeRO-1。
"""

# ==========================================
# 自定义 Autograd Function: 纯 TP (移除 ZeRO-3 的 DP 通信)
# ==========================================
class TP_ColumnLinear(torch.autograd.Function):
    """
    矩阵乘法的梯度推导：
    Y=X@W
    dL/dW = X.T @ dL/dY
    dL/dX = dL/dY @ W.T

    * **Column Parallel**：前向无 TP 通信；反向时对输入 `x` 的梯度需要在 TP 组内做 `All-Reduce SUM`。 
    Y = X @[W1, W2] = [Y1, Y2]
    即:
    Y1= X @ W1
    Y2= X @ W2
    =>
    dL/dW = [dL/dW1, dL/dW2] 
          = [X.T @ dL/dY1, X.T @ dL/dY2]

    dL/dX = dL/dY1 @ W1.T + dL/dY2 @ W2.T , 此处需要All-Reduce SUM

    * **Row Parallel**：前向时对输出结果需要在 TP 组内做 `All-Reduce SUM`；反向无 TP 通信。 
    Z = Y @ [ V1; 
              V2 ]
     = [Y1, Y2] @ [V1; 
                   V2]
     = Z1 + Z2 
     = Y1 @ V1 + Y2 @ V2  # All-Reduce SUM
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

    """
    @staticmethod
    def forward(ctx, x, local_param, tp_group):
        # ZeRO-1 中，local_param 已经是完整的 TP 切块，不需要 all_gather
        # x: [micro_batch_size, num_dim]
        # w1_local: [num_dim, num_dim / tp_size], TP并行按列切分
        # =>
        # out: [micro_batch_size, num_dim/tp_size]
        out = x @ local_param
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        # 1. 计算对输入的梯度 (需要 TP 组内 All-Reduce)
        # grad_out: [micro_batch_size, num_dim/tp_size]
        # w1_local: [num_dim, num_dim / tp_size], TP并行按列切分
        # =>
        # grad_x_partial: [micro_batch_size, num_dim]
        grad_x_partial = grad_out @ ctx.local_param.T
        dist.all_reduce(grad_x_partial, op=dist.ReduceOp.SUM, group=ctx.tp_group)

        # 2. 计算对权重的梯度 (本地直接计算，无需 DP 通信)
        # x: [micro_batch_size, num_dim]
        # grad_out: [micro_batch_size, num_dim/tp_size]
        # grad_local_param: [num_dim, num_dim/tp_size]
        grad_local_param = x.T @ grad_out

        return grad_x_partial, grad_local_param, None

class TP_RowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, tp_group):
        # ZeRO-1 中，local_param 已经是完整的 TP 切块
        out_partial = x @ local_param
        # TP 组内 All-Reduce 聚合结果
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.tp_group = tp_group
        return out_partial

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        # 1. 计算对输入的梯度 (直接计算)
        grad_x = grad_out @ ctx.local_param.T

        # 2. 计算对权重的梯度 (本地直接计算，无需 DP 通信)
        grad_local_param = x.T @ grad_out

        return grad_x, grad_local_param, None

# ==========================================
# P2P 通信辅助函数 (保持不变)
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
        return req
    return None

def p2p_recv_async(tensor, src_rank):
    if src_rank is not None:
        req = dist.irecv(tensor, src_rank)
        return req
    return None

def print_3d_parallel_groups(rank: int, world_size: int, pp_size: int, dp_size: int, tp_size: int):
    """
    打印 3D 并行的 TP, DP, PP 分组情况。
    为了避免控制台混乱，只允许 rank 0 打印。

    示例：
    3D Parallelism Topology (World Size: 8)
    PP_size: 2, DP_size: 2, TP_size: 2
    ============================================================

    [TP Groups] (高频 All-Reduce，通常在同一节点内，利用 NVLink)
    ➤ PP_stage=0, DP_rank=0  =>  TP Group: [0, 1]
    ➤ PP_stage=0, DP_rank=1  =>  TP Group: [2, 3]
    ➤ PP_stage=1, DP_rank=0  =>  TP Group: [4, 5]
    ➤ PP_stage=1, DP_rank=1  =>  TP Group: [6, 7]

    [DP Groups] (ZeRO-1 Reduce-Scatter/All-Gather，可跨节点)
    ➤ PP_stage=0, TP_rank=0  =>  DP Group: [0, 2]
    ➤ PP_stage=0, TP_rank=1  =>  DP Group: [1, 3]
    ➤ PP_stage=1, TP_rank=0  =>  DP Group: [4, 6]
    ➤ PP_stage=1, TP_rank=1  =>  DP Group: [5, 7]

    [PP Groups] (P2P 激活值/梯度通信，最适合跨节点)
    ➤ DP_rank=0, TP_rank=0   =>  PP Group: [0, 4]
    ➤ DP_rank=0, TP_rank=1   =>  PP Group: [1, 5]
    ➤ DP_rank=1, TP_rank=0   =>  PP Group: [2, 6]
    ➤ DP_rank=1, TP_rank=1   =>  PP Group: [3, 7]
    """
    if rank != 0:
        return

    assert world_size == pp_size * dp_size * tp_size, "world_size 必须等于 pp * dp * tp"

    print("=" * 60)
    print(f" 3D Parallelism Topology (World Size: {world_size})")
    print(f" PP_size: {pp_size}, DP_size: {dp_size}, TP_size: {tp_size}")
    print("=" * 60)

    # 1. 打印 TP 组 (PP 和 DP 相同，TP 不同的卡组成一组)
    print("\n[TP Groups] (高频 All-Reduce，通常在同一节点内，利用 NVLink)")
    for pp in range(pp_size):
        for dp in range(dp_size):
            # 百位：dp_size*tp_size, 十位：tp_size, 个位：tp
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for tp in range(tp_size)]
            print(f"  ➤ PP_stage={pp}, DP_rank={dp}  =>  TP Group: {ranks}")

    # 2. 打印 DP 组 (PP 和 TP 相同，DP 不同的卡组成一组)
    print("\n[DP Groups] (ZeRO-1 Reduce-Scatter/All-Gather，可跨节点)")
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for dp in range(dp_size)]
            print(f"  ➤ PP_stage={pp}, TP_rank={tp}  =>  DP Group: {ranks}")

    # 3. 打印 PP 组 (DP 和 TP 相同，PP 不同的卡组成一组)
    print("\n[PP Groups] (P2P 激活值/梯度通信，最适合跨节点)")
    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for pp in range(pp_size)]
            print(f"  ➤ DP_rank={dp}, TP_rank={tp}   =>  PP Group: {ranks}")

    print("=" * 60 + "\n")

# ==========================================
# 主训练逻辑: 3D Parallelism (PP + ZeRO-1 DP + TP) + 1F1B
# ==========================================
def manual_tp_pp_dp_zero1_parallel(rank: int, world_size: int, kwargs: dict):
    tp_size = kwargs['tp_size']
    pp_size = kwargs['pp_size']
    dp_size = kwargs['dp_size']
    data = kwargs['data']
    num_layers = kwargs['num_layers']
    num_steps = kwargs['num_steps']
    num_microbatches = kwargs['num_microbatches']

    setup(rank, world_size, port="15624")
    device = get_device(rank)
    assert world_size == pp_size * tp_size * dp_size

    # 调用工具函数打印拓扑
    print_3d_parallel_groups(rank, world_size, pp_size, dp_size, tp_size)

    # ------------------------------------------
    # 1. 3D 拓扑与通信组初始化 (保持不变)
    # ------------------------------------------
    # 先TP, 再DP, 最后PP 的 rank index
    # 3D 并行的 rank 可以表示为: rank = pp * (dp_size * tp_size) + dp * tp_size + tp
    # 百位：dp_size*tp_size, 十位：tp_size, 个位：tp
    tp_rank_index = rank % tp_size
    dp_rank_index = (rank // tp_size) % dp_size
    pp_rank_index = rank // (tp_size * dp_size)

    prev_pp_rank = rank - (tp_size * dp_size) if pp_rank_index > 0 else None
    next_pp_rank = rank + (tp_size * dp_size) if pp_rank_index < pp_size - 1 else None

    tp_group = None
    for pp in range(pp_size):
        for dp in range(dp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for tp in range(tp_size)]
            group = dist.new_group(ranks)
            if rank in ranks: 
                tp_group = group

    dp_group = None
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = [pp * (dp_size * tp_size) + dp * tp_size + tp for dp in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks: 
                dp_group = group

    if rank == 0:
        print(f"Initialized 3D Parallelism (ZeRO-1): PP={pp_size}, DP={dp_size}, TP={tp_size}")

    # ------------------------------------------
    # 2. 数据切分 (DP 切分 + Micro-batch 切分)
    # ------------------------------------------
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // dp_size
    micro_batch_size = local_batch_size // num_microbatches

    local_data = data[dp_rank_index * local_batch_size : (dp_rank_index + 1) * local_batch_size].to(device)
    micro_batches: Tuple[torch.Tensor] = local_data.split(micro_batch_size)

    # ------------------------------------------
    # 3. 模型参数初始化 (PP 切分 + TP 切分，取消 ZeRO-3 的 DP 切分)
    # ------------------------------------------
    local_params_all_layers: List[Parameter] = []
    # 为了演示 TP，我们将每层定义为一个 MLP Block: ColumnLinear -> GELU -> RowLinear
    # 假设 num_layers 是 MLP Block 的数量, 由于MLP Block 有2层layer = ColumnLinear + GELU + RowLinear, 但GELU无参数, 所以num_layers需要是偶数
    num_mlp_layers = int_divide(num_layers, 2) # # mlp = ColumnLinear + RowLinear, 所以要除以2
    local_mlp_layers = int_divide(num_mlp_layers, pp_size) # 这个是pp_size维度的microbatch

    for layer_idx in range(local_mlp_layers):
        # --- Layer 1: Column Parallel ---
        full_w1 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_out = num_dim // tp_size
        # ZeRO-1: 不再按 DP 切分，直接保留完整的 TP 块。使用 .clone() 保证内存连续性
        # w1_tp: [num_dim, num_dim / tp_size], TP并行按列切分
        w1_tp = full_w1[:, tp_rank_index * tp_chunk_out : (tp_rank_index + 1) * tp_chunk_out].clone()
        local_params_all_layers.append(Parameter(w1_tp)) # 添加ColumnLinear的参数

        # --- Layer 2: Row Parallel ---
        full_w2 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_in = num_dim // tp_size
        # ZeRO-1: 不再按 DP 切分
        # w2_tp: [num_dim/tp_size, num_dim], TP并行按行切分
        w2_tp = full_w2[tp_rank_index * tp_chunk_in : (tp_rank_index + 1) * tp_chunk_in, :].clone()
        local_params_all_layers.append(Parameter(w2_tp)) # 添加RowLinear的参数

    optim_states = {}
    lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 1e-2

    # ------------------------------------------
    # 4. 训练循环 (1F1B 调度)
    # ------------------------------------------
    for step in range(num_steps):
        # 每个microbatch + 每个layer 的前向和反向传播
        saved_tensors_per_micro_batch_per_layer = []
        step_loss = 0.0
        async_reqs = []

        num_warmup = min(num_microbatches, pp_size - pp_rank_index - 1)
        num_1f1b = num_microbatches - num_warmup
        num_cooldown = num_warmup

        def forward_microbatch(mb_idx):
            if pp_rank_index == 0:
                # x: [micro_batch_size, num_dim]
                x = micro_batches[mb_idx].clone().requires_grad_()
            else:
                x = torch.empty(micro_batch_size, num_dim, device=device)
                p2p_recv_sync(x, prev_pp_rank)
                x.requires_grad_()

            # out: [micro_batch_size, num_dim]
            out = x
            for i in range(local_mlp_layers):
                w1_local = local_params_all_layers[2 * i] # ColumnLinear的参数
                w2_local = local_params_all_layers[2 * i + 1] # RowLinear的参数

                # 调用纯 TP 的 Autograd 函数
                # out: [micro_batch_size, num_dim]
                # w1_local: [num_dim, num_dim / tp_size], TP并行按列切分
                out = TP_ColumnLinear.apply(out, w1_local, tp_group)
                out = F.gelu(out)
                # w2_local: [num_dim/tp_size, num_dim], TP并行按行切分
                # out: [micro_batch_size, num_dim]
                out = TP_RowLinear.apply(out, w2_local, tp_group)
                # out: [micro_batch_size, num_dim]
                out = F.gelu(out)
            # end for

            if pp_rank_index == pp_size - 1:
                loss = out.square().mean()
                saved_tensors_per_micro_batch_per_layer.append((x, out, loss))
                return loss.item()
            else:
                # out: [micro_batch_size, num_dim]
                p2p_send_sync(out, next_pp_rank)
                saved_tensors_per_micro_batch_per_layer.append((x, out, None))
                return 0.0

        def backward_microbatch():
            x, out, loss = saved_tensors_per_micro_batch_per_layer.pop(0)

            if pp_rank_index == pp_size - 1:
                loss.backward()
            else:
                grad_out = torch.empty_like(out)
                p2p_recv_sync(grad_out, next_pp_rank)
                out.backward(grad_out)

            if pp_rank_index != 0:
                req = p2p_send_async(x.grad, prev_pp_rank)
                async_reqs.append(req)

        # --- 执行 1F1B 调度 ---
        for i in range(num_warmup): 
            step_loss += forward_microbatch(i)
        for i in range(num_1f1b):
            step_loss += forward_microbatch(num_warmup + i)
            backward_microbatch()
        for i in range(num_cooldown): 
            backward_microbatch()

        # 等待所有异步发送完成，防止内存泄漏
        for req in async_reqs:
            if req is not None: 
                req.wait()

        # ------------------------------------------
        # 5. 优化器更新 (ZeRO-1 核心逻辑)
        # ------------------------------------------
        for layer_param_idx, local_layer_param in enumerate(local_params_all_layers):
            if local_layer_param.grad is None: 
                continue

            # 1. 梯度除以 microbatches 数量
            # 因为 loss 是 mean，且累加了 num_microbatches 次，需要求平均
            # w1_local: [num_dim / dp_size, num_dim / tp_size], 列切分
            # w2_local: [num_dim / tp_size / dp_size, num_dim]，行切分
            grad = local_layer_param.grad.div_(num_microbatches)

            # 2. ZeRO-1: 展平参数和梯度，准备 Reduce-Scatter
            flat_grad = grad.view(-1)
            flat_param = local_layer_param.data.view(-1)

            # 计算每个 DP rank 负责的 chunk 大小
            assert flat_grad.numel() % dp_size == 0, "参数量必须能被 dp_size 整除"
            chunk_size = flat_grad.numel() // dp_size

            # 创建用于接收本地梯度 chunk 的 buffer
            local_grad_chunk = torch.empty(chunk_size, dtype=grad.dtype, device=device)

            # Reduce-Scatter: ZeRO2 将 DP 组内的梯度求平均，并打散到各个 GPU 的 local_grad_chunk 中
            dist.reduce_scatter_tensor(output=local_grad_chunk, input=flat_grad, op=dist.ReduceOp.AVG, group=dp_group)

            # 3. 提取当前 GPU 负责更新的参数 chunk (使用 clone 避免原地修改引发 NCCL 冲突)
            local_param_chunk = flat_param[dp_rank_index * chunk_size : (dp_rank_index + 1) * chunk_size].clone()

            # 4. 仅对本地 chunk 执行 AdamW 优化器更新
            if layer_param_idx not in optim_states:
                optim_states[layer_param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_grad_chunk), # 优化器状态大小仅为 1/dp_size
                    'exp_avg_sq': torch.zeros_like(local_grad_chunk)
                }

            state = optim_states[layer_param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            local_param_chunk.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad_chunk, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad_chunk, local_grad_chunk, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # 更新本地参数 chunk
            local_param_chunk.addcdiv_(exp_avg, denom, value=-step_size)

            # 5. ZeRO-1: All-Gather 将更新后的 chunk 广播回完整的参数中
            dist.all_gather_into_tensor(
                output_tensor=flat_param, 
                input_tensor=local_param_chunk, 
                group=dp_group
            )

            # 清理梯度
            local_layer_param.grad = None 

        if rank == world_size - 1:
            avg_loss = step_loss / num_microbatches
            print(f"[3D(TP+ZeRO1+1F1B) Parallel Rank:{rank}] Step {step:2d}, Loss = {avg_loss:.8f}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    data = generate_sample_data(batch_size=256, num_dim=1024)

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
        manual_tp_pp_dp_zero1_parallel, 
        world_size=8, 
        kwargs=kwargs
    )