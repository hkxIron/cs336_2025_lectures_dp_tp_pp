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
在上一版代码中，虽然我们在优化器更新阶段使用了 `reduce_scatter`，但因为我们是**在所有 micro-batch 的反向传播全部结束之后**，才去遍历参数做 `reduce_scatter`，这意味着在整个反向传播期间，**完整的梯度 (`param.grad`) 一直驻留在显存中**。

从显存占用的角度来看，上一版代码确实退化成了 **ZeRO-1**（只切分了优化器状态，没有真正切分梯度显存）。

要实现真正的 **ZeRO-2**，核心奥义是：**一旦某个参数的梯度计算完毕，立刻触发 `Reduce-Scatter`，并将本地的完整梯度显存释放掉（设为 None），只保留属于自己的那一小块梯度切片。**

为了实现这一点，你提到的 **`register_post_accumulate_grad_hook`** 是 PyTorch 官方提供的最完美、最标准的解决方案！

### 引入 Hook 后的 ZeRO-2 核心逻辑变化：

由于我们结合了 **Pipeline Parallelism (PP)**，有多个 micro-batch，所以一个参数的梯度会被累加 `num_microbatches` 次。因此，Hook 的逻辑必须是：
1. 每次反向传播触发 Hook 时，计数器 +1。
2. 当计数器达到 `num_microbatches` 时（说明该参数在当前 Step 的梯度已经全部累加完毕）：
   - 对梯度求平均。
   - 执行 `Reduce-Scatter`，将切片存入本地 buffer。
   - **立刻释放完整的 `param.grad` 显存！**
   - 计数器清零。

### 核心改动深度解析：

1. **`register_post_accumulate_grad_hook` 的妙用**：
   - 这个 Hook 会在 PyTorch Autograd 引擎计算完某个参数的梯度并累加到 `.grad` 后**立即触发**。
   - 因为我们有 Pipeline Parallelism，一个参数会被反向传播 `num_microbatches` 次。所以我们引入了 `grad_accum_counters`。
   - 只有当计数器达到 `num_microbatches` 时，才执行 `Reduce-Scatter`。

2. **如何真正释放显存？**
   - 在 Hook 的最后，我们 `return None`。
   - 在 PyTorch 的机制中，如果 Hook 返回一个 Tensor，它会替换掉原来的 `.grad`；**如果返回 `None`，PyTorch 会直接丢弃这个梯度，使得 `param.grad = None`**。
   - 这一步极其关键！它保证了在反向传播的过程中，一旦某层的梯度通信完毕，完整的梯度显存立刻被回收，从而实现了真正的 ZeRO-2 显存峰值降低。

3. **优化器逻辑的解耦**：
   - 在最后的优化器更新循环中，我们不再去读取 `local_layer_param.grad`（因为它已经是 None 了）。
   - 我们直接从 `local_grad_chunks` 字典中取出 Hook 帮我们准备好的、大小只有 `1/dp_size` 的梯度切片，进行 AdamW 更新。

通过这种方式，不仅实现了真正的显存分片，而且因为 Hook 是在反向传播过程中逐层触发的，**通信（Reduce-Scatter）和计算（前一层的反向传播）在底层实现了完美的重叠（Overlap）**，这也是工业界（如 DeepSpeed, FSDP）提升训练吞吐量的核心技巧！
下面是修改后的完整代码，真正实现了 **ZeRO-2 的梯度显存分片**与**通信/计算重叠**。

"""

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
# 自定义 Autograd Function: 纯 TP (保持不变)
# ==========================================
class TP_ColumnLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, tp_group):
        out = x @ local_param
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_x_partial = grad_out @ ctx.local_param.T
        dist.all_reduce(grad_x_partial, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        grad_local_param = x.T @ grad_out
        return grad_x_partial, grad_local_param, None

class TP_RowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, tp_group):
        out_partial = x @ local_param
        dist.all_reduce(out_partial, op=dist.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(x)
        ctx.local_param = local_param
        ctx.tp_group = tp_group
        return out_partial

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_x = grad_out @ ctx.local_param.T
        grad_local_param = x.T @ grad_out
        return grad_x, grad_local_param, None

# ==========================================
# P2P 通信辅助函数 (保持不变)
# ==========================================
def p2p_send_sync(tensor, dst_rank):
    if dst_rank is not None: dist.send(tensor, dst_rank)

def p2p_recv_sync(tensor, src_rank):
    if src_rank is not None: dist.recv(tensor, src_rank)

def p2p_send_async(tensor, dst_rank):
    if dst_rank is not None: return dist.isend(tensor, dst_rank)
    return None

def p2p_recv_async(tensor, src_rank):
    if src_rank is not None: return dist.irecv(tensor, src_rank)
    return None

# ==========================================
# 主训练逻辑: 3D Parallelism (PP + 真正的 ZeRO-2 DP + TP) + 1F1B
# ==========================================
def manual_tp_pp_dp_zero2_parallel(rank: int, world_size: int, kwargs: dict):
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

    # 1. 3D 拓扑与通信组初始化
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

    # 2. 数据切分
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // dp_size
    micro_batch_size = local_batch_size // num_microbatches

    local_data = data[dp_rank_index * local_batch_size : (dp_rank_index + 1) * local_batch_size].to(device)
    micro_batches: Tuple[torch.Tensor] = local_data.split(micro_batch_size)

    # 3. 模型参数初始化
    local_params_all_layers: List[Parameter] = []
    num_mlp_layers = int_divide(num_layers, 2)
    local_mlp_layers = int_divide(num_mlp_layers, pp_size)

    for layer_idx in range(local_mlp_layers):
        full_w1 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_out = num_dim // tp_size
        w1_tp = full_w1[:, tp_rank_index * tp_chunk_out : (tp_rank_index + 1) * tp_chunk_out].clone()
        local_params_all_layers.append(Parameter(w1_tp))

        full_w2 = get_init_params(num_dim, num_dim, rank)
        tp_chunk_in = num_dim // tp_size
        w2_tp = full_w2[tp_rank_index * tp_chunk_in : (tp_rank_index + 1) * tp_chunk_in, :].clone()
        local_params_all_layers.append(Parameter(w2_tp))

    # ==========================================
    # 核心修改：真正的 ZeRO-2 梯度 Hook 注册
    # ==========================================
    # 用于记录每个参数累加了多少个 micro-batch 的梯度
    grad_accum_counters = {id(p): 0 for p in local_params_all_layers}
    # ZeRO2: 用于存放 Reduce-Scatter 之后的局部梯度切片 (1/dp_size)
    local_grad_chunks = {}

    def get_zero2_grad_hook(param:torch.Tensor, param_index:int):
        if rank==0:
            print(f"Registering ZeRO-2 Hook for param_idx: {param_index} at rank:{rank}")

        # 注意：这里的入参 p 是 Parameter 本身，而不是梯度！
        def hook(p):
            param_id = id(p)
            grad_accum_counters[param_id] += 1

            # 只有当batch内所有 micro-batch 都反向传播完，才进行梯度的reduce_scatter通信, 然后将完整的梯度显存释放掉
            if grad_accum_counters[param_id] == num_microbatches:
                # 1. 从 parameter 中取出累加好的完整梯度
                grad = p.grad

                # 2. 求平均 (注意：是对 grad 求平均，千万别对 p 求平均)
                grad.div_(num_microbatches)

                # 3. 准备 Reduce-Scatter
                flat_grad = grad.view(-1)
                chunk_size = flat_grad.numel() // dp_size

                if param_id not in local_grad_chunks:
                    local_grad_chunks[param_id] = torch.empty(chunk_size, dtype=grad.dtype, device=device)

                # 4. 执行 Reduce-Scatter, 阅后即焚, 只保留 1/dp_size 的梯度切片
                dist.reduce_scatter_tensor(
                    output=local_grad_chunks[param_id], # 只保留 1/dp_size 的梯度切片
                    input=flat_grad, 
                    op=dist.ReduceOp.AVG, 
                    group=dp_group
                )

                # 5. 计数器清零
                grad_accum_counters[param_id] = 0

                # 6. 核心奥义：显式地将梯度设为 None，真正释放 (dp_size - 1)/dp_size 的显存！
                p.grad = None 
            # end if
        # end hook

        return hook


    # 为所有参数注册 Hook
    for p_idx, p in enumerate(local_params_all_layers):
        p.register_post_accumulate_grad_hook(get_zero2_grad_hook(p, p_idx))

    optim_states = {}
    lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 1e-2

    # 4. 训练循环 (1F1B 调度)
    for step in range(num_steps):
        saved_tensors_per_micro_batch_per_layer = []
        step_loss = 0.0
        async_reqs = []

        num_warmup = min(num_microbatches, pp_size - pp_rank_index - 1)
        num_1f1b = num_microbatches - num_warmup
        num_cooldown = num_warmup

        def forward_microbatch(mb_idx):
            if pp_rank_index == 0:
                x = micro_batches[mb_idx].clone().requires_grad_()
            else:
                x = torch.empty(micro_batch_size, num_dim, device=device)
                p2p_recv_sync(x, prev_pp_rank)
                x.requires_grad_()

            out = x
            for i in range(local_mlp_layers):
                w1_local = local_params_all_layers[2 * i]
                w2_local = local_params_all_layers[2 * i + 1]

                out = TP_ColumnLinear.apply(out, w1_local, tp_group)
                out = F.gelu(out)
                out = TP_RowLinear.apply(out, w2_local, tp_group)
                out = F.gelu(out)

            if pp_rank_index == pp_size - 1:
                loss = out.square().mean()
                saved_tensors_per_micro_batch_per_layer.append((x, out, loss))
                return loss.item()
            else:
                p2p_send_sync(out, next_pp_rank)
                saved_tensors_per_micro_batch_per_layer.append((x, out, None))
                return 0.0

        def backward_microbatch():
            x, out, loss = saved_tensors_per_micro_batch_per_layer.pop(0)

            if pp_rank_index == pp_size - 1:
                # 这里调用 backward 时，底层会自动触发我们注册的 Hook
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

        for req in async_reqs:
            if req is not None: 
                req.wait()

        # ==========================================
        # 5. 优化器更新 (基于 Hook 提取的局部梯度)
        # ==========================================
        for layer_param_idx, local_layer_param in enumerate(local_params_all_layers):
            pid = id(local_layer_param)

            # 注意：此时 local_layer_param.grad 已经是 None 了（被 Hook 释放了）
            # 我们直接使用 Hook 中保存的 local_grad_chunk
            if pid not in local_grad_chunks:
                continue

            local_grad_chunk = local_grad_chunks[pid]
            flat_param = local_layer_param.data.view(-1)
            chunk_size = local_grad_chunk.numel()

            # 提取当前 GPU 负责更新的参数 chunk
            local_param_chunk = flat_param[dp_rank_index * chunk_size : (dp_rank_index + 1) * chunk_size].clone()

            # 仅对本地 chunk 执行 AdamW 优化器更新
            if layer_param_idx not in optim_states:
                optim_states[layer_param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_grad_chunk),
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

            local_param_chunk.addcdiv_(exp_avg, denom, value=-step_size)

            # ZeRO-2: All-Gather 将更新后的 chunk 广播回完整的参数中
            dist.all_gather_into_tensor(
                output_tensor=flat_param, 
                input_tensor=local_param_chunk, 
                group=dp_group
            )

        if rank == world_size - 1:
            avg_loss = step_loss / num_microbatches
            print(f"[3D(TP+ZeRO2+1F1B) Parallel Rank:{rank}] Step {step:2d}, Loss = {avg_loss:.8f}")

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
    spawn_wrapper(manual_tp_pp_dp_zero2_parallel, world_size=8, kwargs=kwargs)
