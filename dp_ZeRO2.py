import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
from typing import List
from dp_tp_pp import generate_sample_data, get_init_params, int_divide, get_device, get_device, setup, cleanup
from torch_util import get_device
from lecture_08_utils import spawn_wrapper, int_divide, summarize_tensor, get_init_params, render_duration

def manual_zero_stage2_gpu(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    """
    ### ZeRO-1 与 ZeRO-2 的核心区别
    * **ZeRO-1**：切分了**优化器状态**。但在反向传播（Backward）期间，每个 Rank 依然会把所有层的**完整梯度**（`param.grad`）一直保存在显存中，直到最后统一进行通信和更新。
    * **ZeRO-2**：在 ZeRO-1 的基础上，进一步切分了**梯度**。它的核心精髓是：**“阅后即焚”**。
    在反向传播时，一旦某一层计算出了完整的梯度，**立刻**进行通信（Reduce-Scatter）拿到属于自己的那 $1/N$ 梯度，然后**立刻把完整的梯度从显存中删除**！这样，显存中永远不会同时存在所有层的完整梯度，极大地节省了峰值显存（Peak Memory）。

    ### 如何在 PyTorch 中实现“阅后即焚”？
    为了在梯度计算出来的瞬间就触发通信并删除它，我们需要使用 PyTorch 的 **Hook（钩子）机制**。在 PyTorch 2.1 及以上版本中，官方提供了一个完美的 API：`register_post_accumulate_grad_hook`。

    ### 深度解析：ZeRO-2 是如何运作的？

    1. **时间线的改变（Overlap）**：
    * 在 **ZeRO-1** 中，通信发生在 `loss.backward()` **之后**。此时，所有层的完整梯度都堆积在显存里，达到了显存占用的最高峰。
    * 在 **ZeRO-2** 中，通信发生在 `loss.backward()` **期间**。假设模型有 4 层，当第 4 层的梯度算完时，Hook 立刻触发，把第 4 层的梯度切分并删掉完整的grad；接着算第 3 层，算完立刻切分并删掉…… 这样，显存中最多只存在**当前正在计算的那一层**的完整梯度。

    2. **`register_post_accumulate_grad_hook` 的妙用**：
    这是 PyTorch 专门为大模型分布式训练（如 FSDP）引入的底层 API。它会在 Autograd 引擎把某一个参数的梯度累加到 `param.grad` 完毕后的**瞬间**被调用。我们在里面执行 `p.grad = None`，就相当于从 C++ 底层直接释放了这块显存。

    3. **通信量有变化吗？**
    **没有变化。** ZeRO-2 的通信量和 ZeRO-1 完全一样（依然是 $1 \times \text{模型参数量}$）。它只是把原本集中在反向传播之后的通信，**打散**到了反向传播的过程中。这种打散不仅节省了显存，还能让**通信和计算重叠（Overlap）**，在真实的 GPU 环境下反而能提升训练速度！
    """

    setup(rank, world_size)
    print(f"[Rank:{rank}]Zero stage 2 (GPU)")
    device = get_device(rank)

    # ==========================================
    # 1. 数据准备
    # ==========================================
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    local_data = data[start_index:end_index].to(device)

    # ==========================================
    # 2. 模型参数初始化
    # ==========================================
    params: List[Parameter] = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
    local_chunk_size = int_divide(num_dim, world_size)

    # ==========================================
    # 3. ZeRO-2 核心：注册反向传播 Hook (阅后即焚)
    # ==========================================
    for param in params:
        def reduce_scatter_and_free_grad_hook(p: Parameter):
            # NOTE: 每算完一层grad，立即触发! 算完层 layer_idx 梯度后立即触发 -> ReduceScatter -> 删掉层 layer_idx 的梯度
            if p.grad is None:
                return

            # 1. 分配一块极小的显存，用于接收属于当前 rank 的那 1/world_size 的梯度, 放在param的device上
            # shape: [local_chunk_size=num_dim/world_size, num_dim]
            local_grad = torch.empty(local_chunk_size, num_dim, dtype=p.grad.dtype, device=p.grad.device)

            # 2. 【纯 GPU 高性能 API】Reduce-Scatter Tensor
            # p.grad: [num_dim, num_dim], p.local_grad: [num_dim/world_size, num_dim]
            # 它会自动将 p.grad 沿 dim=0 切分成 world_size 份，求平均后，把属于当前 rank 的那份直接写入 local_grad
            dist.reduce_scatter_tensor(output=local_grad, input=p.grad, op=dist.ReduceOp.AVG)

            # 3. 保存局部梯度
            # shape: [local_chunk_size=num_dim/world_size, num_dim]
            p.local_grad = local_grad

            # 4. 【ZeRO-2 的灵魂】立刻释放完整的梯度内存！妙！
            p.grad = None 

        # 注册 Hook (PyTorch >= 2.1)
        if hasattr(param, 'register_post_accumulate_grad_hook'):
            param.register_post_accumulate_grad_hook(reduce_scatter_and_free_grad_hook)
        else:
            raise RuntimeError("ZeRO-2 requires PyTorch >= 2.1")

    # ==========================================
    # 4. 优化器状态初始化
    # ==========================================
    optim_states = {}
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-2

    for step in range(num_steps):
        # 前向传播
        # x: [local_batch_size, num_dim]
        x = local_data
        for param in params:
            # x: [local_batch_size, num_dim]
            x = x @ param
            x = F.gelu(x)

        loss = x.square().mean()

        # ==========================================
        # 反向传播
        # ==========================================
        # backward() 执行期间，Hook 会被逐层触发。
        # 算完一层 -> ReduceScatter -> 删掉完整梯度。
        # 显存峰值被完美压平！
        loss.backward()

        # ==========================================
        # 参数更新阶段
        # ==========================================
        for param_idx, param in enumerate(params):
            if not hasattr(param, 'local_grad') or param.local_grad is None:
                continue
            # NOTE:现在使用的是局部梯度param.local_grad，而不是完整的梯度param.grad
            local_grad = param.local_grad

            # 提取当前 rank 负责更新的那部分参数的【视图 View】
            # 注意：这里没有使用 clone()，我们直接在原参数的内存地址上进行原地修改！
            chunk_start = rank * local_chunk_size
            chunk_end = chunk_start + local_chunk_size
            local_param = param.data[chunk_start:chunk_end]

            # 初始化局部优化器状态 (仅占 1/world_size 显存)
            if param_idx not in optim_states:
                optim_states[param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_param, device=local_param.device),
                    'exp_avg_sq': torch.zeros_like(local_param, device=local_param.device)
                }

            state = optim_states[param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # [手动 AdamW 逻辑 - 全部是 In-place 原地操作]
            local_param.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # 更新局部参数 (此时 param.data 中只有当前 rank 负责的那一块被更新了，其他块是旧的)
            local_param.addcdiv_(exp_avg, denom, value=-step_size)

            # --- 【纯 GPU 高性能 API】All-Gather Into Tensor ---
            # 将所有 rank 更新好的 local_param 收集起来，直接按顺序拼接到完整的 param.data 中
            # 零额外内存分配，直接覆盖原显存！
            # all_gather_into_tensor 直接把完整的 param.data 作为接收容器。NCCL 会自动把 Rank 0 的 local_param 塞进第一块，Rank 1 的塞进第二块…… 一步到位，没有任何中间变量。
            # 相对于 all_gather， all_gather_into_tensor 可以节省一次内存分配！
            # NOTE: all_gather_into_tensor 更高效，但要求所有进程的输入张量形状完全相同；而 all_gather 更灵活，可以处理形状不一致的情况，但会输出一个张量列表。
            # local_param: [local_chunk_size=num_dim/world_size, num_dim]
            # => param: [num_dim, num_dim]
            dist.all_gather_into_tensor(output_tensor=param.data, input_tensor=local_param)

            # 清理 local_grad，为下一步做准备
            param.local_grad = None
        # end of param_idx

        # 打印验证
        print(f"[ZeRO-stage2 GPU] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
              f"params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)
    # end of step

    cleanup()

if  __name__ == "__main__":
    data = generate_sample_data()
    spawn_wrapper(manual_zero_stage2_gpu, world_size=4, data=data, num_layers=4, num_steps=20)