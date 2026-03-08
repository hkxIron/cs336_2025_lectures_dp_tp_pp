import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
from typing import List
from dp_tp_pp import generate_sample_data, get_init_params, int_divide, get_device, get_device, setup, cleanup
from torch_util import get_device
from lecture_08_utils import spawn_wrapper, int_divide, summarize_tensor, get_init_params, render_duration

def manual_zero_stage1_gpu(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    """
    ZeRO stage 1原理:
    Step 1. Everyone computes a full gradient on their subset of the batch
    Step 2. ReduceScatter the gradients – incur #params communication cost
    Step 3. Each machine updates their param using their gradient + state.
    Step 4. All Gather the parameters – incur #params communication cost
    解释：
    1.每个rank使用reduce_scatter只处理1/rank的梯度的平均值，然后计算adamw的一阶/二阶冲量（optimizer state），然后仅更新1/rank的本机权重参数
    2.每个rank将本地更新后的参数使用all gather同步到其它rank，至此每个rank都拥有完整的最新参数,注意：未发送optimizer state,每个rank只有一部分optimizer state

    在标准 DP 中，每个 Rank 都保存了**完整**的优化器状态（AdamW 的一阶矩和二阶矩），这会消耗大量的显存（通常是模型参数本身的 2 倍）。
    **ZeRO-1 的核心思想是**：既然每个 Rank 计算出的梯度经过 All-Reduce 后都是一样的，大家都在做完全相同的参数更新计算，那不如**让每个 Rank 只负责更新 $1/N$ 的参数**，并且只保存这 $1/N$ 参数对应的优化器状态。更新完后，大家再把各自更新好的那部分参数广播（All-Gather）给所有人。

    ### 代码原理解析 (对比标准 DP)

    1. **显存优化的核心体现**：
    在标准 DP 中，如果你调用 `torch.optim.AdamW(params)`，PyTorch 会为每一个 `[num_dim, num_dim]` 的参数创建同样大小的 `exp_avg` 和 `exp_avg_sq`。
    在上面的 ZeRO-1 代码中，看这一行：
    ```python
    'exp_avg': torch.zeros_like(local_param)
    ```
    `local_param` 的大小只有 `[num_dim / world_size, num_dim]`。这意味着**优化器状态的显存占用被完美地除以了 `world_size`**。

    2. **通信逻辑的转变**：
    * **标准 DP**：`dist.all_reduce(param.grad)`。通信量是 $1 \times \text{模型参数量}$。
    * **ZeRO-1**：
        * `dist.reduce_scatter(local_grad, grad_chunks)`：通信量是 $0.5 \times \text{模型参数量}$。
        * `dist.all_gather(gathered_params, local_param)`：通信量是 $0.5 \times \text{模型参数量}$。
        * **总通信量依然是 $1 \times \text{模型参数量}$**！ZeRO-1 在**不增加任何通信开销**的前提下，把优化器状态的显存给切分了，这就是它被称为“免费午餐”的原因。

    3. **手动 AdamW 的无缝接入**：
    我们直接把之前手写的 AdamW 逻辑嵌入到了 `Step 3` 中。注意，所有的数学运算（`mul_`, `add_`, `addcmul_`, `addcdiv_`）都是在 `local_param` 和 `local_grad` 上进行的，这保证了计算量也被均摊到了各个 GPU 上。


    addcmul_:
    add   +    c   +    mul   +    _
    ↑         ↑         ↑         ↑
    加法      常数      乘法      原地操作
    (add)   (constant) (multiply) (in-place)

    # addcmul_ 示例
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([2.0, 2.0, 2.0])
    tensor2 = torch.tensor([3.0, 3.0, 3.0])

    # tensor = tensor + 2 * tensor1 * tensor2
    tensor.addcmul_(tensor1, tensor2, value=2.0)
    # 结果: [1+2*2*3, 2+2*2*3, 3+2*2*3] = [13, 14, 15]


    addcdiv_:
    add   +    c   +    div   +    _
    ↑         ↑         ↑         ↑
    加法      常数      除法      原地操作
    (add)   (constant) (division) (in-place)

    完整含义：add constant divided → 加上常数除以
    数学公式：output = input + value * tensor1 / tensor2

    # addcdiv_ 示例
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([6.0, 6.0, 6.0])
    tensor2 = torch.tensor([2.0, 3.0, 4.0])

    # tensor = tensor + 2 * tensor1 / tensor2
    tensor.addcdiv_(tensor1, tensor2, value=2.0)
    # 结果: [1+2*6/2, 2+2*6/3, 3+2*6/4] = [7.0, 6.0, 6.0]

    NOTE:需要使用GPU来运行，PyTorch 的 Gloo 后端不支持 reduce_scatter 这个算子。 （reduce_scatter 通常只有在 GPU 环境下使用 NCCL 后端时才被完全支持）。
    """
    print(f"[Rank:{rank}]Zero stage 1 (GPU)")
    setup(rank, world_size)
    device = get_device(rank)

    # ==========================================
    # 1. 数据准备 (Data Parallelism)
    # ==========================================
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    # 每个 rank 只获取属于自己的那部分数据
    local_data = data[start_index:end_index].to(device)

    # ==========================================
    # 2. 模型参数初始化
    # ==========================================
    # 每个 rank 依然需要完整的模型参数来进行前向和反向传播
    params: List[Parameter] = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]

    # ==========================================
    # 3. ZeRO-1 优化器状态初始化 (核心改变)
    # ==========================================
    # 我们不再使用 torch.optim.AdamW，而是手动维护局部的优化器状态
    # 每个 rank 只维护 1/world_size 的参数对应的 exp_avg 和 exp_avg_sq
    local_chunk_size = int_divide(num_dim, world_size)
    optim_states = {}

    # AdamW 超参数
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-2

    for step in range(num_steps):
        # ==========================================
        # Step 1: Everyone computes a full gradient on their subset of the batch
        # ==========================================
        # 前向传播
        x = local_data
        for param in params:
            x = x @ param
            x = F.gelu(x)

        loss = x.square().mean()

        # 反向传播，计算出完整的梯度 param.grad
        loss.backward()

        # ==========================================
        # ZeRO-1 通信与参数更新阶段
        # ==========================================
        for param_idx, param in enumerate(params):
            # 确保梯度存在
            if param.grad is None:
                continue

            # --- Step 2: ReduceScatter the gradients ---
            # 将完整的梯度 [num_dim, num_dim] 沿 dim=0 切分成 world_size 份
            grad_chunks: List[torch.Tensor] = list(torch.chunk(param.grad, world_size, dim=0))
            # 必须保证内存连续，以便进行底层通信
            grad_chunks = [chunk.contiguous() for chunk in grad_chunks]

            # 准备接收局部梯度的 buffer
            local_grad = torch.empty_like(grad_chunks[0])

            # ReduceScatter: 将所有 rank 的 grad_chunks 对应位置相加求平均，
            # 然后 rank i 只拿走第 i 个 chunk 存入 local_grad
            dist.reduce_scatter(output=local_grad, input_list=grad_chunks, op=dist.ReduceOp.AVG)

            # --- Step 3: Each machine updates their param using their gradient + state ---
            # 提取当前 rank 负责更新的那部分参数 (克隆出来以避免影响原计算图)
            chunk_start = rank * local_chunk_size
            chunk_end = chunk_start + local_chunk_size
            local_param = param.data[chunk_start:chunk_end].clone()

            # 初始化局部优化器状态 (仅在第一步执行)
            if param_idx not in optim_states:
                optim_states[param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_param),    # 显存占用仅为原来的 1/world_size!
                    'exp_avg_sq': torch.zeros_like(local_param)  # 显存占用仅为原来的 1/world_size!
                }

            state = optim_states[param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # [手动 AdamW 逻辑]
            # 1. 解耦的权重衰减 (Decoupled Weight Decay)
            local_param.mul_(1 - lr * weight_decay)

            # 2. 更新一阶和二阶矩
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            # 3. 偏差校正
            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # 4. 更新局部参数
            local_param.addcdiv_(exp_avg, denom, value=-step_size)

            # --- Step 4: All Gather the parameters ---
            # 准备接收所有 rank 更新后的局部参数的 buffer
            gathered_params: List[torch.Tensor] = [torch.empty_like(local_param) for _ in range(world_size)]

            # AllGather: 将每个 rank 的 local_param 收集起来，填入 gathered_params 列表
            dist.all_gather(tensor_list=gathered_params, tensor=local_param)

            # 将收集到的更新后的参数拼接起来，覆盖回原始的完整参数中
            param.data.copy_(torch.cat(gathered_params, dim=0))

            # 清空梯度，为下一步做准备
            param.grad = None

        # 打印验证
        print(f"[ZeRO-stage1] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
              f"params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()

if  __name__ == "__main__":
    data = generate_sample_data()
    spawn_wrapper(manual_zero_stage1_gpu, world_size=4, data=data, num_layers=4, num_steps=20)