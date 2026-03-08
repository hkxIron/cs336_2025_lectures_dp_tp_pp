import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
from typing import List
from dp_tp_pp import generate_sample_data, get_init_params, int_divide, get_device, get_device, setup, cleanup
from torch_util import get_device
from lecture_08_utils import spawn_wrapper, int_divide, summarize_tensor, get_init_params, render_duration

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter
from typing import List

# ==========================================
# 自定义 Autograd Function: ZeRO-3 的核心引擎
# ==========================================
class ZeRO3Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param, world_size, rank, num_dim):
        """
        前向传播：Gather 参数 -> 计算 -> 释放参数
        """
        # 1. 分配临时显存，用于存放完整的参数
        full_param = torch.empty(num_dim, num_dim, dtype=local_param.dtype, device=local_param.device)

        # 2. 【通信】All-Gather: 收集所有 rank 的 local_param 拼成 full_param
        dist.all_gather_into_tensor(output_tensor=full_param, input_tensor=local_param)

        # 3. 【计算】使用完整参数进行前向计算
        out = x @ full_param

        # 4. 保存反向传播需要的上下文
        # 注意：我们只保存输入 x，绝对不保存 full_param！
        ctx.save_for_backward(x)
        ctx.local_param = local_param  # 保存局部参数的引用
        ctx.world_size = world_size
        ctx.num_dim = num_dim

        # 5. 【阅后即焚】full_param 在函数返回后会被 Python 垃圾回收机制自动释放！
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播：Gather 参数 -> 算梯度 -> 释放参数 -> Reduce-Scatter 梯度 -> 释放完整梯度
        """
        x, = ctx.saved_tensors
        local_param = ctx.local_param
        world_size = ctx.world_size
        num_dim = ctx.num_dim

        # 1. 【通信】再次 All-Gather: 反向传播也需要完整的参数
        full_param = torch.empty(num_dim, num_dim, dtype=local_param.dtype, device=local_param.device)
        dist.all_gather_into_tensor(output_tensor=full_param, input_tensor=local_param)

        # 2. 【计算】计算对输入 x 的梯度 (传给上一层)
        grad_x = grad_out @ full_param.T

        # 3. 【计算】计算对完整参数的梯度
        grad_full_param = x.T @ grad_out

        # 4. 【阅后即焚】完整参数用完了，立刻手动释放！
        del full_param 

        # 5. 【通信】Reduce-Scatter: 把完整的参数梯度切分，每个 rank 只拿 1/world_size
        local_chunk_size = num_dim // world_size
        local_grad = torch.empty(local_chunk_size, num_dim, dtype=grad_full_param.dtype, device=grad_full_param.device)
        dist.reduce_scatter_tensor(output=local_grad, input=grad_full_param, op=dist.ReduceOp.AVG)

        # 6. 【阅后即焚】完整的梯度用完了，立刻释放！(函数返回后 grad_full_param 被销毁)

        # 返回的梯度顺序必须与 forward 的输入参数顺序一致
        # x 的梯度是 grad_x, local_param 的梯度是 local_grad
        return grad_x, local_grad, None, None, None


def manual_zero_stage3_gpu(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    """
    **ZeRO Stage 3（参数切分）** 是 DeepSpeed 和 PyTorch FSDP（Fully Sharded Data Parallel）的核心灵魂。
    在 ZeRO-1 和 ZeRO-2 中，每个 GPU 依然在显存中保留了**完整的模型参数**。当模型达到千亿参数（如 GPT-3 175B，仅参数就占 350GB）时，单张 GPU 连参数都放不下，更别提训练了。

    **ZeRO-3 的核心思想是：**
    1. **彻底切分**：每个 GPU 平时只保存 $1/N$ 的参数、$1/N$ 的梯度、$1/N$ 的优化器状态。
    2. **按需组装（Gather）**：当要计算某一层的前向或反向时，临时通过 `All-Gather` 把这层的完整参数拼起来。
    3. **阅后即焚（Free）**：这层计算完，**立刻**把拼出来的完整参数（以及完整梯度）删掉，只保留属于自己的那 $1/N$。

    ### 如何在 PyTorch 中实现“阅后即焚”的参数？
    在标准的 PyTorch 中，前向传播用到的权重会被 Autograd 引擎**自动保存在显存中**，留给反向传播用。为了打破这个机制，实现真正的 ZeRO-3，我们必须自定义一个 `torch.autograd.Function`，手动控制参数的拉取和释放。

    ### 深度解析：ZeRO-3 的终极奥义

    1. **显存的极致压缩**：
    在代码的初始化阶段，`local_param` 的 shape 直接就是 `[local_chunk_size, num_dim]`。
    这意味着：**模型参数、梯度、优化器状态，三者的常驻显存全部被除以了 `world_size`！**
    如果用 8 张 GPU 训练，单卡的静态显存占用直接降为原来的 1/8。

    2. **通信与计算的交响乐 (Custom Autograd Function)**：
    * **前向传播**：算第 $i$ 层时，拉取第 $i$ 层的参数，算完立刻扔掉。
    * **反向传播**：算第 $i$ 层时，再次拉取第 $i$ 层的参数，算出梯度后扔掉参数；接着把梯度 Reduce-Scatter 切分，扔掉完整梯度。
    * **优化器更新**：**零通信！** 因为每个 GPU 手里只有 $1/N$ 的参数和 $1/N$ 的梯度，大家各自闭门更新自己的那一小块即可。不需要像 ZeRO-1/2 那样在最后做 All-Gather 广播参数。

    3. **ZeRO-3 的代价（Trade-off）**：
    天下没有免费的午餐。相比于 ZeRO-1 和 ZeRO-2，ZeRO-3 的通信量增加了 **50%**。
    * ZeRO-1/2 的单层通信量：反向传播的 `Reduce-Scatter` (0.5x) + 更新后的 `All-Gather` (0.5x) = **1x 参数量**。
    * ZeRO-3 的单层通信量：前向传播的 `All-Gather` (0.5x) + 反向传播的 `All-Gather` (0.5x) + 反向传播的 `Reduce-Scatter` (0.5x) = **1.5x 参数量**。
    
    **结论**：ZeRO-3 是用 **1.5 倍的网络带宽**，换取了 **N 倍的显存空间**。在千亿级大模型训练中，显存是绝对的瓶颈，因此这个交易是非常划算的！


    下面是纯 GPU (NCCL) 环境下的 **ZeRO-3** 完整实现：
    """


    setup(rank, world_size)
    print(f"[Rank:{rank}]Zero stage 3 (GPU)")
    device = get_device(rank)

    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    local_data = data[rank * local_batch_size : (rank + 1) * local_batch_size].to(device)

    # ==========================================
    # 1. ZeRO-3 模型参数初始化 (彻底切分)
    # ==========================================
    local_chunk_size = int_divide(num_dim, world_size)
    local_params: List[Parameter] = []

    for _ in range(num_layers):
        # 假设 get_init_params 返回的是全局一致的完整初始权重
        full_init_param = get_init_params(num_dim, num_dim, rank)

        # 每个 rank 只截取属于自己的那 1/world_size 作为真正的 Parameter！
        chunk_start = rank * local_chunk_size
        chunk_end = chunk_start + local_chunk_size

        # 显存占用从一开始就只有 1/world_size
        local_param = Parameter(full_init_param[chunk_start:chunk_end].clone().to(device))
        local_params.append(local_param)

    # ==========================================
    # 2. 优化器状态初始化
    # ==========================================
    optim_states = {}
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-2

    for step in range(num_steps):
        # ==========================================
        # 前向传播 (调用自定义的 ZeRO3Linear)
        # ==========================================
        x = local_data
        for local_param in local_params:
            # 魔法发生在这里：自动 Gather -> 计算 -> 释放
            x = ZeRO3Linear.apply(x, local_param, world_size, rank, num_dim)
            x = F.gelu(x)

        loss = x.square().mean()

        # ==========================================
        # 反向传播
        # ==========================================
        # 魔法再次发生：自动 Gather -> 算梯度 -> 释放 -> ReduceScatter -> 释放
        # 最终，每个 local_param.grad 中只包含 1/world_size 的局部梯度！
        loss.backward()

        # ==========================================
        # 参数更新阶段 (完全没有通信！)
        # ==========================================
        for param_idx, local_param in enumerate(local_params):
            if local_param.grad is None:
                continue

            local_grad = local_param.grad

            # 初始化局部优化器状态
            if param_idx not in optim_states:
                optim_states[param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_param),
                    'exp_avg_sq': torch.zeros_like(local_param)
                }

            state = optim_states[param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # [手动 AdamW 逻辑]
            # 注意：这里所有的操作都是在 1/world_size 的 local_param 上进行的！
            local_param.data.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            local_param.data.addcdiv_(exp_avg, denom, value=-step_size)

            # 清理梯度
            local_param.grad = None

        # 打印验证 (注意：这里打印的 params 已经是切片后的 local_param 了)
        print(f"[ZeRO-3 GPU] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
              f"local_params_shape = {local_params[0].shape}", flush=True)

    cleanup()

if  __name__ == "__main__":
    data = generate_sample_data()
    spawn_wrapper(manual_zero_stage3_gpu, world_size=4, data=data, num_layers=4, num_steps=20)