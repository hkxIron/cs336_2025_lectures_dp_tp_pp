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
# 自定义 Autograd Function: ZeRO-3 + CPU Offload
# ==========================================
class ZeRO3OffloadLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, local_param_cpu, world_size, rank, num_dim):
        """
        前向传播：CPU参数 -> GPU -> All-Gather -> 计算 -> 释放GPU参数
        """
        device = x.device # 获取当前 GPU 设备

        # 1. 【H2D 拷贝】将局部的 1/N 参数从 CPU 内存拷贝到 GPU 显存
        # non_blocking=True 允许异步拷贝，提升 PCIe 传输效率
        local_param_gpu = local_param_cpu.to(device, non_blocking=True)

        # 2. 在 GPU 上分配临时显存，用于存放完整的参数
        full_param_gpu = torch.empty(num_dim, num_dim, dtype=local_param_gpu.dtype, device=device)

        # 3. 【GPU 通信】All-Gather: 收集所有 rank 的 local_param_gpu
        dist.all_gather_into_tensor(output_tensor=full_param_gpu, input_tensor=local_param_gpu)

        # 4. 【GPU 计算】使用完整参数进行前向计算
        out = x @ full_param_gpu

        # 5. 保存上下文
        ctx.save_for_backward(x)
        ctx.local_param_cpu = local_param_cpu # 注意：保存的是 CPU 上的参数引用！
        ctx.world_size = world_size
        ctx.num_dim = num_dim

        # 6. 【阅后即焚】函数返回后，local_param_gpu 和 full_param_gpu 会被自动释放
        # 此时 GPU 显存被清空，参数安全地躺在 CPU 内存里
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播：CPU参数 -> GPU -> All-Gather -> 算梯度 -> Reduce-Scatter -> GPU梯度 -> CPU -> 释放GPU显存
        """
        x, = ctx.saved_tensors
        local_param_cpu = ctx.local_param_cpu
        world_size = ctx.world_size
        num_dim = ctx.num_dim
        device = grad_out.device

        # 1. 【H2D 拷贝】再次将局部参数从 CPU 拷贝到 GPU
        local_param_gpu = local_param_cpu.to(device, non_blocking=True)

        # 2. 【GPU 通信】All-Gather 拼出完整参数
        full_param_gpu = torch.empty(num_dim, num_dim, dtype=local_param_gpu.dtype, device=device)
        dist.all_gather_into_tensor(output_tensor=full_param_gpu, input_tensor=local_param_gpu)

        # 3. 【GPU 计算】计算梯度
        grad_x = grad_out @ full_param_gpu.T
        grad_full_param_gpu = x.T @ grad_out

        # 4. 【阅后即焚】完整参数和局部参数用完了，立刻手动释放 GPU 显存！
        del full_param_gpu
        del local_param_gpu

        # 5. 【GPU 通信】Reduce-Scatter: 切分梯度
        local_chunk_size = num_dim // world_size
        local_grad_gpu = torch.empty(local_chunk_size, num_dim, dtype=grad_full_param_gpu.dtype, device=device)
        dist.reduce_scatter_tensor(output=local_grad_gpu, input=grad_full_param_gpu, op=dist.ReduceOp.AVG)

        # 6. 【D2H 拷贝】将切分后的 1/N 梯度从 GPU 拷贝回 CPU 内存！
        local_grad_cpu = local_grad_gpu.to('cpu', non_blocking=True)

        # 7. 【阅后即焚】释放 GPU 上的梯度
        del grad_full_param_gpu
        del local_grad_gpu

        # 返回的 local_grad_cpu 是一个 CPU Tensor
        return grad_x, local_grad_cpu, None, None, None


def zero3_offload_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    """
    **ZeRO-Offload (CPU Offload)**。

    当模型大到连 ZeRO-3 切分后的 $1/N$ 参数都塞不进 GPU 显存时（比如在单张 24GB 显卡上微调 70B 模型），我们就必须打破 GPU 显存的物理限制，向主板借内存——**将参数、梯度和优化器状态全部存放在 CPU 内存（RAM）中**。

    ### CPU Offload 的核心工作流：
    1. **平时**：所有的 $1/N$ 参数、梯度、优化器状态都在 **CPU 内存** 中。
    2. **前向/反向计算时**：通过 PCIe 总线，把需要的 $1/N$ 参数从 CPU 拷贝到 GPU（Host to Device, H2D），在 GPU 上做 All-Gather 和矩阵乘法。算完立刻把 GPU 上的参数删掉。
    3. **反向传播结束时**：GPU 上算出了 $1/N$ 的梯度，立刻通过 PCIe 拷贝回 CPU（Device to Host, D2H），然后删掉 GPU 上的梯度。
    4. **优化器更新时**：**完全在 CPU 上进行！** CPU 读取内存中的梯度和状态，更新内存中的参数。GPU 在这个阶段完全休息（或者去算下一层）。

    ### 深度解析：CPU Offload 的三大核心细节

    #### 1. 锁页内存（Pinned Memory / Page-Locked Memory）
    在初始化参数和优化器状态时，我加上了 `.pin_memory()`。这是 CPU Offload 性能的生命线。
    * **普通 CPU 内存（Paged Memory）**：操作系统可能会把它交换到硬盘（Swap）上。当 GPU 需要数据时，CPU 必须先把它拷贝到一块临时的锁页内存中，再通过 DMA（直接内存访问）传给 GPU。这多了一次 CPU 内部的拷贝，非常慢。
    * **锁页内存（Pinned Memory）**：告诉操作系统“这块内存绝对不能动”。GPU 的 DMA 控制器可以直接通过 PCIe 总线读取这块内存，**完全不需要 CPU 参与**，速度极快，且支持 `non_blocking=True` 异步传输。

    #### 2. 跨设备的数据流转 (H2D & D2H)
    在 `ZeRO3OffloadLinear` 中，你可以清晰地看到数据的流动：
    * **Forward**: `local_param_cpu` (CPU) $\xrightarrow{\text{PCIe}}$ `local_param_gpu` (GPU) $\xrightarrow{\text{All-Gather}}$ `full_param_gpu` (GPU) $\rightarrow$ 计算 $\rightarrow$ 销毁 GPU 显存。
    * **Backward**: `grad_full_param_gpu` (GPU) $\xrightarrow{\text{Reduce-Scatter}}$ `local_grad_gpu` (GPU) $\xrightarrow{\text{PCIe}}$ `local_grad_cpu` (CPU) $\rightarrow$ 销毁 GPU 显存。

    #### 3. CPU 优化器 (CPU Optimizer)
    在代码的最后一步，AdamW 的更新逻辑完全在 CPU 上运行。
    * **优点**：彻底解放了 GPU 显存。原本占用极大的 `exp_avg` 和 `exp_avg_sq` 现在只消耗廉价的 DDR 内存。
    * **缺点**：CPU 的并行计算能力远不如 GPU，计算 AdamW 可能会比较慢。
    * **工业界解法**：在真实的 DeepSpeed 中，微软用 C++ 和 AVX 指令集手写了一个高度优化的 `DeepSpeedCPUAdam`，利用 CPU 的多线程向量化指令来加速这部分计算，使得 CPU 更新参数的速度几乎能赶上 GPU。

    下面是基于 ZeRO-3 增加 **CPU Offload** 的完整实现：    
    """


    setup(rank, world_size)
    device = get_device(rank)

    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    # 数据依然在 GPU 上 (因为前向计算在 GPU)
    local_data = data[rank * local_batch_size : (rank + 1) * local_batch_size].to(device)

    # ==========================================
    # 1. ZeRO-3 Offload 参数初始化 (存放在 CPU)
    # ==========================================
    local_chunk_size = int_divide(num_dim, world_size)
    local_params: List[Parameter] = []

    for _ in range(num_layers):
        full_init_param = get_init_params(num_dim, num_dim, rank)
        chunk_start = rank * local_chunk_size
        chunk_end = chunk_start + local_chunk_size

        # 【关键优化】：使用 pin_memory() 锁页内存！
        # 锁页内存可以大幅提升 CPU 到 GPU 的 PCIe 传输速度，并允许真正的异步拷贝
        local_param_cpu = full_init_param[chunk_start:chunk_end].clone().cpu().pin_memory()
        local_params.append(Parameter(local_param_cpu))

    # ==========================================
    # 2. 优化器状态初始化 (存放在 CPU)
    # ==========================================
    optim_states = {}
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-2

    for step in range(num_steps):
        # ==========================================
        # 前向传播 (GPU 计算，参数按需从 CPU 拉取)
        # ==========================================
        x = local_data
        for local_param in local_params:
            x = ZeRO3OffloadLinear.apply(x, local_param, world_size, rank, num_dim)
            x = F.gelu(x)

        loss = x.square().mean()

        # ==========================================
        # 反向传播 (GPU 计算，梯度算完后推回 CPU)
        # ==========================================
        loss.backward()

        # ==========================================
        # 参数更新阶段 (完全在 CPU 上执行！)
        # ==========================================
        for param_idx, local_param in enumerate(local_params):
            if local_param.grad is None:
                continue

            # 此时 local_grad 是一个 CPU Tensor
            local_grad = local_param.grad

            # 初始化 CPU 上的优化器状态 (同样使用 pin_memory)
            if param_idx not in optim_states:
                optim_states[param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_param, device='cpu').pin_memory(),
                    'exp_avg_sq': torch.zeros_like(local_param, device='cpu').pin_memory()
                }

            state = optim_states[param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # [手动 AdamW 逻辑 - 纯 CPU 计算]
            # 这里的加减乘除全部由 CPU 的算术逻辑单元 (ALU) 完成，GPU 此时处于空闲状态
            local_param.data.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            local_param.data.addcdiv_(exp_avg, denom, value=-step_size)

            # 清理 CPU 上的梯度
            local_param.grad = None

        print(f"[ZeRO-3 Offload] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
              f"param_device = {local_params[0].device}", flush=True)

    cleanup()


if  __name__ == "__main__":
    data = generate_sample_data()
    spawn_wrapper(zero3_offload_main, world_size=4, data=data, num_layers=4, num_steps=20)