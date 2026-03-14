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

        注意：
            - 前向传播与反向传播都是在 GPU 上进行
            - 数据放在 GPU 上
            - 但参数和梯度都放在 CPU 内存里
        """

        # NOTE: 数据放在GPU上
        device = x.device # 获取当前 GPU 设备

        # 1. 【H2D 拷贝】将局部的 1/N 参数从 CPU 内存拷贝到 GPU 显存
        # non_blocking=True 允许异步拷贝，提升 PCIe 传输效率
        local_param_gpu = local_param_cpu.to(device, non_blocking=True)

        # 2. 在 GPU 上分配临时显存，用于存放完整的参数
        full_param_gpu = torch.empty(num_dim, num_dim, dtype=local_param_gpu.dtype, device=device)

        # 3. 【GPU 通信】All-Gather: 收集所有 rank 的 local_param_gpu
        # local_param_gpu:[num_dim/word_size, num_dim]
        # full_param_gpu:[num_dim, num_dim]
        dist.all_gather_into_tensor(output_tensor=full_param_gpu, input_tensor=local_param_gpu)

        # 4. 【GPU 计算】使用完整参数进行前向计算
        # x: [local_batch_size, num_dim]
        # full_param_gpu:[num_dim, num_dim]
        # out:[local_batch_size, num_dim]
        out = x @ full_param_gpu

        # 5. 保存上下文
        """
        1. 如果要保存张量且该张量在反向传播中只作为数值使用（不计算其梯度） → 使用 save_for_backward()
        2. 如果要保存张量且该张量需要自己的梯度 → 让它作为函数forward的输入/输出，让PyTorch自动处理
        3. 非张量数据 → 直接保存为 ctx 的属性

        不能使用ctx.x = x, 在 backward 中，ctx.x 会包含梯度信息
        当你计算 grad_out @ full_param_gpu.T 时，PyTorch 会认为 ctx.x 也需要梯度
        这会导致不必要的梯度计算和内存占用
        """
        ctx.save_for_backward(x) # 保存输入x，但不追踪其梯度
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

        矩阵乘法的梯度推导：
        Y=X@W
        dL/dW = X.T @ dL/dY
        dL/dX = dL/dY @ W.T
        """
        x, = ctx.saved_tensors # 取出ctx.save_for_backward中的值
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
        x_grad_gpu = grad_out @ full_param_gpu.T
        grad_full_param_gpu = x.T @ grad_out

        # 4. 【阅后即焚】完整参数和局部参数用完了，立刻手动释放 GPU 显存！
        del full_param_gpu
        del local_param_gpu

        # 5. 【GPU 通信】Reduce-Scatter: 切分梯度
        local_chunk_size = num_dim // world_size
        local_grad_gpu = torch.empty(local_chunk_size, num_dim, dtype=grad_full_param_gpu.dtype, device=device)
        dist.reduce_scatter_tensor(output=local_grad_gpu, input=grad_full_param_gpu, op=dist.ReduceOp.AVG)

        """
        同步传输
        CPU 线程会阻塞等待直到整个张量传输完成
        GPU 计算流水线完全停滞，浪费宝贵的 GPU 计算资源
        传输期间 GPU 处于空闲状态

        异步传输（非阻塞式)
        立即返回一个 Future 张量，CPU 线程不阻塞
        GPU 可以立即开始下一步计算（如处理下一层）
        传输在后台通过 PCIe 总线进行
        """
        # 6. 【D2H 拷贝】将切分后的 1/N 梯度从 GPU 拷贝回 CPU 内存！
        local_grad_cpu = local_grad_gpu.to('cpu', non_blocking=True)

        # 7. 【阅后即焚】释放 GPU 上的梯度
        del grad_full_param_gpu
        del local_grad_gpu

        # 返回的 local_grad_cpu 是一个 CPU Tensor, 立即返回，不等待传输完成
        # NOTE: x的梯度放在GPU上， param的梯度放在CPU上
        return x_grad_gpu, local_grad_cpu, None, None, None

def zero3_offload_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    '''
    下面是基于 ZeRO-3 增加 CPU Offload 的完整实现：    
    '''

    setup(rank, world_size)
    device = get_device(rank)

    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    # 数据依然在 GPU 上 (因为前向计算在 GPU)
    local_data_gpu = data[rank * local_batch_size : (rank + 1) * local_batch_size].to(device)

    # ==========================================
    # 1. ZeRO-3 Offload 参数初始化 (存放在 CPU)
    # ==========================================
    local_chunk_size = int_divide(num_dim, world_size)
    local_cpu_params_all_layers: List[Parameter] = []

    for _ in range(num_layers):
        # full_init_param: [num_dim, num_dim], 每层的完整参数均放在CPU上, 然后即时释放
        #full_init_param = get_init_params(num_dim, num_dim, rank)
        full_init_param_cpu_per_layer = get_init_params(num_dim, num_dim, -1) # cpu上
        chunk_start = rank * local_chunk_size
        chunk_end = chunk_start + local_chunk_size

        # 【关键优化】：使用 pin_memory() 锁页内存！
        # 锁页内存可以大幅提升 CPU 到 GPU 的 PCIe 传输速度，并允许真正的异步拷贝
        # local_param_cpu: [num_dim/world_size, num_dim], 只有当前rank上的1/N 的参数都放在CPU
        local_param_cpu = full_init_param_cpu_per_layer[chunk_start:chunk_end].clone().cpu().pin_memory()
        del full_init_param_cpu_per_layer
        local_cpu_params_all_layers.append(Parameter(local_param_cpu))

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
        x = local_data_gpu
        for local_cpu_layer_param in local_cpu_params_all_layers:
            # local_param: [num_dim/world_size, num_dim], 只有当前rank上的1/N 的参数都放在CPU
            x = ZeRO3OffloadLinear.apply(x, local_cpu_layer_param, world_size, rank, num_dim)
            x = F.gelu(x)

        loss = x.square().mean()

        # ==========================================
        # 反向传播 (GPU 计算，梯度算完后推回 CPU)
        # ==========================================
        loss.backward()

        """
        虽然传输是异步的，但最终需要同步, 这个同步是隐式的：
        梯度返回后，在优化器步骤中才会使用,此时传输通常已经完成
        """
        # ==========================================
        # 参数更新阶段 (完全在 CPU 上执行！)
        # ==========================================
        for layer_param_idx, local_cpu_layer_param in enumerate(local_cpu_params_all_layers):
            if local_cpu_layer_param.grad is None:
                continue

            # 此时 local_grad 是一个 CPU Tensor
            # local_grad: [num_dim/world_size, num_dim]
            local_grad = local_cpu_layer_param.grad

            # 初始化 CPU 上的优化器状态 (同样使用 pin_memory)
            if layer_param_idx not in optim_states:
                optim_states[layer_param_idx] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(local_cpu_layer_param, device='cpu').pin_memory(), # 注意：一二阶动量均放在 CPU 上
                    'exp_avg_sq': torch.zeros_like(local_cpu_layer_param, device='cpu').pin_memory()
                }

            state = optim_states[layer_param_idx]
            state['step'] += 1
            t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # [手动 AdamW 逻辑 - 纯 CPU 计算]
            # 这里的加减乘除全部由 CPU 的算术逻辑单元 (ALU) 完成，GPU 此时处于空闲状态
            local_cpu_layer_param.data.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(local_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(local_grad, local_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            local_cpu_layer_param.data.addcdiv_(exp_avg, denom, value=-step_size)

            # 清理 CPU 上的梯度
            local_cpu_layer_param.grad = None

        print(f"[ZeRO-3 Offload] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
              f"param_device = {local_cpu_params_all_layers[0].device}", flush=True)

    cleanup()


if  __name__ == "__main__":
    data = generate_sample_data()
    spawn_wrapper(zero3_offload_main, world_size=4, data=data, num_layers=4, num_steps=20)

    """
    自己测试可以正确运行
    """