import torch.distributed as dist
from inspect import isfunction
from typing import Callable
import sys
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import math
from torch_util import get_device

class DisableDistributed:
    """Context manager that temporarily disables distributed functions (replaces with no-ops)
    这是一个上下文管理器，用于临时禁用分布式功能。

    # 正常模式：会执行分布式通信
    dist.barrier()  # 真正的同步操作

    # 在调试模式下：禁用所有分布式操作
    with DisableDistributed():
        dist.barrier()  # 现在什么也不做，返回None

    主要用途：
        在单进程调试时避免分布式通信错误
        防止调试器（如pdb）在分布式环境中出现问题
        简化单机单卡的测试

    2. spawn 函数
    这是一个包装器函数，用于启动分布式训练进程。 
    """

    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                # 将函数体替换为一个空函数
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        退出上下文时：恢复原来的函数
        """
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    # Note: assume kwargs are in the same order as what main needs
    """
    当检测到调试器（如pdb）时自动触发

    只启动1个进程（rank 0）

    禁用所有分布式功能

    适合单步调试代码
    """
    if sys.gettrace(): # # 检测是否在调试中
        # If we're being traced, run the function directly, since we can't trace through mp.spawn
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
    else:
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)


def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"
