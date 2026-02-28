import torch
import math
import time
import os
import sys
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
from torch.nn.parameter import Parameter
from execute_util import text, image, link, system_text
from torch_util import get_device
from lecture_util import article_link
from lecture_08_utils import spawn_wrapper, int_divide, summarize_tensor, get_init_params, render_duration

"""
stanford cs336: Distributed Training

数据并行
张量并行
流水线并行
"""

def main():
    # torch_distributed()        # How this is implemented in NCCL/PyTorch
    # benchmarking()             # Measure actual NCCL bandwidth

    # data_parallelism()         # Cut up along the batch dimension
    # tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

def torch_distributed():
    print("Let's walk through some examples.")
    spawn_wrapper(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    # NOTE: All-reduce
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True) # tensor([ 0.,  1.,  2.,  3.])
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True) # tensor([ 6., 10., 14., 18.])
    """
    Rank 0 [before all-reduce]: tensor([0., 1., 2., 3.])
    Rank 1 [before all-reduce]: tensor([1., 2., 3., 4.])
    Rank 2 [before all-reduce]: tensor([2., 3., 4., 5.])
    Rank 3 [before all-reduce]: tensor([3., 4., 5., 6.])

    Rank 0 [after all-reduce]: tensor([ 6., 10., 14., 18.])
    Rank 1 [after all-reduce]: tensor([ 6., 10., 14., 18.])
    Rank 2 [after all-reduce]: tensor([ 6., 10., 14., 18.])
    Rank 3 [after all-reduce]: tensor([ 6., 10., 14., 18.]) 
    """

    # NOTE: All-reduce 2D
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    tensor = torch.tensor([[0., 1],
                                        [ 2, 3]], device=get_device(rank)) + rank  # Both input and output
    print(f"Rank {rank} [before all-reduce-2D]: {tensor}", flush=True) # tensor([ 0.,  1.,  2.,  3.])
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce-2D]: {tensor}", flush=True) # tensor([ 6., 10., 14., 18.])
    """
    Rank 0 [before all-reduce-2D]: tensor([[0., 1.],
                                           [2., 3.]])
    Rank 1 [before all-reduce-2D]: tensor([[1., 2.],
                                           [3., 4.]])
    Rank 2 [before all-reduce-2D]: tensor([[2., 3.],
                                           [4., 5.]])
    Rank 3 [before all-reduce-2D]: tensor([[3., 4.],
                                           [5., 6.]])

    Rank 0 [after all-reduce-2D]: tensor([[ 6., 10.],
                                          [14., 18.]])
    Rank 1 [after all-reduce-2D]: tensor([[ 6., 10.],
                                          [14., 18.]])
    Rank 2 [after all-reduce-2D]: tensor([[ 6., 10.],
                                          [14., 18.]])
    Rank 3 [after all-reduce-2D]: tensor([[ 6., 10.],
                                          [14., 18.]])
    """
    #sys.exit(0)

    dist.barrier()
    # NOTE: Reduce-scatter
    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output, 此时output中的值为随机值

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)
    """
    Rank 0 [before reduce-scatter]: input = tensor([0., 1., 2., 3.]), output = tensor([-1.3031e-08])
    Rank 1 [before reduce-scatter]: input = tensor([1., 2., 3., 4.]), output = tensor([-1.5123e+38])
    Rank 2 [before reduce-scatter]: input = tensor([2., 3., 4., 5.]), output = tensor([-1.0475e-25])
    Rank 3 [before reduce-scatter]: input = tensor([3., 4., 5., 6.]), output = tensor([-2.0544e+23])

    Rank 0 [after reduce-scatter]: input = tensor([0., 1., 2., 3.]), output = tensor([6.]) , reduce-scatter将input中的值相加，然后将结果存入output中, 但只取input中的某一列进行sum
    Rank 1 [after reduce-scatter]: input = tensor([1., 2., 3., 4.]), output = tensor([10.])
    Rank 2 [after reduce-scatter]: input = tensor([2., 3., 4., 5.]), output = tensor([14.])
    Rank 3 [after reduce-scatter]: input = tensor([3., 4., 5., 6.]), output = tensor([18.])
    """

    # NOTE: All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)
    """
    Rank 0 [before all-gather]: input = tensor([6.]), output = tensor([0., 0., 0., 0.])
    Rank 1 [before all-gather]: input = tensor([10.]), output = tensor([ 0.0000e+00,  0.0000e+00, -1.4778e+38,  3.0809e-41])
    Rank 2 [before all-gather]: input = tensor([14.]), output = tensor([ 0.0000e+00,  0.0000e+00, -1.0322e+09,  3.0889e-41])
    Rank 3 [before all-gather]: input = tensor([18.]), output = tensor([-1.2843e+30,  4.5793e-41, -1.2843e+30,  4.5793e-41])

    Rank 0 [after all-gather]: input = tensor([6.]), output = tensor([ 6., 10., 14., 18.])
    Rank 1 [after all-gather]: input = tensor([10.]), output = tensor([ 6., 10., 14., 18.])
    Rank 2 [after all-gather]: input = tensor([14.]), output = tensor([ 6., 10., 14., 18.])
    Rank 3 [after all-gather]: input = tensor([18.]), output = tensor([ 6., 10., 14., 18.])
    """

    print("Indeed, all-reduce = reduce-scatter + all-gather!!!")


    dist.barrier()
    # NOTE: Reduce-scatter 2D
    batch_dim=3
    hidden_dim=2
    print(f"world_size={world_size}, batch_dim={batch_dim}, dim={hidden_dim}")
    # NOTE: reduce_scatter中，torch.dist总是从第0维将tensor按word_size进行切分，然后将切分后的tensor按reduce_op进行操作，最后将结果存入output中
    input = rank + torch.arange(world_size* batch_dim*hidden_dim, dtype=torch.float32, device=get_device(rank)).reshape(world_size*batch_dim, hidden_dim) 
    output = torch.empty(batch_dim, hidden_dim, device=get_device(rank))  # Allocate output, 此时output中的值为随机值

    print(f"Rank {rank} [before reduce-scatter-2D]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter-2D]: input = {input}, output = {output}", flush=True)
    """
Rank 0 [before reduce-scatter-2D]: input = tensor([[ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.],
        [ 6.,  7.],
        [ 8.,  9.],
        [10., 11.],
        [12., 13.],
        [14., 15.],
        [16., 17.],
        [18., 19.],
        [20., 21.],
        [22., 23.]]), output = tensor([[ 1.7386e-31,  4.5884e-41],
        [ 1.7386e-31,  4.5884e-41],
        [-2.0180e+18,  3.0749e-41]])
Rank 1 [before reduce-scatter-2D]: input = tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.],
        [13., 14.],
        [15., 16.],
        [17., 18.],
        [19., 20.],
        [21., 22.],
        [23., 24.]]), output = tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
Rank 2 [before reduce-scatter-2D]: input = tensor([[ 2.,  3.],
        [ 4.,  5.],
        [ 6.,  7.],
        [ 8.,  9.],
        [10., 11.],
        [12., 13.],
        [14., 15.],
        [16., 17.],
        [18., 19.],
        [20., 21.],
        [22., 23.],
        [24., 25.]]), output = tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
Rank 3 [before reduce-scatter-2D]: input = tensor([[ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.],
        [13., 14.],
        [15., 16.],
        [17., 18.],
        [19., 20.],
        [21., 22.],
        [23., 24.],
        [25., 26.]]), output = tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])


        
Rank 0 [after reduce-scatter-2D]: input = tensor(        [
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.], # 0~3行发送给rank0上进行reduce
        [ 6.,  7.],
        [ 8.,  9.],
        [10., 11.], # 4~7行发送给rank1上进行reduce
        [12., 13.],
        [14., 15.],
        [16., 17.], # 8~11行发送给rank2上进行reduce
        [18., 19.],
        [20., 21.],
        [22., 23.]]), # 12~15行发送给rank3上进行reduce
                        output = tensor([[ 6., 10.],
                                        [14., 18.],
                                        [22., 26.]])
Rank 1 [after reduce-scatter-2D]: input = tensor(        [
        [ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.],
        [13., 14.],
        [15., 16.],
        [17., 18.],
        [19., 20.],
        [21., 22.],
        [23., 24.]]), output = tensor([[30., 34.],
                                        [38., 42.],
                                        [46., 50.]])
Rank 2 [after reduce-scatter-2D]: input = tensor( 
       [[ 2.,  3.],
        [ 4.,  5.],
        [ 6.,  7.],
        [ 8.,  9.],
        [10., 11.],
        [12., 13.],
        [14., 15.],
        [16., 17.],
        [18., 19.],
        [20., 21.],
        [22., 23.],
        [24., 25.]]), output = tensor([[54., 58.],
                                        [62., 66.],
                                        [70., 74.]])
Rank 3 [after reduce-scatter-2D]: input = tensor([
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.],
        [13., 14.],
        [15., 16.],
        [17., 18.],
        [19., 20.],
        [21., 22.],
        [23., 24.],
        [25., 26.]]), output = tensor([[78., 82.],
                                        [86., 90.],
                                        [94., 98.]])
    """

    cleanup()


def benchmarking():
    print("Let's see how fast communication happens (restrict to one node).")

    # NOTE: All-reduce
    spawn_wrapper(all_reduce, world_size=4, num_elements=10 * 1024**2)
    # NOTE:Reduce-scatter
    spawn_wrapper(reduce_scatter, world_size=4, num_elements_per_rank=10 * 1024**2)
    #sys.exit(0)


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    """ 
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    """
    # 自认为正确的实现
    # 对于 Ring All-Reduce 算法
    # 每个进程发送和接收的数据量
    per_process_bytes_sent = 2 * (world_size - 1) * size_bytes / world_size
    total_bytes_sent = world_size * per_process_bytes_sent
    # 聚合带宽 = 总传输数据量 / 总时间
    bandwidth = total_bytes_sent / (duration * world_size) # 乘以world_size是因为每个进程都同时参与了传输
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    """
    为什么要乘以2
    在 all-reduce 操作中，乘以2是因为每个数据元素需要被传输两次：
    一次发送（scatter-reduce 阶段）：数据从各个进程分散出去进行归约
    一次接收（all-gather 阶段）：归约后的结果再广播回所有进程

    以 Ring All-Reduce 为例详细说明
    假设有 4 个进程（world_size=4），每个进程有大小为 N 的张量：

    阶段1：Scatter-Reduce（数据分散归约）
    text
    进程0: 发送 chunk 给进程1
    进程1: 发送 chunk 给进程2  
    进程2: 发送 chunk 给进程3
    进程3: 发送 chunk 给进程0

    每个进程发送 N/world_size 数据
    总共传输次数：world_size-1 次
    总传输量：(world_size-1) * (N/world_size)
    阶段2：All-Gather（结果收集）
    text
    进程0: 发送结果给进程3
    进程1: 发送结果给进程0
    进程2: 发送结果给进程1
    进程3: 发送结果给进程2

    每个进程再次发送 N/world_size 数据
    总传输量：(world_size-1) * (N/world_size)
    """

    cleanup()

def reduce_scatter(rank: int, world_size: int, num_elements_per_rank: int):
    setup(rank, world_size)

    """
    正确的理解：
    reduce_scatter 操作：将 world_size 个张量（每个大小为 N）reduce 后，再分散给每个 rank（每个得到大小为 N 的结果）
    所以 input 形状应该是 [world_size, N]
    output 形状应该是 [N]
    
    """
    # Create input and outputs
    # reduce scatter二维数据
    hidden_dim=2
    input = torch.randn(world_size*num_elements_per_rank, hidden_dim, device=get_device(rank))  # Each rank has a matrix
    #output = torch.empty(num_elements_per_rank, device=get_device(rank))
    output = torch.empty(num_elements_per_rank, hidden_dim, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements_per_rank}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    per_process_bytes_sent = data_bytes / world_size * (world_size - 1)  # How much needs to be sent (no 2x here)
    #per_process_bytes_sent = 2 * (world_size - 1) * size_bytes / world_size
    total_bytes_sent = world_size * per_process_bytes_sent
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = total_bytes_sent / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def data_parallelism():
    print("Sharding strategy: each rank gets a slice of the data")

    data = generate_sample_data()
    spawn_wrapper(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=10)
    #sys.exit()

    print("Notes:")
    print("- Losses are different across ranks (computed on local data)")
    print("- Gradients are all-reduced to be the same across ranks")
    print("- Therefore, parameters remain the same across ranks")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    print(f"generate_sample_data: batch_size = {batch_size}, num_dim = {num_dim}")
    return data


"""
数据并行, 沿batch维度切分数据, 每个rank计算自己的部分
"""
def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params: List[Parameter] = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        optimizer.zero_grad()
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass, 计算所有的梯度
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        # 将各layer的参数梯度进行平均
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        """
        可以看到，每个rank的loss不一样，但参数是相同的
        [data_parallelism] Rank 0: step = 19, loss = 0.00010565415141172707, params = ['1024x1024[-0.0364...]', '1024x1024[-0.0395...]', '1024x1024[-0.04...]', '1024x1024[-0.036...]']
        [data_parallelism] Rank 1: step = 19, loss = 0.0001070127691491507, params = ['1024x1024[-0.0364...]', '1024x1024[-0.0395...]', '1024x1024[-0.04...]', '1024x1024[-0.036...]']
        [data_parallelism] Rank 2: step = 19, loss = 0.0001068995043169707, params = ['1024x1024[-0.0364...]', '1024x1024[-0.0395...]', '1024x1024[-0.04...]', '1024x1024[-0.036...]']
        [data_parallelism] Rank 3: step = 19, loss = 0.0001085107505787164, params = ['1024x1024[-0.0364...]', '1024x1024[-0.0395...]', '1024x1024[-0.04...]', '1024x1024[-0.036...]']
        """
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()


def tensor_parallelism():
    # 张量并行
    print("Sharding strategy: each rank gets part of each layer, transfer all data/activations")

    data = generate_sample_data()
    #spawn_wrapper(tensor_parallelism_main, world_size=4, data=data, num_layers=4)
    spawn_wrapper(tensor_parallelism_main_auto_grad, world_size=4, data=data, num_layers=4)
    spawn_wrapper(tensor_parallelism_main_no_autograd, world_size=4, data=data, num_layers=4)
    #sys.exit()

def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    # 每个rank只有参数的一部分
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    num_steps = 10
    for step in range(num_steps):
        # Forward pass
        x = data
        for i in range(num_layers):
            # Compute activations (batch_size x local_num_dim)
            x = x @ params[i]  # Note: this is only on a slice of the parameters
            x = F.gelu(x)

            # Allocate memory for activations (world_size x batch_size x local_num_dim)
            activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

            # Send activations via all gather
            dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

            # Concatenate them to get batch_size x num_dim
            x = torch.cat(activations, dim=1)

        print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

        # Backward pass: homework exercise
        loss = x.square().mean()  # Loss function is average squared magnitude
        # CODE HERE
    cleanup()

def tensor_parallelism_main_no_autograd(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)

    # 初始化参数
    params: List[Parameter] = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # 简单的SGD更新，或者仅用于打印梯度验证
    lr = 1e-3

    num_steps = 10
    for step in range(num_steps):
        # ==========================================
        # Forward pass (需修改以缓存中间变量)
        # ==========================================
        x = data

        # 缓存列表，用于反向传播
        # inputs_cache: 存储每一层的输入 X [Batch, Global_Dim]
        # pre_act_cache: 存储每一层激活前的结果 Z [Batch, Local_Dim]
        inputs_cache_per_layer:List[torch.Tensor] = []
        prev_activation_cache_per_layer:List[torch.Tensor] = []

        for i in range(num_layers):
            # 1. 缓存当前层的输入
            inputs_cache_per_layer.append(x)

            # 2. 局部计算
            # x: [B, Global], param: [Global, Local] -> z: [B, Local]
            z = x @ params[i]

            # 3. 缓存激活前的值 (用于计算GELU的导数)
            prev_activation_cache_per_layer.append(z)

            # 4. 激活函数
            local_act = F.gelu(z)

            # 5. 通信 (All-Gather)
            activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]
            dist.all_gather(tensor_list=activations, tensor=local_act, async_op=False)

            # 6. 拼接
            x = torch.cat(activations, dim=1)
            # if rank<=1:
            #     print(f"[tensor_parallelism] Rank {rank} step:{step}: forward pass produced activations {summarize_tensor(x)}", flush=True)

        # ==========================================
        # Backward pass (Manual Implementation)
        # ==========================================

        # 1. 计算 Loss 对最终输出 x 的梯度
        # Loss = mean(x^2) -> dL/dx = 2*x / numel
        # grad_x: [Batch, Global_Dim]
        grad_x = 2 * x / x.numel()

        # 从最后一层向前遍历
        for i in reversed(range(num_layers)):
            # --- Step A: 处理 All-Gather 的反向 (Split/Slice) ---
            # 前向是 Concat(AllGather(local))，反向就是切片取出属于当前rank的那部分梯度
            # grad_x shape: [Batch, Global_Dim]
            start_idx = rank * local_num_dim
            end_idx = (rank + 1) * local_num_dim

            # grad_local_act shape: [Batch, Local_Dim]
            grad_local_act = grad_x[:, start_idx:end_idx]

            # --- Step B: 处理 Activation (GELU) 的反向 ---
            # d(GELU(z))/dz * grad_output
            # 为了简化数学公式书写，这里我们利用PyTorch的自动求导计算局部元素的导数
            # (这在手动实现中是允许的，只要不依赖整个图)
            z = prev_activation_cache_per_layer[i]
            with torch.enable_grad():
                z_temp = z.detach().requires_grad_(True)
                out_temp = F.gelu(z_temp)
                out_temp.backward(grad_local_act)
                grad_z = z_temp.grad # shape: [Batch, Local_Dim]

            # --- Step C: 处理 MatMul (Z = X @ W) 的反向 ---
            # 我们需要计算 dL/dW 和 dL/dX
            input_x = inputs_cache_per_layer[i] # shape: [Batch, Global_Dim]
            weight = params[i]        # shape: [Global_Dim, Local_Dim]

            # z = x @ W
            # 1. 计算权重梯度: dL/dW = X^T @ dL/dZ
            # [Global, Batch] @ [Batch, Local] -> [Global, Local]
            grad_w = input_x.t() @ grad_z

            # 手动保存梯度到 param.grad
            weight.grad = grad_w # 这里直接覆盖，因为没有累积steps

            # 2. 计算输入梯度: dL/dX_partial = dL/dZ @ W^T
            # [Batch, Local] @ [Local, Global] -> grad_x_partial: [Batch, Global]
            # 注意：这只是当前 rank 对输入 X 的梯度贡献
            grad_x_partial = grad_z @ weight.t()

            # --- Step D: 处理输入复制的反向 (All-Reduce) ---
            # 前向传播时，输入 X 被复制到了所有 rank。
            # 反向传播时，我们需要将所有 rank 对 X 的梯度贡献相加。
            dist.all_reduce(grad_x_partial, op=dist.ReduceOp.SUM, async_op=False)

            # 更新 grad_x 用于上一层的计算
            # grad_x: [Batch, Global_Dim]
            grad_x = grad_x_partial

        # ==========================================
        # Optimizer Step (Manual)
        # ==========================================
        loss_val = x.square().mean().item()
        grad_norm = params[0].grad.norm().item() if params[0].grad is not None else 0
        print(f"[tensor_parallel_no_autograd][Rank {rank}] Step {step}: Loss = {loss_val:.6f}, Grad Norm (Layer 0) = {grad_norm:.4f}", flush=True)

        # 简单的 SGD 更新
        with torch.no_grad():
            for param in params:
                param -= lr/math.sqrt(step+1) * param.grad
                # 清零梯度 (可选，因为上面是直接覆盖)
                param.grad = None

    cleanup()

class TP_AllGather(torch.autograd.Function):
    """
    前向传播：All-Gather (将切分的输出拼接成完整的)
    反向传播：Split (将完整的梯度切分回各个rank)

    与数据并行（Data Parallelism）不同，TP 中的参数W_i是切分的，每个rank只有一部分。
    它们各自计算出的梯度也是针对各自部分的，因此不需要像数据并行那样对 param.grad 进行 All-Reduce。直接 optimizer.step() 即可。
    """
    @staticmethod
    def forward(ctx, input, world_size:int) -> torch.Tensor:
        # input: [batch_size, local_dim]
        # output: [batch_size, global_dim]

        # 1. 准备输出张量列表
        # 注意：为了简单起见，这里假设所有rank的维度相同
        local_dim = input.size(1)
        global_dim = local_dim * world_size

        # 创建输出列表, world_size个元素，每个元素都是一个空的张量[batch_size, local_dim]
        tensor_list: List[torch.Tensor] = [torch.empty_like(input) for _ in range(world_size)]

        # 2. 执行 All-Gather
        dist.all_gather(tensor_list, input, async_op=False)

        # 3. 拼接
        # output: [batch_size, global_dim]
        output = torch.cat(tensor_list, dim=1)

        ctx.world_size = world_size
        ctx.local_dim = local_dim
        return output

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        # grad_output: [batch_size, global_dim]
        # return: [batch_size, local_dim]

        # 反向传播时，只需要取出属于当前rank的那一部分梯度
        rank = dist.get_rank()
        start = rank * ctx.local_dim
        end = start + ctx.local_dim

        # Slice操作 (Split)
        # grad_output: [batch_size, global_dim], 在hidden_dim维度上对grad进行切分
        grad_input = grad_output[:, start:end]

        return grad_input, None # 这里的None是对world_size的梯度求导，因为world_size是常数，所以返回None

class TP_CopyTo(torch.autograd.Function):
    """
    前向传播：Identity (不做任何事，输入即输出)
    反向传播：All-Reduce (将各个rank计算出的关于输入的偏导数求和)

    原因：每一层的输入X在所有rank上是相同的（复制的）。
    但在反向传播时，每个rank只计算了X对局部权重W_i的梯度贡献。
    为了得到X的总梯度，必须将所有rank的贡献相加。
    即梯度 G_i = dL/dY_i * W_i
    X 的真实总梯度应该是所有贡献之和：
    G = sum_i(G_i) = sum_i(dL/dY_i * W_i)

    因此，必须在反向传播时执行 All-Reduce (Sum)。这就是 TP_CopyTo 的作用。
    """
    @staticmethod
    def forward(ctx, input:torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> torch.Tensor:
        # Clone梯度以避免原地修改问题
        grad_input = grad_output.clone()
        # 执行 All-Reduce Sum
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)
        return grad_input

def tensor_parallelism_main_auto_grad(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)

    # Create model
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]
    # 需要定义优化器
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    num_steps = 10
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        x = data
        for i in range(num_layers):
            # 1. [关键] 在进入层之前，插入 CopyTo 算子
            # 确保反向传播时，这一处的梯度会被 All-Reduce
            x = TP_CopyTo.apply(x)

            # 2. Local Computation
            # x: [B, D], param: [D, D/world_size] -> result: [B, D/world_size]
            x = x @ params[i]
            x = F.gelu(x)

            # 3. [关键] 使用支持自动求导的 All-Gather
            # result: [B, D]
            x = TP_AllGather.apply(x, world_size)

        # Loss calculation
        loss = x.square().mean()

        # Backward pass
        # 由于我们使用了自定义的 autograd.Function，loss.backward() 会自动处理通信
        loss.backward()

        # Update parameters
        # 在纯张量并行中，参数是切分的，梯度也是切分的，不需要做额外的 All-Reduce
        optimizer.step()

        print(f"[tensor_parallelism_auto_grad] Rank {rank}: step={step}, loss={loss.item():.6f}, "
              f"param_grad_norm={params[0].grad.norm().item():.4f}", flush=True)

    cleanup()

def pipeline_parallelism():
    print("流水线并行，Sharding strategy: each rank gets subset of layers, transfer all data/activations")

    data = generate_sample_data()
    #spawn_wrapper(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)
    spawn_wrapper(pipeline_parallelism_main_gpipe, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    print("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    # Backward pass: homework exercise
    # CODE HERE

    cleanup()

"""
### 代码核心逻辑解析

1.  **`forward_cache` 的作用**：
    *   在流水线并行中，我们不能像普通训练那样一次性跑完 `loss.backward()`。
    *   我们需要保存每个 micro-batch 的 `input_tensor`（作为局部图的起点）和 `output_tensor`（作为局部图的终点）。
    *   `input_tensor` 必须设置 `requires_grad=True`，这样 PyTorch 才会计算关于它的梯度（即我们要传给上一个 Rank 的梯度）。

2.  **前向传播中的 `detach()`**：
    *   `x = recv_buffer.detach().requires_grad_(True)`
    *   这是最关键的一行。因为 `dist.recv` 接收的是纯数据，没有梯度历史。我们需要手动告诉 PyTorch：“从这里开始构建一个新的计算图，并且我需要计算关于这个输入的梯度”。

3.  **反向传播流程**：
    *   **Rank N (最后一层)**: 计算 Loss -> `loss.backward()` -> 生成参数梯度和输入梯度。
    *   **Rank i (中间层)**: 接收来自 Rank i+1 的梯度 -> `output.backward(grad_recv)` -> 生成参数梯度和输入梯度。
    *   **Rank 0 (第一层)**: 接收来自 Rank 1 的梯度 -> `output.backward(grad_recv)` -> 生成参数梯度（不需要再往前传了）。

4.  **梯度累积**：
    *   PyTorch 的 `.backward()` 默认行为是累积梯度（Accumulate）。
    *   我们在循环中对每个 micro-batch 调用 `.backward()`，参数的 `.grad` 属性会自动累加所有 micro-batch 的梯度。
    *   最后调用一次 `optimizer.step()` 即可更新参数。

### 关于 "Bubble" (气泡)
这段代码实现的是最简单的 **GPipe (F-F-F-F | B-B-B-B)** 调度模式。
*   **缺点**：内存占用高（需要缓存所有 micro-batch），且存在明显的空闲时间（Bubble），因为 Rank 0 必须等 Rank N 跑完前向才能开始反向。
*   **改进**：更高级的实现（如 1F1B 策略）会交错执行前向和反向（F-F-B-F-B...），以减少气泡和内存占用，但代码逻辑会复杂得多。
"""
def pipeline_parallelism_main_gpipe(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # 1. Data Preparation
    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Optimizer (needed for parameter updates)
    optimizer = torch.optim.AdamW(local_params, lr=1e-3)

    # Break up into micro batches
    micro_batch_size = int_divide(batch_size, num_micro_batches)

    if rank == 0:
        # Rank 0 holds the source data chunks
        data_chunks = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        data_chunks = None

    # ==========================================
    # Forward Pass (GPipe Schedule: All Forward)
    # ==========================================

    # Cache to store (input, output) for each micro-batch to use in backward
    # input: used to get .grad to send to prev rank
    # output: used to call .backward(grad_from_next_rank)
    num_steps= 10
    for step in range(num_steps):
        forward_cache = []

        #print(f"[Rank {rank}] Starting Forward Pass...")
        for i in range(num_micro_batches):
            optimizer.zero_grad() # Note: usually done once per global batch, but here for safety

            # --- 1. Receive Input / Get Data ---
            if rank == 0:
                # Rank 0 gets data directly
                x = data_chunks[i]
                # Clone to ensure we don't mess up original data, enable grad for local graph
                x = x.clone().detach().requires_grad_(True)
            else:
                # Other ranks receive from previous rank
                recv_buffer = torch.empty(micro_batch_size, num_dim, device=get_device(rank))
                dist.recv(tensor=recv_buffer, src=rank - 1)
                # [Critical] Detach from the communication buffer and start a new computation graph
                x = recv_buffer.detach().requires_grad_(True)

            # Keep reference to input `x` (it's the root of our local graph)
            input_tensor = x

            # --- 2. Local Computation ---
            for param in local_params:
                x = x @ param
                x = F.gelu(x)

            output_tensor = x

            # --- 3. Cache for Backward ---
            forward_cache.append((input_tensor, output_tensor))

            # --- 4. Send Output to Next Rank ---
            if rank < world_size - 1:
                # Send detached tensor (values only, no graph)
                dist.send(tensor=output_tensor.detach(), dst=rank + 1)
                # print(f"[Rank {rank}] Sent MB {i} to Rank {rank+1}")

        # ==========================================
        # Backward Pass (GPipe Schedule: All Backward)
        # ==========================================

        #print(f"[Rank {rank}] Starting Backward Pass...")

        # Iterate in reverse order (LIFO)
        for i in reversed(range(num_micro_batches)):
            input_tensor, output_tensor = forward_cache[i]

            # --- 1. Calculate Gradient w.r.t Output ---
            if rank == world_size - 1:
                # Last rank calculates the loss
                loss = output_tensor.square().mean()
                # Start backward chain
                loss.backward()
                print(f"[Rank {rank}] Step {step} micro_batch {i} Loss: {loss.item():.4f}")
            else:
                # Intermediate ranks receive gradient from next rank
                grad_recv = torch.empty_like(output_tensor)
                dist.recv(tensor=grad_recv, src=rank + 1)

                # Continue backward chain: compute grads for local params and input_tensor
                output_tensor.backward(grad_recv)

            # --- 2. Send Gradient to Previous Rank ---
            if rank > 0:
                # input_tensor.grad contains dL/d(Input)
                # We send this back to rank-1 to serve as their dL/d(Output)
                dist.send(tensor=input_tensor.grad, dst=rank - 1)

        # ==========================================
        # Optimizer Step
        # ==========================================
        # All micro-batches processed, gradients accumulated. Update weights.
        optimizer.step()

        # Print updated weights norm to verify learning
        # print(f"[pipeline paramllel][Rank {rank}] Step:{step}] Update Param[0] grad norm: {local_params[0].grad.norm():.4f}", flush=True)

    cleanup()

############################################################

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        # NCCL: nvdia collective communication library
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
