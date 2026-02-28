import torch
import time
import os
import sys
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
from execute_util import text, image, link, system_text
from torch_util import get_device
from lecture_util import article_link
from lecture_08_utils import spawn, int_divide, summarize_tensor, get_init_params, render_duration

def main():
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Measure actual NCCL bandwidth

    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

def torch_distributed():
    print("Let's walk through some examples.")
    spawn(collective_operations_main, world_size=4)


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
    """
    sys.exit(0)

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

    print("Indeed, all-reduce = reduce-scatter + all-gather!")
    cleanup()


def benchmarking():
    print("Let's see how fast communication happens (restrict to one node).")

    # NOTE: All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)
    # NOTE:Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements_per_rank=10 * 1024**2 )
    import sys;sys.exit(0)


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
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

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
    dim=2
    input = torch.randn(num_elements_per_rank * world_size, dim, device=get_device(rank))  # Each rank has a matrix
    #output = torch.empty(num_elements_per_rank, device=get_device(rank))
    output = torch.empty(num_elements_per_rank, dim, device=get_device(rank))

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
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def data_parallelism():
    print("Sharding strategy: each rank gets a slice of the data")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    print("Notes:")
    print("- Losses are different across ranks (computed on local data)")
    print("- Gradients are all-reduced to be the same across ranks")
    print("- Therefore, parameters remain the same across ranks")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


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
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()


def tensor_parallelism():
    # 张量并行
    print("Sharding strategy: each rank gets part of each layer, transfer all data/activations")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

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

    cleanup()


def pipeline_parallelism():
    print("流水线并行，Sharding strategy: each rank gets subset of layers, transfer all data/activations")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


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
