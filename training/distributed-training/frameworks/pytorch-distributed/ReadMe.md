# PyTorch Distributed

## Summary

PyTorch Distributed provides native support for distributed training through two main APIs: DistributedDataParallel (DDP) for data parallelism and FullyShardedDataParallel (FSDP) for memory-efficient sharded training. As the built-in solution in PyTorch, these APIs offer tight integration with the PyTorch ecosystem, good performance, and active development.

Key points to remember:

- DDP is the standard for data-parallel training when model fits in GPU memory
- FSDP implements ZeRO-3 style sharding for large models
- Both require torch.distributed initialization with appropriate backend
- NCCL backend preferred for GPU training
- torchrun simplifies launching distributed jobs
- Process groups enable flexible communication patterns
- Native integration with PyTorch profiler, checkpointing, and mixed precision
- Active development with regular improvements

## Initialization

### Basic Setup

```python
import torch
import torch.distributed as dist

def setup(rank, world_size):
    # Set device
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(
        backend="nccl",           # NCCL for GPU
        init_method="env://",     # Get config from environment
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()
```

### Launch Methods

**torchrun (recommended)**:
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train.py

# Multiple nodes
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=123 --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 train.py
```

**torch.multiprocessing**:
```python
import torch.multiprocessing as mp

def train(rank, world_size):
    setup(rank, world_size)
    # Training code
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### Environment Variables

torchrun sets these automatically:
```python
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

## DistributedDataParallel (DDP)

### Basic Usage

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# Create model
model = MyModel().to(rank)

# Wrap with DDP
model = DDP(model, device_ids=[rank])

# Create distributed sampler
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=world_size, rank=rank
)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for shuffling
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### DDP Options

```python
model = DDP(
    model,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=False,  # Set True if some params unused
    broadcast_buffers=True,        # Sync buffers (e.g., BatchNorm)
    bucket_cap_mb=25,              # Gradient bucket size
    gradient_as_bucket_view=True,  # Memory optimization
    static_graph=False,            # Set True for fixed computation graph
)
```

### Gradient Accumulation with DDP

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    is_accumulating = (i + 1) % accumulation_steps != 0

    # Skip gradient sync during accumulation
    with model.no_sync() if is_accumulating else nullcontext():
        loss = model(batch) / accumulation_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

### Communication Hooks

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

# FP16 compression
model.register_comm_hook(
    state=None,
    hook=default_hooks.fp16_compress_hook
)

# PowerSGD compression
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
model.register_comm_hook(
    state=powerSGD_hook.PowerSGDState(process_group=None),
    hook=powerSGD_hook.powerSGD_hook
)
```

## FullyShardedDataParallel (FSDP)

### Basic Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel()
model = FSDP(model)

# Training loop same as DDP
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Sharding Strategies

```python
from torch.distributed.fsdp import ShardingStrategy

# Full sharding (ZeRO-3)
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# Shard gradients and optimizer only (ZeRO-2)
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

# Hybrid: full shard within node, replicate across nodes
model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)
```

### Auto Wrapping

```python
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

# Wrap by layer type
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerEncoderLayer}
)

# Wrap by parameter count
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

### Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(model, mixed_precision=bf16_policy)
```

### CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

### Checkpointing

```python
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# Full state dict (for inference)
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "model.pt")

# Sharded state dict (for training checkpoints)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    # Each rank saves its shard
```

## Process Groups

### Creating Sub-Groups

```python
# Create group with subset of ranks
group = dist.new_group(ranks=[0, 1, 2, 3])

# Use group for collective operations
dist.all_reduce(tensor, group=group)
```

### Hierarchical Groups

```python
# Example: 8 GPUs, 2 nodes, 4 GPUs per node
# Create intra-node and inter-node groups

world_size = 8
local_size = 4  # GPUs per node

# Intra-node group
local_ranks = list(range(
    (rank // local_size) * local_size,
    (rank // local_size + 1) * local_size
))
local_group = dist.new_group(ranks=local_ranks)

# Inter-node group (one rank per node)
global_ranks = list(range(0, world_size, local_size))
global_group = dist.new_group(ranks=global_ranks)
```

## Communication Primitives

### Point-to-Point

```python
# Send
if rank == 0:
    tensor = torch.tensor([1, 2, 3], device='cuda')
    dist.send(tensor, dst=1)

# Receive
if rank == 1:
    tensor = torch.empty(3, device='cuda')
    dist.recv(tensor, src=0)
```

### Collectives

```python
# All-reduce: sum across all ranks
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast: rank 0 to all
dist.broadcast(tensor, src=0)

# All-gather: collect from all ranks
gathered = [torch.empty_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered, tensor)

# Reduce-scatter: reduce and scatter chunks
output = torch.empty(tensor.size(0) // world_size, device='cuda')
dist.reduce_scatter(output, [tensor])

# Barrier: synchronize all ranks
dist.barrier()
```

## Mixed Precision with DDP

```python
from torch.cuda.amp import autocast, GradScaler

model = DDP(model, device_ids=[rank])
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Debugging

### Verify Synchronization

```python
# Check all ranks have same parameters
for name, param in model.named_parameters():
    local_param = param.data.clone()
    dist.all_reduce(local_param)
    local_param /= world_size

    if not torch.allclose(param.data, local_param, atol=1e-5):
        print(f"Rank {rank}: {name} is out of sync")
```

### Detect Hangs

```python
import os

# Enable async error handling
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Set timeout
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
```

### Memory Debugging

```python
# Check memory after each operation
def log_memory(msg):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[Rank {rank}] {msg}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

## Performance Tips

### DDP Performance

1. **Use static_graph=True** if computation graph is fixed
2. **Increase bucket_cap_mb** for large models
3. **Enable gradient_as_bucket_view** to reduce memory
4. **Profile with torch.profiler** to identify bottlenecks

### FSDP Performance

1. **Enable forward_prefetch** and **backward_prefetch**
2. **Choose appropriate sharding_strategy** for your network
3. **Tune auto_wrap_policy** for optimal granularity
4. **Use mixed_precision** for memory and speed

### General

1. **Pin memory** in DataLoader: `pin_memory=True`
2. **Use multiple workers**: `num_workers=4`
3. **Overlap data loading**: Use prefetch
4. **Profile regularly**: torch.profiler.profile()

## Common Patterns

### Training Script Template

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Get distributed info from environment
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Create model
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create data
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        train_epoch(model, dataloader, optimizer)

        # Save checkpoint (rank 0 only)
        if rank == 0:
            torch.save(model.module.state_dict(), f"model_{epoch}.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Launch

```bash
torchrun --nproc_per_node=4 train.py
```
