# Gloo

## Summary

Gloo is Facebook's collective communications library, providing efficient collective operations for both CPU and GPU tensors. While NCCL is preferred for GPU-to-GPU communication, Gloo serves as an important fallback and is the primary choice for CPU-based distributed training. Gloo is integrated into PyTorch as one of the supported distributed backends.

Key points to remember:

- Primary backend for CPU distributed training
- Fallback for GPU when NCCL is unavailable
- Cross-platform support (Linux, macOS, Windows)
- Supports TCP/IP and shared memory transports
- Lower performance than NCCL for GPU workloads
- Good for heterogeneous or CPU-only environments
- Built into PyTorch, no separate installation needed
- Useful for debugging distributed code

## Use Cases

### When to Use Gloo

1. **CPU-only training**: No GPU available or CPU training preferred
2. **Heterogeneous systems**: Mixed GPU vendors
3. **Development and debugging**: Easier to debug than NCCL
4. **Fallback**: When NCCL initialization fails
5. **Object communication**: Sending Python objects (pickled data)

### When to Use NCCL Instead

1. GPU training on NVIDIA hardware
2. Multi-node GPU training
3. Performance-critical applications

## Usage in PyTorch

### Initialization

```python
import torch.distributed as dist

# CPU distributed training
dist.init_process_group(
    backend="gloo",
    init_method="env://",
    world_size=world_size,
    rank=rank
)

# Or with TCP init
dist.init_process_group(
    backend="gloo",
    init_method="tcp://master_ip:port",
    world_size=world_size,
    rank=rank
)

# Or with file-based init
dist.init_process_group(
    backend="gloo",
    init_method="file:///shared/path/init_file",
    world_size=world_size,
    rank=rank
)
```

### Collective Operations

```python
# All-reduce (CPU tensor)
tensor = torch.randn(1000)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast
dist.broadcast(tensor, src=0)

# All-gather
gathered = [torch.empty_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered, tensor)

# Reduce-scatter
output = torch.empty(tensor.size(0) // world_size)
input_list = list(tensor.chunk(world_size))
dist.reduce_scatter(output, input_list)
```

### GPU Tensors with Gloo

```python
# Gloo can handle GPU tensors but is slower than NCCL
gpu_tensor = torch.randn(1000, device='cuda')

# This works but is not recommended for performance
dist.all_reduce(gpu_tensor)  # Uses Gloo for GPU tensor

# Better: Use NCCL for GPU, Gloo for CPU
dist.init_process_group(backend="nccl")  # Primary for GPU
gloo_group = dist.new_group(backend="gloo")  # Secondary for CPU

# GPU tensor with NCCL (default group)
dist.all_reduce(gpu_tensor)

# CPU tensor with Gloo
cpu_tensor = torch.randn(1000)
dist.all_reduce(cpu_tensor, group=gloo_group)
```

### Object Communication

Gloo supports sending Python objects (pickled):

```python
# Send arbitrary Python objects
if rank == 0:
    objects = [{"data": [1, 2, 3]}, "hello", 42]
else:
    objects = [None] * 3

dist.broadcast_object_list(objects, src=0)
print(f"Rank {rank}: {objects}")

# Gather objects
output = [None] * world_size if rank == 0 else None
obj = {"rank": rank, "data": torch.randn(10).tolist()}
dist.gather_object(obj, output, dst=0)
```

## Environment Variables

```bash
# Network interface
export GLOO_SOCKET_IFNAME=eth0

# Timeout (milliseconds)
export GLOO_SOCKET_TIMEOUT_MS=60000

# Debug output
export GLOO_LOG_LEVEL=DEBUG
```

## Transport Options

### TCP Transport

Default transport, works over any network:

```python
dist.init_process_group(
    backend="gloo",
    init_method="tcp://master:29500"
)
```

### Shared Memory (Single Node)

For processes on the same machine:

```python
# Gloo automatically uses shared memory for local communication
# when processes are on the same node
```

## Performance Characteristics

### Bandwidth Comparison

| Operation | Gloo (CPU) | Gloo (GPU) | NCCL (GPU) |
|-----------|------------|------------|------------|
| All-reduce 1GB | ~1 GB/s | ~5 GB/s | ~50 GB/s |

Gloo is 5-50x slower than NCCL for GPU operations.

### Latency

Gloo has higher latency than NCCL:
- Gloo: 50-200 microseconds
- NCCL: 10-50 microseconds

## Multi-Backend Setup

### Combining NCCL and Gloo

```python
import torch.distributed as dist

# Initialize with NCCL as default (for GPU)
dist.init_process_group(backend="nccl")

# Create Gloo group for CPU operations
cpu_group = dist.new_group(backend="gloo")

def sync_tensor(tensor):
    if tensor.is_cuda:
        dist.all_reduce(tensor)  # Uses NCCL
    else:
        dist.all_reduce(tensor, group=cpu_group)  # Uses Gloo
```

### Process Group Management

```python
# Create subgroups with specific backends
ranks_0_1 = [0, 1]
gloo_subgroup = dist.new_group(ranks=ranks_0_1, backend="gloo")

# Operations on subgroup
if dist.get_rank() in ranks_0_1:
    tensor = torch.randn(100)
    dist.all_reduce(tensor, group=gloo_subgroup)
```

## Debugging with Gloo

### Advantages for Debugging

1. **Clearer error messages**: More readable than NCCL
2. **Works without GPU**: Debug distributed logic on CPU
3. **Simpler setup**: No CUDA/NCCL dependencies

### Debug Example

```python
import torch.distributed as dist
import os

def debug_distributed():
    # Use Gloo for easier debugging
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}: Initialized")

    # Test operations
    tensor = torch.tensor([rank], dtype=torch.float)
    print(f"Rank {rank}: Before all-reduce: {tensor}")

    dist.all_reduce(tensor)
    print(f"Rank {rank}: After all-reduce: {tensor}")

    expected = sum(range(world_size))
    assert tensor.item() == expected, f"Expected {expected}, got {tensor.item()}"

    print(f"Rank {rank}: Success!")
    dist.destroy_process_group()
```

## Common Issues

### Timeout Errors

```python
# Increase timeout
os.environ["GLOO_SOCKET_TIMEOUT_MS"] = "300000"  # 5 minutes

# Or in init
import datetime
dist.init_process_group(
    backend="gloo",
    timeout=datetime.timedelta(minutes=5)
)
```

### Network Interface Selection

```bash
# Specify correct interface
export GLOO_SOCKET_IFNAME=eth0

# Or in code
os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
```

### File Init Method Issues

```python
# Ensure shared filesystem and unique file
init_file = f"/shared/init_{job_id}"

# Clean up old files
if rank == 0 and os.path.exists(init_file):
    os.remove(init_file)
dist.barrier()  # Ensure cleanup before init

dist.init_process_group(
    backend="gloo",
    init_method=f"file://{init_file}"
)
```

## Best Practices

1. **Use NCCL for GPU training**: Gloo is significantly slower
2. **Gloo for CPU workloads**: Appropriate choice for CPU-only
3. **Debug with Gloo first**: Easier to diagnose issues
4. **Set timeouts**: Prevent indefinite hangs
5. **Specify network interface**: Avoid wrong interface selection
6. **Use for object passing**: Good for Python objects between ranks
