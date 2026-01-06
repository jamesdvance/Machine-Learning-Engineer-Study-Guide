# All-Reduce

## Summary

All-reduce is a collective communication operation that aggregates values from all processes and distributes the result back to all processes. In distributed deep learning, all-reduce is primarily used to synchronize gradients across data-parallel workers, ensuring all replicas have identical parameter updates. Understanding all-reduce is fundamental to understanding distributed training performance.

Key points to remember:

- Combines values from all processes using a reduction operation (usually sum)
- Result is distributed to all processes (unlike reduce which sends to one)
- Primary use: gradient synchronization in data parallelism
- Ring-allreduce is the bandwidth-optimal implementation
- Communication volume: 2(N-1)/N times data size (approaches 2x for large N)
- Latency scales with number of processes
- NCCL provides highly optimized GPU implementation
- Can be overlapped with computation for better performance

## The All-Reduce Operation

### Definition

Given N processes, each with a local tensor:
```
Process 0: x_0
Process 1: x_1
...
Process N-1: x_{N-1}
```

After all-reduce with SUM:
```
All processes have: x_0 + x_1 + ... + x_{N-1}
```

### Supported Operations

| Operation | Result |
|-----------|--------|
| SUM | Sum of all tensors |
| PRODUCT | Product of all tensors |
| MIN | Element-wise minimum |
| MAX | Element-wise maximum |
| AVG | Average (SUM/N) |

For gradient synchronization, SUM or AVG is used.

### Visual Representation

```
Before all-reduce:
  Process 0: [1, 2, 3]
  Process 1: [4, 5, 6]
  Process 2: [7, 8, 9]

All-reduce (SUM):
  Process 0: [12, 15, 18]
  Process 1: [12, 15, 18]
  Process 2: [12, 15, 18]
```

## Role in Distributed Training

### Data Parallel Gradient Sync

In data parallel training:

```python
# Each process computes local gradients
loss = model(local_batch)
loss.backward()  # Computes local gradients

# All-reduce to get average gradient
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size  # Average

# Now all processes have identical gradients
optimizer.step()  # Identical updates
```

### Why All-Reduce (Not Reduce)

**Reduce**: Only one process gets the result
```
Process 0 gets result -> broadcasts to others -> extra step
```

**All-reduce**: All processes get the result directly
```
All processes get result -> no broadcast needed
```

All-reduce is more efficient for the data parallel pattern.

## Implementation Approaches

### Naive All-Reduce

```python
def naive_all_reduce(tensor, world_size, rank):
    # Each process sends to rank 0
    if rank != 0:
        dist.send(tensor, dst=0)
    else:
        result = tensor.clone()
        for i in range(1, world_size):
            recv_tensor = torch.empty_like(tensor)
            dist.recv(recv_tensor, src=i)
            result += recv_tensor

    # Rank 0 broadcasts result
    if rank == 0:
        for i in range(1, world_size):
            dist.send(result, dst=i)
    else:
        dist.recv(tensor, src=0)

    return tensor
```

**Problem**: Rank 0 is a bottleneck. Communication: O(N x data_size)

### Tree-Based All-Reduce

Organize processes in a tree for parallel aggregation:

```
        [Result]
       /        \
    [Sum01]    [Sum23]
    /    \     /    \
  [P0]  [P1] [P2]  [P3]
```

Better parallelism but still not bandwidth-optimal.

### Ring-AllReduce

Bandwidth-optimal approach (see Ring-AllReduce chapter for details):

```
Phase 1: Reduce-scatter
  Each process ends with 1/N of the final result

Phase 2: All-gather
  Distribute chunks to all processes
```

Communication: 2(N-1)/N x data_size, independent of N for large N.

## PyTorch Implementation

### Basic Usage

```python
import torch
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")

# Create tensor
tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")

# All-reduce (in-place)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# tensor now contains sum across all processes
```

### Async All-Reduce

```python
# Non-blocking all-reduce
handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

# Do other work while communication happens
other_computation()

# Wait for completion
handle.wait()
```

### Grouped All-Reduce

```python
# All-reduce multiple tensors efficiently
tensors = [tensor1, tensor2, tensor3]

handles = []
for t in tensors:
    h = dist.all_reduce(t, async_op=True)
    handles.append(h)

# Wait for all
for h in handles:
    h.wait()
```

## Communication Costs

### Time Complexity

```
All-reduce time = latency + data_size / bandwidth

For ring-allreduce:
  latency = 2(N-1) x alpha  (message startups)
  bandwidth = 2(N-1)/N x data_size / beta

Where:
  N = number of processes
  alpha = per-message latency
  beta = bandwidth
```

### Practical Measurements

| Data Size | 8 GPUs (NVLink) | 8 Nodes (InfiniBand) |
|-----------|-----------------|----------------------|
| 1 MB | ~0.1 ms | ~1 ms |
| 100 MB | ~5 ms | ~50 ms |
| 1 GB | ~50 ms | ~500 ms |

Actual times depend heavily on hardware and implementation.

### Scaling Behavior

For ring-allreduce with N processes:

- **Latency**: Increases linearly with N
- **Bandwidth utilization**: Approaches optimal (2x data) for large data
- **Total time**: Dominated by bandwidth for large tensors

## Optimization Techniques

### Gradient Bucketing

Group small gradients into buckets:

```python
# Instead of many small all-reduces:
for param in model.parameters():
    dist.all_reduce(param.grad)  # Many calls

# Bucket into larger tensors:
bucket_size = 25 * 1024 * 1024  # 25 MB
bucket = []
bucket_bytes = 0

for param in model.parameters():
    bucket.append(param.grad.view(-1))
    bucket_bytes += param.grad.numel() * param.grad.element_size()

    if bucket_bytes >= bucket_size:
        flat = torch.cat(bucket)
        dist.all_reduce(flat)
        # Unflatten and copy back
        bucket = []
        bucket_bytes = 0
```

PyTorch DDP does this automatically.

### Overlap with Computation

Start all-reduce while backward pass continues:

```
Layer N:   Backward -> Start all-reduce for grad_N
Layer N-1: Backward -> Start all-reduce for grad_{N-1}
...
```

DDP implements this via backward hooks.

### Compression

Reduce communication volume:

```python
# FP16 compression
def fp16_compress_hook(state, bucket):
    compressed = bucket.buffer().half()
    dist.all_reduce(compressed)
    return compressed.float()

model.register_comm_hook(state=None, hook=fp16_compress_hook)
```

## Related Collective Operations

### Reduce

Sum to one process only:
```
Before: P0:[1], P1:[2], P2:[3]
After:  P0:[6], P1:[2], P2:[3]  (only P0 has result)
```

### Broadcast

One process sends to all:
```
Before: P0:[5], P1:[?], P2:[?]
After:  P0:[5], P1:[5], P2:[5]
```

### All-Gather

Collect all data to all processes:
```
Before: P0:[A], P1:[B], P2:[C]
After:  P0:[A,B,C], P1:[A,B,C], P2:[A,B,C]
```

### Reduce-Scatter

Reduce and scatter different chunks:
```
Before: P0:[1,2], P1:[3,4], P2:[5,6]
Reduce: [1+3+5, 2+4+6] = [9, 12]
After:  P0:[9], P1:[12], P2:[remaining]
```

Used in FSDP/ZeRO for gradient sharding.

## Debugging

### Verify All-Reduce Results

```python
# Check that all ranks have same result
local_tensor = torch.tensor([rank], device="cuda", dtype=torch.float)
dist.all_reduce(local_tensor)

expected = sum(range(world_size))
assert local_tensor.item() == expected, f"Rank {rank}: got {local_tensor.item()}"
```

### Profile All-Reduce Time

```python
import time

torch.cuda.synchronize()
start = time.time()

dist.all_reduce(tensor)

torch.cuda.synchronize()
elapsed = time.time() - start

if rank == 0:
    size_mb = tensor.numel() * tensor.element_size() / 1e6
    print(f"All-reduce: {size_mb:.1f} MB in {elapsed*1000:.1f} ms")
    print(f"Effective bandwidth: {2 * size_mb / elapsed / 1000:.1f} GB/s")
```

### Common Issues

**Hang/deadlock**:
- All ranks must participate in collective
- Check for conditional execution that differs across ranks

**Wrong results**:
- Check tensor dtype and device consistency
- Verify same tensor shapes across ranks

**Slow performance**:
- Check network bandwidth
- Profile to identify if latency or bandwidth bound
- Consider bucketing small tensors
