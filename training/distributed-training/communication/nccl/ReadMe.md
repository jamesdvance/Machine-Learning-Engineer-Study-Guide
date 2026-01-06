# NCCL (NVIDIA Collective Communications Library)

## Summary

NCCL is NVIDIA's library for multi-GPU and multi-node collective communication, optimized for NVIDIA hardware. It provides highly efficient implementations of collective operations like all-reduce, all-gather, and broadcast, automatically utilizing the best available interconnects (NVLink, NVSwitch, InfiniBand). NCCL is the standard communication backend for GPU-based distributed deep learning.

Key points to remember:

- Optimized for NVIDIA GPUs with automatic topology detection
- Utilizes NVLink, NVSwitch, PCIe, and InfiniBand
- Implements ring-allreduce and tree-based algorithms
- Automatic algorithm selection based on message size and topology
- Asynchronous operations with CUDA stream integration
- Standard backend for PyTorch, TensorFlow distributed training
- Extensive tuning via environment variables
- Critical for achieving good scaling efficiency

## Key Features

### Automatic Topology Detection

NCCL automatically detects:
- GPU interconnect topology (NVLink, PCIe)
- Network topology (InfiniBand, Ethernet)
- Optimal communication paths

### Supported Operations

| Operation | Description |
|-----------|-------------|
| AllReduce | Reduce + broadcast to all |
| Broadcast | One to all |
| Reduce | All to one |
| AllGather | Gather to all |
| ReduceScatter | Reduce and scatter |
| Send/Recv | Point-to-point |

### Algorithm Selection

NCCL chooses algorithms based on:
- Message size (tree for small, ring for large)
- Number of GPUs
- Network topology
- Available bandwidth

## Usage in PyTorch

### Initialization

```python
import torch.distributed as dist

# Initialize with NCCL backend
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=world_size,
    rank=rank
)
```

### Collective Operations

```python
# All-reduce
tensor = torch.randn(1000, 1000, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast
if rank == 0:
    tensor = torch.randn(1000, device='cuda')
else:
    tensor = torch.empty(1000, device='cuda')
dist.broadcast(tensor, src=0)

# All-gather
output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
dist.all_gather(output_tensors, tensor)

# Reduce-scatter
input_tensor = torch.randn(world_size * 100, device='cuda')
output_tensor = torch.empty(100, device='cuda')
dist.reduce_scatter(output_tensor, [input_tensor])
```

### Asynchronous Operations

```python
# Non-blocking all-reduce
handle = dist.all_reduce(tensor, async_op=True)

# Do other work
other_computation()

# Wait for completion
handle.wait()
```

## Environment Variables

### Debug and Logging

```bash
# Debug output level
export NCCL_DEBUG=INFO        # INFO, WARN, TRACE
export NCCL_DEBUG_SUBSYS=ALL  # INIT, COLL, P2P, SHM, NET, GRAPH

# Log to file
export NCCL_DEBUG_FILE=/tmp/nccl_debug.%h.%p
```

### Network Configuration

```bash
# InfiniBand settings
export NCCL_IB_DISABLE=0           # Enable InfiniBand (default)
export NCCL_IB_HCA=mlx5_0          # Specific HCA device
export NCCL_IB_GID_INDEX=3         # GID index for RoCE

# Socket settings
export NCCL_SOCKET_IFNAME=eth0     # Network interface
export NCCL_SOCKET_NTHREADS=4      # Socket threads

# GPU Direct RDMA
export NCCL_NET_GDR_LEVEL=5        # GPU Direct RDMA level
export NCCL_NET_GDR_READ=1         # Enable GDR read
```

### Algorithm Tuning

```bash
# Algorithm selection
export NCCL_ALGO=Ring              # Ring, Tree, CollnetDirect, CollnetChain
export NCCL_PROTO=Simple           # Simple, LL, LL128

# Tree algorithm settings
export NCCL_TREE_THRESHOLD=0       # Force tree for small messages

# Buffer sizes
export NCCL_BUFFSIZE=4194304       # 4MB buffer
```

### Performance Tuning

```bash
# Thread settings
export NCCL_NTHREADS=512           # Threads per block
export NCCL_MAX_NCHANNELS=32       # Maximum channels

# Cross-NIC settings
export NCCL_CROSS_NIC=1            # Enable cross-NIC communication

# P2P settings
export NCCL_P2P_DISABLE=0          # Enable P2P (NVLink)
export NCCL_P2P_LEVEL=SYS          # P2P level
export NCCL_SHM_DISABLE=0          # Enable shared memory
```

## Topology and Performance

### NVLink Topology

DGX A100 8-GPU topology:
```
GPU0 -- GPU1 -- GPU2 -- GPU3
  |       |       |       |
GPU4 -- GPU5 -- GPU6 -- GPU7
```

NCCL automatically uses optimal paths.

### Bandwidth Expectations

| Connection | Bandwidth |
|------------|-----------|
| NVLink 3.0 | 600 GB/s total |
| NVLink 4.0 | 900 GB/s total |
| PCIe 4.0 x16 | 32 GB/s |
| InfiniBand HDR | 200 Gbps (~25 GB/s) |
| InfiniBand NDR | 400 Gbps (~50 GB/s) |

### Performance Profiling

```python
def profile_nccl_bandwidth(sizes=[1e6, 1e7, 1e8, 1e9]):
    for size in sizes:
        tensor = torch.randn(int(size), device='cuda')

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Profile
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            dist.all_reduce(tensor)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 10 / 1000  # seconds

        size_gb = size * 4 / 1e9  # FP32
        # All-reduce moves 2x data (ring algorithm)
        bandwidth = 2 * size_gb / elapsed

        if dist.get_rank() == 0:
            print(f"Size: {size_gb:.2f}GB, Time: {elapsed*1000:.1f}ms, BW: {bandwidth:.1f}GB/s")
```

## NCCL Communication Patterns

### Ring All-Reduce

For large messages, NCCL uses ring-allreduce:
```
GPU0 -> GPU1 -> GPU2 -> GPU3 -> GPU0

Phase 1: Reduce-scatter (N-1 steps)
Phase 2: All-gather (N-1 steps)
```

### Tree All-Reduce

For small messages, tree algorithms have lower latency:
```
        GPU0
       /    \
    GPU1    GPU2
    /  \    /  \
 GPU3 GPU4 GPU5 GPU6
```

### Double Binary Tree

NCCL uses double binary trees for best latency:
```
Two overlapping trees provide full bandwidth utilization
while maintaining logarithmic latency
```

## Multi-Node Communication

### InfiniBand Setup

```bash
# Verify IB devices
ibstat

# Check connectivity
ibping -S  # Server
ibping -L <lid>  # Client

# NCCL configuration
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1
export NCCL_NET_GDR_LEVEL=5
```

### RoCE (RDMA over Converged Ethernet)

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3  # RoCE v2
```

### TCP/IP Fallback

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

## Debugging

### Common Issues

**Initialization hang**:
```bash
# Enable debug output
export NCCL_DEBUG=INFO
# Check for network issues
export NCCL_SOCKET_IFNAME=<correct_interface>
```

**Timeout**:
```bash
# Increase timeout
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

**Performance issues**:
```bash
# Check algorithm selection
export NCCL_DEBUG=INFO
# Look for "Using" messages in output
```

### Verifying NCCL Setup

```python
import torch
import torch.distributed as dist

def verify_nccl():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Test all-reduce
    tensor = torch.tensor([rank], device='cuda', dtype=torch.float)
    dist.all_reduce(tensor)
    expected = sum(range(world_size))

    if tensor.item() == expected:
        print(f"Rank {rank}: NCCL all-reduce verified!")
    else:
        print(f"Rank {rank}: NCCL error! Expected {expected}, got {tensor.item()}")

    dist.destroy_process_group()
```

## Best Practices

1. **Use NVLink when available**: Ensure NCCL detects NVLink topology
2. **Enable GPU Direct RDMA**: For multi-node with InfiniBand
3. **Profile before tuning**: Measure baseline performance first
4. **Match NCCL version**: Ensure compatible versions across nodes
5. **Set correct interfaces**: Specify NCCL_SOCKET_IFNAME for multi-NIC
6. **Monitor with debug**: Use NCCL_DEBUG=INFO during development
7. **Use async operations**: Overlap communication with computation

## Version Compatibility

| CUDA | Recommended NCCL |
|------|-----------------|
| 12.x | 2.18+ |
| 11.x | 2.14+ |

Check NCCL version:
```python
import torch
print(torch.cuda.nccl.version())
```
