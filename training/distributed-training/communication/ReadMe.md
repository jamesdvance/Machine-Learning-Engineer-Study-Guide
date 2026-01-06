# Communication Backends

## Summary

Communication backends provide the low-level primitives for inter-process communication in distributed training. The choice of backend significantly impacts training performance, especially at scale. Modern deep learning primarily uses three backends: NCCL for GPU-to-GPU communication, Gloo for CPU and mixed workloads, and MPI for HPC environments.

Key points to remember:

- NCCL is the standard for GPU training, highly optimized for NVIDIA hardware
- Gloo provides CPU communication and GPU fallback
- MPI offers portability and HPC integration
- Backend choice affects collective operation performance
- NCCL required for efficient multi-GPU training
- Gloo useful for CPU-only or heterogeneous setups
- MPI common in traditional HPC environments
- Most frameworks default to NCCL for GPU workloads

## Backend Comparison

| Backend | Primary Use | Strengths | Limitations |
|---------|-------------|-----------|-------------|
| NCCL | GPU-to-GPU | Fastest for NVIDIA GPUs | NVIDIA-only |
| Gloo | CPU, fallback | Cross-platform | Slower for GPU |
| MPI | HPC clusters | Portability, features | Setup complexity |

## Collective Operations

All backends implement standard collective operations:

### Point-to-Point
- **Send/Recv**: Direct transfer between two processes

### Collectives
- **Broadcast**: One-to-all distribution
- **Reduce**: Many-to-one aggregation
- **All-reduce**: Reduce + broadcast (all processes get result)
- **All-gather**: Gather data from all to all
- **Reduce-scatter**: Reduce and scatter chunks
- **Barrier**: Synchronization point

## Backend Selection

### PyTorch

```python
import torch.distributed as dist

# GPU training: NCCL
dist.init_process_group(backend="nccl")

# CPU training: Gloo
dist.init_process_group(backend="gloo")

# HPC environment: MPI (if built with MPI support)
dist.init_process_group(backend="mpi")
```

### Multiple Backends

PyTorch supports different backends for different operations:

```python
# NCCL for GPU collectives, Gloo for CPU
dist.init_process_group(backend="nccl")

# Create Gloo process group for CPU tensors
gloo_group = dist.new_group(backend="gloo")

# Use appropriate backend based on tensor device
if tensor.is_cuda:
    dist.all_reduce(tensor)  # Uses NCCL
else:
    dist.all_reduce(tensor, group=gloo_group)  # Uses Gloo
```

## Performance Characteristics

### Bandwidth

Typical bandwidth for all-reduce (large tensors):

| Backend | Intra-node (NVLink) | Inter-node (IB) |
|---------|---------------------|-----------------|
| NCCL | 500+ GB/s | 20-50 GB/s |
| Gloo (GPU) | 50-100 GB/s | 10-20 GB/s |
| MPI | Varies | Varies |

### Latency

For small tensors (latency-dominated):

| Backend | Typical Latency |
|---------|----------------|
| NCCL | 10-50 microseconds |
| Gloo | 50-200 microseconds |
| MPI | 1-10 microseconds |

MPI has the lowest latency for small messages.

## Configuration

### NCCL Environment Variables

```bash
# Debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Network configuration
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0    # Network interface

# Performance tuning
export NCCL_ALGO=Ring             # Algorithm selection
export NCCL_PROTO=Simple          # Protocol selection
```

### Gloo Environment Variables

```bash
# Network interface
export GLOO_SOCKET_IFNAME=eth0

# Timeout
export GLOO_SOCKET_TIMEOUT_MS=60000
```

### MPI Configuration

```bash
# OpenMPI example
mpirun --mca btl_tcp_if_include eth0 python train.py
```

## Debugging Communication Issues

### NCCL Debugging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_debug.%h.%p

# Check for errors
grep -i error /tmp/nccl_debug.*
```

### Identifying Bottlenecks

```python
import time
import torch.distributed as dist

def profile_allreduce(tensor, name=""):
    torch.cuda.synchronize()
    start = time.time()

    dist.all_reduce(tensor)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    size_mb = tensor.numel() * tensor.element_size() / 1e6
    bandwidth = 2 * size_mb / elapsed / 1000  # GB/s

    if dist.get_rank() == 0:
        print(f"{name}: {size_mb:.1f}MB, {elapsed*1000:.1f}ms, {bandwidth:.1f}GB/s")
```

### Common Issues

**Hang/deadlock**:
- All ranks must call collective operations
- Check for conditional code that differs between ranks
- Set NCCL_BLOCKING_WAIT=1 for timeout

**Performance degradation**:
- Check network interface selection
- Verify InfiniBand/NVLink availability
- Profile to identify slow operations

**Memory errors**:
- Ensure tensors on correct devices
- Check for shape mismatches
- Verify contiguous memory

## Further Reading

Detailed coverage of each backend:
- [NCCL](nccl/ReadMe.md): NVIDIA Collective Communications Library
- [Gloo](gloo/ReadMe.md): Facebook's collective communications library
- [MPI](mpi/ReadMe.md): Message Passing Interface
