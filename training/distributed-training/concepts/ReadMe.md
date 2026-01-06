# Distributed Training Concepts

## Summary

Distributed training concepts form the foundation for understanding how to scale model training across multiple devices and machines. The core ideas revolve around partitioning work (data, model, or both), managing communication overhead, and optimizing memory usage. Understanding these concepts enables informed decisions about which parallelism strategies and frameworks to use for specific training scenarios.

Key points to remember:

- Data parallelism is the simplest and most common approach; use when model fits in memory
- Model and tensor parallelism address memory constraints when models are too large
- Pipeline parallelism improves utilization by overlapping computation across stages
- ZeRO and FSDP reduce memory by sharding optimizer states and optionally parameters
- Gradient accumulation simulates larger batch sizes without additional memory
- All-reduce and ring-allreduce are fundamental operations for gradient synchronization
- Choice of parallelism strategy depends on model size, hardware topology, and batch requirements
- Hybrid approaches (3D parallelism) combine strategies for maximum scale

## Parallelism Strategy Overview

### When to Use Each Strategy

| Strategy | Use Case | Memory Benefit | Communication Cost |
|----------|----------|----------------|-------------------|
| Data Parallelism | Model fits in GPU | None | Gradients once per step |
| Model Parallelism | Deep models | Splits activations | Activations per layer |
| Tensor Parallelism | Wide layers | Splits layer memory | Within each layer |
| Pipeline Parallelism | Deep models | Limits micro-batch memory | Activations at boundaries |
| ZeRO/FSDP | Large optimizer states | Up to 8x reduction | Parameters before forward |

### Decision Framework

```
Does model fit in single GPU?
  Yes -> Data Parallelism (DDP)
  No -> Are optimizer states the bottleneck?
          Yes -> ZeRO Stage 1-2 or FSDP
          No -> Is model too deep or too wide?
                  Too deep -> Pipeline Parallelism
                  Too wide -> Tensor Parallelism
                  Both -> 3D Parallelism
```

### Combining Strategies

Strategies can and often should be combined:

**Data + ZeRO**: Most common for medium-large models
- ZeRO reduces memory within data-parallel group
- Maintains data parallelism benefits

**Tensor + Pipeline**: Common for very large models
- Tensor parallel within node (fast interconnect)
- Pipeline parallel across nodes

**Data + Tensor + Pipeline (3D)**: Maximum scale
- Data parallel across pipeline replicas
- Tensor parallel within pipeline stages
- Pipeline parallel across stage groups

## Memory Breakdown

Understanding memory consumption helps choose strategies:

### Per-GPU Memory Components

For a model with P parameters:

| Component | FP32 | FP16/BF16 | Mixed Precision |
|-----------|------|-----------|-----------------|
| Parameters | 4P | 2P | 2P (FP16) + 4P (master) |
| Gradients | 4P | 2P | 2P |
| Adam momentum | 4P | 4P | 4P |
| Adam variance | 4P | 4P | 4P |
| **Subtotal** | 16P | 12P | 16P |

Activations add significantly more, depending on batch size and sequence length.

### Example: 7B Parameter Model

| Component | Memory |
|-----------|--------|
| Parameters (FP16) | 14 GB |
| Master weights (FP32) | 28 GB |
| Gradients (FP16) | 14 GB |
| Adam states | 56 GB |
| **Total (no activations)** | 112 GB |

This already exceeds 80GB A100 capacity, requiring distribution.

## Communication Overhead

### Bandwidth Requirements

| Operation | Data Volume | Frequency |
|-----------|-------------|-----------|
| All-reduce (DDP) | 2 x model size | Once per step |
| All-gather (FSDP) | Model size | Before each forward |
| Reduce-scatter (FSDP) | Model size | After each backward |
| Point-to-point (Pipeline) | Activation size | Per micro-batch, per stage |

### Network Topology Impact

**Intra-node** (NVLink):
- 600+ GB/s bidirectional
- Low latency
- Ideal for tensor parallelism

**Inter-node** (InfiniBand):
- 200-400 Gbps (25-50 GB/s)
- Higher latency
- Suitable for data/pipeline parallelism

**Ethernet**:
- 10-100 Gbps (1-12 GB/s)
- Highest latency
- Limits scaling efficiency

### Optimizing Communication

**Overlap computation and communication**:
- Start gradient all-reduce while still computing later layers
- Pre-fetch parameters in FSDP during forward pass

**Gradient compression**:
- Reduce precision for communication
- Use sparse gradients where applicable

**Batching collectives**:
- Group small tensors into larger operations
- Reduces number of synchronization points

## Synchronization Semantics

### Synchronous Training

All devices wait for each other:
```
Step N:
  All devices: Forward -> Backward -> All-reduce -> Update
  |          synchronized           |
Step N+1:
  ...
```

**Pros**: Equivalent to single-device training mathematically
**Cons**: Slowest device determines speed

### Asynchronous Training

Devices proceed independently:
```
Device 0: F -> B -> Update -> F -> B -> Update
Device 1:   F -> B -> Update -> F -> B -> Update
Device 2:     F -> B -> Update -> F -> B -> Update
```

**Pros**: No waiting on stragglers
**Cons**: Stale gradients, convergence issues

Most modern systems use synchronous training with techniques to mitigate straggler effects.

### Local SGD

Hybrid approach:
1. Devices train independently for K steps
2. Synchronize parameters every K steps
3. Average parameters across devices

Reduces communication while maintaining convergence for many tasks.

## Batch Size Considerations

### Effective Batch Size

```
effective_batch = per_device_batch x devices x gradient_accumulation
```

### Scaling Rules

When increasing batch size, learning rate often needs adjustment:

**Linear scaling**: LR proportional to batch size
- Works well up to moderate batch sizes
- May diverge at very large batches

**Square root scaling**: LR proportional to sqrt(batch size)
- More conservative
- Better stability at large batches

**Warmup**: Gradually increase LR at start
- Helps with large batch stability
- Typically 1-5% of total steps

### Batch Size Limits

Very large batches can hurt generalization:
- Effective batch sizes above 32K-64K often degrade
- Model and task dependent
- May need to reduce LR or use other techniques

## Practical Considerations

### Checkpointing Across Ranks

Distributed checkpointing patterns:

**Full checkpoint per rank**:
- Simple but redundant
- Good for small models

**Sharded checkpoints**:
- Each rank saves its shard
- Required for FSDP/ZeRO
- Needs consolidation for inference

**Checkpoint on rank 0**:
- Gather to single rank
- Memory pressure on rank 0
- Simpler for inference deployment

### Reproducibility

Ensuring reproducible distributed training:

```python
# Set seeds on all ranks
torch.manual_seed(seed + rank)  # Different data per rank
torch.cuda.manual_seed(seed)    # Same initialization

# Or for identical behavior
torch.manual_seed(seed)         # Same everywhere
# Control data sampling separately
```

### Debugging Distributed Code

Common debugging approaches:

**Print with rank**:
```python
if dist.get_rank() == 0:
    print(f"Loss: {loss.item()}")
```

**Verify synchronization**:
```python
local_tensor = torch.tensor([value], device=device)
dist.all_reduce(local_tensor)
assert local_tensor.item() == expected_sum
```

**Check for NaN**:
```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Rank {rank}: NaN/Inf detected")
    dist.barrier()  # Sync before potentially crashing
```

## Further Reading

### Core Parallelism Strategies
- [Data Parallelism](data-parallelism/ReadMe.md): Replicating models across devices
- [Model Parallelism](model-parallelism/ReadMe.md): Splitting layers across devices
- [Tensor Parallelism](tensor-parallelism/ReadMe.md): Splitting operations within layers
- [Pipeline Parallelism](pipeline-parallelism/ReadMe.md): Micro-batch pipelining

### Memory Optimization
- [FSDP](fsdp/ReadMe.md): Fully Sharded Data Parallel
- [ZeRO Optimization](zero-optimization/ReadMe.md): Zero Redundancy Optimizer
- [Gradient Accumulation](gradient-accumulation/ReadMe.md): Simulating large batches

### Communication Operations
- [All-Reduce](all-reduce/ReadMe.md): Gradient aggregation primitive
- [Ring-AllReduce](ring-allreduce/ReadMe.md): Bandwidth-optimal reduction
