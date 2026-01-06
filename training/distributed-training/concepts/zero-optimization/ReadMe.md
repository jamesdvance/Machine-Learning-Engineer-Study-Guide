# ZeRO Optimization

## Summary

ZeRO (Zero Redundancy Optimizer) is a memory optimization technique developed by Microsoft that eliminates redundant storage of model states across data-parallel processes. Standard data parallelism replicates the entire model on each GPU. ZeRO progressively partitions optimizer states, gradients, and parameters across GPUs, dramatically reducing per-device memory while maintaining data parallelism semantics.

Key points to remember:

- ZeRO has three stages with increasing memory savings
- Stage 1: Partition optimizer states (4x memory reduction for Adam)
- Stage 2: Partition gradients (adds 2x reduction)
- Stage 3: Partition parameters (adds another 2x reduction)
- Higher stages increase communication but reduce memory
- DeepSpeed is the primary implementation
- PyTorch FSDP implements ZeRO-3 natively
- ZeRO-Offload extends to CPU/NVMe for extreme memory savings
- ZeRO-Infinity enables training trillion-parameter models

## The Memory Problem

### Standard Data Parallelism Memory

For a model with P parameters using Adam optimizer:

| Component | Memory per GPU | Purpose |
|-----------|----------------|---------|
| Parameters | 4P (FP32) or 2P (FP16) | Model weights |
| Gradients | 4P (FP32) or 2P (FP16) | Computed gradients |
| Optimizer momentum | 4P | Adam first moment |
| Optimizer variance | 4P | Adam second moment |
| Master weights | 4P | FP32 copy for mixed precision |

Total: 16-20 bytes per parameter with mixed precision training.

### The Redundancy

With N GPUs in data parallelism:
- Each GPU stores identical copies
- Total storage: N x 16P bytes
- Redundancy: N copies of the same data

ZeRO eliminates this redundancy by partitioning instead of replicating.

## ZeRO Stages

### ZeRO Stage 1: Optimizer State Partitioning

Each GPU stores only 1/N of the optimizer states:

```
Standard:
GPU 0: [params] [grads] [adam_m] [adam_v]
GPU 1: [params] [grads] [adam_m] [adam_v]
GPU 2: [params] [grads] [adam_m] [adam_v]
GPU 3: [params] [grads] [adam_m] [adam_v]

ZeRO-1:
GPU 0: [params] [grads] [adam_m[0:N/4]] [adam_v[0:N/4]]
GPU 1: [params] [grads] [adam_m[N/4:N/2]] [adam_v[N/4:N/2]]
GPU 2: [params] [grads] [adam_m[N/2:3N/4]] [adam_v[N/2:3N/4]]
GPU 3: [params] [grads] [adam_m[3N/4:N]] [adam_v[3N/4:N]]
```

**Memory reduction**: 4x for optimizer states

**Additional communication**: All-gather optimizer states before update

### ZeRO Stage 2: Gradient Partitioning

Each GPU also stores only 1/N of gradients:

```
ZeRO-2:
GPU 0: [params] [grads[0:N/4]] [adam_m[0:N/4]] [adam_v[0:N/4]]
GPU 1: [params] [grads[N/4:N/2]] [adam_m[N/4:N/2]] [adam_v[N/4:N/2]]
GPU 2: [params] [grads[N/2:3N/4]] [adam_m[N/2:3N/4]] [adam_v[N/2:3N/4]]
GPU 3: [params] [grads[3N/4:N]] [adam_m[3N/4:N]] [adam_v[3N/4:N]]
```

**Memory reduction**: 8x total (4x optimizer + 2x gradients)

**Communication change**: Reduce-scatter instead of all-reduce for gradients

### ZeRO Stage 3: Parameter Partitioning

Each GPU stores only 1/N of parameters:

```
ZeRO-3:
GPU 0: [params[0:N/4]] [grads[0:N/4]] [adam_m[0:N/4]] [adam_v[0:N/4]]
GPU 1: [params[N/4:N/2]] [grads[N/4:N/2]] [adam_m[N/4:N/2]] [adam_v[N/4:N/2]]
GPU 2: [params[N/2:3N/4]] [grads[N/2:3N/4]] [adam_m[N/2:3N/4]] [adam_v[N/2:3N/4]]
GPU 3: [params[3N/4:N]] [grads[3N/4:N]] [adam_m[3N/4:N]] [adam_v[3N/4:N]]
```

**Memory reduction**: 64x total with 8 GPUs (linear in GPU count)

**Communication**: All-gather parameters before forward and backward

## Communication Analysis

### Communication Volume

| Stage | Communication per Step |
|-------|----------------------|
| DDP | 2P (all-reduce gradients) |
| ZeRO-1 | 2P (all-reduce) + P (all-gather optimizer) |
| ZeRO-2 | P (reduce-scatter) + P (all-gather updates) |
| ZeRO-3 | 3P (all-gather forward + backward + reduce-scatter) |

### Latency Considerations

ZeRO-3 has more synchronization points:
- All-gather before each layer forward
- All-gather before each layer backward
- Reduce-scatter after each layer backward

This increases latency sensitivity but can be hidden with prefetching.

## DeepSpeed Implementation

### ZeRO Stage 1 Configuration

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### ZeRO Stage 2 Configuration

```python
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

### ZeRO Stage 3 Configuration

```python
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

### DeepSpeed Training Code

```python
import deepspeed
import torch

model = MyModel()
optimizer = None  # DeepSpeed creates optimizer

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

## ZeRO-Offload

### CPU Offloading

Moves optimizer states and optionally parameters to CPU:

```python
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Benefits**: Further reduces GPU memory
**Cost**: CPU-GPU transfers add latency

### NVMe Offloading (ZeRO-Infinity)

For extreme cases, offload to NVMe storage:

```python
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        }
    }
}
```

**Use case**: Training models that exceed combined GPU + CPU memory

## Optimization Techniques

### Communication Overlap

```python
{
    "zero_optimization": {
        "overlap_comm": true
    }
}
```

Overlaps communication with computation when possible.

### Contiguous Gradients

```python
{
    "zero_optimization": {
        "contiguous_gradients": true
    }
}
```

Allocates gradients in contiguous memory for efficient reduce operations.

### Bucketing

```python
{
    "zero_optimization": {
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

Groups parameters into buckets for more efficient collective operations.

### Prefetching (Stage 3)

```python
{
    "zero_optimization": {
        "stage3_prefetch_bucket_size": 5e7
    }
}
```

Prefetches parameters before they are needed, hiding communication latency.

## Choosing a ZeRO Stage

### Decision Framework

```
Does model fit in GPU with DDP?
  Yes -> Use DDP (simplest)
  No -> How much memory reduction needed?
          Moderate (4x) -> ZeRO-1
          More (8x) -> ZeRO-2
          Maximum -> ZeRO-3

Is communication bandwidth limited?
  Yes -> Stay at lower stages
  No -> Higher stages acceptable

Need to train very large model?
  Yes -> ZeRO-3 + Offload
```

### Stage Comparison

| Aspect | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|--------|
| Memory reduction | 4x | 8x | Linear in N |
| Communication | Low overhead | Moderate | Higher |
| Implementation complexity | Low | Low | Moderate |
| Best for | Medium models | Large models | Very large models |

## Integration with Other Techniques

### ZeRO + Tensor Parallelism

```
Within node: Tensor Parallelism (NVLink)
Across nodes: ZeRO (InfiniBand)
```

ZeRO handles optimizer state sharding, tensor parallelism handles compute.

### ZeRO + Pipeline Parallelism

```python
# DeepSpeed pipeline with ZeRO
{
    "zero_optimization": {
        "stage": 1  # ZeRO-1 compatible with pipeline
    },
    "pipeline": {
        "pipe_partitioned": true,
        "grad_partitioned": true
    }
}
```

Note: ZeRO-2/3 interaction with pipeline parallelism requires careful configuration.

### ZeRO + Gradient Checkpointing

Combine for maximum memory savings:

```python
{
    "zero_optimization": {
        "stage": 3
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true
    }
}
```

## Debugging and Profiling

### Memory Profiling

```python
import deepspeed.comm as dist

# Print memory stats
if dist.get_rank() == 0:
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Communication Profiling

Enable DeepSpeed logging:

```python
{
    "wall_clock_breakdown": true,
    "comms_logger": {
        "enabled": true,
        "verbose": true
    }
}
```

### Common Issues

**Out of memory with ZeRO-3**:
- Reduce batch size
- Enable activation checkpointing
- Use CPU offloading

**Slow training**:
- Check communication bandwidth
- Enable overlap_comm
- Adjust bucket sizes

**Convergence issues**:
- ZeRO should not affect convergence
- Verify same effective batch size
- Check learning rate scaling

## Best Practices

1. **Start simple**: Try ZeRO-1 before jumping to ZeRO-3

2. **Profile first**: Understand memory breakdown before optimizing

3. **Tune bucket sizes**: Match to your network bandwidth

4. **Use overlap**: Enable communication overlap when possible

5. **Combine wisely**: ZeRO + checkpointing + mixed precision for best results

6. **Monitor throughput**: Higher ZeRO stages should not drastically reduce throughput with good network

7. **Test checkpointing**: Verify save/load works before long training runs
