# Right-Sizing GPU Selection

## Summary

Choosing the right GPU for ML training involves balancing performance, memory, and cost. Over-provisioning wastes money while under-provisioning causes training failures or inefficiency. Understanding GPU characteristics and matching them to workload requirements enables optimal cost-performance tradeoffs.

Key points to remember:

- GPU memory determines maximum model/batch size
- Compute throughput (TFLOPS) affects training speed
- Memory bandwidth impacts memory-bound operations
- Interconnect speed matters for multi-GPU training
- Newer architectures have better efficiency
- Match GPU tier to model complexity
- Consider availability and spot pricing
- Profile before scaling up

## GPU Comparison

### Current Generation GPUs

| GPU | Memory | FP16 TFLOPS | Memory BW | Use Case |
|-----|--------|-------------|-----------|----------|
| T4 | 16 GB | 65 | 320 GB/s | Inference, small training |
| A10G | 24 GB | 125 | 600 GB/s | Medium training |
| L4 | 24 GB | 121 | 300 GB/s | Inference, training |
| A100 40GB | 40 GB | 312 | 1.6 TB/s | Large training |
| A100 80GB | 80 GB | 312 | 2.0 TB/s | Very large training |
| H100 | 80 GB | 990 | 3.4 TB/s | Cutting-edge training |

### Price-Performance Ratio

| GPU | $/hr (Spot) | TFLOPS/$ | GB/$ |
|-----|-------------|----------|------|
| T4 | $0.12 | 540 | 133 |
| A10G | $0.35 | 357 | 69 |
| A100 40GB | $1.00 | 312 | 40 |
| A100 80GB | $1.30 | 240 | 62 |
| H100 | $1.50 | 660 | 53 |

*T4 best $/TFLOPS, but H100 best for large models*

## Selection Framework

### By Model Size

```
Model Parameters -> Recommended GPU

< 500M:
  -> T4 (16GB) sufficient
  -> Best cost efficiency

500M - 3B:
  -> A10G (24GB) or T4 with gradient checkpointing
  -> Balance of cost and capability

3B - 7B:
  -> A100 40GB or A10G with memory optimization
  -> May need multiple A10Gs

7B - 13B:
  -> A100 80GB recommended
  -> A100 40GB with extensive optimization

13B - 30B:
  -> Multiple A100 80GB with tensor parallelism
  -> H100 for faster training

30B+:
  -> Multiple H100s with 3D parallelism
  -> Consider cloud TPUs
```

### By Workload Type

```
Training from scratch:
  -> Maximize compute (A100, H100)
  -> Training time dominates cost

Fine-tuning:
  -> Memory-focused (fits model + optimizer)
  -> A10G often sufficient

Inference:
  -> Optimize for latency or throughput
  -> T4/L4 for cost, A10G/A100 for speed

Batch inference:
  -> Maximize throughput
  -> Multiple smaller GPUs may be better
```

## Memory Requirements

### Estimate Training Memory

```python
def estimate_gpu_memory(
    num_params,
    batch_size,
    seq_len,
    hidden_dim,
    num_layers,
    precision='fp16',
    optimizer='adam',
    gradient_checkpointing=False
):
    """Estimate GPU memory for training."""
    bytes_per_param = 2 if precision == 'fp16' else 4

    # Model components
    params = num_params * bytes_per_param
    grads = num_params * bytes_per_param

    # Optimizer state
    if optimizer == 'adam':
        opt_state = num_params * 8  # FP32 momentum + variance
    elif optimizer == 'sgd':
        opt_state = num_params * 4  # FP32 momentum
    else:
        opt_state = 0

    # Activations (rough estimate)
    if gradient_checkpointing:
        # Only store layer inputs
        activations = batch_size * seq_len * hidden_dim * num_layers * bytes_per_param
    else:
        # Store all intermediate activations
        activations = batch_size * seq_len * hidden_dim * num_layers * 10 * bytes_per_param

    total = params + grads + opt_state + activations

    print(f"Parameters: {params / 1e9:.2f} GB")
    print(f"Gradients: {grads / 1e9:.2f} GB")
    print(f"Optimizer: {opt_state / 1e9:.2f} GB")
    print(f"Activations: {activations / 1e9:.2f} GB")
    print(f"Total: {total / 1e9:.2f} GB")

    return total

# Example: 7B model
estimate_gpu_memory(
    num_params=7e9,
    batch_size=4,
    seq_len=2048,
    hidden_dim=4096,
    num_layers=32,
    gradient_checkpointing=True
)
```

### Memory-Constrained Options

```python
# If model doesn't fit, options in order of preference:

# 1. Enable gradient checkpointing (30% compute overhead)
model.gradient_checkpointing_enable()

# 2. Use mixed precision (halves activation memory)
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    output = model(input)

# 3. Reduce batch size (may affect convergence)
batch_size = batch_size // 2

# 4. Use smaller model variant
# 7B -> 3B if quality acceptable

# 5. Use model parallelism (adds complexity)
from torch.distributed.fsdp import FullyShardedDataParallel
```

## Compute Efficiency

### Measure Utilization

```python
import torch
import time

def benchmark_gpu(model, input_shape, num_iterations=100):
    """Measure GPU utilization."""
    device = next(model.parameters()).device

    # Warmup
    x = torch.randn(*input_shape, device=device)
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        output = model(x)
        output.sum().backward()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate throughput
    samples_per_second = num_iterations * input_shape[0] / elapsed
    print(f"Throughput: {samples_per_second:.1f} samples/sec")

    return samples_per_second
```

### GPU Utilization Guidelines

```
Target utilization: > 80%

Low utilization causes:
- Data loading bottleneck -> Use more workers
- Small batch size -> Increase batch or use gradient accumulation
- CPU-bound preprocessing -> Prefetch data
- Memory bandwidth bound -> Use tensor cores (FP16/BF16)

Check utilization:
$ nvidia-smi dmon -s u
```

## Multi-GPU Considerations

### Scaling Efficiency

```
GPUs  |  Ideal Speedup  |  Typical Speedup  |  Efficiency
------|-----------------|-------------------|-------------
1     |  1.0x           |  1.0x             |  100%
2     |  2.0x           |  1.9x             |  95%
4     |  4.0x           |  3.6x             |  90%
8     |  8.0x           |  6.8x             |  85%
16    |  16.0x          |  12.8x            |  80%
```

### When to Scale Out vs Up

```
Scale Up (bigger GPU):
- Memory-limited workloads
- Better efficiency (no communication)
- Simpler infrastructure

Scale Out (more GPUs):
- Compute-limited workloads
- Need distributed parallelism
- Cost optimization (spot availability)
```

## Cloud-Specific Guidance

### AWS Instance Selection

| Instance | GPUs | GPU Type | Use Case |
|----------|------|----------|----------|
| g4dn.xlarge | 1 | T4 | Development |
| g5.xlarge | 1 | A10G | Small training |
| p3.2xlarge | 1 | V100 | Medium training |
| p4d.24xlarge | 8 | A100 | Large training |
| p5.48xlarge | 8 | H100 | Cutting-edge |

### GCP Instance Selection

| Instance | GPUs | GPU Type | Use Case |
|----------|------|----------|----------|
| n1-standard-4 + T4 | 1-4 | T4 | Development |
| a2-highgpu-1g | 1 | A100 | Training |
| a2-ultragpu-1g | 1 | A100 80GB | Large training |
| a3-highgpu-8g | 8 | H100 | Cutting-edge |

## Decision Flowchart

```
Start
  |
  v
Does model fit in 16GB with optimizations?
  |
  Yes -> T4 (cheapest)
  No
  |
  v
Does model fit in 24GB?
  |
  Yes -> A10G (good value)
  No
  |
  v
Does model fit in 40GB?
  |
  Yes -> A100 40GB
  No
  |
  v
Does model fit in 80GB?
  |
  Yes -> A100 80GB or H100
  No
  |
  v
Multi-GPU with model parallelism
```

## Best Practices

1. **Profile first**: Measure actual memory usage
2. **Start small**: Validate on smaller GPU
3. **Consider spot pricing**: Price varies by GPU
4. **Check availability**: Some GPUs scarce
5. **Optimize before scaling**: Often cheaper
6. **Test scaling efficiency**: Measure actual speedup
7. **Monitor utilization**: Ensure GPUs are used
8. **Re-evaluate periodically**: Prices change
