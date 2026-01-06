# Offloading

## Summary

Offloading extends GPU memory by utilizing CPU RAM or NVMe storage to hold model states that do not fit in GPU memory. While offloading enables training models that would otherwise be impossible, it comes with significant performance costs due to data transfer overhead. Offloading should be considered a last resort when other memory optimizations are insufficient.

Key points to remember:

- Extends effective memory beyond GPU capacity
- CPU offloading: Move optimizer states and/or parameters to CPU RAM
- NVMe offloading: Move to SSD storage for even larger models
- Significant performance penalty (2-10x slowdown typical)
- DeepSpeed ZeRO-Offload and ZeRO-Infinity are primary implementations
- FSDP supports CPU offloading
- Best for very large models that cannot fit otherwise
- Requires careful tuning to minimize overhead

## Types of Offloading

### CPU Offloading

Move data to CPU RAM:

| What to Offload | Memory Savings | Speed Impact |
|-----------------|---------------|--------------|
| Optimizer states | ~8 bytes/param | Moderate |
| Gradients | ~4 bytes/param | Moderate |
| Parameters | ~4 bytes/param | High |

### NVMe Offloading

Move data to SSD storage:

| What to Offload | Effective Memory | Speed Impact |
|-----------------|-----------------|--------------|
| All model states | Disk-limited | Very high |

NVMe is 10-100x slower than CPU RAM.

### Trade-offs

```
Memory efficiency:  GPU < CPU < NVMe
Speed:             GPU > CPU > NVMe

Choose based on:
1. Model size vs available GPU memory
2. Acceptable slowdown
3. Available CPU RAM
4. NVMe speed
```

## DeepSpeed ZeRO-Offload

### Optimizer State Offloading

```json
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

Moves Adam momentum and variance to CPU.

### Parameter Offloading

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

Moves parameters and optimizer states to CPU.

### NVMe Offloading (ZeRO-Infinity)

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme"
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    }
}
```

## FSDP CPU Offloading

### Basic Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

### When Parameters are Offloaded

```
Forward pass:
  1. Move parameters from CPU to GPU (all-gather)
  2. Compute forward
  3. Move parameters back to CPU (free GPU memory)

Backward pass:
  1. Move parameters from CPU to GPU (all-gather)
  2. Compute backward
  3. Move gradients/parameters as needed
```

## Performance Optimization

### Pin Memory

Pin CPU memory for faster transfers:

```python
# DeepSpeed
{
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true  # Critical for performance
    }
}

# FSDP
# Handled automatically
```

### Overlap Communication

Overlap CPU-GPU transfers with computation:

```json
{
    "zero_optimization": {
        "overlap_comm": true
    }
}
```

### NVMe Tuning

```json
{
    "aio": {
        "block_size": 1048576,      // 1MB blocks
        "queue_depth": 8,            // Concurrent IO operations
        "thread_count": 4,           // IO threads
        "single_submit": false,      // Batch submissions
        "overlap_events": true       // Overlap IO with compute
    }
}
```

### NVMe Hardware Recommendations

| NVMe Type | Sequential Read | Training Speed |
|-----------|-----------------|----------------|
| Consumer SSD | 3-5 GB/s | Slow |
| Enterprise NVMe | 5-7 GB/s | Moderate |
| PCIe 4.0 NVMe | 7+ GB/s | Better |

## Speed Impact

### Typical Slowdowns

| Configuration | Relative Speed |
|---------------|---------------|
| GPU only | 1.0x |
| Optimizer to CPU | 0.5-0.8x |
| Params + Optimizer to CPU | 0.3-0.5x |
| Everything to NVMe | 0.1-0.3x |

### When Offloading is Worth It

```
Without offloading: Cannot train (OOM)
With offloading: Slow but possible

Training time:
  No offloading + smaller batch: May be faster
  Offloading + larger batch: May be comparable

Evaluate both options.
```

## Memory Analysis

### Calculate Memory Requirements

```python
def estimate_memory(num_params, batch_size, seq_len, hidden_size):
    # Parameters (FP16)
    param_mem = num_params * 2

    # Gradients (FP16)
    grad_mem = num_params * 2

    # Optimizer states (FP32)
    optimizer_mem = num_params * 8  # Adam: 2 states

    # Activations (rough estimate)
    activation_mem = batch_size * seq_len * hidden_size * 4 * num_layers

    return {
        'params': param_mem / 1e9,
        'grads': grad_mem / 1e9,
        'optimizer': optimizer_mem / 1e9,
        'activations': activation_mem / 1e9
    }
```

### Decide What to Offload

```python
gpu_memory = 80  # GB (A100)
model_memory = estimate_memory(7e9, 8, 2048, 4096)

# Priority order for offloading:
# 1. Optimizer states (least impact on speed)
# 2. Parameters
# 3. Activations (use gradient checkpointing instead)
```

## Implementation Details

### CPU Memory Management

```python
# Ensure sufficient CPU memory
import psutil

available_ram = psutil.virtual_memory().available / 1e9
required_ram = model_size * 2  # Parameters + optimizer

if required_ram > available_ram * 0.8:
    print("Warning: Insufficient CPU memory for offloading")
```

### GPU Memory Monitoring

```python
import torch

def monitor_offload_effectiveness():
    # Before training step
    gpu_before = torch.cuda.memory_allocated()

    # Training step
    train_step()

    # After training step
    gpu_after = torch.cuda.memory_allocated()

    print(f"GPU memory used: {gpu_after / 1e9:.2f} GB")
    # Should be significantly lower than without offloading
```

## Best Practices

### 1. Try Other Optimizations First

```
Order of preference:
1. Mixed precision
2. Gradient checkpointing
3. Flash Attention
4. ZeRO-2 (no offload)
5. ZeRO-3 (no offload)
6. CPU offloading (optimizer only)
7. CPU offloading (full)
8. NVMe offloading
```

### 2. Profile Before and After

```python
# Compare throughput
samples_per_second_no_offload = measure_throughput(model_no_offload)
samples_per_second_offload = measure_throughput(model_offload)

speedup = samples_per_second_no_offload / samples_per_second_offload
print(f"Offloading slowdown: {speedup:.2f}x")
```

### 3. Consider Gradient Accumulation

```python
# Instead of offloading for larger batch:
# Use smaller batch with gradient accumulation

# This may be faster than offloading
effective_batch = micro_batch * accumulation_steps
```

### 4. NVMe Configuration

```bash
# Check NVMe performance
fio --name=test --rw=read --bs=1M --size=1G --numjobs=4

# Ensure high IOPS and bandwidth
# Target: >5 GB/s sequential read
```

## Debugging

### Memory Not Reduced

```python
# Verify offloading is active
if hasattr(model, 'offload_device'):
    print(f"Offloading to: {model.offload_device}")

# Check where parameters are
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
```

### Slow Training

```python
# Profile CPU-GPU transfers
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    train_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Look for cudaMemcpy operations
```

### OOM Despite Offloading

```python
# Activation memory may still be on GPU
# Solution: Add gradient checkpointing

model.gradient_checkpointing_enable()
```

## When to Use Offloading

### Good Use Cases

1. **Training models > 100B parameters**: Necessary regardless of GPU count
2. **Single GPU training of large models**: Only way to fit
3. **Research/prototyping**: Speed less important than capability
4. **Memory-intensive fine-tuning**: LoRA may be better alternative

### Avoid When

1. **Production training at scale**: Consider more GPUs instead
2. **Time-sensitive training**: Slowdown may be unacceptable
3. **Model fits with other optimizations**: Unnecessary overhead
4. **Inference**: Use quantization instead
