# Memory Optimization

## Summary

Memory optimization techniques enable training models that would otherwise exceed GPU memory capacity. These techniques span multiple approaches: reducing precision (mixed precision), eliminating redundancy (ZeRO/FSDP), recomputing instead of storing (gradient checkpointing), and extending memory to other devices (offloading). Understanding when and how to apply each technique is essential for large model training.

Key points to remember:

- Flash Attention reduces attention memory from O(S^2) to O(S)
- Gradient checkpointing trades compute (~30%) for activation memory
- Mixed precision (FP16/BF16) halves parameter and activation memory
- ZeRO/FSDP shard optimizer states across GPUs
- Offloading extends memory to CPU or NVMe at cost of speed
- Gradient accumulation enables larger effective batches without more memory
- Techniques stack: use multiple for maximum effect
- Profile to understand actual memory breakdown before optimizing

## Memory Breakdown

### Training Memory Components

For a model with P parameters, batch size B, sequence length S:

| Component | Size | Notes |
|-----------|------|-------|
| Parameters | 4P (FP32) / 2P (FP16) | Model weights |
| Gradients | 4P (FP32) / 2P (FP16) | Same size as params |
| Optimizer | 8P (Adam) | Momentum + variance |
| Activations | O(B * S * H * L) | Batch, seq, hidden, layers |

### Typical Breakdown (7B Model, FP16)

| Component | Memory |
|-----------|--------|
| Parameters | 14 GB |
| Gradients | 14 GB |
| Optimizer (Adam) | 56 GB |
| Activations | Variable (10-50+ GB) |
| **Total** | 94+ GB |

### Where to Optimize

1. **Activations dominate**: Gradient checkpointing, Flash Attention
2. **Optimizer dominates**: ZeRO Stage 1-2, FSDP
3. **Parameters dominate**: ZeRO Stage 3, tensor parallelism
4. **All constrained**: Combine multiple techniques

## Optimization Techniques Overview

### Quick Reference

| Technique | Memory Savings | Overhead | Complexity |
|-----------|---------------|----------|------------|
| Mixed precision | ~50% | Minimal | Low |
| Flash Attention | ~50% attention | Speed gain | Low |
| Gradient checkpointing | ~70% activations | ~30% compute | Low |
| ZeRO-1 | ~4x optimizer | Minimal | Low |
| ZeRO-2 | ~8x total | Low | Low |
| ZeRO-3 | Linear in GPUs | Moderate | Moderate |
| CPU offload | Extends to RAM | Significant | Moderate |
| NVMe offload | Extends to disk | Very high | High |

### Stacking Techniques

Techniques can be combined:

```
Base: FP32, no optimization
  -> Mixed precision: 2x savings
  -> + Flash Attention: Additional attention savings
  -> + Gradient checkpointing: Reduce activation memory
  -> + ZeRO-2: Shard optimizer states
  -> + Offloading: Extend to CPU (if needed)
```

## Memory Profiling

### PyTorch Memory Tracking

```python
import torch

def print_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Max allocated: {max_allocated:.2f} GB")

# Profile training step
torch.cuda.reset_peak_memory_stats()

# Forward
output = model(input)
print("After forward:")
print_memory_stats()

# Backward
output.sum().backward()
print("After backward:")
print_memory_stats()

# Optimizer step
optimizer.step()
print("After optimizer:")
print_memory_stats()
```

### Memory Snapshot

```python
# Record memory allocations
torch.cuda.memory._record_memory_history()

# Run training step
output = model(input)
output.sum().backward()

# Dump snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze with PyTorch Memory Visualizer
```

### Per-Layer Memory

```python
def count_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} params, {param.numel() * param.element_size() / 1e6:.2f} MB")

count_parameters(model)
```

## Optimization Strategy

### Step-by-Step Approach

1. **Measure baseline**
   ```python
   # Train step without optimizations
   # Record peak memory
   ```

2. **Enable mixed precision**
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       output = model(input)
   ```

3. **Add Flash Attention**
   ```python
   # Use transformer with Flash Attention
   model = model_with_flash_attention()
   ```

4. **Enable gradient checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

5. **Add optimizer sharding if needed**
   ```python
   model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
   ```

6. **Add offloading if still needed**
   ```python
   model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
   ```

### Decision Tree

```
Does model fit with mixed precision?
  Yes -> Done (or optimize for throughput)
  No -> Add gradient checkpointing
        |
        Does it fit now?
        Yes -> Done
        No -> Add ZeRO/FSDP sharding
              |
              Does it fit now?
              Yes -> Done
              No -> Add CPU offloading
                    |
                    Does it fit now?
                    Yes -> Done (expect slowdown)
                    No -> Need more GPUs or reduce model size
```

## Common Memory Issues

### Out of Memory (OOM)

```python
# Common causes:
# 1. Batch size too large
# 2. Sequence length too long
# 3. Model doesn't fit

# Solutions:
# 1. Reduce batch size, use gradient accumulation
# 2. Reduce sequence length or use Flash Attention
# 3. Use FSDP/ZeRO or smaller model
```

### Memory Fragmentation

```python
# Symptoms: OOM despite apparent free memory
# Solution: Reserve memory upfront

# Reserve 90% of GPU memory for PyTorch
import torch
torch.cuda.set_per_process_memory_fraction(0.9)

# Or clear cache periodically
torch.cuda.empty_cache()
```

### Memory Leaks

```python
# Common causes:
# 1. Storing tensors in lists
# 2. Not detaching for logging
# 3. Growing computation graphs

# Solution 1: Detach logged values
loss_value = loss.detach().item()

# Solution 2: Clear stored tensors
stored_activations.clear()

# Solution 3: Use torch.no_grad for evaluation
with torch.no_grad():
    val_output = model(val_input)
```

## Further Reading

Memory optimization techniques:
- [Flash Attention](flash-attention/ReadMe.md): Memory-efficient attention
- [Offloading](offloading/ReadMe.md): CPU and NVMe offloading
- [Gradient Accumulation](gradient-accumulation/ReadMe.md): Larger effective batches

Related topics:
- [Gradient Checkpointing](../gradient-checkpointing/ReadMe.md): Trading compute for memory
- [Mixed Precision](../mixed-precision-training/ReadMe.md): Lower precision training
- [FSDP](../../distributed-training/concepts/fsdp/ReadMe.md): Fully sharded data parallel
- [ZeRO](../../distributed-training/concepts/zero-optimization/ReadMe.md): Zero redundancy optimizer
