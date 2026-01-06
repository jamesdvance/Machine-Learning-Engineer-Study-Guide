# Training Performance

## Summary

Training performance optimization focuses on maximizing throughput, minimizing memory usage, and reducing training time. Modern deep learning training combines multiple optimization techniques: mixed precision reduces memory and increases speed, gradient checkpointing trades compute for memory, and compiler optimizations fuse operations for efficiency. Understanding these techniques is essential for training large models within hardware constraints.

Key points to remember:

- Mixed precision (FP16/BF16) nearly doubles throughput with minimal accuracy impact
- BF16 preferred on modern hardware (A100, H100) for stability
- Gradient checkpointing reduces activation memory at cost of recomputation
- Flash Attention provides memory-efficient attention with better speed
- torch.compile and XLA can provide 20-50% speedups
- Offloading extends memory to CPU/NVMe for very large models
- Techniques are cumulative: combine for maximum benefit
- Profile before optimizing to identify actual bottlenecks

## Performance Optimization Landscape

### Memory vs Compute Trade-offs

| Technique | Memory Savings | Compute Cost | Use Case |
|-----------|---------------|--------------|----------|
| Mixed precision | ~50% | Slight gain | Always use |
| Gradient checkpointing | ~60-80% | ~33% overhead | Large models |
| Activation checkpointing | ~60-80% | ~33% overhead | Same as above |
| Flash Attention | ~50% attention | Speed gain | Transformers |
| CPU offloading | Extended | Significant | Extreme cases |

### When to Apply Each

**Mixed precision**: Always, unless specific numerical stability concerns.

**Gradient checkpointing**: When activation memory is the bottleneck.

**Flash Attention**: For any transformer with attention.

**Compile optimizations**: When training loop is stable and optimized.

**Offloading**: Only when other techniques insufficient.

## Performance Metrics

### Key Metrics

**Throughput**: Samples/second or tokens/second
```python
throughput = batch_size * num_gpus / step_time
```

**GPU Utilization**: Percentage of compute capacity used
```python
# Check with nvidia-smi
nvidia-smi --query-gpu=utilization.gpu --format=csv
```

**Memory Efficiency**: Actual vs peak memory
```python
allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
efficiency = allocated / reserved
```

**MFU (Model FLOPs Utilization)**: Actual vs theoretical FLOPs
```python
mfu = actual_flops / (peak_flops * time)
```

### Profiling

```python
# PyTorch Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

## Optimization Strategy

### Step 1: Measure Baseline

```python
import torch
import time

def benchmark_step(model, batch, num_warmup=5, num_runs=20):
    # Warmup
    for _ in range(num_warmup):
        with torch.cuda.amp.autocast():
            loss = model(batch)
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.cuda.amp.autocast():
            loss = model(batch)
        loss.backward()
    torch.cuda.synchronize()

    elapsed = time.time() - start
    return elapsed / num_runs
```

### Step 2: Apply Low-Hanging Fruit

1. **Enable mixed precision**
2. **Use Flash Attention**
3. **Optimize data loading** (num_workers, pin_memory)
4. **Increase batch size** to GPU capacity

### Step 3: Address Memory Bottlenecks

1. **Enable gradient checkpointing** if OOM
2. **Use FSDP/ZeRO** for large models
3. **Consider offloading** as last resort

### Step 4: Compiler Optimizations

1. **Enable torch.compile** for stable training
2. **Profile and tune** compilation settings

## Common Bottlenecks

### Compute Bound

Symptoms:
- GPU utilization near 100%
- Memory has headroom
- Larger batch does not help

Solutions:
- torch.compile for kernel fusion
- Flash Attention
- Mixed precision

### Memory Bound

Symptoms:
- OOM errors
- Cannot increase batch size
- GPU memory near full

Solutions:
- Gradient checkpointing
- Mixed precision
- FSDP/ZeRO sharding
- Reduce batch size

### Data Loading Bound

Symptoms:
- GPU utilization spiky
- CPU utilization high
- GPU waits for data

Solutions:
- Increase num_workers
- Enable pin_memory
- Use faster storage
- Prefetch more batches

### Communication Bound

Symptoms:
- Low GPU utilization in distributed
- Scaling efficiency poor
- Network bandwidth saturated

Solutions:
- Gradient compression
- Overlap communication with compute
- Better network (InfiniBand)
- Reduce communication frequency

## Further Reading

### Mixed Precision Training
- [Overview](mixed-precision-training/ReadMe.md): Concepts and implementation
- [FP16](mixed-precision-training/fp16/ReadMe.md): Half precision training
- [BF16](mixed-precision-training/bf16/ReadMe.md): Brain floating point
- [FP8](mixed-precision-training/fp8/ReadMe.md): Latest precision format

### Memory Optimization
- [Overview](memory-optimization/ReadMe.md): Memory techniques
- [Flash Attention](memory-optimization/flash-attention/ReadMe.md): Efficient attention
- [Offloading](memory-optimization/offloading/ReadMe.md): CPU/NVMe offload
- [Gradient Accumulation](memory-optimization/gradient-accumulation/ReadMe.md): Larger effective batches

### Checkpointing
- [Gradient Checkpointing](gradient-checkpointing/ReadMe.md): Trading compute for memory
- [Activation Checkpointing](activation-checkpointing/ReadMe.md): Selective recomputation

### Compile Optimizations
- [Overview](compile-optimizations/ReadMe.md): Compilation techniques
- [torch.compile](compile-optimizations/torch-compile/ReadMe.md): PyTorch 2.0 compilation
- [XLA](compile-optimizations/xla/ReadMe.md): Accelerated Linear Algebra
- [TensorRT](compile-optimizations/tensorrt/ReadMe.md): NVIDIA optimization
