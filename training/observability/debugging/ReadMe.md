# Debugging Training Issues

## Summary

Debugging deep learning training involves identifying and resolving issues that prevent models from learning effectively. Common problems include numerical instabilities (NaN/Inf), gradient pathologies (explosion/vanishing), memory issues, and performance bottlenecks. Effective debugging requires systematic approaches and appropriate tooling to diagnose issues before they waste compute resources.

Key points to remember:

- Monitor gradients and loss from the start
- NaN/Inf usually indicate learning rate or initialization issues
- Gradient clipping prevents explosion
- Memory profiling identifies leaks and inefficiencies
- Use deterministic mode for reproducible debugging
- Start small (single GPU, small batch) then scale
- Profile early to identify bottlenecks
- Log extensively during debugging

## Common Training Issues

### Issue Categories

| Category | Symptoms | Likely Causes |
|----------|----------|---------------|
| Numerical | NaN loss, Inf values | LR too high, bad init, unstable ops |
| Gradient | No learning, explosion | Vanishing/exploding gradients |
| Memory | OOM errors, slowdown | Leaks, fragmentation, batch size |
| Performance | Slow training | I/O bottleneck, inefficient ops |
| Convergence | Loss plateau | LR, architecture, data issues |

### Debugging Workflow

```
1. Reproduce the issue consistently
2. Minimize the problem (small batch, few steps)
3. Enable verbose logging
4. Check basic sanity (data, shapes, dtypes)
5. Profile and identify bottleneck
6. Isolate the problematic component
7. Fix and verify
8. Test at scale
```

## Numerical Debugging

### Detect NaN/Inf

```python
import torch

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

# In training loop
loss = model(batch)
check_nan_inf(loss, "loss")

loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        check_nan_inf(param.grad, f"{name}.grad")
```

### Enable PyTorch Anomaly Detection

```python
import torch

# Detect NaN in backward pass
torch.autograd.set_detect_anomaly(True)

# Throws error with traceback when NaN/Inf encountered
loss = model(input)
loss.backward()  # Will show source of NaN

# Disable for production (overhead)
torch.autograd.set_detect_anomaly(False)
```

### Common NaN Causes

| Cause | Solution |
|-------|----------|
| Learning rate too high | Reduce LR, use warmup |
| Bad initialization | Use appropriate init (Xavier, Kaiming) |
| Log of zero/negative | Add epsilon, use clamp |
| Exp overflow | Use log-sum-exp trick |
| Division by zero | Add epsilon to denominator |
| Large input values | Normalize inputs |

## Gradient Debugging

### Monitor Gradient Norms

```python
def compute_grad_norm(model, norm_type=2):
    """Compute total gradient norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(norm_type) ** norm_type
    return total_norm ** (1.0 / norm_type)

# In training loop
loss.backward()
grad_norm = compute_grad_norm(model)
print(f"Gradient norm: {grad_norm:.4f}")
```

### Per-Layer Gradient Statistics

```python
def log_gradient_stats(model):
    """Log per-layer gradient statistics."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name}:")
            print(f"  mean: {grad.mean():.6f}")
            print(f"  std: {grad.std():.6f}")
            print(f"  min: {grad.min():.6f}")
            print(f"  max: {grad.max():.6f}")
            print(f"  norm: {grad.norm():.6f}")
```

### Gradient Clipping

```python
import torch

# Clip by norm (recommended)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# In training loop
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if grad_norm > 100:
    print(f"Warning: Large gradient norm {grad_norm:.2f}")
optimizer.step()
```

## Memory Debugging

### Track Memory Usage

```python
import torch

def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()

# In training loop
output = model(batch)
print_memory_stats()
loss.backward()
print_memory_stats()
```

### Memory Snapshot

```python
import torch

# Record memory allocations
torch.cuda.memory._record_memory_history()

# Run code
output = model(batch)
output.sum().backward()

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze with PyTorch Memory Visualizer
# https://pytorch.org/memory_viz
```

### Common Memory Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Growing memory | OOM over time | Clear cache, detach tensors |
| Fragmentation | OOM with free memory | Preallocate, reduce variation |
| Large activations | OOM on forward | Gradient checkpointing |
| Optimizer state | High baseline memory | Optimizer sharding |

## Reproducibility

### Deterministic Mode

```python
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Full determinism (PyTorch 2.0+)
    torch.use_deterministic_algorithms(True)

set_seed(42)
```

### Debugging Non-Determinism

```python
# Enable warnings for non-deterministic ops
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
```

## Performance Debugging

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=tensorboard_trace_handler('./logs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Identify Bottlenecks

```python
import time

class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        torch.cuda.synchronize()
        self.times[name] = time.time()

    def stop(self, name):
        torch.cuda.synchronize()
        elapsed = time.time() - self.times[name]
        print(f"{name}: {elapsed*1000:.2f}ms")
        return elapsed

timer = Timer()

timer.start("forward")
output = model(batch)
timer.stop("forward")

timer.start("backward")
output.sum().backward()
timer.stop("backward")

timer.start("optimizer")
optimizer.step()
timer.stop("optimizer")
```

## Sanity Checks

### Overfit Single Batch

```python
def overfit_single_batch(model, batch, num_steps=100):
    """Model should achieve near-zero loss on single batch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.6f}")

    assert loss.item() < 0.01, "Model cannot overfit single batch!"
```

### Check Data Pipeline

```python
def verify_data(dataloader):
    """Check data is loaded correctly."""
    batch = next(iter(dataloader))

    print(f"Batch keys: {batch.keys()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(f"  min={value.min():.4f}, max={value.max():.4f}")
            print(f"  has_nan={torch.isnan(value).any()}")
```

### Verify Gradient Flow

```python
def check_gradient_flow(model, input):
    """Verify gradients flow to all parameters."""
    model.zero_grad()
    output = model(input)
    output.sum().backward()

    no_grad = []
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.abs().sum() == 0:
            no_grad.append(name)

    if no_grad:
        print(f"Warning: No gradient for: {no_grad}")
    else:
        print("All parameters receiving gradients")
```

## Framework Tools

### PyTorch Lightning

```python
trainer = pl.Trainer(
    # Debugging
    fast_dev_run=True,         # Run 1 batch
    overfit_batches=1,         # Overfit 1 batch
    detect_anomaly=True,       # NaN detection

    # Profiling
    profiler='simple',         # Basic profiling
    # profiler='advanced',     # Detailed profiling
)
```

### Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Debug info
print(accelerator.state)
print(f"Device: {accelerator.device}")
print(f"Distributed: {accelerator.distributed_type}")
```

## Best Practices

1. **Start simple**: Debug on single GPU, small batch
2. **Log everything**: Metrics, gradients, memory
3. **Use deterministic mode**: For reproducible debugging
4. **Overfit first**: Verify model can learn
5. **Profile early**: Find bottlenecks before scaling
6. **Check data first**: Many issues are data bugs
7. **Use anomaly detection**: During development
8. **Verify checkpoint loading**: Test save/load cycle

## Further Reading

- [Memory Profiling](memory-profiling/ReadMe.md): GPU memory debugging
- [NaN Detection](nan-detection/ReadMe.md): Numerical stability
- [Gradient Monitoring](gradient-monitoring/ReadMe.md): Gradient health
