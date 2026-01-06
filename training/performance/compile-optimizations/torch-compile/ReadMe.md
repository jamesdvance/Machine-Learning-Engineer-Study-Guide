# torch.compile

## Summary

torch.compile is PyTorch 2.0's flagship feature that automatically optimizes models through graph capture and compilation. By using TorchDynamo for graph capture and TorchInductor for code generation, torch.compile provides significant speedups with minimal code changes. It represents PyTorch's shift toward compiled execution while maintaining the eager-mode development experience.

Key points to remember:

- Single line change: `model = torch.compile(model)`
- 1.5-2x speedup typical for training and inference
- Uses TorchDynamo to capture Python bytecode as graphs
- TorchInductor generates optimized Triton/C++ kernels
- Handles dynamic shapes and control flow gracefully
- Graph breaks allow mixing compiled and eager execution
- Multiple modes: default, reduce-overhead, max-autotune
- Backward pass is automatically compiled with forward

## How torch.compile Works

### Architecture

```
Python Code
    |
    v
TorchDynamo (Graph Capture)
    - Captures Python bytecode
    - Identifies tensor operations
    - Handles graph breaks gracefully
    |
    v
FX Graph (Intermediate Representation)
    - Symbolic representation
    - Operator-level graph
    |
    v
AOTAutograd (Backward Graph)
    - Generates backward graph
    - Joint forward-backward optimization
    |
    v
TorchInductor (Code Generation)
    - Generates Triton kernels (GPU)
    - Generates C++/OpenMP (CPU)
    - Applies optimizations
    |
    v
Compiled Kernel
```

### Key Components

| Component | Role |
|-----------|------|
| TorchDynamo | Python bytecode analysis, graph capture |
| FX | Graph intermediate representation |
| AOTAutograd | Forward/backward graph generation |
| TorchInductor | Triton/C++ code generation |
| Triton | GPU kernel language |

## Basic Usage

### Simple Compilation

```python
import torch

model = MyModel()

# Compile the model
compiled_model = torch.compile(model)

# Use exactly like before
output = compiled_model(input)
```

### Compilation Modes

```python
# Default: Balance between compile time and runtime
model = torch.compile(model, mode="default")

# Reduce overhead: Minimize latency (good for inference)
model = torch.compile(model, mode="reduce-overhead")

# Max autotune: Maximum performance, longer compile
model = torch.compile(model, mode="max-autotune")

# Max autotune without pointwise: Focus on matmuls
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

### Mode Comparison

| Mode | Compile Time | Runtime | Best For |
|------|-------------|---------|----------|
| default | Fast | Good | Development, training |
| reduce-overhead | Medium | Better | Inference |
| max-autotune | Slow | Best | Production inference |

## Training Integration

### Basic Training Loop

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Compile the model (backward is automatic)
model = torch.compile(model)

for epoch in range(num_epochs):
    for batch, labels in dataloader:
        optimizer.zero_grad()

        output = model(batch)
        loss = loss_fn(output, labels)

        loss.backward()  # Also compiled
        optimizer.step()
```

### With Mixed Precision

```python
model = torch.compile(model)

for batch, labels in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = loss_fn(output, labels)

    loss.backward()
    optimizer.step()
```

### With Gradient Scaler

```python
from torch.cuda.amp import GradScaler

model = torch.compile(model)
scaler = GradScaler()

for batch, labels in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(batch)
        loss = loss_fn(output, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Distributed Training

### With DDP

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Compile after DDP wrapping
model = torch.compile(model)
```

### With FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)

# Compile after FSDP wrapping
model = torch.compile(model)
```

### Best Order

```
1. Move to device
2. Wrap with DDP/FSDP
3. Compile with torch.compile
```

## Backend Selection

### Available Backends

```python
# Inductor (default, recommended)
model = torch.compile(model, backend="inductor")

# Eager (no optimization, for debugging)
model = torch.compile(model, backend="eager")

# AOT Eager (captures graph but doesn't optimize)
model = torch.compile(model, backend="aot_eager")

# Custom backend
model = torch.compile(model, backend=my_custom_backend)
```

### Backend Comparison

| Backend | Optimization | Use Case |
|---------|-------------|----------|
| inductor | Full | Production |
| eager | None | Debugging |
| aot_eager | Graph capture only | Testing |
| cudagraphs | CUDA graph capture | Low-latency inference |

## Dynamic Shapes

### Handling Variable Shapes

```python
# Dynamic shapes are supported by default
model = torch.compile(model, dynamic=True)

# Explicit dynamic dimensions
model = torch.compile(model, dynamic=None)  # Auto-detect
```

### Shape Specialization

```python
# Force static shapes (may improve performance)
model = torch.compile(model, dynamic=False)

# Each unique shape triggers recompilation
```

### Dynamic Batch Size

```python
# Works with varying batch sizes
compiled_model = torch.compile(model)

# Different batch sizes work without recompilation
out1 = compiled_model(torch.randn(32, 512))
out2 = compiled_model(torch.randn(64, 512))  # Still uses cached
```

## Graph Breaks

### What Causes Graph Breaks

```python
import torch._dynamo as dynamo

@torch.compile
def problematic(x):
    y = x * 2

    # Graph break: print statement
    print(y.shape)

    # Graph break: data-dependent control flow
    if y.sum() > 0:
        y = y + 1

    # Graph break: calling into numpy
    z = y.numpy()

    return y
```

### Identifying Graph Breaks

```python
# Enable verbose logging
torch._dynamo.config.verbose = True

# Or use explain mode
explanation = torch._dynamo.explain(model)(sample_input)
print(explanation)
```

### Minimizing Graph Breaks

```python
# Bad: Data-dependent control flow
def forward(x):
    if x.sum() > 0:  # Graph break!
        return x * 2
    return x

# Better: Use torch.where
def forward(x):
    return torch.where(x.sum() > 0, x * 2, x)

# Bad: Python print
def forward(x):
    print(f"Shape: {x.shape}")  # Graph break!
    return x * 2

# Better: Remove or use torch logging
def forward(x):
    return x * 2
```

## Performance Tuning

### Autotune Settings

```python
import torch._inductor.config

# Enable more aggressive autotuning
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm = True

model = torch.compile(model, mode="max-autotune")
```

### CUDA Graphs Integration

```python
# Reduce-overhead uses CUDA graphs internally
model = torch.compile(model, mode="reduce-overhead")

# Or enable explicitly
torch._inductor.config.triton.cudagraphs = True
```

### Memory Optimization

```python
# Reduce memory usage during compilation
torch._inductor.config.memory_planning = True

# Pattern matching for memory reuse
torch._inductor.config.pattern_matcher = True
```

## Debugging

### Disable Compilation

```python
# Disable for debugging
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Or completely disable
torch._dynamo.disable()
```

### Inspect Generated Code

```python
# See generated Triton code
import torch._inductor.config
torch._inductor.config.debug = True

@torch.compile
def f(x):
    return torch.relu(x @ x.T)

f(torch.randn(100, 100, device='cuda'))
# Prints generated kernel code
```

### Compare Outputs

```python
# Verify correctness
input = torch.randn(32, 512, device='cuda')

eager_output = model(input)
compiled_output = compiled_model(input)

torch.testing.assert_close(eager_output, compiled_output)
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    compiled_model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Common Patterns

### Caching Compiled Models

```python
import torch._dynamo

# Increase cache size
torch._dynamo.config.cache_size_limit = 64

# Persistent cache directory
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/path/to/cache"
```

### Compiling Loss Functions

```python
# Compile model and loss together
@torch.compile
def train_step(model, batch, labels, loss_fn):
    output = model(batch)
    return loss_fn(output, labels)

# Use in training loop
loss = train_step(model, batch, labels, loss_fn)
```

### Selective Compilation

```python
# Compile specific submodules
model.encoder = torch.compile(model.encoder)
model.decoder = torch.compile(model.decoder)

# Or skip specific layers
model.slow_layer._torch_compile_skip = True
```

## Limitations

### Known Issues

- First iteration is slow (compilation)
- Some operations not yet supported
- Graph breaks reduce benefit
- Memory overhead during compilation

### Unsupported Operations

```python
# Some operations cause fallback to eager:
# - Some inplace operations
# - Certain custom CUDA extensions
# - Some control flow patterns

# Check if operation is supported
torch._dynamo.list_backends()
```

### Workarounds

```python
# Mark regions to skip compilation
with torch._dynamo.disable():
    output = unsupported_operation(x)

# Or use graph break intentionally
torch._dynamo.graph_break()
```

## Best Practices

1. **Start with defaults**: `torch.compile(model)` first
2. **Measure**: Always benchmark before and after
3. **Fix graph breaks**: Audit with verbose mode
4. **Use appropriate mode**: default for training, reduce-overhead for inference
5. **Warm up**: Exclude first iteration from benchmarks
6. **Compile after wrapping**: DDP/FSDP first, then compile
7. **Enable caching**: Use persistent cache for repeated runs

## Benchmarking Template

```python
import torch
import time

def benchmark(model, input, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        model(input)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        model(input)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations * 1000  # ms

# Compare
eager_time = benchmark(model, input)
compiled_time = benchmark(compiled_model, input)

print(f"Eager: {eager_time:.2f}ms")
print(f"Compiled: {compiled_time:.2f}ms")
print(f"Speedup: {eager_time/compiled_time:.2f}x")
```
