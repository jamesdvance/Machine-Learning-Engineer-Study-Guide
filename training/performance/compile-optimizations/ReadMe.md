# Compile Optimizations

## Summary

Compile optimizations transform dynamic deep learning models into optimized static graphs that execute faster on target hardware. By analyzing computation patterns, fusing operations, and generating hardware-specific kernels, compilers can achieve 1.5-3x speedups over eager execution. Modern frameworks increasingly rely on compilation for both training and inference performance.

Key points to remember:

- Compilation converts dynamic graphs to optimized static representations
- torch.compile (PyTorch 2.0+) is now the primary compilation strategy for PyTorch
- XLA (Accelerated Linear Algebra) powers JAX and TensorFlow optimizations
- TensorRT provides maximum inference performance on NVIDIA GPUs
- Compilation benefits: operation fusion, memory optimization, kernel generation
- Trade-offs: initial compilation overhead, debugging complexity
- Graph breaks reduce optimization opportunities
- Compilation is most beneficial for repeated operations (training loops)

## Why Compilation Matters

### Eager vs Compiled Execution

```
Eager Execution:
  - Each operation dispatched individually
  - Python overhead between operations
  - No cross-operation optimization
  - Easy to debug

Compiled Execution:
  - Multiple operations fused
  - Single kernel launch for fused ops
  - Memory access patterns optimized
  - Harder to debug, faster to run
```

### Optimization Opportunities

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| Operator fusion | Combine multiple ops into one kernel | Reduced memory bandwidth |
| Memory planning | Reuse tensor allocations | Lower memory usage |
| Layout optimization | Optimal tensor memory layouts | Faster access patterns |
| Kernel selection | Choose best implementation | Hardware utilization |
| Dead code elimination | Remove unused computations | Reduced work |

## Compilation Approaches

### Just-In-Time (JIT) Compilation

```python
import torch

# PyTorch dynamo captures at runtime
@torch.compile
def forward(x, w):
    return torch.relu(x @ w)

# First call: trace and compile
# Subsequent calls: use compiled version
```

Advantages: Works with dynamic shapes, minimal code changes
Disadvantages: Initial compilation overhead

### Ahead-of-Time (AOT) Compilation

```python
# TensorRT: compile before deployment
import torch_tensorrt

compiled = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16}
)

# Save for later use
torch.jit.save(compiled, "model.ts")
```

Advantages: No runtime compilation cost
Disadvantages: Less flexible, requires shape specification

## Compiler Comparison

### Quick Reference

| Compiler | Framework | Best For | Speedup |
|----------|-----------|----------|---------|
| torch.compile | PyTorch | Training, flexible inference | 1.5-2x |
| TorchScript | PyTorch | Deployment, model export | 1.1-1.5x |
| XLA | JAX/TensorFlow | TPU, distributed training | 1.5-3x |
| TensorRT | Any (ONNX) | NVIDIA inference | 2-5x |
| ONNX Runtime | Any (ONNX) | Cross-platform inference | 1.5-2x |

### When to Use Each

```
Training on GPU:
  -> torch.compile (PyTorch)
  -> XLA (JAX/TensorFlow)

Training on TPU:
  -> XLA (required)

Inference on NVIDIA:
  -> TensorRT (maximum performance)
  -> torch.compile (flexibility)

Inference cross-platform:
  -> ONNX Runtime
  -> TorchScript
```

## Common Patterns

### Compiling Training Loops

```python
import torch

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters())

# Compile the model
compiled_model = torch.compile(model)

for batch in dataloader:
    optimizer.zero_grad()
    loss = compiled_model(batch)  # Uses compiled forward
    loss.backward()               # Backward is also compiled
    optimizer.step()
```

### Compiling Inference

```python
# Mode optimized for inference
compiled_model = torch.compile(
    model,
    mode="reduce-overhead"  # Minimize latency
)

# Warm up (trigger compilation)
with torch.no_grad():
    compiled_model(sample_input)

# Now compiled for fast inference
```

### Conditional Compilation

```python
# Compile only if CUDA available
if torch.cuda.is_available():
    model = torch.compile(model)

# Or check for specific backends
if torch._dynamo.is_compiling():
    print("Currently in compilation mode")
```

## Performance Considerations

### Compilation Overhead

```python
import time

model = LargeModel()
x = torch.randn(32, 1024)

# Eager baseline
start = time.time()
for _ in range(100):
    model(x)
eager_time = time.time() - start

# Compiled (including compilation time)
compiled = torch.compile(model)
start = time.time()
for _ in range(100):
    compiled(x)  # First iteration compiles
compiled_time = time.time() - start

print(f"Eager: {eager_time:.2f}s")
print(f"Compiled (with overhead): {compiled_time:.2f}s")
```

### Amortizing Compilation Cost

```
Compilation time: O(seconds to minutes)
Per-iteration savings: O(milliseconds)

Break-even: compilation_time / savings_per_iter iterations

For training (thousands of iterations): Always worth it
For inference (few iterations): Pre-compile or use cache
```

### Caching Compiled Graphs

```python
import torch._dynamo

# Enable persistent cache (across runs)
torch._dynamo.config.cache_size_limit = 64

# Or use environment variable
# TORCHINDUCTOR_CACHE_DIR=/path/to/cache
```

## Debugging Compilation

### Understanding Graph Breaks

```python
import torch._dynamo as dynamo

# See what causes graph breaks
dynamo.config.verbose = True

@torch.compile
def forward(x):
    y = x * 2
    print(f"Value: {y}")  # Graph break! Python side effect
    return y + 1

# Log will show: "Graph break: call_function print"
```

### Common Graph Break Causes

| Cause | Example | Solution |
|-------|---------|----------|
| Print statements | `print(tensor)` | Remove or use logging |
| Data-dependent control flow | `if x.sum() > 0` | Restructure logic |
| Unsupported ops | Some numpy calls | Use torch equivalents |
| Dynamic shapes | Varying sequence lengths | Use padding |

### Inspecting Generated Code

```python
import torch._inductor.config

# See generated Triton/C++ code
torch._inductor.config.debug = True

@torch.compile
def matmul_relu(a, b):
    return torch.relu(a @ b)

# Will print generated kernel code
```

## Integration Tips

### With Mixed Precision

```python
model = torch.compile(model)

# Autocast works with compiled models
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
```

### With Distributed Training

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap with FSDP first, then compile
model = FSDP(model)
model = torch.compile(model)
```

### With Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
model = torch.compile(model)  # Works together
```

## Best Practices

1. **Start simple**: Enable compilation with defaults first
2. **Measure**: Always benchmark before and after
3. **Minimize graph breaks**: Audit code for break causes
4. **Use appropriate mode**: `default` for training, `reduce-overhead` for inference
5. **Warm up**: Run a few iterations before benchmarking
6. **Cache graphs**: Enable persistent caching for repeated runs
7. **Profile**: Use profilers to verify optimization gains

## Further Reading

Compilation technologies:
- [torch.compile](torch-compile/ReadMe.md): PyTorch 2.0 compilation
- [XLA](xla/ReadMe.md): Accelerated Linear Algebra for JAX/TensorFlow
- [TensorRT](tensorrt/ReadMe.md): NVIDIA inference optimization

Related topics:
- [Mixed Precision](../mixed-precision-training/ReadMe.md): Combine with compilation
- [Flash Attention](../memory-optimization/flash-attention/ReadMe.md): Optimized attention
