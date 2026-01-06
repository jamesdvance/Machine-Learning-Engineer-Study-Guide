# XLA (Accelerated Linear Algebra)

## Summary

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that optimizes TensorFlow and JAX computations. By analyzing entire computation graphs, XLA fuses operations, optimizes memory layout, and generates efficient code for CPUs, GPUs, and TPUs. XLA is essential for TPU training and provides significant speedups on GPU workloads.

Key points to remember:

- XLA compiles computation graphs into optimized machine code
- Required for TPU training (JAX and TensorFlow)
- Provides 1.5-3x speedups on GPU for suitable workloads
- Performs operation fusion to reduce memory bandwidth
- Optimizes memory layout for target hardware
- JAX uses XLA by default (JIT compilation)
- TensorFlow can enable XLA via `tf.function(jit_compile=True)`
- PyTorch/XLA enables XLA for PyTorch models

## How XLA Works

### Compilation Pipeline

```
High-Level Operations (MatMul, Conv, etc.)
    |
    v
HLO (High Level Optimizer)
    - Platform-independent optimizations
    - Algebraic simplifications
    - Operation fusion
    |
    v
LLO (Low Level Optimizer)
    - Target-specific optimizations
    - Memory layout optimization
    - Buffer assignment
    |
    v
Target Code Generation
    - GPU: PTX/CUDA
    - TPU: TPU instructions
    - CPU: LLVM
```

### Key Optimizations

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| Op fusion | Combine elementwise ops | Reduced memory I/O |
| Layout optimization | Optimal tensor layout for HW | Faster access |
| Buffer sharing | Reuse memory allocations | Lower memory use |
| Constant folding | Compute constants at compile | Less runtime work |
| Dead code elimination | Remove unused ops | Cleaner graphs |

## JAX and XLA

### JIT Compilation in JAX

```python
import jax
import jax.numpy as jnp

# JIT compile a function
@jax.jit
def forward(params, x):
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return x

# First call triggers XLA compilation
output = forward(params, input_data)
```

### Static vs Dynamic Shapes

```python
# XLA requires static shapes at compile time
@jax.jit
def fixed_shape(x):
    return x * 2  # Shape must be known

# For dynamic shapes, use static_argnums
@jax.jit
def dynamic_batch(x, batch_size):
    return x[:batch_size]

# Or use jax.ensure_compile_time_eval
```

### Tracing and Compilation

```python
# See the compiled HLO
from jax import make_jaxpr

jaxpr = make_jaxpr(forward)(params, sample_input)
print(jaxpr)

# Lower to HLO
lowered = jax.jit(forward).lower(params, sample_input)
print(lowered.as_text())
```

## TensorFlow and XLA

### Enabling XLA

```python
import tensorflow as tf

# Method 1: JIT compile specific functions
@tf.function(jit_compile=True)
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Method 2: Global XLA auto-clustering
tf.config.optimizer.set_jit(True)
```

### Keras Integration

```python
# Compile with XLA
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    jit_compile=True  # Enable XLA
)

model.fit(train_data, epochs=10)
```

### Mixed Precision with XLA

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# XLA optimizes mixed precision operations
@tf.function(jit_compile=True)
def forward(x):
    return model(x)
```

## PyTorch/XLA

### Basic Usage

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Get XLA device (TPU or GPU)
device = xm.xla_device()

# Move model and data to XLA device
model = model.to(device)
input = input.to(device)

# Forward pass (lazy execution)
output = model(input)

# Mark step to execute accumulated operations
xm.mark_step()
```

### Training Loop

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, labels)
        loss.backward()

        # XLA optimizer step
        xm.optimizer_step(optimizer)
```

### Distributed Training with PyTorch/XLA

```python
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(index):
    device = xm.xla_device()

    # Wrap dataloader for distributed loading
    train_loader = pl.MpDeviceLoader(train_loader, device)

    for batch, labels in train_loader:
        # Training step
        pass

# Launch on all TPU cores
xmp.spawn(train_fn, nprocs=8)
```

## TPU Training

### TPU-Specific Considerations

```python
import jax

# Check available TPU devices
print(jax.devices())  # Lists TPU cores

# TPU pods: multiple TPU hosts
# Use pjit for multi-host parallelism
from jax.experimental import pjit
from jax.experimental.maps import Mesh
```

### Optimal Batch Sizes

```
TPU v4 has 8 cores per chip
Each core has 16GB HBM

Batch size recommendations:
- Per-core batch: 8, 16, 32 (power of 2)
- Total batch: per_core * num_cores
- Larger batches = better TPU utilization
```

### Avoiding Recompilation

```python
# XLA recompiles for new shapes
# Use padding to maintain fixed shapes

def pad_to_fixed(batch, max_len):
    """Pad sequences to fixed length."""
    padded = jnp.zeros((batch.shape[0], max_len))
    padded = padded.at[:, :batch.shape[1]].set(batch)
    return padded

# Or use bucketing
buckets = [64, 128, 256, 512]
```

## Performance Optimization

### Fusion Patterns

```python
# XLA automatically fuses these patterns:

# Pattern 1: Elementwise chain
def fused_elementwise(x):
    return jax.nn.relu(x * 2 + 1)  # Single kernel

# Pattern 2: Reduce + broadcast
def fused_normalize(x):
    mean = x.mean(axis=-1, keepdims=True)
    return x - mean  # Fused

# Pattern 3: MatMul + bias + activation
def fused_linear(x, w, b):
    return jax.nn.relu(x @ w + b)  # Fused
```

### Memory Optimization

```python
# XLA optimizes memory automatically, but you can help:

# 1. Prefer in-place patterns
x = x.at[0].set(1)  # In-place update syntax

# 2. Use gradient checkpointing
from jax.checkpoint import checkpoint

@checkpoint
def memory_efficient_layer(x):
    # Recomputed during backward
    return heavy_computation(x)
```

### Profiling XLA

```python
# JAX profiling
with jax.profiler.trace("/tmp/tensorboard"):
    output = model(input)

# TensorFlow XLA profiling
tf.profiler.experimental.start('/tmp/logs')
train_step(model, x, y)
tf.profiler.experimental.stop()
```

## Debugging XLA

### Viewing HLO

```python
# JAX: View compiled HLO
computation = jax.jit(forward).lower(params, input)
print(computation.compile().as_text())

# TensorFlow: Enable HLO dumping
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'
```

### Disabling XLA

```python
# JAX: Disable JIT for debugging
with jax.disable_jit():
    output = forward(params, input)

# TensorFlow: Run eagerly
tf.config.run_functions_eagerly(True)
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow first step | Compilation | Warm up, cache |
| Shape errors | Dynamic shapes | Pad to fixed sizes |
| OOM during compile | Large graphs | Reduce model, checkpoint |
| Different results | Numerical precision | Check dtype, reductions |

## XLA vs torch.compile

### Comparison

| Aspect | XLA | torch.compile |
|--------|-----|---------------|
| Frameworks | JAX, TensorFlow | PyTorch |
| TPU support | Native | Via PyTorch/XLA |
| Dynamic shapes | Limited | Better support |
| Debugging | Harder | Easier |
| Maturity | Older, stable | Newer, evolving |

### When to Use Each

```
Use XLA (JAX/TensorFlow):
- TPU training (required)
- JAX ecosystem
- TensorFlow production

Use torch.compile:
- PyTorch ecosystem
- Dynamic shapes needed
- Rapid prototyping
```

## Best Practices

1. **Use static shapes**: Pad inputs to fixed sizes
2. **Batch appropriately**: Larger batches amortize compilation
3. **Profile compilation**: Identify slow compilations
4. **Cache compiled graphs**: Avoid recompilation
5. **Warm up**: Exclude first step from timing
6. **Prefer JIT-compatible operations**: Avoid Python callbacks
7. **Use appropriate precision**: BF16 on TPU, FP16 on GPU

## Example: Full JAX Training

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Initialize
model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 784]))

# Create optimizer
tx = optax.adam(1e-3)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

# JIT-compiled training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
for batch in dataloader:
    state, loss = train_step(state, batch)
```
