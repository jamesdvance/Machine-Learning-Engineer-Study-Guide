# JAX/Flax

## Summary

JAX is a high-performance numerical computing library that combines NumPy's familiar API with automatic differentiation, GPU/TPU acceleration, and powerful program transformations. Flax is the recommended neural network library built on JAX, providing a flexible and performant way to define models. JAX's functional approach and XLA compilation make it particularly well-suited for research and large-scale training.

Key points to remember:

- JAX provides NumPy API with autodiff and XLA compilation
- Flax is the primary neural network library for JAX
- Functional programming: pure functions, explicit state
- Key transformations: jit, grad, vmap, pmap
- Native TPU support with excellent performance
- Explicit random number handling with PRNGKeys
- State is external to models (params dict)
- Optax provides optimizers for JAX

## JAX Fundamentals

### Basic JAX Operations

```python
import jax
import jax.numpy as jnp

# NumPy-like API
x = jnp.array([1, 2, 3])
y = jnp.dot(x, x)

# Automatic differentiation
def f(x):
    return jnp.sum(x ** 2)

grad_f = jax.grad(f)
print(grad_f(jnp.array([1.0, 2.0, 3.0])))  # [2., 4., 6.]
```

### Key Transformations

```python
import jax
import jax.numpy as jnp

# JIT compilation
@jax.jit
def fast_function(x):
    return jnp.dot(x, x.T)

# Automatic differentiation
@jax.grad
def loss_grad(params, x, y):
    pred = model(params, x)
    return jnp.mean((pred - y) ** 2)

# Vectorization (auto-batching)
@jax.vmap
def batched_predict(params, x):
    return model(params, x)

# Parallelization across devices
@jax.pmap
def parallel_step(params, batch):
    return train_step(params, batch)
```

### Random Numbers

```python
import jax
import jax.random as random

# JAX requires explicit random key management
key = random.PRNGKey(42)

# Split keys for different operations
key, subkey1, subkey2 = random.split(key, 3)

# Use subkeys for randomness
x = random.normal(subkey1, shape=(10,))
y = random.uniform(subkey2, shape=(5,))

# Never reuse keys!
```

## Flax Basics

### Defining Models with Linen

```python
from flax import linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# Create model
model = MLP(hidden_dim=256, output_dim=10)
```

### Initializing Parameters

```python
import jax
from flax import linen as nn

model = MLP(hidden_dim=256, output_dim=10)

# Initialize with dummy input
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 784))

# params is a nested dict
params = model.init(key, dummy_input)

# Inspect structure
print(jax.tree_util.tree_map(lambda x: x.shape, params))
```

### Forward Pass

```python
# Apply model to input
output = model.apply(params, input_data)

# With dropout (needs rngs)
output = model.apply(
    params,
    input_data,
    training=True,
    rngs={'dropout': dropout_key}
)
```

## Training with Optax

### Basic Training Loop

```python
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Create optimizer
tx = optax.adam(learning_rate=1e-3)

# Create train state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

# Define training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        state, loss = train_step(state, batch)
```

### Learning Rate Schedules

```python
import optax

# Warmup + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000
)

optimizer = optax.adam(learning_rate=schedule)

# Or chain transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3)
)
```

### Gradient Accumulation

```python
import optax

# Accumulate over N steps
optimizer = optax.MultiSteps(
    optax.adam(learning_rate=1e-3),
    every_k_schedule=4  # Accumulate 4 steps
)
```

## Distributed Training

### Data Parallelism with pmap

```python
import jax
from jax import pmap

# Replicate state across devices
state = jax.device_put_replicated(state, jax.devices())

# pmap the training step
@jax.pmap
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state, loss

# Shard batch across devices
def shard_batch(batch):
    num_devices = jax.device_count()
    batch_size = batch['image'].shape[0]
    per_device = batch_size // num_devices
    return jax.tree_util.tree_map(
        lambda x: x.reshape(num_devices, per_device, *x.shape[1:]),
        batch
    )

# Training loop
for batch in dataloader:
    sharded_batch = shard_batch(batch)
    state, loss = train_step(state, sharded_batch)
```

### Model Parallelism with pjit

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit

# Create device mesh
devices = mesh_utils.create_device_mesh((2, 4))  # 2x4 mesh
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding for parameters
def shard_params(params):
    # Shard along model axis
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, NamedSharding(mesh, PartitionSpec('model'))),
        params
    )

# Use pjit for training
@pjit
def train_step(state, batch):
    # Training logic
    pass
```

## Common Patterns

### Dropout and BatchNorm

```python
from flax import linen as nn

class ModelWithDropout(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(10)(x)
        return x

# Initialize
key = jax.random.PRNGKey(0)
params = model.init(key, dummy_input)

# Forward with dropout
key, dropout_key = jax.random.split(key)
output = model.apply(params, x, training=True, rngs={'dropout': dropout_key})

# Forward without dropout (inference)
output = model.apply(params, x, training=False)
```

### BatchNorm with Mutable State

```python
class ModelWithBN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(256)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return x

# Initialize returns params and batch_stats
variables = model.init(key, dummy_input)
params, batch_stats = variables['params'], variables['batch_stats']

# Training: update batch_stats
output, updates = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x,
    training=True,
    mutable=['batch_stats']
)
batch_stats = updates['batch_stats']

# Inference: use running stats
output = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x,
    training=False
)
```

### Gradient Checkpointing

```python
from jax.checkpoint import checkpoint

class EfficientModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        for i in range(12):
            x = checkpoint(self.transformer_block)(x)
        return x

    def transformer_block(self, x):
        # Memory-intensive computation
        return nn.Dense(512)(nn.relu(nn.Dense(512)(x)))
```

## Checkpointing

### Saving and Loading

```python
from flax.training import checkpoints

# Save checkpoint
checkpoints.save_checkpoint(
    ckpt_dir='checkpoints/',
    target=state,
    step=step,
    overwrite=True
)

# Load checkpoint
state = checkpoints.restore_checkpoint(
    ckpt_dir='checkpoints/',
    target=state
)
```

### Orbax (Recommended)

```python
import orbax.checkpoint as ocp

# Create checkpointer
checkpointer = ocp.PyTreeCheckpointer()

# Save
checkpointer.save(
    'checkpoint/path',
    state,
    save_args=ocp.SaveArgs(aggregate=True)
)

# Load
state = checkpointer.restore('checkpoint/path', target=state)
```

## Metrics and Evaluation

### Computing Metrics

```python
import jax
import jax.numpy as jnp

@jax.jit
def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

# Aggregate across devices
def aggregate_metrics(metrics):
    return jax.tree_util.tree_map(lambda x: x.mean(), metrics)
```

### CLU Metrics Library

```python
from clu import metrics

@flax.struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

# Update metrics
state_metrics = Metrics.single_from_model_output(
    logits=logits,
    labels=labels,
    loss=loss
)

# Merge across batches
all_metrics = all_metrics.merge(state_metrics)

# Compute final values
print(all_metrics.compute())
```

## Transformers with Flax

### Self-Attention Layer

```python
from flax import linen as nn
import jax.numpy as jnp

class SelfAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, mask=None):
        B, L, D = x.shape
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = jnp.einsum('blhd,bmhd->bhlm', q, k) * scale

        if mask is not None:
            attn = jnp.where(mask, attn, -1e9)

        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum('bhlm,bmhd->blhd', attn, v)
        out = out.reshape(B, L, -1)

        return nn.Dense(D)(out)
```

### Transformer Block

```python
class TransformerBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Self-attention
        attn_out = SelfAttention(self.num_heads, x.shape[-1] // self.num_heads)(x)
        attn_out = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_out)
        x = nn.LayerNorm()(x + attn_out)

        # MLP
        mlp_out = nn.Dense(self.mlp_dim)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(x.shape[-1])(mlp_out)
        mlp_out = nn.Dropout(self.dropout_rate, deterministic=not training)(mlp_out)
        x = nn.LayerNorm()(x + mlp_out)

        return x
```

## Mixed Precision

### Using bfloat16

```python
import jax.numpy as jnp

# Cast inputs
x = x.astype(jnp.bfloat16)

# Use bf16 for computation
@jax.jit
def forward(params, x):
    # Params stay fp32, compute in bf16
    x = x.astype(jnp.bfloat16)
    out = model.apply(params, x)
    return out.astype(jnp.float32)

# Or use policy
from flax import linen as nn
nn.set_default_dtype(jnp.bfloat16)
```

## Debugging

### Disable JIT

```python
# For debugging
with jax.disable_jit():
    output = train_step(state, batch)

# Or globally
jax.config.update('jax_disable_jit', True)
```

### Print Intermediate Values

```python
from jax import debug

@jax.jit
def debug_function(x):
    y = x * 2
    jax.debug.print("y = {}", y)  # Works inside JIT
    return y
```

### Check NaN/Inf

```python
jax.config.update("jax_debug_nans", True)

# Will raise error on NaN
output = model.apply(params, input)
```

## Best Practices

1. **Use pure functions**: No side effects
2. **Manage PRNG carefully**: Split keys, never reuse
3. **JIT everything**: Compile training step
4. **Use vmap for batching**: Cleaner than manual batching
5. **Profile with jax.profiler**: Find bottlenecks
6. **Use Orbax for checkpoints**: Modern checkpointing
7. **Leverage pmap for multi-GPU**: Simple data parallelism
8. **Use optax for optimizers**: Composable gradient transforms
