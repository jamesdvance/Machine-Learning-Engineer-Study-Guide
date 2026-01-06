# Training Frameworks

## Summary

Training frameworks provide high-level abstractions that simplify model development, distributed training, and experiment management. They handle boilerplate code for training loops, logging, checkpointing, and multi-GPU scaling, allowing engineers to focus on model architecture and experimentation. Choosing the right framework depends on your ecosystem, scale requirements, and team expertise.

Key points to remember:

- PyTorch Lightning: Structured PyTorch with minimal boilerplate
- Hugging Face Accelerate: Lightweight distributed training for PyTorch
- Keras: High-level API for TensorFlow (and now JAX/PyTorch)
- JAX/Flax: Functional programming for high-performance ML
- TensorFlow: Production-oriented with comprehensive tooling
- Frameworks trade flexibility for convenience
- Most support mixed precision, multi-GPU, and checkpointing
- Consider team expertise and existing infrastructure

## Framework Comparison

### Quick Reference

| Framework | Base Library | Learning Curve | Flexibility | Production |
|-----------|--------------|----------------|-------------|------------|
| PyTorch Lightning | PyTorch | Low | High | Good |
| Accelerate | PyTorch | Very Low | Very High | Good |
| Keras | TF/JAX/PyTorch | Very Low | Medium | Excellent |
| JAX/Flax | JAX | High | Very High | Good |
| TensorFlow | TensorFlow | Medium | High | Excellent |

### Use Case Recommendations

```
Research with PyTorch:
  -> PyTorch Lightning (structured)
  -> Accelerate (minimal changes)

Hugging Face ecosystem:
  -> Accelerate + Transformers

TPU training:
  -> JAX/Flax (native)
  -> TensorFlow (supported)

Production deployment:
  -> TensorFlow/Keras (TF Serving)
  -> PyTorch + TorchServe

Rapid prototyping:
  -> Keras (simplest API)

Custom training loops:
  -> Accelerate (most control)
  -> Raw PyTorch/TensorFlow
```

## Core Features Comparison

### Distributed Training

| Framework | DDP | FSDP | DeepSpeed | TPU |
|-----------|-----|------|-----------|-----|
| Lightning | Yes | Yes | Yes | Yes |
| Accelerate | Yes | Yes | Yes | Yes |
| Keras | Via strategy | Via strategy | No | Yes |
| JAX/Flax | pmap/pjit | pjit | No | Native |
| TensorFlow | MirroredStrategy | - | No | TPUStrategy |

### Mixed Precision

| Framework | FP16 | BF16 | FP8 |
|-----------|------|------|-----|
| Lightning | Yes | Yes | Via plugins |
| Accelerate | Yes | Yes | Yes |
| Keras | Yes | Yes | Limited |
| JAX/Flax | Yes | Yes | Limited |
| TensorFlow | Yes | Yes | Limited |

### Checkpointing

| Framework | Save/Resume | Best Model | Distributed |
|-----------|-------------|------------|-------------|
| Lightning | Automatic | ModelCheckpoint | Handled |
| Accelerate | Manual | Manual | Handled |
| Keras | Callbacks | ModelCheckpoint | Handled |
| JAX/Flax | Orbax/Flax | Manual | Manual |
| TensorFlow | Callbacks | Automatic | Handled |

## Minimal Examples

### PyTorch Lightning

```python
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(784, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=4)
trainer.fit(model, train_loader)
```

### Hugging Face Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch).loss
    accelerator.backward(loss)
    optimizer.step()
```

### Keras

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)
```

### JAX/Flax

```python
import jax
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(10)(nn.relu(nn.Dense(512)(x)))

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['x'])
        return optax.softmax_cross_entropy(logits, batch['y']).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
```

### TensorFlow

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

with tf.distribute.MirroredStrategy().scope():
    model.fit(train_dataset, epochs=10)
```

## When to Use Each Framework

### PyTorch Lightning

**Best for:**
- Teams wanting structured PyTorch code
- Research requiring reproducibility
- Projects needing multiple accelerator support
- Complex training procedures (GANs, RL)

**Avoid when:**
- Need maximum flexibility
- Simple scripts suffice
- Minimal dependencies required

### Hugging Face Accelerate

**Best for:**
- Existing PyTorch code needing distributed support
- Hugging Face Transformers workflows
- Minimal framework overhead
- Gradual migration to distributed

**Avoid when:**
- Need extensive logging/callbacks
- Want opinionated structure
- Non-PyTorch workflows

### Keras

**Best for:**
- Rapid prototyping
- Beginners to deep learning
- Standard architectures
- TensorFlow production pipelines

**Avoid when:**
- Need fine-grained control
- Custom training dynamics
- PyTorch ecosystem preferred

### JAX/Flax

**Best for:**
- TPU training at scale
- Functional programming preference
- Research requiring transformations (vmap, pmap)
- Performance-critical applications

**Avoid when:**
- Team unfamiliar with functional programming
- Need extensive debugging tools
- Want plug-and-play simplicity

### TensorFlow

**Best for:**
- Production ML pipelines
- Mobile/edge deployment
- Enterprise environments
- Comprehensive tooling needs

**Avoid when:**
- Research flexibility required
- PyTorch ecosystem preferred
- Team prefers imperative programming

## Migration Paths

### PyTorch to Lightning

```python
# Before: Raw PyTorch
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

# After: Lightning
class Model(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        return self.model(batch)

trainer = pl.Trainer(max_epochs=epochs)
trainer.fit(model, loader)
```

### PyTorch to Accelerate

```python
# Before: Single GPU
model = Model().cuda()
for batch in loader:
    loss = model(batch.cuda())

# After: Multi-GPU ready
accelerator = Accelerator()
model, loader = accelerator.prepare(model, loader)
for batch in loader:
    loss = model(batch)
```

## Further Reading

Framework deep dives:
- [PyTorch Lightning](pytorch-lightning/ReadMe.md): Structured PyTorch training
- [Hugging Face Accelerate](hugging-face-accelerate/ReadMe.md): Lightweight distributed training
- [Keras](keras/ReadMe.md): High-level neural network API
- [JAX/Flax](jax-flax/ReadMe.md): Functional ML with XLA
- [TensorFlow](tensorflow/ReadMe.md): End-to-end ML platform

Related topics:
- [Distributed Training](../distributed-training/ReadMe.md): Multi-GPU/TPU strategies
- [Performance](../performance/ReadMe.md): Optimization techniques
