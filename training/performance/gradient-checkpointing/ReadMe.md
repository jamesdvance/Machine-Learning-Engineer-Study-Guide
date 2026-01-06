# Gradient Checkpointing

## Summary

Gradient checkpointing (also called activation checkpointing) trades computation for memory by discarding intermediate activations during the forward pass and recomputing them during the backward pass. This technique can dramatically reduce memory usage at the cost of additional computation, enabling training of models that would otherwise not fit in GPU memory.

Key points to remember:

- Discards activations during forward pass, recomputes during backward
- Reduces activation memory from O(n) to O(sqrt(n)) for n layers
- Typically adds ~30-33% compute overhead
- Essential for training very large models
- Can be applied selectively to specific layers
- Combines well with other memory optimizations
- Supported natively in PyTorch and most frameworks
- Different from model checkpointing (saving to disk)

## How It Works

### Standard Training

Without checkpointing, all activations are stored:

```
Forward:  Input -> L1 -> L2 -> L3 -> L4 -> Loss
          Store   A1    A2    A3    A4

Backward: dLoss -> dL4 -> dL3 -> dL2 -> dL1
          Use A4   Use A3  Use A2  Use A1

Memory: O(n) activations for n layers
```

### With Gradient Checkpointing

Checkpointed activations are recomputed:

```
Forward:  Input -> L1 -> L2 -> L3 -> L4 -> Loss
          Store A1 (checkpoint), discard others

Backward: Need A4? Recompute from A1: A1 -> L2 -> L3 -> L4
          Need A3? Recompute from A1: A1 -> L2 -> L3
          ...

Memory: O(1) per checkpoint segment
```

### Optimal Checkpointing

With sqrt(n) checkpoints for n layers:

```
Memory: O(sqrt(n)) activations
Compute: ~33% additional forward passes
```

## PyTorch Implementation

### Basic Usage

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 1024)

    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint(self.layer3, x, use_reentrant=False)
        x = self.layer4(x)  # Last layer: no checkpoint needed
        return x
```

### Checkpointing Sequential Blocks

```python
from torch.utils.checkpoint import checkpoint_sequential

class SequentialModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(1024, 1024) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Checkpoint every 2 layers
        segments = 2
        return checkpoint_sequential(self.layers, segments, x)
```

### Custom Checkpoint Function

```python
from torch.utils.checkpoint import checkpoint

def checkpoint_layer(layer, x, use_checkpoint=True):
    if use_checkpoint and x.requires_grad:
        return checkpoint(layer, x, use_reentrant=False)
    else:
        return layer(x)

class FlexibleModel(nn.Module):
    def __init__(self, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(12)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint_layer(layer, x, self.use_checkpointing)
        return x
```

## Reentrant vs Non-Reentrant

### use_reentrant=False (Recommended)

```python
# Preferred: Safer and more compatible
x = checkpoint(layer, x, use_reentrant=False)
```

Benefits:
- Works with torch.compile
- Better error messages
- No issues with in-place operations
- Handles multiple outputs correctly

### use_reentrant=True (Legacy)

```python
# Legacy mode: May have issues
x = checkpoint(layer, x, use_reentrant=True)
```

Issues:
- Requires all inputs to have requires_grad=True
- Problems with torch.compile
- In-place operation restrictions

**Always use `use_reentrant=False` for new code.**

## Framework Support

### Hugging Face Transformers

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
model.gradient_checkpointing_enable()

# Training works normally
output = model(input_ids)
loss = output.loss
loss.backward()
```

### PyTorch Lightning

```python
import pytorch_lightning as pl

class MyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model()

    def configure_gradient_clipping(self):
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
```

### DeepSpeed

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "cpu_checkpointing": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    }
}
```

### FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

# Apply checkpointing to specific layers
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda module: isinstance(module, TransformerLayer),
)

model = FSDP(model)
```

## Memory Analysis

### Without Checkpointing

For a transformer with L layers, hidden size H, sequence length S, batch B:

```
Activations per layer: ~B * S * H * 4 (attention) + B * S * H * 2 (MLP)
Total: O(L * B * S * H)
```

For L=24, H=1024, S=512, B=8:
```
Per layer: ~8 * 512 * 1024 * 6 * 4 bytes = 100 MB
Total: 24 * 100 MB = 2.4 GB activations
```

### With Checkpointing

```
Stored activations: O(sqrt(L) * B * S * H)
With sqrt(24) = ~5 checkpoints: 500 MB activations
```

Savings: 2.4 GB -> 500 MB = ~80% reduction

## Selective Checkpointing

### Checkpoint Expensive Layers Only

```python
class SelectiveCheckpointing(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(...)  # Expensive
        self.ffn = FeedForward(...)                # Expensive
        self.norm1 = LayerNorm(...)                # Cheap
        self.norm2 = LayerNorm(...)                # Cheap

    def forward(self, x):
        # Checkpoint expensive parts
        attn_out = checkpoint(self.attention, self.norm1(x), use_reentrant=False)
        x = x + attn_out

        ffn_out = checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        x = x + ffn_out

        return x
```

### Skip Every N Layers

```python
class EveryNCheckpoint(nn.Module):
    def __init__(self, layers, checkpoint_every=2):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

## Performance Trade-offs

### Compute Overhead

Checkpointing adds recomputation:
- Each checkpointed segment is computed twice
- Overhead: ~20-33% depending on checkpoint frequency

```python
# Measure overhead
import time

# Without checkpointing
start = time.time()
for _ in range(100):
    loss = model_no_ckpt(batch).sum()
    loss.backward()
no_ckpt_time = time.time() - start

# With checkpointing
start = time.time()
for _ in range(100):
    loss = model_with_ckpt(batch).sum()
    loss.backward()
ckpt_time = time.time() - start

print(f"Overhead: {(ckpt_time - no_ckpt_time) / no_ckpt_time * 100:.1f}%")
```

### Memory-Compute Trade-off Curve

| Checkpoints | Memory Reduction | Compute Overhead |
|-------------|-----------------|------------------|
| 0 | 0% | 0% |
| sqrt(n) | ~70% | ~33% |
| n (every layer) | ~90% | ~100% |

## Debugging

### Verify Checkpointing is Active

```python
def log_memory(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"{stage}: {allocated:.2f} GB")

log_memory("Before forward")
output = model(input)
log_memory("After forward")
loss = output.sum()
log_memory("After loss")
loss.backward()
log_memory("After backward")
```

### Common Issues

**Gradients are None**:
```python
# Ensure requires_grad is set
input = input.requires_grad_(True)
```

**In-place operation errors**:
```python
# Avoid in-place operations in checkpointed functions
# Wrong:
x += self.residual(x)
# Right:
x = x + self.residual(x)
```

**torch.compile incompatibility**:
```python
# Use use_reentrant=False
x = checkpoint(layer, x, use_reentrant=False)
```

## Best Practices

1. **Use non-reentrant mode**: `use_reentrant=False`
2. **Checkpoint expensive layers**: Attention, large MLPs
3. **Don't checkpoint cheap layers**: LayerNorm, dropout
4. **Combine with other techniques**: Mixed precision, FSDP
5. **Profile first**: Verify memory is actually the bottleneck
6. **Test accuracy**: Ensure numerically identical to non-checkpointed
