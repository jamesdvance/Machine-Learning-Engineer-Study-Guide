# Activation Checkpointing

## Summary

Activation checkpointing is synonymous with gradient checkpointing - both terms refer to the same technique of saving memory by recomputing activations during the backward pass instead of storing them during the forward pass. This chapter covers additional considerations and framework-specific implementations beyond the core gradient checkpointing concepts.

Key points to remember:

- Same technique as gradient checkpointing, different terminology
- "Activation checkpointing" emphasizes what is being checkpointed (activations)
- "Gradient checkpointing" emphasizes why (for gradient computation)
- Frameworks use different terminology (DeepSpeed: activation checkpointing)
- All concepts from gradient checkpointing apply here
- Focus on which activations to checkpoint for best memory/compute trade-off

## Terminology Clarification

| Framework | Term Used |
|-----------|-----------|
| PyTorch | gradient checkpointing |
| DeepSpeed | activation checkpointing |
| Megatron-LM | activation checkpointing |
| JAX | checkpoint / remat |
| TensorFlow | gradient checkpointing |

The implementation is identical across all.

## What Gets Checkpointed

### Activations in a Transformer Layer

```
Input x
    |
LayerNorm -> h1 (stored for backward if no checkpointing)
    |
Attention:
    Q, K, V projections -> q, k, v (stored)
    Attention scores -> scores (stored)
    Attention output -> attn_out (stored)
    |
Residual add
    |
LayerNorm -> h2 (stored)
    |
MLP:
    Linear 1 -> mlp1 (stored)
    Activation -> act (stored)
    Linear 2 -> mlp2 (stored)
    |
Residual add -> output
```

With checkpointing: Only checkpoint boundaries stored (e.g., input x).

### Memory Per Activation

For sequence length S, batch B, hidden H:

| Activation | Shape | Memory (FP16) |
|------------|-------|---------------|
| Input/Output | B x S x H | BSH x 2 bytes |
| Q, K, V | 3 x B x heads x S x head_dim | 3BSH x 2 |
| Attention scores | B x heads x S x S | BhS^2 x 2 |
| MLP intermediate | B x S x 4H | 4BSH x 2 |

Attention scores grow quadratically with sequence length.

## Selective Activation Checkpointing

### Checkpoint Only Attention

Attention has high memory cost (scores are O(S^2)):

```python
class SelectiveCheckpointLayer(nn.Module):
    def forward(self, x):
        # Checkpoint attention (high memory)
        attn_out = checkpoint(self.attention, x, use_reentrant=False)
        x = x + attn_out

        # Don't checkpoint MLP (moderate memory)
        mlp_out = self.mlp(x)
        x = x + mlp_out

        return x
```

### Megatron-LM Selective Checkpointing

```python
# Megatron supports selective activation checkpointing
--recompute-activations
--recompute-granularity selective  # Only attention
--recompute-granularity full       # Everything
```

### DeepSpeed Partitioned Activation Checkpointing

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true
    }
}
```

Partitions activations across data parallel ranks for additional memory savings.

## CPU Checkpointing

### Offload Activations to CPU

For extreme memory constraints, store checkpoints on CPU:

```python
# DeepSpeed CPU checkpointing
{
    "activation_checkpointing": {
        "cpu_checkpointing": true
    }
}
```

### Manual CPU Offload

```python
class CPUCheckpoint(nn.Module):
    def forward(self, x):
        # Store checkpoint on CPU
        checkpoint_cpu = x.cpu()

        # Continue forward pass
        x = self.expensive_layers(x)

        return x, checkpoint_cpu

    def backward_recompute(self, checkpoint_cpu, grad):
        # Move back to GPU for recomputation
        x = checkpoint_cpu.cuda()
        x = self.expensive_layers(x)
        return grad
```

**Trade-off**: PCIe transfer time vs GPU memory savings.

## Activation Compression

### Compress Checkpointed Activations

```python
import torch

def compress_activation(activation):
    """Quantize to int8 for storage."""
    scale = activation.abs().max() / 127
    quantized = (activation / scale).to(torch.int8)
    return quantized, scale

def decompress_activation(quantized, scale):
    """Dequantize back to float."""
    return quantized.float() * scale
```

### Random Projection Compression

```python
class CompressedCheckpoint:
    def __init__(self, compression_ratio=4):
        self.compression_ratio = compression_ratio

    def compress(self, activation):
        # Random projection to lower dimension
        B, S, H = activation.shape
        proj_dim = H // self.compression_ratio

        # Generate random projection matrix (deterministic seed)
        proj = torch.randn(H, proj_dim, device=activation.device)
        proj = proj / (H ** 0.5)

        compressed = activation @ proj
        return compressed, proj

    def decompress(self, compressed, proj):
        # Project back (approximate)
        return compressed @ proj.T
```

## Framework Implementations

### PyTorch Native

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Single function
output = checkpoint(layer, input, use_reentrant=False)

# Sequential layers
output = checkpoint_sequential(layers, segments, input)
```

### JAX/Flax

```python
import jax
from jax import checkpoint  # Also called remat

@checkpoint
def checkpointed_layer(params, x):
    return layer(params, x)

# Or with custom policy
from jax.ad_checkpoint import checkpoint as remat

@remat(policy=jax.checkpoint_policies.checkpoint_dots)
def layer(params, x):
    ...
```

### TensorFlow

```python
import tensorflow as tf

@tf.recompute_grad
def checkpointed_layer(x):
    return layer(x)
```

## Checkpoint Granularity

### Layer-Level (Coarse)

```python
# Checkpoint entire transformer layers
for layer in self.layers:
    x = checkpoint(layer, x, use_reentrant=False)
```

Memory: ~1 activation per layer
Compute: Recompute entire layer

### Sub-Layer Level (Fine)

```python
# Checkpoint attention and MLP separately
attn_out = checkpoint(self.attention, x, use_reentrant=False)
x = x + attn_out
mlp_out = checkpoint(self.mlp, x, use_reentrant=False)
x = x + mlp_out
```

Memory: More checkpoints, smaller segments
Compute: Recompute smaller segments

### Operation-Level (Very Fine)

```python
# Checkpoint individual operations
q = checkpoint(self.q_proj, x, use_reentrant=False)
k = checkpoint(self.k_proj, x, use_reentrant=False)
v = checkpoint(self.v_proj, x, use_reentrant=False)
```

Memory: Minimal storage
Compute: Maximum recomputation

### Choosing Granularity

| Granularity | Memory Savings | Compute Overhead | Complexity |
|-------------|---------------|------------------|------------|
| Layer | Moderate | Low (~20%) | Simple |
| Sub-layer | High | Moderate (~30%) | Moderate |
| Operation | Maximum | High (~50%+) | Complex |

**Recommendation**: Start with layer-level, refine if needed.

## Integration with Flash Attention

Flash Attention reduces attention memory without checkpointing:

```python
# Without Flash Attention: Need to checkpoint attention
attn_out = checkpoint(self.attention, x, use_reentrant=False)

# With Flash Attention: May not need checkpointing
from flash_attn import flash_attn_func
attn_out = flash_attn_func(q, k, v)  # Memory efficient already
```

Combining both provides maximum memory efficiency.

## Debugging Activation Checkpointing

### Memory Profiling

```python
import torch

def profile_activations(model, input):
    torch.cuda.reset_peak_memory_stats()

    output = model(input)
    forward_mem = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    output.sum().backward()
    backward_mem = torch.cuda.max_memory_allocated()

    print(f"Forward peak: {forward_mem / 1e9:.2f} GB")
    print(f"Backward peak: {backward_mem / 1e9:.2f} GB")
```

### Verify Recomputation

```python
# Add hooks to see when layers are called
def make_hook(name):
    def hook(module, input, output):
        print(f"{name} called")
    return hook

for name, module in model.named_modules():
    module.register_forward_hook(make_hook(name))

# Should see layers called twice (forward + recompute)
output = model(input)
output.sum().backward()
```

## Best Practices

1. **Profile first**: Understand where memory is used
2. **Start coarse**: Layer-level checkpointing first
3. **Combine with Flash Attention**: Reduces need for attention checkpointing
4. **Use non-reentrant mode**: Better compatibility
5. **Don't over-checkpoint**: Balance memory and compute
6. **Test numerics**: Verify identical gradients
