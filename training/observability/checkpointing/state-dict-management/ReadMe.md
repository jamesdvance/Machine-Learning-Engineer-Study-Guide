# State Dict Management

## Summary

PyTorch's state_dict is a Python dictionary that maps each layer to its parameter tensor. Understanding state dict structure and manipulation is essential for saving, loading, transferring, and modifying model weights. Proper state dict management enables model checkpointing, fine-tuning, model surgery, and deployment.

Key points to remember:

- state_dict contains learnable parameters (weights, biases)
- Keys are parameter names, values are tensors
- Use strict=False for partial loading
- Map locations handle device placement
- Prefix handling for wrapped models (DDP, FSDP)
- Buffers (non-learnable state) are included
- Optimizer has its own state_dict
- Named parameters provide name-to-parameter mapping

## State Dict Basics

### Model State Dict

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Get state dict
state_dict = model.state_dict()

# Inspect structure
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.shape}")

# Output:
# 0.weight: torch.Size([256, 784])
# 0.bias: torch.Size([256])
# 2.weight: torch.Size([10, 256])
# 2.bias: torch.Size([10])
```

### Optimizer State Dict

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After some training steps
loss.backward()
optimizer.step()

# Get optimizer state
opt_state = optimizer.state_dict()

# Contains:
# - 'state': per-parameter optimizer state (momentum, etc.)
# - 'param_groups': optimizer configuration
```

### Save and Load

```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model.load_state_dict(torch.load('model.pt'))
```

## Loading Options

### Strict Loading (Default)

```python
# All keys must match exactly
model.load_state_dict(state_dict)  # strict=True by default

# Raises error if:
# - Missing keys in checkpoint
# - Unexpected keys in checkpoint
```

### Non-Strict Loading

```python
# Allow partial loading
model.load_state_dict(state_dict, strict=False)

# Useful for:
# - Fine-tuning (loading pretrained weights)
# - Transfer learning (different architecture)
# - Adding new layers
```

### Inspecting Missing/Unexpected Keys

```python
# Returns named tuple with missing and unexpected keys
result = model.load_state_dict(state_dict, strict=False)

print(f"Missing keys: {result.missing_keys}")
print(f"Unexpected keys: {result.unexpected_keys}")
```

## Device Management

### Map Location

```python
# Load to specific device
state_dict = torch.load('model.pt', map_location='cpu')
state_dict = torch.load('model.pt', map_location='cuda:0')
state_dict = torch.load('model.pt', map_location='cuda:1')

# Dynamic mapping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('model.pt', map_location=device)
```

### Moving State Between Devices

```python
# Load to CPU first (safe approach)
state_dict = torch.load('model.pt', map_location='cpu')

# Load into model
model.load_state_dict(state_dict)

# Then move model to target device
model = model.to('cuda:0')
```

### Multi-GPU to Single-GPU

```python
# State dict saved from DataParallel has 'module.' prefix
state_dict = torch.load('ddp_model.pt')

# Remove prefix
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
```

## Handling Wrapped Models

### DataParallel / DistributedDataParallel

```python
# DDP wraps model with 'module.' prefix
model = nn.DataParallel(model)
# or
model = DistributedDataParallel(model)

# Keys become: 'module.layer.weight'

# Save unwrapped
torch.save(model.module.state_dict(), 'model.pt')

# Or save wrapped and handle on load
def remove_prefix(state_dict, prefix='module.'):
    return {k.replace(prefix, ''): v for k, v in state_dict.items()}

# Or add prefix for loading into wrapped model
def add_prefix(state_dict, prefix='module.'):
    return {prefix + k: v for k, v in state_dict.items()}
```

### FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Configure full state dict gathering
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state_dict = model.state_dict()

    if rank == 0:
        torch.save(state_dict, 'model.pt')
```

## Partial Loading

### Load Specific Layers

```python
# Load only encoder weights
pretrained_dict = torch.load('pretrained.pt')
model_dict = model.state_dict()

# Filter to only encoder keys
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k.startswith('encoder.')}

# Update model dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### Freeze Loaded Weights

```python
# Load pretrained weights
model.load_state_dict(torch.load('pretrained.pt'), strict=False)

# Freeze encoder
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
```

### Skip Mismatched Shapes

```python
def load_matching_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    # Filter out mismatched shapes
    compatible_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            compatible_dict[k] = v
        else:
            print(f"Skipping {k}: shape mismatch")

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
```

## State Dict Surgery

### Rename Keys

```python
def rename_keys(state_dict, mapping):
    """Rename state dict keys.

    Args:
        state_dict: Original state dict
        mapping: Dict of old_name -> new_name
    """
    new_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in mapping.items():
            new_key = new_key.replace(old, new)
        new_dict[new_key] = value
    return new_dict

# Example: adapt checkpoint from different codebase
mapping = {
    'backbone.': 'encoder.',
    'head.': 'classifier.',
}
adapted_dict = rename_keys(state_dict, mapping)
```

### Reshape Weights

```python
def reshape_embedding(state_dict, old_size, new_size):
    """Reshape embedding layer for different vocabulary."""
    embed_key = 'embedding.weight'

    if embed_key in state_dict:
        old_embed = state_dict[embed_key]

        if old_embed.shape[0] != new_size:
            # Initialize new embedding
            embed_dim = old_embed.shape[1]
            new_embed = torch.zeros(new_size, embed_dim)

            # Copy existing weights
            copy_size = min(old_size, new_size)
            new_embed[:copy_size] = old_embed[:copy_size]

            state_dict[embed_key] = new_embed

    return state_dict
```

## Buffers vs Parameters

### Understanding Buffers

```python
class ModelWithBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Has parameters

        # Buffers: non-learnable state
        self.register_buffer('running_mean', torch.zeros(5))
        self.register_buffer('num_batches', torch.tensor(0))

# state_dict includes both
state_dict = model.state_dict()
# Contains: linear.weight, linear.bias, running_mean, num_batches
```

### Buffer Handling

```python
# Check if key is a parameter or buffer
for name, param in model.named_parameters():
    print(f"Parameter: {name}")

for name, buffer in model.named_buffers():
    print(f"Buffer: {name}")
```

## Advanced Patterns

### Checkpoint Averaging

```python
def average_checkpoints(checkpoint_paths):
    """Average multiple checkpoints for improved performance."""
    state_dicts = [torch.load(p, map_location='cpu') for p in checkpoint_paths]

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state_dict[key] = torch.stack(tensors).mean(0)

    return avg_state_dict

averaged = average_checkpoints([
    'checkpoint_8.pt',
    'checkpoint_9.pt',
    'checkpoint_10.pt'
])
```

### EMA Weights

```python
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
```

## Best Practices

1. **Always use map_location**: Explicit device handling
2. **Check for module prefix**: Handle DDP wrapping
3. **Use strict=False carefully**: May hide issues
4. **Verify key matching**: Print missing/unexpected keys
5. **Load to CPU first**: Then move to GPU
6. **Save unwrapped models**: Or handle prefix on load
7. **Include optimizer state**: For training resumption
8. **Version your checkpoints**: Include architecture info
