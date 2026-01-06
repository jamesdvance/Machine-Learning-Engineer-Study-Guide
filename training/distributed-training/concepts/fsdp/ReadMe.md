# FSDP (Fully Sharded Data Parallel)

## Summary

Fully Sharded Data Parallel (FSDP) is PyTorch's native implementation of ZeRO-style memory optimization for distributed training. FSDP shards model parameters, gradients, and optimizer states across data-parallel workers, dramatically reducing per-GPU memory requirements. Unlike standard data parallelism where each GPU holds a complete model copy, FSDP allows training models that would otherwise not fit in memory.

Key points to remember:

- Shards parameters, gradients, and optimizer states across ranks
- Equivalent to ZeRO Stage 3 in DeepSpeed
- Native PyTorch implementation with good integration
- All-gather before forward, reduce-scatter after backward
- Supports mixed precision, activation checkpointing, CPU offloading
- Requires careful wrapping of model layers for optimal performance
- FULL_SHARD maximizes memory savings, SHARD_GRAD_OP reduces communication
- Backward-compatible checkpointing requires consolidation

## Memory Savings

### Standard DDP Memory

Each GPU stores:
- Full model parameters: P
- Full gradients: P
- Full optimizer states: 2P (Adam momentum + variance)
- Total: 4P per GPU (plus activations)

### FSDP Memory

Each GPU stores:
- Sharded parameters: P/N
- Sharded gradients: P/N
- Sharded optimizer states: 2P/N
- Total: 4P/N per GPU (plus activations)

For 8 GPUs, FSDP uses 1/8th the parameter-related memory per GPU.

### Practical Impact

| Model Size | DDP (per GPU) | FSDP 8 GPUs | FSDP 64 GPUs |
|------------|---------------|-------------|--------------|
| 1B params | 16 GB | 2 GB | 0.25 GB |
| 7B params | 112 GB | 14 GB | 1.75 GB |
| 70B params | 1.12 TB | 140 GB | 17.5 GB |

FSDP makes training very large models practical on commodity hardware.

## How FSDP Works

### Sharding Strategy

FSDP wraps model modules and manages their parameter lifecycle:

```
Initialization:
  - Shard parameters across ranks
  - Each rank keeps only P/N of each wrapped module

Forward Pass:
  1. All-gather: Collect full parameters from all ranks
  2. Compute forward pass with full parameters
  3. Discard non-local parameters (free memory)

Backward Pass:
  1. All-gather: Collect full parameters again
  2. Compute gradients with full parameters
  3. Reduce-scatter: Each rank gets its gradient shard
  4. Discard non-local parameters
```

### Communication Pattern

```
Forward (per FSDP unit):
  All-gather parameters: P * (N-1) / N bytes communicated
  Computation: Forward pass
  Free remote parameters

Backward (per FSDP unit):
  All-gather parameters: P * (N-1) / N bytes
  Computation: Backward pass
  Reduce-scatter gradients: P * (N-1) / N bytes
  Free remote parameters
```

Total communication: approximately 3P per FSDP unit per step.

## Basic Usage

### Simple Wrapping

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Initialize distributed
torch.distributed.init_process_group(backend="nccl")

# Create model
model = MyLargeModel()

# Wrap with FSDP
model = FSDP(model)

# Training loop (same as DDP)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Sharding Strategies

```python
from torch.distributed.fsdp import ShardingStrategy

# FULL_SHARD: ZeRO-3, maximum memory savings
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# SHARD_GRAD_OP: ZeRO-2, shard gradients and optimizer only
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

# NO_SHARD: Like DDP, no sharding
model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)

# HYBRID_SHARD: Full shard within node, replicate across nodes
model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)
```

**Choosing a strategy**:
- FULL_SHARD: Default, best memory efficiency
- SHARD_GRAD_OP: When memory is sufficient, reduces communication
- HYBRID_SHARD: Multi-node training, balances communication

## Auto Wrapping

### Transformer Auto Wrap

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Wrap at transformer layer granularity
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

### Size-Based Wrapping

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap modules with at least 100M parameters
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

### Manual Wrapping

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            FSDP(TransformerLayer(hidden_size))  # Wrap each layer
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)

# Wrap the outer model
model = FSDP(MyModel())
```

## Mixed Precision

### Configuring Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

# BF16 mixed precision (recommended for modern GPUs)
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)

# FP16 mixed precision
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16
)

model = FSDP(
    model,
    mixed_precision=bf16_policy
)
```

### Precision Components

- **param_dtype**: Precision for parameters during compute
- **reduce_dtype**: Precision for gradient reduction
- **buffer_dtype**: Precision for buffers (e.g., batch norm stats)

### Gradient Scaling with FP16

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

BF16 typically does not require gradient scaling due to wider dynamic range.

## CPU Offloading

### Offload Configuration

```python
from torch.distributed.fsdp import CPUOffload

# Offload parameters to CPU when not in use
model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

### When to Use Offloading

**Benefits**:
- Further reduces GPU memory
- Enables training even larger models

**Costs**:
- Significant slowdown (CPU-GPU transfer)
- Only useful when truly memory-constrained

Typically 3-5x slower with offloading. Use only when necessary.

## Activation Checkpointing

### Enabling Checkpointing

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return checkpoint(self.layer, x, use_reentrant=False)

# Wrap layers with checkpointing before FSDP
model.layers = nn.ModuleList([
    CheckpointedLayer(layer) for layer in model.layers
])
model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

### FSDP Native Checkpointing

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig

# Apply checkpointing to specific layers
def apply_fsdp_checkpointing(model):
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, TransformerLayer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn
    )
```

## Checkpointing (Saving/Loading)

### Full State Dict (for inference)

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Save full state dict (only on rank 0)
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "model.pt")
```

### Sharded State Dict (for training)

```python
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

# Save sharded state dict (each rank saves its shard)
sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
    state_dict = model.state_dict()
    torch.save(state_dict, f"model_rank{rank}.pt")
```

### Loading Checkpoints

```python
# Load full state dict
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state_dict = torch.load("model.pt")
    model.load_state_dict(state_dict)

# Load sharded state dict
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = torch.load(f"model_rank{rank}.pt")
    model.load_state_dict(state_dict)
```

## Performance Optimization

### Prefetching

```python
from torch.distributed.fsdp import BackwardPrefetch

# Prefetch parameters during backward
model = FSDP(
    model,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE
)
```

Options:
- BACKWARD_PRE: Prefetch before backward (default, best performance)
- BACKWARD_POST: Prefetch after backward (lower memory)
- None: No prefetching

### Forward Prefetching

```python
model = FSDP(
    model,
    forward_prefetch=True  # Prefetch next layer during forward
)
```

### Limiting All-Gathers

```python
model = FSDP(
    model,
    limit_all_gathers=True  # Limit concurrent all-gathers
)
```

Reduces memory at cost of some performance.

## Common Configuration

### Recommended Setup

```python
import functools
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def setup_fsdp(model, transformer_layer_cls):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls}
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        device_id=torch.cuda.current_device(),
    )

    return model
```

## Debugging

### Memory Tracking

```python
def log_memory(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

log_memory("Before FSDP")
model = FSDP(model)
log_memory("After FSDP")
```

### Verifying Sharding

```python
def check_sharding(model):
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, device={param.device}")
```

### Common Issues

**Parameters not sharded**:
- Check auto_wrap_policy matches your layer types
- Verify model structure before wrapping

**Memory still high**:
- Activation memory dominates; use checkpointing
- Check batch size and sequence length

**Training slow**:
- Reduce communication with SHARD_GRAD_OP
- Ensure fast interconnect (NVLink, InfiniBand)

## Comparison with DeepSpeed ZeRO

| Aspect | FSDP | DeepSpeed ZeRO-3 |
|--------|------|------------------|
| Integration | Native PyTorch | Separate library |
| Maturity | Newer | More mature |
| Features | Core sharding | More optimization features |
| Ease of use | Moderate | Moderate |
| Performance | Competitive | Highly optimized |
| Community | Growing | Large |

**Choose FSDP when**:
- Want native PyTorch solution
- Using PyTorch ecosystem tools
- Simpler dependencies

**Choose DeepSpeed when**:
- Need advanced features (ZeRO-Infinity)
- Existing DeepSpeed infrastructure
- Maximum performance critical
