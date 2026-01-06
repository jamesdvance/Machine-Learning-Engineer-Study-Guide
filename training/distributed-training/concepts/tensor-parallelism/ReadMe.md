# Tensor Parallelism

## Summary

Tensor parallelism splits individual tensor operations across multiple devices, allowing single layers to use the combined memory and compute of multiple GPUs. Unlike model parallelism which assigns entire layers to devices, tensor parallelism divides the computation within each layer. This approach is essential for training models with very large hidden dimensions that cannot fit individual layer weights on a single GPU.

Key points to remember:

- Splits weight matrices within layers across devices
- Enables layers larger than single GPU memory
- Requires high-bandwidth interconnect (NVLink strongly preferred)
- Communication happens within every layer forward and backward
- Column-parallel and row-parallel are the two main patterns
- Typically limited to single-node (8 GPUs) due to communication demands
- Megatron-LM popularized tensor parallelism for transformers
- Often combined with data and pipeline parallelism for maximum scale

## How Tensor Parallelism Works

### The Basic Idea

A matrix multiplication Y = XW can be split:

```
Original:
X: [batch, hidden]
W: [hidden, output]
Y = X @ W: [batch, output]

Column-parallel (split W columns):
W = [W1 | W2]  where W1, W2: [hidden, output/2]
Y1 = X @ W1   on GPU 0
Y2 = X @ W2   on GPU 1
Y = [Y1 | Y2]  concatenate

Row-parallel (split W rows):
W = [W1]   where W1, W2: [hidden/2, output]
    [W2]
X = [X1 | X2]  split input
Y = X1 @ W1 + X2 @ W2  reduce
```

### Why Both Patterns

Transformer layers use both patterns to minimize communication:

```
Attention:
  Q, K, V projections: Column-parallel (split heads)
  Output projection: Row-parallel (gather heads)

MLP:
  First linear: Column-parallel (split hidden)
  Second linear: Row-parallel (reduce hidden)
```

This arrangement requires only two all-reduce operations per transformer layer.

## Column-Parallel Linear

### Implementation

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Each rank holds out_features/world_size columns
        self.out_features_per_rank = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        self.bias = nn.Parameter(
            torch.empty(self.out_features_per_rank)
        )

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, in_features] - same on all ranks
        # output: [batch, out_features_per_rank] - different on each rank
        return nn.functional.linear(x, self.weight, self.bias)
```

### Communication Pattern

```
Input x (identical on all ranks)
    |
    v
GPU 0: x @ W[:, 0:N/2] -> Y0
GPU 1: x @ W[:, N/2:N] -> Y1
    |
    v
No communication needed after column-parallel!
(Unless you need the full output)
```

The output is partitioned: each rank has different columns of Y.

## Row-Parallel Linear

### Implementation

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Each rank holds in_features/world_size rows
        self.in_features_per_rank = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        # Bias only on rank 0, or split and reduce
        self.bias = nn.Parameter(torch.empty(out_features)) if rank == 0 else None

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, in_features_per_rank] - different on each rank
        # Need to gather x first or assume input is pre-partitioned

        local_output = nn.functional.linear(x, self.weight)

        # All-reduce to sum partial outputs
        dist.all_reduce(local_output)

        if self.bias is not None:
            local_output = local_output + self.bias

        return local_output
```

### Communication Pattern

```
Input x (partitioned: each rank has different columns)
    |
    v
GPU 0: x0 @ W[0:N/2, :] -> partial_Y0
GPU 1: x1 @ W[N/2:N, :] -> partial_Y1
    |
    v
All-reduce (sum): Y = partial_Y0 + partial_Y1
    |
    v
Output Y (identical on all ranks)
```

## Tensor Parallel Transformer

### Self-Attention

```python
class TensorParallelSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.num_heads = num_heads
        self.num_heads_per_rank = num_heads // world_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections: column-parallel
        # Each rank computes a subset of heads
        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,  # Q, K, V concatenated
            world_size,
            rank
        )

        # Output projection: row-parallel
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            world_size,
            rank
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Column-parallel QKV projection
        qkv = self.qkv(x)  # [batch, seq, 3 * hidden/world_size]

        # Split into Q, K, V
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads_per_rank, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Attention computation (local per rank)
        attn_output = scaled_dot_product_attention(q, k, v)

        # Reshape for output projection
        attn_output = attn_output.reshape(batch, seq_len, -1)

        # Row-parallel output projection (includes all-reduce)
        output = self.out_proj(attn_output)

        return output
```

### MLP Block

```python
class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank):
        super().__init__()

        # First projection: column-parallel
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            world_size,
            rank
        )

        # Second projection: row-parallel
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            world_size,
            rank
        )

    def forward(self, x):
        # Column-parallel: output is partitioned
        x = self.fc1(x)
        x = nn.functional.gelu(x)

        # Row-parallel: includes all-reduce, output is replicated
        x = self.fc2(x)

        return x
```

### Full Transformer Layer

```
Input x (replicated on all ranks)
    |
    +-- LayerNorm (replicated)
    |
    +-- Tensor-Parallel Attention
    |     Column-parallel QKV (no comm)
    |     Local attention computation
    |     Row-parallel output (all-reduce)  <-- Communication
    |
    +-- Residual add (replicated)
    |
    +-- LayerNorm (replicated)
    |
    +-- Tensor-Parallel MLP
    |     Column-parallel FC1 (no comm)
    |     Activation (local)
    |     Row-parallel FC2 (all-reduce)  <-- Communication
    |
    +-- Residual add (replicated)
    |
Output x (replicated on all ranks)
```

Total: 2 all-reduce operations per transformer layer.

## Communication Requirements

### Bandwidth Analysis

For a transformer layer with hidden size H and batch B, sequence length S:

All-reduce data volume per layer:
```
2 * B * S * H * element_size
```

For H=4096, B=1, S=2048, FP16:
```
2 * 1 * 2048 * 4096 * 2 bytes = 33.5 MB per layer
```

With 32 layers and 1ms per iteration target:
```
32 * 33.5 MB = 1.07 GB in 1ms
Requires: 1.07 GB / 0.001s = 1.07 TB/s
```

This is only achievable with NVLink (600+ GB/s bidirectional).

### Latency Impact

Each all-reduce adds latency. With 2 all-reduces per layer:

| Layers | All-reduces | Added latency (0.1ms each) |
|--------|-------------|---------------------------|
| 12 | 24 | 2.4 ms |
| 24 | 48 | 4.8 ms |
| 48 | 96 | 9.6 ms |

This significantly impacts throughput, especially for inference.

## Sequence Parallelism

Extends tensor parallelism to parallelize along sequence dimension:

### Motivation

Between attention and MLP blocks, activations are replicated:
```
After all-reduce: all ranks have same [batch, seq, hidden] tensor
```

This wastes memory. Sequence parallelism partitions along sequence.

### Implementation

```python
# After attention all-reduce, scatter along sequence
# Before MLP column-parallel, gather along sequence

def scatter_to_sequence_parallel(tensor, world_size, rank):
    """Scatter replicated tensor along sequence dimension."""
    seq_len = tensor.size(1)
    seq_per_rank = seq_len // world_size
    start = rank * seq_per_rank
    end = start + seq_per_rank
    return tensor[:, start:end, :]

def gather_from_sequence_parallel(tensor, world_size):
    """Gather sequence-parallel tensor."""
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=1)
```

Megatron-LM implements this as part of their tensor parallelism.

## Comparison with Model Parallelism

| Aspect | Tensor Parallelism | Model Parallelism |
|--------|-------------------|-------------------|
| Granularity | Within layers | Across layers |
| Communication | Per layer (2x per transformer) | At layer boundaries |
| Memory per device | Layer split across devices | Full layers on each |
| Bandwidth needs | Very high | Moderate |
| Typical scale | 2-8 GPUs (single node) | Any number |
| Implementation | Complex | Moderate |

### When to Choose Each

**Tensor parallelism**:
- Hidden dimensions too large for single GPU
- NVLink available within node
- Combined with pipeline parallelism across nodes

**Model parallelism** (alone):
- Deep but narrow models
- Lower bandwidth interconnect
- Usually combined with pipeline parallelism

**Both** (3D parallelism):
- Maximum scale training
- Tensor parallel within node, pipeline across nodes

## Framework Support

### Megatron-LM

```python
from megatron.core import tensor_parallel

# Megatron provides optimized implementations
qkv_weight = tensor_parallel.ColumnParallelLinear(
    hidden_size,
    3 * hidden_size,
    bias=True,
    gather_output=False  # Keep partitioned
)
```

### DeepSpeed

```python
# DeepSpeed inference supports tensor parallelism
import deepspeed

model = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 8}
)
```

### FairScale

```python
from fairscale.nn.model_parallel import ColumnParallelLinear

linear = ColumnParallelLinear(
    in_features=1024,
    out_features=4096,
    bias=True,
    gather_output=True
)
```

## Best Practices

### Sizing

1. **Divisibility**: Hidden size must be divisible by tensor parallel size
2. **Head alignment**: Number of attention heads must be divisible by TP size
3. **Power of 2**: TP size is typically 2, 4, or 8

### Performance

1. **NVLink required**: Do not use tensor parallelism across nodes
2. **Batch size**: Larger batches amortize communication overhead
3. **Fused kernels**: Use fused attention/MLP implementations

### Debugging

```python
# Verify tensor parallel is working
def check_tensor_parallel(tensor, name):
    local_sum = tensor.sum()
    dist.all_reduce(local_sum)
    print(f"Rank {rank}, {name}: local_shape={tensor.shape}, global_sum={local_sum}")
```

### Common Issues

**Shape mismatches**: Ensure all dimensions properly divided
```python
assert hidden_size % world_size == 0
assert num_heads % world_size == 0
```

**Silent correctness bugs**: Validate against single-GPU implementation
```python
# Run small batch through both implementations
# Compare outputs with torch.allclose()
```
