# Flash Attention

## Summary

Flash Attention is an IO-aware attention algorithm that reduces memory usage from O(N^2) to O(N) while also improving speed. By computing attention in tiles and avoiding materialization of the full attention matrix, Flash Attention enables training with much longer sequences and larger batch sizes. It has become essential for efficient transformer training and inference.

Key points to remember:

- Reduces attention memory from O(N^2) to O(N) for sequence length N
- Often faster than standard attention due to reduced memory IO
- Computes attention in tiles without materializing full attention matrix
- Uses online softmax algorithm for numerical stability
- Supports causal masking and various attention patterns
- Available in PyTorch 2.0+ via scaled_dot_product_attention
- FlashAttention-2 provides further optimizations
- Critical for long-context models

## The Memory Problem

### Standard Attention Memory

```python
# Standard attention
Q, K, V = x @ W_q, x @ W_k, x @ W_v  # [B, N, d]
scores = Q @ K.T / sqrt(d)           # [B, N, N] - O(N^2) memory!
probs = softmax(scores)              # [B, N, N]
output = probs @ V                   # [B, N, d]
```

For sequence length N=8192, batch B=8, heads H=32:
```
Attention matrix: 8 * 32 * 8192 * 8192 * 2 bytes = 32 GB
```

This often exceeds GPU memory.

### Flash Attention Solution

```
Instead of materializing full N x N matrix:
1. Split Q, K, V into blocks
2. Compute attention block by block
3. Accumulate output using online softmax
4. Never store full attention matrix
```

Memory: O(N) for storing Q, K, V, output only.

## How Flash Attention Works

### Tiled Computation

```
Q: [B, N, d] split into blocks of size B_q
K: [B, N, d] split into blocks of size B_k
V: [B, N, d] split into blocks of size B_k

For each Q block:
    Initialize output accumulator and softmax statistics
    For each K, V block:
        Compute block attention scores
        Update output using online softmax
```

### Online Softmax

Key insight: Can compute softmax incrementally:

```python
# Standard softmax
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

# Online softmax: Update running statistics
m_new = max(m_old, max(block))
d_new = d_old * exp(m_old - m_new) + sum(exp(block - m_new))
output_new = (output_old * d_old * exp(m_old - m_new) + block_output) / d_new
```

This allows computing attention block by block without storing the full matrix.

## Usage

### PyTorch Native (2.0+)

```python
import torch.nn.functional as F

# Automatic Flash Attention selection
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)

# With causal masking (for autoregressive)
output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True
)
```

### Flash Attention Package

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Separate Q, K, V
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True
)

# Packed QKV
qkv = torch.stack([q, k, v], dim=2)  # [B, N, 3, H, d]
output = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    causal=True
)
```

### Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM

# Flash Attention enabled automatically if available
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"  # Explicit selection
)
```

## Memory Comparison

### Standard vs Flash Attention

| Sequence Length | Standard (per head) | Flash Attention |
|-----------------|---------------------|-----------------|
| 512 | 1 MB | 4 KB |
| 2048 | 16 MB | 16 KB |
| 8192 | 256 MB | 64 KB |
| 32768 | 4 GB | 256 KB |

### Practical Impact

For a model with 32 heads, batch 8:

```
Standard attention @ N=8192:
32 * 8 * 256 MB = 64 GB (doesn't fit!)

Flash Attention @ N=8192:
32 * 8 * 64 KB = 16 MB (easily fits)
```

## Speed Comparison

Flash Attention is often faster due to reduced memory IO:

| Operation | Memory IO | Compute |
|-----------|-----------|---------|
| Standard | High (O(N^2)) | O(N^2) |
| Flash | Low (O(N)) | O(N^2) |

Memory bandwidth is often the bottleneck, so Flash Attention provides speedup despite same compute.

### Benchmarks (A100)

| Sequence Length | Standard | Flash Attention | Speedup |
|-----------------|----------|-----------------|---------|
| 512 | 1.0x | 1.2x | 1.2x |
| 2048 | 1.0x | 2.0x | 2.0x |
| 8192 | OOM | Works | - |

## Flash Attention 2

### Improvements over Flash Attention 1

1. **Better parallelism**: Parallelizes over sequence length
2. **Reduced non-matmul FLOPs**: Optimized softmax
3. **Better occupancy**: Improved block sizes

### Usage

```python
from flash_attn import flash_attn_func

# Flash Attention 2 is the default in recent versions
output = flash_attn_func(q, k, v, causal=True)
```

### Speed Improvements

Flash Attention 2 is typically 1.5-2x faster than Flash Attention 1.

## Attention Patterns

### Causal Attention

```python
# For autoregressive models
output = flash_attn_func(q, k, v, causal=True)

# Or with PyTorch
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Bidirectional Attention

```python
# For BERT-style models
output = flash_attn_func(q, k, v, causal=False)
```

### Custom Masks

Flash Attention has limited mask support. For complex masks:

```python
# May need to fall back to standard attention
# Or use block-sparse attention variants
```

## Integration with Training

### Basic Training Loop

```python
import torch.nn.functional as F

class FlashAttentionLayer(nn.Module):
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention
        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # Flash Attention
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        output = output.transpose(1, 2).contiguous().view(B, N, H*d)
        return self.out_proj(output)
```

### With Gradient Checkpointing

Flash Attention is compatible with gradient checkpointing:

```python
from torch.utils.checkpoint import checkpoint

def attention_with_checkpoint(q, k, v):
    return checkpoint(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        q, k, v,
        use_reentrant=False
    )
```

## Limitations

### Mask Support

- Full attention masks not supported in Flash Attention
- Block-sparse patterns may require different implementations
- Use standard attention for complex masking

### Precision

- Primarily FP16/BF16
- May have slight numerical differences from standard attention
- Usually acceptable for training

### Hardware Requirements

- CUDA compute capability 8.0+ (A100, H100)
- Older GPUs fall back to standard attention
- AMD support via PyTorch

## Debugging

### Verify Flash Attention is Active

```python
# PyTorch 2.0+
import torch

# Check which implementation is used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    # This will error if Flash Attention unavailable
    output = F.scaled_dot_product_attention(q, k, v)
```

### Compare Outputs

```python
# Standard attention
scores = q @ k.transpose(-2, -1) / math.sqrt(d)
probs = F.softmax(scores, dim=-1)
standard_output = probs @ v

# Flash attention
flash_output = F.scaled_dot_product_attention(q, k, v)

# Should be close
print(f"Max diff: {(standard_output - flash_output).abs().max()}")
```

## Best Practices

1. **Always use Flash Attention**: Unless specific mask patterns required
2. **Use PyTorch 2.0+**: Built-in support, automatic selection
3. **Combine with BF16**: Maximum memory efficiency
4. **Enable for both training and inference**: Benefits both
5. **Check hardware support**: Falls back gracefully on older GPUs
6. **Profile memory**: Verify expected savings achieved
