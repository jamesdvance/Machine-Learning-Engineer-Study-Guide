# Sliding Window Attention

## Summary

Sliding Window Attention (SWA) reduces the computational complexity of self-attention from O(n^2) to O(n*w) by limiting each token to attend only to a fixed window of preceding tokens. Despite this local constraint, information propagates across the full sequence through multiple transformer layers, creating an effective receptive field much larger than the window size. This approach, popularized by Mistral, enables efficient processing of very long sequences while maintaining strong performance through multi-layer information flow.

Key points to remember:

- Local attention: Each token attends to only w previous tokens
- Complexity reduction: O(n*w) instead of O(n^2)
- Information propagation: Effective receptive field = window * layers
- Memory efficiency: KV-cache bounded by window size
- Fixed memory: Same memory usage regardless of sequence length
- Combines with full attention: Some layers can use full attention for global context
- Trade-off: Slightly reduced direct long-range attention for major efficiency gains

## The Attention Bottleneck

### Standard Attention Complexity

```
Full self-attention:
- Each token attends to all previous tokens
- Complexity: O(n^2) in sequence length
- Memory: O(n) for KV-cache

For n = 32,000 tokens:
- Attention operations: ~1 billion per layer
- KV-cache: ~4GB (LLaMA 7B, FP16)
```

### Sliding Window Solution

```
Sliding window attention (w = 4096):
- Each token attends to w previous tokens
- Complexity: O(n * w) = O(n) for fixed w
- Memory: O(w) for KV-cache

For n = 32,000 with w = 4096:
- Attention operations: ~131 million per layer (8x reduction)
- KV-cache: ~0.5GB (87.5% reduction)
```

## How Sliding Window Works

### Attention Pattern

```
Standard Causal Attention:         Sliding Window (w=4):
Token 7 attends to: 1,2,3,4,5,6    Token 7 attends to: 4,5,6
Token 6 attends to: 1,2,3,4,5      Token 6 attends to: 3,4,5
Token 5 attends to: 1,2,3,4        Token 5 attends to: 2,3,4
Token 4 attends to: 1,2,3          Token 4 attends to: 1,2,3
Token 3 attends to: 1,2            Token 3 attends to: 1,2
Token 2 attends to: 1              Token 2 attends to: 1
Token 1 attends to: (none)         Token 1 attends to: (none)
```

### Information Propagation

Despite local attention, information flows globally through layers:

```
Layer 1: Token 8 sees [5,6,7] directly
Layer 2: Token 8 sees [2,3,4,5,6,7] (via layer 1 representations)
Layer 3: Token 8 sees [1,2,3,4,5,6,7] (via layer 2)
...

Effective receptive field = min(sequence_length, window_size * n_layers)

Mistral 7B: window=4096, layers=32
-> Theoretical receptive field: 4096 * 32 = 131,072 tokens
```

## Implementation

### Basic Sliding Window

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, start_pos=0):
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Create sliding window mask
        mask = self._create_sliding_window_mask(T, start_pos)

        # Attention with mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)

    def _create_sliding_window_mask(self, seq_len, start_pos=0):
        """Create causal sliding window mask."""
        # Position indices
        positions = torch.arange(seq_len)

        # Distance matrix
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Valid: causal (distance >= 0) and within window (distance < window_size)
        valid = (distances >= 0) & (distances < self.window_size)

        # Convert to attention mask (0 for valid, -inf for invalid)
        mask = torch.where(valid, 0.0, float('-inf'))

        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
```

### Rolling Buffer KV-Cache

```python
class RollingKVCache:
    """Fixed-size circular buffer for KV-cache."""

    def __init__(self, window_size, n_layers, n_heads, d_head, device):
        self.window_size = window_size
        self.n_layers = n_layers

        # Circular buffers for K and V
        self.k_cache = torch.zeros(
            n_layers, 1, window_size, n_heads, d_head,
            device=device
        )
        self.v_cache = torch.zeros(
            n_layers, 1, window_size, n_heads, d_head,
            device=device
        )
        self.positions = [0] * n_layers

    def update(self, layer_idx, k, v, pos):
        """Update cache with new K, V at given position."""
        # Circular index
        cache_pos = pos % self.window_size

        self.k_cache[layer_idx, :, cache_pos] = k
        self.v_cache[layer_idx, :, cache_pos] = v
        self.positions[layer_idx] = pos + 1

    def get(self, layer_idx, current_pos):
        """Get valid cache entries for attention."""
        # Determine valid range
        start = max(0, current_pos - self.window_size + 1)
        end = current_pos + 1

        # Gather entries (handling wrap-around)
        indices = [i % self.window_size for i in range(start, end)]
        indices = torch.tensor(indices, device=self.k_cache.device)

        k = self.k_cache[layer_idx].index_select(1, indices)
        v = self.v_cache[layer_idx].index_select(1, indices)

        return k, v


class SlidingWindowWithCache(SlidingWindowAttention):
    def forward_with_cache(self, x, cache, layer_idx, start_pos):
        """Efficient inference with rolling cache."""
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)

        # Update cache
        for i in range(T):
            cache.update(layer_idx, k[:, i], v[:, i], start_pos + i)

        # Get cached K, V
        cached_k, cached_v = cache.get(layer_idx, start_pos + T - 1)

        # Attention against cached values
        q = q.transpose(1, 2)
        cached_k = cached_k.transpose(1, 2)
        cached_v = cached_v.transpose(1, 2)

        scores = torch.matmul(q, cached_k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Create mask for valid positions
        mask = self._create_cache_mask(T, cached_k.size(2))
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, cached_v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)
```

## Combining with Full Attention

### Interleaved Pattern

Some architectures use sliding window for most layers with occasional full attention:

```python
class HybridAttentionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(config.n_layers):
            if i % config.full_attention_stride == 0:
                # Every N layers, use full attention
                self.layers.append(FullAttentionLayer(config))
            else:
                # Most layers use sliding window
                self.layers.append(SlidingWindowLayer(config))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example: Mistral-like pattern
# Layers 0, 8, 16, 24: Full attention (global context)
# Other layers: Sliding window (local processing)
```

### Longformer-Style Pattern

```
Layer pattern:
- Local sliding window for most positions
- Global attention for special tokens (CLS, task tokens)

[CLS] attends to: ALL positions
Token 1 attends to: [CLS], local window
Token 2 attends to: [CLS], local window
...
```

## Performance Analysis

### Complexity Comparison

| Method | Attention Ops | KV-Cache | When Faster |
|--------|--------------|----------|-------------|
| Full | O(n^2) | O(n) | Short sequences |
| Sliding Window | O(n*w) | O(w) | Long sequences |
| Crossover | - | - | n > 2*w typically |

### Memory Savings

```python
def compare_memory(seq_len, window_size, n_layers, n_kv_heads, d_head):
    """Compare KV-cache memory usage."""
    bytes_per_param = 2  # FP16

    # Full attention cache
    full_cache = seq_len * n_layers * n_kv_heads * d_head * 2 * bytes_per_param

    # Sliding window cache
    window_cache = window_size * n_layers * n_kv_heads * d_head * 2 * bytes_per_param

    print(f"Sequence length: {seq_len}")
    print(f"Full cache: {full_cache / 1e9:.2f} GB")
    print(f"Window cache: {window_cache / 1e9:.2f} GB")
    print(f"Savings: {(1 - window_cache/full_cache) * 100:.1f}%")

# Mistral 7B, 32K context
compare_memory(32768, 4096, 32, 8, 128)
# Full cache: 4.29 GB
# Window cache: 0.54 GB
# Savings: 87.5%
```

## Quality Trade-offs

### What's Preserved

- Local context: Full attention within window
- Multi-hop reasoning: Via layer-by-layer propagation
- Most benchmarks: Minimal impact on standard evals

### What's Affected

- Direct long-range attention: Must go through intermediate layers
- Very long-range dependencies: May be weaker
- Needle-in-haystack at distance: Performance can degrade

### Mitigation Strategies

```python
# 1. Larger windows for critical applications
config.window_size = 8192  # vs default 4096

# 2. Interleave with full attention layers
config.full_attention_every = 4  # Every 4th layer

# 3. Use RAG for very long documents
if len(document) > MAX_RELIABLE_CONTEXT:
    chunks = chunk_document(document)
    relevant = retrieve_relevant(chunks, query)
    context = concat(relevant)
```

## Practical Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM

# Mistral uses sliding window by default
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Model config shows window size
print(model.config.sliding_window)  # 4096
```

### Extending Window Size

```python
# Increase window for longer context
model.config.sliding_window = 8192

# Note: May need RoPE scaling too
model.config.rope_scaling = {
    "type": "dynamic",
    "factor": 2.0
}
```

## Key Takeaways

1. **O(n*w) scales linearly**: Fixed window means linear cost in sequence length.

2. **Information flows through layers**: Effective receptive field = window * layers.

3. **Rolling buffer bounds memory**: Fixed-size KV-cache regardless of sequence.

4. **Trade-off is manageable**: Most tasks don't need direct attention across 32K tokens.

5. **Combine with full attention**: Interleaving patterns provide global context where needed.

6. **Enables long context**: Makes 100K+ token contexts practical.

7. **Standard in modern LLMs**: Mistral and derivatives use this extensively.
