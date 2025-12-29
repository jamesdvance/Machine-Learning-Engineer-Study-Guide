# KV-Cache Optimization

## Summary

The KV-cache stores key and value projections from previous tokens during autoregressive generation, avoiding redundant computation. Without caching, generating N tokens requires O(N^2) total compute as each new token recomputes attention over all previous tokens. With KV-cache, generation is O(N) in compute but requires O(N) memory per layer. This memory cost becomes the primary bottleneck for long sequences and large batches. Optimizations include grouped-query attention (GQA), multi-query attention (MQA), paged attention, and sliding window approaches.

Key points to remember:

- Avoids recomputation: Store key/value projections instead of recomputing
- Memory bottleneck: KV-cache often dominates inference memory
- Size calculation: 2 * layers * heads * head_dim * seq_len * batch * precision
- GQA/MQA: Reduce cache size by sharing KV heads across query heads
- Paged attention: Efficient memory management with blocks
- Sliding window: Bounded cache for streaming applications
- Critical for throughput: Cache management determines max batch size

## Why KV-Cache Matters

### Without Cache: Quadratic Cost

```python
def generate_without_cache(model, prompt, max_tokens):
    """Naive generation - recomputes all attention each step."""
    tokens = prompt.copy()

    for _ in range(max_tokens):
        # Recompute attention over ALL previous tokens
        # Token 100: Attend to tokens 0-99 (100 operations)
        # Token 101: Attend to tokens 0-100 (101 operations)
        # Total: 1 + 2 + ... + N = O(N^2)

        logits = model(tokens)  # Full forward pass
        next_token = sample(logits[-1])
        tokens.append(next_token)

    return tokens

# Generating 1000 tokens: ~500K attention operations
```

### With Cache: Linear Cost

```python
def generate_with_cache(model, prompt, max_tokens):
    """Cached generation - each step is O(1) in compute."""
    tokens = prompt.copy()
    kv_cache = None

    # Prefill: Process prompt, build initial cache
    logits, kv_cache = model(tokens, kv_cache=None)

    for _ in range(max_tokens):
        # Only process last token, attend to cached KV
        # Each step: O(seq_len) attention, O(1) new computation
        logits, kv_cache = model(
            [tokens[-1]],  # Only new token
            kv_cache=kv_cache
        )
        next_token = sample(logits[-1])
        tokens.append(next_token)

    return tokens

# Generating 1000 tokens: ~1000 attention operations (1000x faster)
```

## KV-Cache Structure

### Memory Layout

```python
class KVCache:
    def __init__(self, config, batch_size, max_seq_len):
        # Cache dimensions
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Allocate cache: [layers, 2, batch, heads, seq_len, head_dim]
        # Factor of 2 for key and value
        self.cache = torch.zeros(
            self.num_layers,
            2,  # Key and Value
            batch_size,
            self.num_kv_heads,
            max_seq_len,
            self.head_dim,
            dtype=torch.float16,
            device="cuda"
        )

        self.seq_len = 0  # Current sequence position

    def update(self, layer_idx, key, value):
        """Append new KV to cache."""
        seq_len = key.shape[2]  # New tokens
        start = self.seq_len
        end = start + seq_len

        self.cache[layer_idx, 0, :, :, start:end, :] = key
        self.cache[layer_idx, 1, :, :, start:end, :] = value

        if layer_idx == self.num_layers - 1:
            self.seq_len = end

    def get(self, layer_idx):
        """Get cached KV up to current position."""
        return (
            self.cache[layer_idx, 0, :, :, :self.seq_len, :],  # Keys
            self.cache[layer_idx, 1, :, :, :self.seq_len, :]   # Values
        )
```

### Memory Calculation

```python
def calculate_kv_cache_size(config, batch_size, seq_len, precision_bytes=2):
    """
    Calculate KV-cache memory in bytes.

    For LLaMA-2-7B with 4K context, batch=1:
    32 layers * 32 heads * 128 head_dim * 4096 seq * 2 (KV) * 2 bytes
    = 2.1 GB per request
    """
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Total elements: layers * 2 (KV) * batch * heads * seq * head_dim
    elements = (
        num_layers * 2 * batch_size *
        num_kv_heads * seq_len * head_dim
    )

    return elements * precision_bytes

# Examples:
# LLaMA-2-7B (32 layers, 32 heads, 128 dim):
#   - 4K context, batch 1: 2.1 GB
#   - 4K context, batch 16: 34 GB
#   - Model weights: 14 GB
#   -> KV cache dominates at large batch/context

# LLaMA-2-70B (80 layers, 64 heads, 128 dim):
#   - 4K context, batch 1: 10.5 GB
#   - 4K context, batch 4: 42 GB
```

## Attention Head Optimization

### Multi-Query Attention (MQA)

```python
class MultiQueryAttention(nn.Module):
    """
    Single KV head shared across all query heads.
    Reduces KV cache by num_heads factor.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        # Many query heads, ONE KV head
        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim)  # 1 head
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim)  # 1 head
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, kv_cache=None):
        batch, seq_len, _ = hidden_states.shape

        # Project queries (all heads)
        q = self.q_proj(hidden_states)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)

        # Project keys and values (single head)
        k = self.k_proj(hidden_states).unsqueeze(2)  # Add head dim
        v = self.v_proj(hidden_states).unsqueeze(2)

        # Broadcast single KV to all query heads
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)

        # Standard attention computation
        attn_output = scaled_dot_product_attention(q, k, v)

        return self.o_proj(attn_output.flatten(-2))

# KV cache reduction: 32x smaller for 32-head model
# Quality trade-off: Slight degradation vs MHA
```

### Grouped-Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    """
    Multiple KV heads shared among groups of query heads.
    Balance between MQA efficiency and MHA quality.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads  # e.g., 32
        self.num_kv_heads = config.num_key_value_heads  # e.g., 8
        self.num_groups = self.num_heads // self.num_kv_heads  # 4
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, kv_cache=None):
        batch, seq_len, _ = hidden_states.shape

        # Project all
        q = self.q_proj(hidden_states).view(
            batch, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )

        # Expand KV heads to match query heads
        # Each KV head serves num_groups query heads
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)

        # Update cache
        if kv_cache is not None:
            k, v = self.update_cache(kv_cache, k, v)

        attn_output = scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn_output.flatten(-2))

# LLaMA-2 uses GQA: 32 query heads, 8 KV heads
# KV cache reduction: 4x smaller
# Quality: Nearly matches MHA
```

### Comparison

| Method | KV Heads | Cache Size | Quality | Used By |
|--------|----------|------------|---------|---------|
| MHA | All | 1x | Best | GPT-3, older models |
| GQA | Groups | 4-8x smaller | Near-MHA | LLaMA-2, Mistral |
| MQA | One | 32x smaller | Slightly lower | Falcon, PaLM-2 |

## Paged Attention

### Block-Based Memory Management

```python
class PagedKVCache:
    """
    Manage KV cache as pages/blocks like OS virtual memory.
    Eliminates fragmentation, enables efficient memory sharing.
    """

    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim):
        self.block_size = block_size  # Tokens per block (e.g., 16)

        # Physical blocks pool
        # Shape: [num_blocks, 2, num_layers, num_heads, block_size, head_dim]
        self.blocks = torch.zeros(
            num_blocks, 2, num_layers, num_heads, block_size, head_dim,
            dtype=torch.float16, device="cuda"
        )

        self.free_blocks = set(range(num_blocks))
        self.block_tables = {}  # request_id -> list of block indices

    def allocate_blocks(self, request_id, num_tokens):
        """Allocate blocks for request."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return None  # Out of memory

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self.block_tables[request_id] = allocated
        return allocated

    def read_kv(self, request_id, layer_idx):
        """Read all KV for a request from scattered blocks."""
        block_ids = self.block_tables[request_id]

        # Gather from physical blocks
        keys = torch.cat([
            self.blocks[bid, 0, layer_idx] for bid in block_ids
        ], dim=1)  # Concat along seq dimension

        values = torch.cat([
            self.blocks[bid, 1, layer_idx] for bid in block_ids
        ], dim=1)

        return keys, values

    def write_kv(self, request_id, layer_idx, position, key, value):
        """Write KV for new tokens."""
        block_idx = position // self.block_size
        slot_idx = position % self.block_size

        # Allocate new block if needed
        if block_idx >= len(self.block_tables[request_id]):
            new_block = self.free_blocks.pop()
            self.block_tables[request_id].append(new_block)

        block_id = self.block_tables[request_id][block_idx]
        self.blocks[block_id, 0, layer_idx, :, slot_idx, :] = key
        self.blocks[block_id, 1, layer_idx, :, slot_idx, :] = value
```

### Copy-on-Write for Beam Search

```python
def fork_sequence(self, src_request_id, dst_request_id):
    """
    Share blocks between sequences (beam search, parallel sampling).
    Copy-on-write: Only copy when modified.
    """
    src_blocks = self.block_tables[src_request_id]

    # Share existing blocks (no copy)
    self.block_tables[dst_request_id] = src_blocks.copy()

    # Track reference counts
    for block_id in src_blocks:
        self.ref_counts[block_id] += 1

def write_with_cow(self, request_id, block_idx, ...):
    """Copy block if shared before writing."""
    block_id = self.block_tables[request_id][block_idx]

    if self.ref_counts[block_id] > 1:
        # Block is shared, make a copy
        new_block_id = self.allocate_new_block()
        self.copy_block(block_id, new_block_id)
        self.block_tables[request_id][block_idx] = new_block_id
        self.ref_counts[block_id] -= 1
        block_id = new_block_id

    # Now safe to write
    self.write_to_block(block_id, ...)
```

## Sliding Window Cache

### Bounded Memory for Long Sequences

```python
class SlidingWindowCache:
    """
    Only cache last W tokens. Memory bounded regardless of sequence length.
    Used by Mistral with W=4096.
    """

    def __init__(self, config, window_size=4096):
        self.window_size = window_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Fixed-size circular buffer
        self.cache = torch.zeros(
            self.num_layers, 2, 1,  # batch
            self.num_heads, window_size, self.head_dim,
            dtype=torch.float16, device="cuda"
        )
        self.position = 0

    def update(self, layer_idx, key, value):
        """Write to circular buffer."""
        seq_len = key.shape[2]

        for i in range(seq_len):
            slot = (self.position + i) % self.window_size
            self.cache[layer_idx, 0, :, :, slot, :] = key[:, :, i, :]
            self.cache[layer_idx, 1, :, :, slot, :] = value[:, :, i, :]

        if layer_idx == self.num_layers - 1:
            self.position = (self.position + seq_len) % self.window_size

    def get_attention_mask(self, query_pos, seq_len):
        """Mask for sliding window attention."""
        # Only attend to tokens within window
        positions = torch.arange(seq_len)
        mask = (query_pos - positions) < self.window_size
        mask = mask & (positions <= query_pos)  # Causal
        return mask
```

### Rolling Buffer Implementation

```python
class RollingBufferCache:
    """
    Efficient implementation using position modulo.
    No memory copies needed for sliding.
    """

    def __init__(self, window_size, num_layers, num_heads, head_dim):
        self.window_size = window_size
        self.cache = torch.zeros(
            num_layers, 2, num_heads, window_size, head_dim
        )

    def get_slot(self, absolute_position):
        """Map absolute position to buffer slot."""
        return absolute_position % self.window_size

    def write(self, layer, position, key, value):
        """Write to rolling position."""
        slot = self.get_slot(position)
        self.cache[layer, 0, :, slot, :] = key
        self.cache[layer, 1, :, slot, :] = value

    def read_window(self, layer, current_position):
        """Read valid window of KV."""
        start = max(0, current_position - self.window_size + 1)
        indices = [self.get_slot(p) for p in range(start, current_position + 1)]

        keys = self.cache[layer, 0, :, indices, :]
        values = self.cache[layer, 1, :, indices, :]
        return keys, values
```

## Optimization Techniques

### KV Cache Quantization

```python
def quantize_kv_cache(cache, bits=8):
    """
    Quantize KV cache to reduce memory.
    INT8 halves cache memory with minimal quality loss.
    """
    # Compute per-head scales
    scales = cache.abs().amax(dim=-1, keepdim=True) / 127

    # Quantize
    quantized = torch.round(cache / scales).to(torch.int8)

    return quantized, scales

def dequantize_for_attention(quantized, scales):
    """Dequantize during attention computation."""
    return quantized.float() * scales

# Memory savings:
# FP16 cache: 2 bytes per element
# INT8 cache: 1 byte per element + scales
# ~50% reduction with <1% quality impact
```

### Cache Compression

```python
class CompressedKVCache:
    """
    Compress older cache entries more aggressively.
    Recent tokens: Full precision
    Older tokens: Quantized or summarized
    """

    def __init__(self, config, compress_after=1024):
        self.compress_after = compress_after
        self.recent_cache = []  # Full precision
        self.compressed_cache = []  # Quantized

    def update(self, layer_idx, key, value, position):
        # Add to recent
        self.recent_cache.append((key, value))

        # Compress old entries
        if len(self.recent_cache) > self.compress_after:
            old_kv = self.recent_cache.pop(0)
            compressed = self.compress(old_kv)
            self.compressed_cache.append(compressed)

    def compress(self, kv):
        """Compress using quantization or pooling."""
        key, value = kv
        # Option 1: Quantize to INT4
        q_key = quantize_int4(key)
        q_value = quantize_int4(value)
        return (q_key, q_value)

        # Option 2: Pool adjacent tokens
        # return (key.mean(dim=-2), value.mean(dim=-2))
```

### Prefix Caching

```python
class PrefixCache:
    """
    Cache KV for common prefixes (system prompts).
    Reuse across requests.
    """

    def __init__(self, model):
        self.model = model
        self.prefix_cache = {}  # prefix_hash -> kv_cache

    def get_or_compute(self, prefix_tokens):
        """Get cached prefix KV or compute and cache."""
        prefix_hash = hash(tuple(prefix_tokens))

        if prefix_hash not in self.prefix_cache:
            # Compute KV for prefix
            with torch.no_grad():
                _, kv_cache = self.model(
                    prefix_tokens,
                    use_cache=True
                )
            self.prefix_cache[prefix_hash] = kv_cache

        return self.prefix_cache[prefix_hash].clone()

    def generate(self, prefix_tokens, continuation_tokens):
        """Generate using cached prefix."""
        # Get pre-computed prefix KV
        kv_cache = self.get_or_compute(prefix_tokens)

        # Continue from cached state
        return self.model.generate(
            continuation_tokens,
            past_key_values=kv_cache
        )

# Benefit: Skip prefill for repeated system prompts
# Common: Same system prompt across many requests
```

## Key Takeaways

1. **Quadratic to linear**: KV-cache converts O(N^2) generation to O(N).

2. **Memory bottleneck**: Cache often exceeds model weights for long sequences.

3. **Size formula**: 2 * layers * heads * head_dim * seq_len * batch * precision.

4. **GQA is standard**: 4-8x cache reduction with minimal quality loss.

5. **Paged attention**: Block-based allocation eliminates fragmentation.

6. **Sliding window**: Bounded memory for streaming/long sequences.

7. **Quantization helps**: INT8 cache halves memory with minimal impact.
