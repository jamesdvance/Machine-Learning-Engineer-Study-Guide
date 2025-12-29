# Mistral Architecture

## Summary

Mistral is Mistral AI's family of efficient decoder-only transformer models that achieve strong performance through architectural innovations focused on inference efficiency. Mistral 7B introduced Sliding Window Attention (SWA) combined with sparse attention patterns, enabling efficient processing of long sequences. The architecture maintains LLaMA's core improvements (RoPE, SwiGLU, RMSNorm, GQA) while adding attention optimizations that reduce memory and computation for long contexts. Mixtral extends this with Mixture of Experts (MoE) for better quality-to-compute ratios.

Key points to remember:

- Sliding Window Attention: Each token attends to a fixed window, reducing O(n^2) to O(n*w)
- Rolling Buffer KV-Cache: Fixed-size cache that wraps around for memory efficiency
- Grouped Query Attention: 8 KV heads for 32 query heads (4x reduction)
- Builds on LLaMA: RoPE, SwiGLU, RMSNorm, pre-normalization
- Mixtral: Mixture of Experts variant with 8 experts, 2 active per token
- Context extension: SWA enables efficient long-context inference
- Strong efficiency: 7B model outperforms many larger models

## Architecture Overview

### Mistral 7B Configuration

| Component | Value |
|-----------|-------|
| Parameters | 7.3B |
| Layers | 32 |
| d_model | 4096 |
| Heads | 32 |
| KV Heads | 8 |
| d_ff | 14336 |
| Vocab size | 32,000 |
| Window size | 4096 |
| Max context | 32K+ (with SWA) |

### Key Innovations

```
+------------------------+
|    Mistral 7B          |
+------------------------+
|                        |
| Sliding Window         | <- O(n*w) attention, not O(n^2)
| Attention              |
|                        |
| Rolling Buffer         | <- Fixed memory for any length
| KV Cache               |
|                        |
| Grouped Query          | <- 4x KV memory reduction
| Attention              |
|                        |
| RoPE + RMSNorm         | <- LLaMA foundations
| + SwiGLU               |
|                        |
+------------------------+
```

## Sliding Window Attention (SWA)

### Concept

Standard attention: each token attends to all previous tokens (O(n^2))
Sliding window: each token attends to only W previous tokens (O(n*W))

```
Standard Causal Attention:        Sliding Window (W=3):

Token 5 attends to: 1,2,3,4       Token 5 attends to: 3,4
Token 4 attends to: 1,2,3         Token 4 attends to: 2,3
Token 3 attends to: 1,2           Token 3 attends to: 1,2
Token 2 attends to: 1             Token 2 attends to: 1
Token 1 attends to: (none)        Token 1 attends to: (none)
```

### Information Propagation

Despite limited window, information propagates through layers:

```
Layer 1: Token 5 sees tokens 3-4
Layer 2: Token 5 sees tokens 1-4 (via layer 1 representations)
Layer 3: Token 5 sees tokens 1-4 with 2-hop connections
...

Effective receptive field = window_size * n_layers
For Mistral: 4096 * 32 = 131K tokens theoretically
```

### Implementation

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.window_size = config.window_size

        self.wq = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # Handle KV cache
        if cache is not None:
            k, v = self._update_cache(cache, k, v)

        # Expand KV heads for GQA
        n_groups = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_groups, dim=2)
        v = v.repeat_interleave(n_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Sliding window attention scores
        scores = self._sliding_window_scores(q, k)

        # Apply mask and softmax
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)

    def _sliding_window_scores(self, q, k):
        """Compute attention scores with sliding window."""
        B, H, T_q, D = q.shape
        _, _, T_k, _ = k.shape

        # Full attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # Create sliding window mask
        # Each query position can only attend to window_size keys before it
        positions = torch.arange(T_q, device=q.device)
        key_positions = torch.arange(T_k, device=k.device)

        # Distance matrix
        distances = positions.unsqueeze(1) - key_positions.unsqueeze(0)

        # Valid positions: within window and causal
        valid = (distances >= 0) & (distances < self.window_size)

        # Apply mask
        mask = torch.where(valid, 0.0, float('-inf'))
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        return scores
```

## Rolling Buffer KV-Cache

### Concept

Instead of growing cache indefinitely, use a fixed-size circular buffer:

```
Traditional KV Cache (grows with sequence):
Position:  1  2  3  4  5  6  7  8  9  10 ...
Cache:    [K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 ...]
Memory: O(sequence_length)

Rolling Buffer (fixed size = window):
Position:  1  2  3  4  5  6  7  8  9  10 ...
Cache:    [K9 K10 K3 K4 K5 K6 K7 K8]  (window=8)
           ^-- wraps around
Memory: O(window_size)
```

### Implementation

```python
class RollingKVCache:
    def __init__(self, max_batch_size, n_layers, n_kv_heads, d_head, window_size, device):
        self.window_size = window_size
        self.cache_k = torch.zeros(
            n_layers, max_batch_size, window_size, n_kv_heads, d_head,
            device=device
        )
        self.cache_v = torch.zeros(
            n_layers, max_batch_size, window_size, n_kv_heads, d_head,
            device=device
        )
        self.positions = torch.zeros(n_layers, dtype=torch.long, device=device)

    def update(self, layer_idx, k, v, start_pos):
        """Update cache with new keys/values using circular indexing."""
        B, T, H, D = k.shape

        for i in range(T):
            pos = start_pos + i
            cache_pos = pos % self.window_size  # Circular index

            self.cache_k[layer_idx, :B, cache_pos] = k[:, i]
            self.cache_v[layer_idx, :B, cache_pos] = v[:, i]

        self.positions[layer_idx] = start_pos + T
        return self.cache_k[layer_idx, :B], self.cache_v[layer_idx, :B]

    def get_valid_cache(self, layer_idx, batch_size, current_pos):
        """Get cache entries within the sliding window."""
        window_start = max(0, current_pos - self.window_size)
        valid_length = min(current_pos, self.window_size)

        # Gather entries in correct order
        indices = [(window_start + i) % self.window_size
                   for i in range(valid_length)]
        indices = torch.tensor(indices, device=self.cache_k.device)

        k = self.cache_k[layer_idx, :batch_size].index_select(1, indices)
        v = self.cache_v[layer_idx, :batch_size].index_select(1, indices)

        return k, v
```

### Memory Comparison

```python
def compare_cache_memory(seq_length, window_size, n_layers, n_kv_heads, d_head):
    """Compare memory usage of cache strategies."""
    bytes_per_param = 2  # FP16

    # Standard cache: grows with sequence
    standard_memory = seq_length * n_layers * n_kv_heads * d_head * 2 * bytes_per_param

    # Rolling buffer: fixed size
    rolling_memory = window_size * n_layers * n_kv_heads * d_head * 2 * bytes_per_param

    print(f"Sequence length: {seq_length}")
    print(f"Standard cache: {standard_memory / 1e9:.2f} GB")
    print(f"Rolling buffer: {rolling_memory / 1e9:.2f} GB")
    print(f"Savings: {(1 - rolling_memory/standard_memory) * 100:.1f}%")

# Example: Mistral 7B with 32K context
compare_cache_memory(
    seq_length=32768,
    window_size=4096,
    n_layers=32,
    n_kv_heads=8,
    d_head=128
)
# Standard cache: ~4.3 GB
# Rolling buffer: ~0.5 GB
# Savings: 87.5%
```

## Grouped Query Attention in Mistral

Mistral uses 8 KV heads for 32 query heads:

```python
class MistralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = 32      # Query heads
        self.n_kv_heads = 8    # Key-value heads
        self.n_groups = 4      # Each KV head serves 4 query heads

        self.d_head = config.d_model // self.n_heads

        self.wq = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(self.n_heads * self.d_head, config.d_model, bias=False)
```

## Complete Mistral Block

```python
class MistralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SlidingWindowAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        # Pre-norm attention
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos, freqs_sin,
            mask, cache
        )

        # Pre-norm FFN
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


class Mistral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            MistralBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # RoPE cache
        freqs_cos, freqs_sin = precompute_rope_cache(
            config.d_model // config.n_heads,
            config.max_seq_len * 2  # Allow for longer inference
        )
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def forward(self, tokens, start_pos=0, cache=None):
        B, T = tokens.shape
        h = self.embed_tokens(tokens)

        freqs_cos = self.freqs_cos[start_pos:start_pos + T]
        freqs_sin = self.freqs_sin[start_pos:start_pos + T]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, cache=cache)

        h = self.norm(h)
        return self.lm_head(h)
```

## Mixtral: Mixture of Experts

### Overview

Mixtral 8x7B extends Mistral with Sparse Mixture of Experts:
- 8 expert FFN networks per layer
- Router selects top-2 experts per token
- Total parameters: 46.7B
- Active parameters per token: ~12.9B (effectively 2x7B)

### Expert Router

```python
class MoERouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        """Route tokens to experts."""
        # x: (batch, seq, d_model)
        B, T, D = x.shape

        # Compute routing scores
        router_logits = self.gate(x)  # (B, T, n_experts)

        # Select top-k experts per token
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # Both: (B, T, top_k)

        # Normalize weights
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, selected_experts
```

### Sparse MoE Layer

```python
class MixtralMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = 8
        self.top_k = 2

        # Create expert networks (each is a SwiGLU FFN)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.d_ff)
            for _ in range(self.n_experts)
        ])

        self.router = MoERouter(config.d_model, self.n_experts, self.top_k)

    def forward(self, x):
        B, T, D = x.shape

        # Get routing decisions
        routing_weights, selected_experts = self.router(x)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = x[expert_mask]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert's contribution
            weights_for_expert = torch.where(
                selected_experts == expert_idx,
                routing_weights,
                torch.zeros_like(routing_weights)
            ).sum(dim=-1)

            # Weighted contribution
            output[expert_mask] += (
                expert_output * weights_for_expert[expert_mask].unsqueeze(-1)
            )

        return output
```

### Mixtral Block

```python
class MixtralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SlidingWindowAttention(config)
        self.moe = MixtralMoE(config)  # MoE replaces FFN
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos, freqs_sin, mask, cache
        )
        out = h + self.moe(self.ffn_norm(h))
        return out
```

## Practical Usage

### Loading Mistral with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Generate
prompt = "Explain how sliding window attention works:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Mixtral Usage

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # Recommended for memory
)
```

### Instruction-Tuned Variants

```python
# Mistral Instruct
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# Chat format
messages = [
    {"role": "user", "content": "What is machine learning?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

## Performance Characteristics

### Efficiency Gains

| Aspect | Standard Attention | Mistral SWA |
|--------|-------------------|-------------|
| Attention complexity | O(n^2) | O(n * w) |
| KV cache size | O(n * L * d) | O(w * L * d) |
| Memory for 32K context | ~4.3 GB | ~0.5 GB |
| Throughput at 32K | Limited by memory | Near full speed |

### Quality-Efficiency Tradeoff

| Model | Parameters | Performance |
|-------|------------|-------------|
| LLaMA 2 7B | 7B | Baseline |
| Mistral 7B | 7.3B | > LLaMA 2 13B |
| LLaMA 2 13B | 13B | Baseline+ |
| Mixtral 8x7B | 46.7B (12.9B active) | > LLaMA 2 70B |

## Key Takeaways

1. **Sliding window enables scale**: O(n*w) attention makes long contexts practical.

2. **Rolling buffer bounds memory**: Fixed cache size regardless of sequence length.

3. **Information still propagates**: Multiple layers provide effective global attention.

4. **GQA complements SWA**: Together they dramatically reduce memory requirements.

5. **MoE multiplies capacity**: Mixtral gets 8x parameters with 2x compute.

6. **Efficiency without quality loss**: Mistral 7B outperforms larger models.

7. **Practical for deployment**: Optimizations make real-world serving feasible.
