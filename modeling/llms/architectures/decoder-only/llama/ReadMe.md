# LLaMA Architecture

## Summary

LLaMA (Large Language Model Meta AI) is Meta's family of open-weight decoder-only transformer models that brought GPT-3-level capabilities to the open-source community. LLaMA introduced several architectural improvements over GPT that have become standard in modern LLMs: RMSNorm for faster normalization, SwiGLU activation for better representation learning, and Rotary Position Embeddings (RoPE) for improved position encoding. LLaMA 2 added Grouped Query Attention (GQA) for efficient inference, and LLaMA 3 further refined these techniques.

Key points to remember:

- RMSNorm: Faster than LayerNorm, removes mean centering
- RoPE: Rotary positional embeddings encode relative positions in attention
- SwiGLU: Gated activation function improves over GELU
- Pre-normalization: Normalizes before attention and FFN (like GPT-2)
- GQA (LLaMA 2+): Groups key-value heads to reduce KV-cache memory
- No bias terms: Most linear layers have no bias for efficiency
- Open weights: Made high-quality LLMs accessible for research and fine-tuning

## Architecture Comparison

### LLaMA vs GPT

| Component | GPT | LLaMA |
|-----------|-----|-------|
| Normalization | LayerNorm | RMSNorm |
| Position encoding | Learned absolute | RoPE |
| Activation | GELU | SwiGLU |
| Attention | Standard MHA | MHA (LLaMA 1) / GQA (LLaMA 2+) |
| Bias | Most layers | No bias (except some layers) |
| Vocab size | 50,257 | 32,000 (LLaMA 1/2) / 128,000 (LLaMA 3) |

### Model Dimensions

| Model | Parameters | Layers | d_model | Heads | KV Heads | d_ff |
|-------|------------|--------|---------|-------|----------|------|
| LLaMA 7B | 6.7B | 32 | 4096 | 32 | 32 | 11008 |
| LLaMA 13B | 13B | 40 | 5120 | 40 | 40 | 13824 |
| LLaMA 2 7B | 6.7B | 32 | 4096 | 32 | 32 | 11008 |
| LLaMA 2 70B | 70B | 80 | 8192 | 64 | 8 | 28672 |
| LLaMA 3 8B | 8B | 32 | 4096 | 32 | 8 | 14336 |
| LLaMA 3 70B | 70B | 80 | 8192 | 64 | 8 | 28672 |

## Key Innovations

### RMSNorm (Root Mean Square Normalization)

RMSNorm removes the mean-centering of LayerNorm, using only variance normalization:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight
```

**Why RMSNorm?**
- 10-15% faster than LayerNorm (no mean computation)
- Empirically matches LayerNorm quality
- Fewer operations in the critical path

### Rotary Position Embeddings (RoPE)

RoPE encodes positions by rotating query and key vectors:

```python
def precompute_rope_cache(dim, max_seq_len, base=10000):
    """Precompute rotation matrices for RoPE."""
    # Frequency for each dimension pair
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Position indices
    t = torch.arange(max_seq_len)

    # Outer product: positions x frequencies
    freqs = torch.outer(t, freqs)

    # Complex exponentials for rotation
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def apply_rope(q, k, freqs_cos, freqs_sin):
    """Apply rotary embeddings to queries and keys."""
    # Reshape to pairs
    q_pairs = q.view(*q.shape[:-1], -1, 2)
    k_pairs = k.view(*k.shape[:-1], -1, 2)

    # Extract real and imaginary parts
    q_r, q_i = q_pairs[..., 0], q_pairs[..., 1]
    k_r, k_i = k_pairs[..., 0], k_pairs[..., 1]

    # Apply rotation
    # (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    q_out_r = q_r * freqs_cos - q_i * freqs_sin
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos

    # Combine back
    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)

    return q_out, k_out
```

**Why RoPE?**
- Encodes relative positions (position m-n in attention)
- Naturally extrapolates to longer sequences
- More efficient than learned absolute embeddings
- Better than sinusoidal for long contexts

### SwiGLU Activation

SwiGLU combines Swish (SiLU) activation with a gating mechanism:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        # SwiGLU uses 2/3 of the hidden dim for each projection
        # to maintain parameter count similar to standard FFN
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection

    def forward(self, x):
        # SwiGLU: SiLU(xW1) * (xW3)
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)
```

**Why SwiGLU?**
- Better gradient flow than GELU
- Gating allows selective information passing
- Empirically improves model quality
- Note: Requires 3 projections vs 2, so d_ff is adjusted

### Grouped Query Attention (GQA)

GQA (LLaMA 2+) shares key-value heads across multiple query heads:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # Handle KV cache
        if cache is not None:
            k = torch.cat([cache['k'], k], dim=1)
            v = torch.cat([cache['v'], v], dim=1)
        cache = {'k': k, 'v': v}

        # Expand KV heads to match query heads
        # (B, T, n_kv_heads, d_head) -> (B, T, n_heads, d_head)
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Transpose for attention: (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), cache
```

**Why GQA?**
- Reduces KV-cache memory by factor of `n_heads / n_kv_heads`
- LLaMA 2 70B uses 8 KV heads for 64 query heads (8x reduction)
- Minimal quality impact vs full MHA
- Essential for serving large models efficiently

## Full LLaMA Implementation

### Transformer Block

```python
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(
            config.d_model,
            config.n_heads,
            config.n_kv_heads
        )
        self.feed_forward = SwiGLU(
            config.d_model,
            config.d_ff
        )
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        # Pre-norm attention with residual
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos, freqs_sin,
            mask, cache
        )[0]

        # Pre-norm FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

### Complete Model

```python
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings (no position embedding - RoPE is applied in attention)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_rope_cache(
            config.d_model // config.n_heads,
            config.max_seq_len
        )
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def forward(self, tokens, start_pos=0):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)

        # Get position-specific RoPE values
        freqs_cos = self.freqs_cos[start_pos:start_pos + T]
        freqs_sin = self.freqs_sin[start_pos:start_pos + T]

        # Causal mask
        mask = torch.full((T, T), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)

        h = self.norm(h)
        logits = self.output(h)

        return logits
```

### Configuration

```python
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32  # For GQA; set < n_heads for LLaMA 2
    d_ff: int = 11008     # ~2.67 * d_model for SwiGLU
    vocab_size: int = 32000
    max_seq_len: int = 4096
    norm_eps: float = 1e-6

# LLaMA 2 configurations
LLAMA2_7B = LlamaConfig()
LLAMA2_13B = LlamaConfig(d_model=5120, n_layers=40, n_heads=40, n_kv_heads=40, d_ff=13824)
LLAMA2_70B = LlamaConfig(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, d_ff=28672)

# LLaMA 3 uses larger vocab (128K) and adjusted architecture
LLAMA3_8B = LlamaConfig(
    d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
    d_ff=14336, vocab_size=128000, max_seq_len=8192
)
```

## LLaMA Evolution

### LLaMA 1 to LLaMA 3

| Feature | LLaMA 1 | LLaMA 2 | LLaMA 3 |
|---------|---------|---------|---------|
| Context length | 2048 | 4096 | 8192+ |
| Attention | MHA | GQA | GQA |
| Vocab size | 32,000 | 32,000 | 128,000 |
| Training data | 1.4T tokens | 2T tokens | 15T+ tokens |
| Chat fine-tuning | No | Yes | Yes |
| Safety training | Minimal | RLHF | RLHF + DPO |

### Key Improvements

```
LLaMA 1:       Base architecture with RoPE, SwiGLU, RMSNorm
               |
LLaMA 2:       + Grouped Query Attention (GQA)
               + Extended context (2K -> 4K)
               + More training data (1.4T -> 2T tokens)
               + Chat versions with RLHF
               |
LLaMA 3:       + Larger vocabulary (32K -> 128K)
               + Extended context (4K -> 8K+)
               + Much more training data (15T+ tokens)
               + Improved tokenizer
               + Instruction following improvements
```

## Practical Usage

### Loading with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate
inputs = tokenizer("The key to machine learning is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Efficient Inference with 4-bit Quantization

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Fine-tuning with LoRA

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Now fine-tune with much fewer trainable parameters
```

## Key Takeaways

1. **RMSNorm over LayerNorm**: Simpler, faster, equally effective.

2. **RoPE enables flexibility**: Better position encoding with length extrapolation.

3. **SwiGLU activation**: Gated activation improves quality (at cost of extra parameters).

4. **GQA is essential for scale**: Reduces KV-cache memory for large models.

5. **No biases**: Most linear layers omit bias terms for efficiency.

6. **Open weights matter**: LLaMA democratized access to high-quality LLMs.

7. **Architecture convergence**: Modern LLMs (Mistral, Qwen, etc.) largely adopt LLaMA's design choices.
