# Decoder-Only Transformer Architectures

## Summary

Decoder-only transformers are the dominant architecture for modern large language models. Unlike encoder-decoder models that process input and output separately, decoder-only models use a single stack of transformer layers with causal (left-to-right) attention masking for autoregressive text generation. This unified design simplifies training, enables efficient in-context learning, and scales effectively. GPT established the paradigm, LLaMA refined it with modern innovations, and Mistral added efficiency optimizations for practical deployment.

Key points to remember:

- Causal attention: Each token only attends to previous tokens, enabling autoregressive generation
- Single stack: No separate encoder; input and output share the same transformer layers
- Pre-training: Next-token prediction (causal language modeling)
- In-context learning: Models learn from examples in the prompt without parameter updates
- Scaling: Quality improves predictably with compute, data, and parameters
- Modern innovations: RoPE, GQA, SwiGLU, RMSNorm have become standard
- Inference optimization: KV-cache, sliding window attention enable practical deployment

## Why Decoder-Only?

### Historical Context

```
2017: Original Transformer (Encoder-Decoder)
      - Designed for sequence-to-sequence tasks (translation)
      |
2018: GPT-1 (Decoder-Only)
      - Showed unsupervised pre-training works
      - Simpler architecture, train on raw text
      |
2018: BERT (Encoder-Only)
      - Bidirectional attention for understanding tasks
      - Required task-specific heads for generation
      |
2020: GPT-3 (Decoder-Only at Scale)
      - Demonstrated emergent abilities
      - In-context learning without fine-tuning
      - Decoder-only becomes dominant paradigm
```

### Decoder-Only Advantages

| Advantage | Explanation |
|-----------|-------------|
| Simplicity | One architecture for all tasks |
| Training efficiency | Single objective (next-token prediction) |
| In-context learning | Learns from prompt examples |
| Unified input-output | No separation needed |
| Scalability | Simpler to scale to billions of parameters |
| Versatility | Same model for generation, classification, reasoning |

### When to Use Decoder-Only

- Text generation (stories, code, emails)
- Conversational AI (chatbots, assistants)
- In-context learning tasks
- Reasoning and problem-solving
- Code completion and generation
- General-purpose language tasks

## Core Architecture

### Structure

```
Input: "The weather is"
        |
        v
+----------------+
| Token Embed    |  Map tokens to vectors
+----------------+
        |
        v
+----------------+
| Position       |  Add positional information
| Encoding       |  (learned absolute or RoPE)
+----------------+
        |
        v
+----------------+
| Transformer    |
| Block 1        |  x N layers
|  - Attention   |  (causal masked)
|  - FFN         |
|  - Residual    |
|  - Normalize   |
+----------------+
        |
        v
+----------------+
| Output Head    |  Project to vocabulary
+----------------+
        |
        v
Output: logits -> "nice" (next token)
```

### Causal Attention Masking

```python
def causal_attention(Q, K, V, mask=None):
    """Attention with causal masking."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Causal mask: prevent attending to future tokens
    seq_len = Q.shape[-2]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    )
    scores.masked_fill_(causal_mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)
```

### Autoregressive Generation

```python
def generate(model, prompt_tokens, max_tokens):
    """Generate text autoregressively."""
    tokens = prompt_tokens.clone()

    for _ in range(max_tokens):
        # Forward pass (only need logits for last position)
        logits = model(tokens)[:, -1, :]

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token == EOS_TOKEN:
            break

    return tokens
```

## Architecture Evolution

### From GPT to Modern LLMs

| Component | GPT | GPT-2/3 | LLaMA | Mistral |
|-----------|-----|---------|-------|---------|
| Normalization | Post-norm | Pre-norm | RMSNorm | RMSNorm |
| Position | Learned | Learned | RoPE | RoPE |
| Activation | GELU | GELU | SwiGLU | SwiGLU |
| Attention | MHA | MHA | MHA/GQA | GQA + SWA |
| Bias | Yes | Yes | No | No |

### Key Innovations

```
GPT (2018)          LLaMA (2023)           Mistral (2023)
-----------         ------------           --------------
LayerNorm    -->    RMSNorm          -->   RMSNorm
                    (faster)               (same)

Learned pos  -->    RoPE             -->   RoPE
                    (extrapolates)         (same)

GELU         -->    SwiGLU           -->   SwiGLU
                    (better quality)       (same)

Full MHA     -->    GQA              -->   GQA + Sliding
                    (less memory)          (long context)
```

## Model Comparison

### Architecture Details

| Model | Params | Layers | d_model | Heads | KV Heads | d_ff | Context |
|-------|--------|--------|---------|-------|----------|------|---------|
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 25 | 6400 | 1024 |
| GPT-3 | 175B | 96 | 12288 | 96 | 96 | 49152 | 2048 |
| LLaMA 7B | 6.7B | 32 | 4096 | 32 | 32 | 11008 | 2048 |
| LLaMA 2 70B | 70B | 80 | 8192 | 64 | 8 | 28672 | 4096 |
| Mistral 7B | 7.3B | 32 | 4096 | 32 | 8 | 14336 | 32K+ |

### Efficiency Trade-offs

| Feature | Benefit | Cost |
|---------|---------|------|
| GQA (8 KV heads) | 4-8x less KV cache | Slight quality drop |
| RMSNorm | 10-15% faster | None |
| SwiGLU | Better quality | 50% more FFN params |
| Sliding Window | O(n) vs O(n^2) | Limited direct attention |
| No bias | Fewer params | None |

## Training

### Pre-training Objective

All decoder-only models use causal language modeling:

```python
def compute_loss(model, batch):
    """Next-token prediction loss."""
    input_ids = batch['input_ids']

    # Shift for next-token prediction
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=PAD_TOKEN
    )
    return loss
```

### Scaling Laws

Performance improves predictably with scale:

```
Loss ~ C^(-±)

Where:
- C = compute budget (FLOPs)
- ± H 0.05-0.1

Key findings:
- Compute-optimal: balance model size and training data
- Chinchilla scaling: ~20 tokens per parameter
- Emergent abilities: some capabilities appear suddenly at scale
```

## Inference Optimization

### KV-Cache

```python
class CachedAttention:
    """Cache key/value projections for efficient generation."""

    def forward(self, x, cache=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if cache is not None:
            # Append new K, V to cache
            k = torch.cat([cache['k'], k], dim=1)
            v = torch.cat([cache['v'], v], dim=1)

        new_cache = {'k': k, 'v': v}

        # Attention with full K, V history
        output = attention(q, k, v)

        return output, new_cache
```

### Memory Optimization

| Technique | Memory Savings | Quality Impact |
|-----------|---------------|----------------|
| GQA (8 heads) | 4-8x KV cache | Minimal |
| Sliding Window | Fixed cache size | Some long-range |
| Quantization (4-bit) | 4x model size | Small |
| PagedAttention | Variable | None |

## Choosing a Model

### Decision Framework

```
What's your use case?
|
+-- Maximum quality needed?
|   --> Largest model you can serve (70B+)
|
+-- Limited compute/memory?
|   --> Mistral 7B or Phi-2/3
|
+-- Long context required?
|   --> Mistral or LLaMA 3 (extended context)
|
+-- Need to fine-tune?
|   --> LLaMA 2/3 (open weights, commercial license)
|
+-- Research/experimentation?
|   --> LLaMA (most ecosystem support)
```

### Model Selection Matrix

| Requirement | Recommended | Reason |
|-------------|-------------|--------|
| Best open weights | LLaMA 3 70B | Quality + license |
| Efficient serving | Mistral 7B | GQA + SWA |
| Long context | Mistral/LLaMA 3 | 8K-128K context |
| Fine-tuning | LLaMA 2 | Ecosystem, tools |
| Mixture of experts | Mixtral 8x7B | Quality/compute ratio |

## Common Patterns

### Loading and Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
inputs = tokenizer("The key to success is", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0]))
```

### Efficient Serving

```python
# 4-bit quantization for memory efficiency
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Further Reading

For detailed coverage of specific architectures, see:

- [GPT](gpt/ReadMe.md) - The foundational decoder-only architecture
- [LLaMA](llama/ReadMe.md) - Modern improvements: RoPE, SwiGLU, RMSNorm, GQA
- [Mistral](mistral/ReadMe.md) - Efficiency optimizations: sliding window attention, MoE
