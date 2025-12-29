# GPT Architecture

## Summary

GPT (Generative Pre-trained Transformer) is the foundational decoder-only transformer architecture that established the paradigm for modern large language models. Introduced by OpenAI, GPT models use causal (left-to-right) self-attention to generate text autoregressively. The architecture has evolved through GPT-1, GPT-2, GPT-3, and GPT-4, with each generation increasing scale and capability while maintaining the core decoder-only design that has become the standard for generative language models.

Key points to remember:

- Decoder-only transformer: Uses causal masking for autoregressive generation
- Pre-training objective: Next-token prediction (causal language modeling)
- Key innovation: Showed that scale + unsupervised pre-training produces capable models
- Architecture components: Token embeddings, positional embeddings, transformer blocks, output head
- Each block: Masked self-attention, layer norm, feed-forward network
- GPT-2 introduced pre-norm (LayerNorm before attention/FFN)
- GPT-3 demonstrated emergent abilities from scale (175B parameters)
- Inference: Autoregressive token-by-token generation with KV-cache optimization

## Architecture Overview

### Core Structure

```
Input: "The cat sat"
       |
       v
+------------------+
| Token Embedding  |  (vocab_size, d_model)
+------------------+
       |
       v
+------------------+
| Position Embed   |  (max_seq_len, d_model)
+------------------+
       |
       v
+------------------+
| Transformer      |
| Block 1          |  x N layers
| - Masked Attn    |
| - LayerNorm      |
| - FFN            |
| - LayerNorm      |
+------------------+
       |
       v
+------------------+
| Final LayerNorm  |
+------------------+
       |
       v
+------------------+
| LM Head          |  (d_model, vocab_size)
+------------------+
       |
       v
Output: logits -> "on" (next token)
```

### Key Dimensions

| Model | Parameters | Layers | d_model | Heads | d_ff |
|-------|------------|--------|---------|-------|------|
| GPT-1 | 117M | 12 | 768 | 12 | 3072 |
| GPT-2 Small | 117M | 12 | 768 | 12 | 3072 |
| GPT-2 Medium | 345M | 24 | 1024 | 16 | 4096 |
| GPT-2 Large | 762M | 36 | 1280 | 20 | 5120 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 6400 |
| GPT-3 | 175B | 96 | 12288 | 96 | 49152 |

## Implementation

### Minimal GPT Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # Combined QKV projection
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        # Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)  # GPT-2 uses GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-norm (GPT-2 style)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### Full GPT Model

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (GPT-2 style)
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_seq_len

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
```

### Configuration

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072  # Usually 4 * d_model
    dropout: float = 0.1

# GPT-2 variants
GPT2_SMALL = GPTConfig()
GPT2_MEDIUM = GPTConfig(d_model=1024, n_layers=24, n_heads=16, d_ff=4096)
GPT2_LARGE = GPTConfig(d_model=1280, n_layers=36, n_heads=20, d_ff=5120)
GPT2_XL = GPTConfig(d_model=1600, n_layers=48, n_heads=25, d_ff=6400)
```

## Inference

### Autoregressive Generation

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Generate text autoregressively."""
    model.eval()

    for _ in range(max_new_tokens):
        # Crop to max sequence length
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len \
                   else idx[:, -model.config.max_seq_len:]

        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### KV-Cache for Efficient Inference

```python
class CausalSelfAttentionWithCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, past_kv=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Append to cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Store for next iteration
        present_kv = (k, v)

        # Attention (only query new positions against all K, V)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Causal mask for new tokens only
        T_full = k.size(2)
        mask = torch.tril(torch.ones(T, T_full, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), present_kv
```

## Pre-training

### Training Objective

Causal language modeling minimizes cross-entropy on next-token prediction:

```python
def compute_loss(model, batch):
    """Standard causal LM loss."""
    input_ids = batch['input_ids']

    # Shift for next-token prediction
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits, _ = model(inputs)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100  # Padding token
    )

    return loss
```

### Training Loop

```python
def train_gpt(model, dataloader, optimizer, scheduler, epochs):
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            loss = compute_loss(model, batch)
            loss.backward()

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Evolution of GPT

### GPT-1 to GPT-4

| Version | Year | Key Innovations |
|---------|------|-----------------|
| GPT-1 | 2018 | Unsupervised pre-training + fine-tuning paradigm |
| GPT-2 | 2019 | Scale (1.5B), pre-norm, zero-shot task transfer |
| GPT-3 | 2020 | Massive scale (175B), in-context learning, few-shot |
| GPT-4 | 2023 | Multimodal, improved reasoning, RLHF alignment |

### Architectural Changes

```
GPT-1:          Post-norm (LayerNorm after attention/FFN)
                |
GPT-2:          Pre-norm (LayerNorm before attention/FFN)
                + Weight tying (embedding = LM head)
                |
GPT-3:          Sparse attention (some layers)
                + Alternating dense/sparse patterns
                |
GPT-4:          Multimodal encoder
                + Mixture of Experts (rumored)
                + Extended context window
```

## Comparison to Other Architectures

| Aspect | GPT | LLaMA | Mistral |
|--------|-----|-------|---------|
| Positional encoding | Learned absolute | RoPE | RoPE |
| Normalization | Pre-norm | RMSNorm | RMSNorm |
| Activation | GELU | SiLU/SwiGLU | SiLU |
| Attention | Full | Full | Sliding window + full |
| KV cache | Standard | Grouped query | Grouped query |

## Practical Usage

### Loading Pre-trained GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors='pt')
output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    do_sample=True
)
print(tokenizer.decode(output[0]))
```

### Fine-tuning

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

## Key Takeaways

1. **Decoder-only design**: Causal masking enables efficient autoregressive generation.

2. **Pre-norm matters**: GPT-2's switch to pre-norm improved training stability.

3. **Scale unlocks capabilities**: GPT-3 showed emergent abilities from parameter scale.

4. **KV-cache is essential**: Without caching, inference is quadratic in sequence length.

5. **Weight tying**: Sharing embedding and output projection reduces parameters.

6. **Foundation for modern LLMs**: GPT established the paradigm that LLaMA, Mistral, and others build upon.

7. **Training at scale**: Requires gradient clipping, learning rate warmup, and careful initialization.
