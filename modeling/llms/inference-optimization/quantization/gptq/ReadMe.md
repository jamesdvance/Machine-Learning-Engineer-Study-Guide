# GPTQ Quantization

## Summary

GPTQ (GPT-Quantization) is a post-training quantization method that achieves high-quality 4-bit and 3-bit quantization by using calibration data to minimize quantization error. Unlike simple round-to-nearest quantization, GPTQ solves an optimization problem: find quantized weights that minimize output error on representative inputs. It processes weights column by column, updating remaining weights to compensate for quantization error. This approach achieves near-lossless 4-bit quantization for large language models.

Key points to remember:

- Data-driven quantization: Uses calibration samples to minimize error
- Layer-wise optimization: Quantizes and compensates column by column
- Hessian-based: Leverages second-order information for better updates
- One-shot: No iterative training, fast quantization process
- High quality: Near-lossless at 4-bit for large models
- Activation order: Quantizes columns by importance (Hessian diagonal)
- Industry standard: Widely used for model distribution

## The GPTQ Algorithm

### Core Idea

```
Traditional quantization: Just round weights
q(W) = round(W / scale) * scale

GPTQ insight: Compensate for errors by updating unquantized weights
When quantizing column j:
1. Quantize column j
2. Compute quantization error
3. Update columns j+1, j+2, ... to compensate
4. Move to next column
```

### Mathematical Foundation

```python
# Goal: Find quantized weights Q that minimize:
# ||WX - QX||^2  (output error on calibration data X)

# Key insight: This equals
# ||W - Q||_H^2  where H = XX^T (Hessian)

# Column-wise solution:
# For each column j:
#   1. Quantize: q_j = quant(w_j)
#   2. Error: delta = w_j - q_j
#   3. Update remaining columns:
#      W[:, j+1:] += delta * H[j, j+1:] / H[j, j]
```

### Algorithm Steps

```python
def gptq_quantize(W, X, n_bits=4, group_size=128):
    """
    GPTQ quantization algorithm.

    W: Weight matrix (out_features, in_features)
    X: Calibration data (batch, in_features)
    """
    n_out, n_in = W.shape

    # Compute Hessian (correlation of inputs)
    H = X.T @ X  # (in_features, in_features)
    H = H / X.shape[0]  # Normalize

    # Add damping for numerical stability
    damp = 0.01 * torch.diag(H).mean()
    H += damp * torch.eye(n_in, device=H.device)

    # Compute inverse Hessian (for updates)
    H_inv = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_inv)

    # Determine quantization order (by Hessian diagonal)
    perm = torch.argsort(torch.diag(H))  # Smallest first

    # Apply permutation
    W = W[:, perm]
    H_inv = H_inv[perm][:, perm]

    # Quantize column by column
    Q = torch.zeros_like(W)
    Losses = torch.zeros(n_out, device=W.device)

    for j in range(n_in):
        # Current column
        w_j = W[:, j]

        # Quantize with per-group scales
        group_idx = j // group_size
        q_j, scale = quantize_column(w_j, n_bits, group_idx)
        Q[:, j] = q_j

        # Quantization error
        err = (w_j - q_j) / H_inv[j, j]
        Losses += err ** 2 * H_inv[j, j]

        # Update remaining columns to compensate
        W[:, j+1:] -= err.unsqueeze(1) @ H_inv[j, j+1:].unsqueeze(0)

    # Undo permutation
    invperm = torch.argsort(perm)
    Q = Q[:, invperm]

    return Q, Losses.sum()
```

## Calibration Data

### Requirements

```python
def prepare_calibration_data(tokenizer, n_samples=128, seq_len=2048):
    """Prepare calibration dataset for GPTQ."""
    from datasets import load_dataset

    # Load diverse text data
    dataset = load_dataset("c4", "en", split="train", streaming=True)

    samples = []
    for sample in dataset:
        tokens = tokenizer(
            sample["text"],
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )
        if tokens.input_ids.shape[1] == seq_len:
            samples.append(tokens.input_ids)

        if len(samples) >= n_samples:
            break

    return torch.cat(samples, dim=0)
```

### Calibration Guidelines

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| n_samples | 128-512 | More samples = better but slower |
| seq_len | 2048-4096 | Match model's training length |
| Dataset | C4, RedPajama | Diverse text works best |
| Damping | 0.01-0.1 | Prevents numerical issues |

## Group Quantization in GPTQ

### Per-Group Scales

```python
def quantize_column(w, n_bits, group_size=128):
    """Quantize a column with group-wise scales."""
    n_groups = (len(w) + group_size - 1) // group_size
    q = torch.zeros_like(w)
    scales = []

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, len(w))
        w_group = w[start:end]

        # Symmetric quantization
        max_val = w_group.abs().max()
        scale = max_val / (2 ** (n_bits - 1) - 1)
        scales.append(scale)

        # Quantize and dequantize
        q_group = torch.round(w_group / scale)
        q_group = q_group.clamp(-2**(n_bits-1), 2**(n_bits-1) - 1)
        q[start:end] = q_group * scale

    return q, scales


# Group size trade-off:
# Smaller groups -> More scales -> Better quality -> More memory
# Larger groups -> Fewer scales -> Lower quality -> Less memory
```

## Practical Usage

### With AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # Activation order
    damp_percent=0.01
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

# Prepare calibration data
calibration_data = prepare_calibration_data(tokenizer)

# Quantize (takes ~30 min for 7B)
model.quantize(calibration_data)

# Save quantized model
model.save_quantized("llama-2-7b-gptq")
```

### Loading Quantized Models

```python
from auto_gptq import AutoGPTQForCausalLM

# Load pre-quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_safetensors=True
)

# Or with transformers
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)
```

### With Transformers Integration

```python
from transformers import AutoModelForCausalLM, GPTQConfig

# Quantize with transformers
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)
```

## Activation Order (desc_act)

### Why Order Matters

```
Column quantization order affects final quality:
- Early columns: Errors propagate to many later columns
- Later columns: Fewer remaining columns to compensate

desc_act=True (descending activation):
- Quantize columns with largest Hessian diagonal last
- Most important columns have more compensation available
- Higher quality but requires reordering during inference
```

### Implementation

```python
def activation_order(H):
    """Order columns by Hessian diagonal (importance)."""
    # Larger diagonal = more important = quantize later
    diag = torch.diag(H)
    order = torch.argsort(diag, descending=False)  # Ascending
    return order

# During inference, need to permute activations
# This adds overhead but improves quality
```

## Quality Comparison

### GPTQ vs Other Methods

| Method | 4-bit Quality | Speed | Calibration |
|--------|---------------|-------|-------------|
| Round-to-nearest | Poor | Fast | None |
| GPTQ | Excellent | Medium | Required |
| AWQ | Excellent | Medium | Required |
| bitsandbytes NF4 | Good | Fast | None |

### Perplexity Results (WikiText-2)

| Model | FP16 | GPTQ-4bit | GPTQ-3bit |
|-------|------|-----------|-----------|
| LLaMA-7B | 5.68 | 5.85 | 6.61 |
| LLaMA-13B | 5.09 | 5.20 | 5.69 |
| LLaMA-30B | 4.10 | 4.17 | 4.54 |
| LLaMA-65B | 3.53 | 3.60 | 4.17 |

## Optimizations

### Lazy Batch Updates

```python
def gptq_with_batching(W, H_inv, block_size=128):
    """
    Process columns in blocks for efficiency.
    Update remaining weights only after each block.
    """
    n_out, n_in = W.shape
    Q = torch.zeros_like(W)

    for block_start in range(0, n_in, block_size):
        block_end = min(block_start + block_size, n_in)

        # Quantize block
        for j in range(block_start, block_end):
            w_j = W[:, j]
            q_j = quantize(w_j)
            Q[:, j] = q_j

            err = (w_j - q_j) / H_inv[j, j]

            # Update only within block
            W[:, j+1:block_end] -= err.unsqueeze(1) @ H_inv[j, j+1:block_end].unsqueeze(0)

        # Block update to remaining columns
        err_block = W[:, block_start:block_end] - Q[:, block_start:block_end]
        W[:, block_end:] -= err_block @ H_inv[block_start:block_end, block_end:]

    return Q
```

### GPU Optimization

```python
# Key optimizations:
# 1. Process multiple layers in parallel
# 2. Keep calibration data on GPU
# 3. Use efficient matrix operations
# 4. Block-wise updates reduce memory

# Typical quantization time:
# 7B model: 20-40 minutes
# 13B model: 40-80 minutes
# 70B model: 3-6 hours
```

## Inference with GPTQ

### Using ExLlama

```python
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer

# ExLlama: Optimized inference for GPTQ models
config = ExLlamaConfig("llama-2-7b-gptq")
model = ExLlama(config)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer("llama-2-7b-gptq")

# Fast inference
input_ids = tokenizer.encode("Hello, world!")
output = model.generate(input_ids, cache, max_tokens=100)
```

### Throughput Considerations

| Backend | Tokens/sec (7B) | Notes |
|---------|-----------------|-------|
| Transformers | 30-50 | Basic integration |
| AutoGPTQ | 40-60 | Optimized kernels |
| ExLlama | 80-120 | Highly optimized |
| ExLlama2 | 100-150 | Latest optimizations |

## Key Takeaways

1. **Calibration-based**: Uses data to minimize output error, not just weight error.

2. **Column-by-column**: Compensates for errors by updating remaining weights.

3. **Hessian importance**: Orders columns by importance for better results.

4. **Near-lossless 4-bit**: Minimal quality loss on large models.

5. **Group size matters**: Smaller groups = better quality = more memory.

6. **One-shot process**: Fast quantization, no iterative training.

7. **Industry standard**: Widely used for model distribution and deployment.
