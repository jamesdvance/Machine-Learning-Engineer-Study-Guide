# INT4 Quantization

## Summary

INT4 quantization reduces model weights to 4 bits, achieving 4x memory reduction compared to FP16. This aggressive compression enables running 70B+ models on consumer GPUs. The main challenge is maintaining quality with only 16 quantization levels. Solutions include group quantization (separate scales for weight groups), NF4 (normalized float format optimized for neural network weight distributions), and double quantization (quantizing the scales themselves). INT4 is essential for deploying large models on limited hardware.

Key points to remember:

- 4x memory reduction: Critical for large model deployment
- Only 16 levels: Requires careful quantization design
- Group quantization: Separate scales per 32-128 weights
- NF4: Normalized 4-bit format optimized for normal distributions
- Double quantization: Further compress by quantizing scales
- Quality trade-off: More degradation than INT8, but usable
- Enables large models: 70B on single GPU, 7B on consumer hardware

## The 4-Bit Challenge

### Limited Precision

```
INT8: 256 levels (-128 to 127)
INT4: 16 levels (-8 to 7 or 0 to 15)

Example weight distribution:
Original: [0.12, -0.45, 0.03, 0.78, -0.22, ...]

INT8 quantized (scale=0.01):
[12, -45, 3, 78, -22, ...] -> Good precision

INT4 quantized (scale=0.1):
[1, -5, 0, 7, -2, ...] -> Significant precision loss
```

### Why INT4 Still Works

```
Neural networks are:
1. Highly redundant: Many near-zero weights
2. Noise tolerant: Small errors average out
3. Non-uniform: Most weights cluster around zero

Weight distribution (typical):
     |
  *  |  *
 *** | ***
*****|*****
-----+-----
     0

Most weights near zero -> Most quantization levels used for small values
```

## Group Quantization

### Per-Group Scales

```python
def group_quantize_int4(weight, group_size=128):
    """Quantize with separate scale per group of weights."""
    # Reshape into groups
    original_shape = weight.shape
    weight = weight.reshape(-1, group_size)

    # Compute per-group scales
    scales = weight.abs().max(dim=1).values / 7  # INT4 symmetric: -7 to 7

    # Quantize
    quantized = torch.round(weight / scales.unsqueeze(1))
    quantized = quantized.clamp(-8, 7).to(torch.int8)  # Store as int8

    # Pack two INT4 values into one INT8
    packed = pack_int4(quantized)

    return packed, scales, original_shape


def pack_int4(tensor):
    """Pack two INT4 values into one INT8."""
    # tensor shape: (n, group_size) with values in [-8, 7]
    tensor = tensor + 8  # Shift to [0, 15]

    # Pack pairs
    even = tensor[:, 0::2]
    odd = tensor[:, 1::2]
    packed = (even << 4) | odd

    return packed.to(torch.uint8)


def unpack_int4(packed):
    """Unpack INT8 to two INT4 values."""
    high = (packed >> 4) - 8
    low = (packed & 0x0F) - 8
    return torch.stack([high, low], dim=-1).flatten(-2)
```

### Optimal Group Size

| Group Size | Memory Overhead | Quality | Notes |
|------------|-----------------|---------|-------|
| 32 | Higher | Best | More scales |
| 64 | Medium | Good | Common choice |
| 128 | Lower | Good | QLoRA default |
| 256 | Lowest | Lower | More error |

```python
# Memory calculation
# For group_size=128, we store:
# - 4 bits per weight
# - 16 bits (FP16) scale per 128 weights
# Overhead: 16 / 128 / 4 = 3.1% extra for scales
```

## NF4 (Normalized Float 4-bit)

### Concept

NF4 uses non-uniform quantization levels optimized for normally distributed weights:

```python
# Standard INT4 levels (uniform):
# [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
# Equal spacing doesn't match weight distribution

# NF4 levels (non-uniform):
# Computed from quantiles of normal distribution
NF4_LEVELS = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
]
# More levels near zero where weights concentrate
```

### NF4 Implementation

```python
def compute_nf4_levels():
    """Compute NF4 quantization levels from normal distribution."""
    from scipy.stats import norm

    # 16 quantiles of N(0,1) with symmetric split
    quantiles = []
    for i in range(8):
        # Negative side
        q = norm.ppf((i + 0.5) / 16)
        quantiles.append(q)

    # Add positive mirror
    quantiles = quantiles + [-q for q in reversed(quantiles[:-1])] + [0]
    quantiles = sorted(set(quantiles))

    # Normalize to [-1, 1]
    max_val = max(abs(min(quantiles)), max(quantiles))
    return [q / max_val for q in quantiles]


def quantize_nf4(tensor, group_size=64):
    """Quantize using NF4 format."""
    # Reshape to groups
    original_shape = tensor.shape
    tensor = tensor.reshape(-1, group_size)

    # Compute absmax scale per group
    scales = tensor.abs().max(dim=1).values

    # Normalize to [-1, 1]
    normalized = tensor / scales.unsqueeze(1)

    # Find nearest NF4 level for each value
    nf4_levels = torch.tensor(NF4_LEVELS, device=tensor.device)
    distances = (normalized.unsqueeze(-1) - nf4_levels).abs()
    indices = distances.argmin(dim=-1)

    return indices.to(torch.uint8), scales, original_shape
```

## Double Quantization

### Quantizing the Scales

```python
def double_quantize(weight, group_size=64, block_size=256):
    """
    Double quantization: quantize weights, then quantize scales.

    Memory savings:
    - Without: 4 bits/weight + 16 bits/scale per group
    - With: 4 bits/weight + 8 bits/scale + 32 bits/block
    """
    # First quantization: weights to NF4 with FP16 scales
    indices, scales, shape = quantize_nf4(weight, group_size)

    # Second quantization: FP16 scales to INT8
    # Group scales into blocks
    scales = scales.reshape(-1, block_size // group_size)
    scale_scales = scales.abs().max(dim=1).values
    scales_int8 = torch.round(scales / scale_scales.unsqueeze(1) * 127)
    scales_int8 = scales_int8.clamp(-128, 127).to(torch.int8)

    return {
        'indices': indices,        # 4-bit weight indices
        'scales_int8': scales_int8,  # 8-bit quantized scales
        'scale_scales': scale_scales,  # FP32 scales for scales
        'shape': shape
    }

# Memory comparison for 1M weights, group_size=64:
# Without double quant: 4M bits + 16k * 16 bits = 4.26M bits
# With double quant: 4M bits + 16k * 8 bits + 1k * 32 bits = 4.16M bits
# ~2.4% additional savings
```

## Practical Usage

### With bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit NF4 configuration
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=nf4_config,
    device_map="auto"
)

# 70B model now fits on ~40GB GPU memory
```

### QLoRA Training

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=nf4_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Base model in NF4, LoRA adapters in FP16
```

## Memory Comparison

### Model Size by Precision

| Model | FP16 | INT8 | INT4 |
|-------|------|------|------|
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |

### GPU Requirements

| Model | FP16 | INT4 | Enables |
|-------|------|------|---------|
| 7B | A100 40GB | RTX 3090 24GB | Consumer deployment |
| 13B | A100 80GB | RTX 4090 24GB | Consumer deployment |
| 70B | 2x A100 80GB | A100 40GB | Single GPU |

## Quality Impact

### Benchmark Comparison

| Model | FP16 | INT8 | INT4-NF4 | Degradation |
|-------|------|------|----------|-------------|
| LLaMA-7B (MMLU) | 35.1 | 34.8 | 33.9 | -3.4% |
| LLaMA-13B (MMLU) | 46.9 | 46.5 | 45.8 | -2.3% |
| LLaMA-7B (PPL) | 5.68 | 5.72 | 5.89 | +3.7% |

### Quality Considerations

```
Where INT4 works well:
- General chat/instruction following
- Summarization
- Translation
- Creative writing

Where to be careful:
- Precise calculations
- Code generation (subtle bugs)
- Factual retrieval
- Small models (<3B)
```

## Advanced Techniques

### Activation-Aware Quantization

```python
def activation_aware_grouping(weight, activations):
    """Group weights based on activation patterns."""
    # Weights with larger activation impact get smaller groups
    activation_importance = activations.abs().mean(dim=0)

    # Assign group sizes inversely proportional to importance
    group_sizes = []
    for i in range(weight.shape[1]):
        if activation_importance[i] > threshold_high:
            group_sizes.append(32)  # Small groups for important
        elif activation_importance[i] > threshold_low:
            group_sizes.append(64)
        else:
            group_sizes.append(128)  # Large groups for less important

    return variable_group_quantize(weight, group_sizes)
```

### Mixed Precision by Layer

```python
def mixed_precision_quantize(model):
    """Different precision for different layers."""
    for name, module in model.named_modules():
        if 'embed' in name or 'lm_head' in name:
            # Keep embeddings in FP16
            pass
        elif 'attention' in name:
            # 8-bit for attention (more sensitive)
            quantize_int8(module)
        else:
            # 4-bit for FFN (more robust)
            quantize_int4(module)
```

## Key Takeaways

1. **4x memory reduction**: Essential for large model deployment.

2. **Group quantization is key**: Per-group scales preserve accuracy.

3. **NF4 > INT4**: Non-uniform levels match weight distributions.

4. **Double quantization**: Additional 2-3% savings, worth it.

5. **QLoRA enables training**: Fine-tune 70B on consumer hardware.

6. **Quality trade-off exists**: ~2-5% degradation, usually acceptable.

7. **Not for all tasks**: Be cautious with precision-sensitive applications.
