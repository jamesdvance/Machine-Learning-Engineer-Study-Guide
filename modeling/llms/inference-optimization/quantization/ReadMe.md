# Quantization for LLM Inference

## Summary

Quantization reduces model precision from 16/32-bit floating point to lower bit-widths (8-bit, 4-bit), dramatically decreasing memory usage and often improving inference speed. For LLMs, this enables running 70B+ parameter models on consumer GPUs and reduces serving costs at scale. The main approaches are post-training quantization (PTQ) methods like GPTQ and AWQ that quantize pre-trained models using calibration data, and quantization-aware training (QAT) that incorporates quantization during training. Modern techniques like mixed-precision decomposition and activation-aware scaling achieve near-lossless 4-bit quantization.

Key points to remember:

- Memory reduction: 4-bit is 4x smaller than FP16, enabling large model deployment
- PTQ dominates: GPTQ, AWQ, and bitsandbytes are the main practical methods
- Calibration matters: Quality depends on representative calibration data
- Trade-offs exist: Lower bits = more compression but potential quality loss
- Hardware acceleration: INT8/INT4 operations are faster on modern GPUs
- Group quantization: Per-group scales preserve accuracy with minimal overhead
- Production ready: vLLM, TensorRT-LLM, and others have native quantization support

## Quantization Landscape

### Method Comparison

| Method | Bits | Approach | Calibration | Quality | Speed |
|--------|------|----------|-------------|---------|-------|
| FP16 | 16 | Baseline | None | Best | 1x |
| LLM.int8() | 8 | Mixed-precision PTQ | None | Excellent | 1.5x |
| GPTQ | 4/3 | Hessian-based PTQ | Required | Excellent | Fast quant |
| AWQ | 4 | Activation-aware PTQ | Required | Excellent | Faster quant |
| NF4 | 4 | Normalized float PTQ | None | Very good | Fast |
| GGUF | 2-8 | CPU-optimized PTQ | Optional | Varies | CPU-focused |

### Memory Impact

```
Model Size by Precision:
                FP32      FP16      INT8      INT4
7B params:     28 GB     14 GB     7 GB      3.5 GB
13B params:    52 GB     26 GB     13 GB     6.5 GB
70B params:    280 GB    140 GB    70 GB     35 GB

Practical Impact:
- 7B INT4: Fits on RTX 3060 (12GB)
- 70B INT4: Fits on single A100 (40GB)
- 70B FP16: Requires 2x A100 (80GB each)
```

## Core Concepts

### The Quantization Formula

```python
# Symmetric quantization
def quantize_symmetric(tensor, n_bits):
    """Map float values to integer range."""
    qmax = 2 ** (n_bits - 1) - 1  # 127 for INT8, 7 for INT4
    scale = tensor.abs().max() / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax-1, qmax)
    return quantized.to(torch.int8), scale

def dequantize(quantized, scale):
    """Restore to floating point."""
    return quantized.float() * scale

# Asymmetric quantization (uses full range)
def quantize_asymmetric(tensor, n_bits):
    """Map to unsigned integer range with zero-point."""
    qmax = 2 ** n_bits - 1  # 255 for UINT8
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / qmax
    zero_point = round(-min_val / scale)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, qmax)
    return quantized.to(torch.uint8), scale, zero_point
```

### Group Quantization

```python
def group_quantize(weight, n_bits=4, group_size=128):
    """
    Quantize with separate scale per group of weights.
    Smaller groups = better quality, more scale storage.
    """
    original_shape = weight.shape
    weight = weight.reshape(-1, group_size)

    # Per-group scales
    scales = weight.abs().max(dim=1).values / (2 ** (n_bits - 1) - 1)

    # Quantize each group
    quantized = torch.round(weight / scales.unsqueeze(1))
    quantized = quantized.clamp(-2**(n_bits-1), 2**(n_bits-1) - 1)

    return quantized.to(torch.int8), scales, original_shape

# Group size trade-offs:
# 32:  Best quality, 0.5 bits/weight overhead for scales
# 64:  Good balance, 0.25 bits/weight overhead
# 128: Common default, 0.125 bits/weight overhead
# 256: Lower quality, minimal overhead
```

### Packing Low-Bit Values

```python
def pack_int4(tensor):
    """Pack two INT4 values into one INT8."""
    # Shift values to unsigned [0, 15]
    tensor = tensor + 8
    # Pack pairs
    even = tensor[:, 0::2]
    odd = tensor[:, 1::2]
    packed = (even << 4) | odd
    return packed.to(torch.uint8)

def unpack_int4(packed):
    """Unpack INT8 to two INT4 values."""
    high = (packed >> 4) - 8  # Upper 4 bits
    low = (packed & 0x0F) - 8  # Lower 4 bits
    return torch.stack([high, low], dim=-1).flatten(-2)
```

## Method Deep Dive

### GPTQ: Optimal Weight Quantization

GPTQ minimizes output error using Hessian information:

```python
# GPTQ core idea:
# 1. Quantize weights column by column
# 2. After quantizing column j, update remaining columns
# 3. Updates compensate for quantization error
# 4. Order columns by Hessian diagonal (importance)

def gptq_quantize_layer(weight, H, n_bits=4):
    """
    GPTQ algorithm for one layer.
    H: Hessian matrix (X @ X.T from calibration)
    """
    n_out, n_in = weight.shape
    Q = torch.zeros_like(weight)

    # Inverse Hessian for update formula
    H_inv = torch.linalg.inv(H + 0.01 * torch.eye(n_in))

    for j in range(n_in):
        # Quantize column j
        q_j = quantize_column(weight[:, j], n_bits)
        Q[:, j] = q_j

        # Compute error
        err = (weight[:, j] - q_j) / H_inv[j, j]

        # Update remaining columns to compensate
        weight[:, j+1:] -= err.unsqueeze(1) * H_inv[j, j+1:].unsqueeze(0)

    return Q
```

### AWQ: Activation-Aware Quantization

AWQ protects important weights by scaling:

```python
# AWQ core idea:
# 1. Find weights that process large activations
# 2. Scale these weights up before quantization
# 3. Scale activations down to compensate (fused at runtime)
# 4. Scaled weights have smaller relative quantization error

def awq_quantize_layer(weight, activations, n_bits=4):
    """AWQ algorithm for one layer."""
    # Compute importance from activation magnitudes
    importance = activations.abs().mean(dim=0)

    # Scale factor: protect important channels
    scales = (importance ** 0.5).clamp(min=0.1, max=10.0)
    scales = scales / scales.mean()  # Normalize

    # Scale weights
    scaled_weight = weight * scales.unsqueeze(0)

    # Quantize scaled weights
    quantized = group_quantize(scaled_weight, n_bits)

    # Store scales for runtime activation compensation
    return quantized, scales
```

### LLM.int8(): Mixed-Precision

Handles outliers by keeping them in FP16:

```python
def llm_int8_forward(x, weight_int8, scale, threshold=6.0):
    """Mixed-precision matmul for outlier handling."""
    # Find outlier features
    outlier_mask = x.abs().max(dim=0).values > threshold

    if outlier_mask.any():
        # Split computation
        x_normal = x[:, ~outlier_mask]
        x_outlier = x[:, outlier_mask]

        # INT8 for normal features
        w_normal = weight_int8[:, ~outlier_mask]
        out_normal = int8_matmul(x_normal, w_normal, scale)

        # FP16 for outliers
        w_outlier = dequantize(weight_int8[:, outlier_mask], scale)
        out_outlier = x_outlier.half() @ w_outlier.T

        return out_normal + out_outlier
    else:
        return int8_matmul(x, weight_int8, scale)
```

## Practical Usage

### With Transformers + bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8 loading
model_int8 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# NF4 (4-bit normalized float)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Quantize the scales too
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=nf4_config,
    device_map="auto"
)
```

### With AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize a model
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # Activation ordering
    damp_percent=0.01
)

model = AutoGPTQForCausalLM.from_pretrained("model-path", quantize_config)
model.quantize(calibration_data)
model.save_quantized("model-gptq")

# Load pre-quantized
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0"
)
```

### With AutoAWQ

```python
from awq import AutoAWQForCausalLM

# Quantize
model = AutoAWQForCausalLM.from_pretrained("model-path")
model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
model.save_quantized("model-awq")

# Load with fusion for faster inference
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,
    device_map="auto"
)
```

### With vLLM

```python
from vllm import LLM

# Native AWQ/GPTQ support
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",  # or "gptq"
    dtype="half"
)

outputs = llm.generate(prompts, sampling_params)
```

## Quality Benchmarks

### Perplexity Comparison (WikiText-2)

| Model | FP16 | GPTQ-4bit | AWQ-4bit | NF4 |
|-------|------|-----------|----------|-----|
| LLaMA-7B | 5.68 | 5.85 | 5.78 | 5.89 |
| LLaMA-13B | 5.09 | 5.20 | 5.14 | 5.25 |
| LLaMA-70B | 3.32 | 3.38 | 3.35 | 3.42 |

### Task Performance (MMLU Accuracy)

| Model | FP16 | INT8 | INT4 |
|-------|------|------|------|
| LLaMA-7B | 35.1% | 34.8% | 33.9% |
| LLaMA-13B | 46.9% | 46.5% | 45.8% |
| LLaMA-70B | 63.4% | 63.1% | 62.5% |

### When Quantization Hurts

```
Higher degradation in:
- Small models (<3B): Less redundancy to absorb error
- Math/reasoning: Precision-sensitive operations
- Code generation: Subtle bugs from quantization
- Extreme quantization (2-3 bit): Significant quality loss

Recommendations:
- Use INT8 for quality-critical applications
- Use INT4 when memory-constrained
- Avoid sub-4-bit for production
- Test on your specific use case
```

## Choosing a Method

### Decision Tree

```
Need to quantize?

  Memory constrained?
     Yes ’ Use 4-bit
        Fast quant needed? ’ AWQ
        Maximum quality? ’ GPTQ
        Simple setup? ’ NF4 (bitsandbytes)
   
     No ’ Use 8-bit
         LLM.int8() (bitsandbytes)

  Inference backend?
     vLLM/TensorRT ’ AWQ or GPTQ
     Transformers ’ bitsandbytes or GPTQ
     llama.cpp ’ GGUF
     ExLlama ’ GPTQ

  Fine-tuning needed?
      Yes ’ QLoRA with NF4
```

### Method Strengths

| Use Case | Best Method |
|----------|-------------|
| Simple deployment | bitsandbytes NF4 |
| Production inference | AWQ + vLLM |
| Maximum quality | GPTQ with desc_act |
| Fine-tuning | QLoRA (NF4 base + LoRA) |
| CPU inference | GGUF |
| Edge devices | GPTQ/AWQ + TensorRT |

## Key Takeaways

1. **4-bit is viable**: GPTQ and AWQ achieve near-lossless 4-bit quantization.

2. **Calibration improves quality**: GPTQ and AWQ use data to minimize error.

3. **Group quantization is key**: Per-group scales preserve accuracy.

4. **Mixed-precision handles outliers**: LLM.int8() keeps outliers in FP16.

5. **Choose based on deployment**: Different methods suit different backends.

6. **Test your use case**: Quality impact varies by task and model.

7. **Infrastructure support**: vLLM, TensorRT-LLM have native quantization.

## Further Reading

For detailed coverage of specific quantization methods, see:

- [INT8 Quantization](int8/ReadMe.md) - LLM.int8() mixed-precision approach
- [INT4 Quantization](int4/ReadMe.md) - Group quantization, NF4, double quantization
- [GPTQ](gptq/ReadMe.md) - Hessian-based optimal quantization
- [AWQ](awq/ReadMe.md) - Activation-aware weight quantization
