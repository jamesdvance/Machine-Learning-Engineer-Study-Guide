# AWQ (Activation-Aware Weight Quantization)

## Summary

AWQ (Activation-Aware Weight Quantization) achieves high-quality 4-bit quantization by identifying and protecting "salient" weights - those that process large activation values and thus have outsized impact on model output. Rather than modifying weights after quantization like GPTQ, AWQ scales salient weight channels before quantization to reduce their relative quantization error. This simpler approach achieves comparable quality to GPTQ while being faster to apply and better suited for efficient inference kernels.

Key points to remember:

- Activation-aware: Identifies important weights via activation magnitudes
- Scaling-based: Multiplies salient channels to reduce relative error
- No weight modification: Simpler than GPTQ's column-by-column updates
- Hardware-friendly: Better suited for optimized inference kernels
- Comparable quality: Matches or beats GPTQ on benchmarks
- Faster quantization: Less compute than GPTQ
- Efficient inference: Enables fast 4-bit kernels

## Core Insight

### Why Activation Matters

```
Not all weights are equal:
- Some weights process large activations -> Large output impact
- Some weights process small activations -> Small output impact

Quantization error impact:
Error in output H Weight error × Activation magnitude

High activation channels:
- Small weight error -> Large output error
- Need more quantization precision

Low activation channels:
- Large weight error -> Small output error
- Can tolerate more quantization error
```

### The Scaling Trick

```python
# Original: W @ X = Y
# With scaling: (W * s) @ (X / s) = Y  (mathematically equivalent)

# Key insight:
# - Scale up salient weight channels by s > 1
# - After quantization, relative error is smaller for scaled weights
# - Fuse the 1/s scaling into activations (no runtime cost)
```

## Algorithm

### Step 1: Identify Salient Channels

```python
def compute_saliency(model, calibration_data):
    """Identify salient weight channels via activation magnitudes."""
    activation_stats = {}

    def hook_fn(name):
        def fn(module, input, output):
            # Record activation magnitudes per channel
            x = input[0]
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(x.abs().mean(dim=(0, 1)))  # Per channel
        return fn

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate statistics
    saliency = {}
    for name, stats in activation_stats.items():
        saliency[name] = torch.stack(stats).mean(dim=0)

    return saliency
```

### Step 2: Compute Scaling Factors

```python
def compute_scales(weight, activations, alpha=0.5):
    """
    Compute per-channel scaling factors.

    alpha: Balance between protecting salient weights and not over-scaling
    """
    # Saliency: how much each input channel affects output
    saliency = activations.abs()

    # Weight magnitude per input channel
    weight_magnitude = weight.abs().mean(dim=0)

    # Scaling: protect channels with high activation × weight impact
    # But don't scale too much (causes other issues)
    scales = (saliency ** alpha) / (weight_magnitude ** (1 - alpha))

    # Normalize to avoid changing overall scale
    scales = scales / scales.mean()

    # Clamp to reasonable range
    scales = scales.clamp(min=0.1, max=10.0)

    return scales
```

### Step 3: Apply Scaling and Quantize

```python
def awq_quantize(weight, activations, n_bits=4, group_size=128):
    """Apply AWQ quantization."""
    # Compute optimal scales
    scales = compute_scales(weight, activations)

    # Scale weights (multiply columns by scales)
    scaled_weight = weight * scales.unsqueeze(0)

    # Quantize the scaled weights
    quantized = group_quantize(scaled_weight, n_bits, group_size)

    # Store scales for activation compensation
    return quantized, scales


def apply_awq_linear(x, quantized_weight, scales, quant_scales):
    """AWQ linear forward pass."""
    # Compensate activations (divide by scales)
    x_compensated = x / scales

    # Dequantize and compute
    weight = dequantize(quantized_weight, quant_scales)
    return x_compensated @ weight.T
```

## Optimal Scale Search

### Grid Search

```python
def search_optimal_scales(weight, activations, n_bits=4, n_grid=20):
    """Search for optimal scaling factors."""
    best_scales = torch.ones(weight.shape[1])
    best_error = float('inf')

    # For each channel, search for best scale
    for ch in range(weight.shape[1]):
        original_error = compute_quantization_error(
            weight, activations, n_bits
        )

        for scale in torch.linspace(0.1, 2.0, n_grid):
            # Try this scale
            test_weight = weight.clone()
            test_weight[:, ch] *= scale

            error = compute_quantization_error(
                test_weight, activations / scale, n_bits
            )

            if error < best_error:
                best_error = error
                best_scales[ch] = scale

    return best_scales


def compute_quantization_error(weight, activations, n_bits):
    """Compute output error from quantization."""
    # Quantize
    q_weight = quantize(weight, n_bits)

    # Compute output difference
    original_output = activations @ weight.T
    quantized_output = activations @ q_weight.T

    return (original_output - quantized_output).pow(2).mean()
```

### Efficient Group-wise Search

```python
def awq_group_search(weight, activations, group_size=128, n_grid=20):
    """
    Search scales for groups of channels together.
    More efficient than per-channel search.
    """
    n_in = weight.shape[1]
    n_groups = (n_in + group_size - 1) // group_size
    scales = torch.ones(n_in)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_in)

        # Search for best scale for this group
        best_scale = 1.0
        best_error = float('inf')

        for scale in torch.linspace(0.5, 1.5, n_grid):
            # Apply scale to group
            test_weight = weight.clone()
            test_weight[:, start:end] *= scale

            test_acts = activations.clone()
            test_acts[:, start:end] /= scale

            error = compute_quantization_error(test_weight, test_acts, 4)

            if error < best_error:
                best_error = error
                best_scale = scale

        scales[start:end] = best_scale

    return scales
```

## Practical Usage

### With AutoAWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # or "GEMV" for smaller batch
}

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### Loading AWQ Models

```python
from awq import AutoAWQForCausalLM

# Load quantized model
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,  # Fuse for faster inference
    device_map="auto"
)

# Or with transformers
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    device_map="auto"
)
```

### With vLLM

```python
from vllm import LLM

# vLLM has native AWQ support
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half"
)

# Efficient batch inference
outputs = llm.generate(prompts, sampling_params)
```

## AWQ vs GPTQ

### Algorithm Comparison

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| Approach | Scale before quantize | Update after quantize |
| Complexity | Simpler | More complex |
| Quant time | Faster | Slower |
| Weight modification | Scaling only | Error compensation |
| Hardware friendliness | Better | Good |

### Quality Comparison

| Model | Method | WikiText-2 PPL |
|-------|--------|----------------|
| LLaMA-7B | FP16 | 5.68 |
| LLaMA-7B | GPTQ-4bit | 5.85 |
| LLaMA-7B | AWQ-4bit | 5.78 |
| LLaMA-13B | FP16 | 5.09 |
| LLaMA-13B | GPTQ-4bit | 5.20 |
| LLaMA-13B | AWQ-4bit | 5.14 |

### When to Choose Each

```
Choose AWQ when:
- Need fast quantization
- Using optimized inference (vLLM, TensorRT)
- Want simpler deployment
- Batch inference is primary use case

Choose GPTQ when:
- Need 3-bit quantization
- Using ExLlama for inference
- Have time for longer quantization
- Maximum quality is critical
```

## Inference Optimization

### Kernel Fusion

```python
# AWQ enables efficient kernel fusion:
# 1. Dequantization
# 2. Scale compensation
# 3. Matrix multiplication
# All in one kernel, no intermediate tensors

# Standard (slow):
x_scaled = x / scales           # Memory write
w_dequant = dequantize(w)       # Memory write
output = x_scaled @ w_dequant   # Memory read + compute

# Fused (fast):
output = awq_gemm(x, w, scales, quant_params)  # Single kernel
```

### GEMM vs GEMV

```python
# GEMM: General Matrix-Matrix Multiplication
# - Batch size > 1
# - Better GPU utilization
quant_config = {"version": "GEMM"}

# GEMV: General Matrix-Vector Multiplication
# - Batch size = 1
# - Lower latency for single requests
quant_config = {"version": "GEMV"}
```

## Memory and Performance

### Memory Comparison

| Model | FP16 | AWQ-4bit | Reduction |
|-------|------|----------|-----------|
| 7B | 14 GB | 4 GB | 3.5x |
| 13B | 26 GB | 7.5 GB | 3.5x |
| 70B | 140 GB | 40 GB | 3.5x |

### Throughput (vLLM, A100)

| Model | FP16 tok/s | AWQ tok/s | Speedup |
|-------|------------|-----------|---------|
| 7B | 2000 | 3500 | 1.75x |
| 13B | 1200 | 2100 | 1.75x |
| 70B | 300 | 600 | 2x |

## Advanced Topics

### Activation-Aware Group Size

```python
def adaptive_group_size(weight, activations, min_group=32, max_group=256):
    """Use smaller groups for salient channels."""
    saliency = activations.abs().mean(dim=0)
    threshold = saliency.quantile(0.9)

    group_sizes = []
    for ch in range(weight.shape[1]):
        if saliency[ch] > threshold:
            group_sizes.append(min_group)  # Small group for important
        else:
            group_sizes.append(max_group)  # Large group for less important

    return group_sizes
```

### Combining with Other Techniques

```python
# AWQ + Activation Quantization
def awq_plus_activation_quant(model):
    # AWQ for weights
    weight_quantized = awq_quantize(model)

    # INT8 for activations
    activation_quantized = quantize_activations_int8(model)

    return weight_quantized, activation_quantized

# Enables even faster inference on INT8 hardware
```

## Key Takeaways

1. **Activation-aware**: Protects weights that process large activations.

2. **Scaling trick**: Mathematically equivalent transformation reduces error.

3. **Simpler than GPTQ**: No iterative weight updates.

4. **Hardware-friendly**: Enables efficient fused kernels.

5. **Fast quantization**: Minutes instead of hours.

6. **Great for inference**: Works excellently with vLLM, TensorRT.

7. **Comparable quality**: Matches GPTQ on most benchmarks.
