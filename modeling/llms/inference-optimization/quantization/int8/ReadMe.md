# INT8 Quantization

## Summary

INT8 quantization reduces model weights from 16-bit (FP16) or 32-bit (FP32) to 8-bit integers, halving memory usage while maintaining most model quality. This technique enables running larger models on limited hardware and improves inference throughput. The key innovation in LLM.int8() is handling outlier features separately to preserve quality. INT8 serves as a practical middle ground between full precision and more aggressive 4-bit quantization.

Key points to remember:

- 2x memory reduction: FP16 to INT8 halves model size
- Mixed-precision: Outlier features stay in FP16 for quality
- Minimal quality loss: <1% degradation on most benchmarks
- Hardware support: INT8 operations accelerated on modern GPUs
- Dynamic quantization: Compute scales at runtime
- LLM.int8(): Handles emergent outliers in large models
- Good default: Lower risk than 4-bit, significant savings

## Quantization Fundamentals

### The Quantization Formula

```python
def quantize_symmetric(tensor, n_bits=8):
    """Symmetric quantization to n-bit integers."""
    # Find scale factor
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (n_bits - 1) - 1)  # For INT8: / 127

    # Quantize
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)

    return quantized, scale

def dequantize(quantized, scale):
    """Convert back to floating point."""
    return quantized.float() * scale
```

### Symmetric vs Asymmetric

```python
# Symmetric: zero maps to zero
# Range: [-127, 127] for INT8
# Formula: q = round(x / scale)

def quantize_asymmetric(tensor, n_bits=8):
    """Asymmetric quantization with zero-point."""
    min_val = tensor.min()
    max_val = tensor.max()

    # Compute scale and zero-point
    scale = (max_val - min_val) / (2 ** n_bits - 1)  # / 255 for INT8
    zero_point = round(-min_val / scale)

    # Quantize
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)

    return quantized, scale, zero_point
```

## LLM.int8() Mixed-Precision

### The Outlier Problem

Large language models develop "outlier features" - activation values that are orders of magnitude larger than others:

```
Normal activations: [-0.5, 0.3, -0.1, 0.2, ...]
Outlier activations: [-0.5, 0.3, 47.2, 0.2, ...]  # 47.2 is an outlier

Problem: Quantizing with outliers wastes precision
- Scale factor dominated by outlier
- Normal values get very few quantization levels
```

### Mixed-Precision Decomposition

LLM.int8() separates outliers for full-precision computation:

```python
class LLMInt8Linear(nn.Module):
    def __init__(self, weight, threshold=6.0):
        super().__init__()
        self.threshold = threshold

        # Quantize weight to INT8
        self.weight_int8, self.weight_scale = quantize_symmetric(weight)

    def forward(self, x):
        # Find outlier features (magnitude > threshold)
        outlier_mask = x.abs().max(dim=0).values > self.threshold

        if outlier_mask.any():
            # Split input
            x_outlier = x[:, outlier_mask]
            x_normal = x[:, ~outlier_mask]

            # INT8 matmul for normal features
            w_normal = self.weight_int8[:, ~outlier_mask]
            out_normal = int8_matmul(x_normal, w_normal, self.weight_scale)

            # FP16 matmul for outliers
            w_outlier = dequantize(
                self.weight_int8[:, outlier_mask],
                self.weight_scale
            )
            out_outlier = torch.matmul(x_outlier.half(), w_outlier.t())

            # Combine
            return out_normal + out_outlier
        else:
            # All INT8
            return int8_matmul(x, self.weight_int8, self.weight_scale)
```

### Implementation with bitsandbytes

```python
import torch
from transformers import AutoModelForCausalLM

# Load model in INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# Check memory usage
print(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
# ~7GB for 7B model (vs ~14GB in FP16)
```

## Quantization Approaches

### Post-Training Quantization (PTQ)

```python
def post_training_quantization(model, calibration_data=None):
    """Quantize pre-trained model without retraining."""
    quantized_state_dict = {}

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Quantize weight matrices
            q_weight, scale = quantize_symmetric(param.data)
            quantized_state_dict[name] = (q_weight, scale)
        else:
            # Keep biases and other params in FP16
            quantized_state_dict[name] = param.data.half()

    return quantized_state_dict
```

### Dynamic Quantization

```python
# Weights quantized statically, activations quantized dynamically
import torch.quantization as quant

# Apply dynamic quantization
model_int8 = quant.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)
```

### Static Quantization (with calibration)

```python
def calibrate_and_quantize(model, calibration_loader):
    """Static quantization with calibration data."""
    # Collect activation statistics
    activation_stats = {}

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {'min': [], 'max': []}
            activation_stats[name]['min'].append(output.min().item())
            activation_stats[name]['max'].append(output.max().item())
        return fn

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute scales from statistics
    scales = {}
    for name, stats in activation_stats.items():
        max_val = max(abs(min(stats['min'])), max(stats['max']))
        scales[name] = max_val / 127

    return scales
```

## Per-Channel vs Per-Tensor

### Per-Tensor Quantization

```python
def per_tensor_quantize(weight):
    """Single scale for entire tensor."""
    scale = weight.abs().max() / 127
    quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale

# Simple but suboptimal for varying weight distributions
```

### Per-Channel Quantization

```python
def per_channel_quantize(weight):
    """Separate scale per output channel."""
    # weight: (out_features, in_features)
    scales = weight.abs().max(dim=1).values / 127

    # Quantize with per-channel scales
    quantized = torch.round(weight / scales.unsqueeze(1))
    quantized = quantized.clamp(-128, 127).to(torch.int8)

    return quantized, scales

# Better quality: each channel uses full INT8 range
```

## Memory and Performance

### Memory Calculation

```python
def calculate_memory(model, dtype):
    """Estimate model memory in bytes."""
    total_params = sum(p.numel() for p in model.parameters())

    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5
    }

    return total_params * bytes_per_param[dtype]

# Example: 7B model
# FP32: 28 GB
# FP16: 14 GB
# INT8: 7 GB
# INT4: 3.5 GB
```

### Throughput Considerations

| Operation | FP16 | INT8 | Notes |
|-----------|------|------|-------|
| Memory bandwidth | 1x | 2x | Move half the data |
| Compute | 1x | 1-2x | Hardware dependent |
| Overall speedup | 1x | 1.5-2x | Memory-bound benefits most |

## Quality Impact

### Benchmark Results (Typical)

| Model | FP16 | INT8 | Degradation |
|-------|------|------|-------------|
| LLaMA-7B (MMLU) | 35.1 | 34.8 | -0.3% |
| LLaMA-13B (MMLU) | 46.9 | 46.5 | -0.4% |
| LLaMA-7B (Perplexity) | 5.68 | 5.72 | +0.7% |

### When INT8 Struggles

```
Problematic cases:
- Very small models (<1B): Less redundancy
- Extreme precision tasks: Math, code
- Models with unusual distributions

Solutions:
- Use mixed-precision (LLM.int8())
- Keep sensitive layers in FP16
- Consider 4-bit only for memory-critical cases
```

## Practical Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8 configuration
int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outlier threshold
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=int8_config,
    device_map="auto"
)
```

### Fine-tuning Quantized Models

```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Prepare INT8 model for training
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Train with adapters in FP16, base model stays INT8
```

## Comparison with Other Bit Widths

| Precision | Memory | Quality | Use Case |
|-----------|--------|---------|----------|
| FP16 | Baseline | Best | Training, quality-critical |
| INT8 | 2x smaller | Very good | Good default for inference |
| INT4 | 4x smaller | Good | Memory-constrained |
| NF4 | 4x smaller | Better than INT4 | QLoRA training |

## Key Takeaways

1. **2x memory reduction**: INT8 halves memory with minimal quality loss.

2. **LLM.int8() handles outliers**: Mixed-precision preserves quality.

3. **Per-channel is better**: Individual scales per output channel.

4. **Good default choice**: Lower risk than 4-bit, meaningful savings.

5. **Hardware accelerated**: Modern GPUs optimize INT8 operations.

6. **Enables fine-tuning**: QLoRA with INT8 base model works well.

7. **Dynamic works well**: No calibration needed for basic usage.
