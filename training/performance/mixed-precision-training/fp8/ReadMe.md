# FP8 (8-bit Floating Point)

## Summary

FP8 is an emerging 8-bit floating point format designed for deep learning training and inference. By further reducing precision from 16 bits to 8 bits, FP8 promises additional memory savings and throughput improvements on specialized hardware. FP8 support is available on NVIDIA H100 GPUs and is actively being integrated into training frameworks.

Key points to remember:

- 8-bit format with two variants: E4M3 and E5M2
- E4M3 (4 exponent, 3 mantissa): Better precision, narrower range
- E5M2 (5 exponent, 2 mantissa): Wider range, lower precision
- Native hardware support on H100 GPUs
- 2x memory savings compared to FP16/BF16
- Requires per-tensor or per-channel scaling
- Still maturing, less stable than FP16/BF16
- Best suited for inference currently

## FP8 Formats

### E4M3 Format

```
E4M3: [S][EEEE][MMM]
       1   4     3

Range: ~1.5e-9 to 448
Precision: ~2 mantissa bits effective
```

Use case: Weights and activations (need precision)

### E5M2 Format

```
E5M2: [S][EEEEE][MM]
       1   5      2

Range: ~6e-8 to 57344
Precision: ~1 mantissa bit effective
```

Use case: Gradients (need range)

### Comparison

| Format | Exponent | Mantissa | Range | Use Case |
|--------|----------|----------|-------|----------|
| E4M3 | 4 | 3 | 448 | Forward pass |
| E5M2 | 5 | 2 | 57344 | Backward pass |
| FP16 | 5 | 10 | 65504 | Reference |
| BF16 | 8 | 7 | 3e38 | Reference |

## Hardware Support

### NVIDIA H100

H100 provides native FP8 tensor cores:

| Precision | H100 Performance |
|-----------|-----------------|
| FP8 | 2000 TFLOPS |
| FP16/BF16 | 1000 TFLOPS |
| FP32 | 67 TFLOPS |

FP8 provides 2x throughput over FP16/BF16.

### Other Hardware

- A100: No FP8 support
- AMD MI300: FP8 support planned
- TPUs: FP8 variants being added

## Implementation

### Transformer Engine

NVIDIA's Transformer Engine library provides FP8 support:

```python
import transformer_engine.pytorch as te

# FP8 linear layer
layer = te.Linear(1024, 1024)

# Training with FP8
with te.fp8_autocast(enabled=True):
    output = layer(input)
```

### PyTorch (Experimental)

```python
import torch

# FP8 tensors (experimental)
x_fp8 = x.to(torch.float8_e4m3fn)

# Or using scaled FP8
from torch._scaled_float8 import Float8Tensor
x_fp8 = Float8Tensor.to_fp8(x, scale=1.0, fp8_dtype=torch.float8_e4m3fn)
```

### DeepSpeed (Experimental)

```json
{
    "fp8": {
        "enabled": true,
        "fp8_format": "hybrid"
    }
}
```

## Per-Tensor Scaling

FP8's limited range requires scaling:

```python
def quantize_to_fp8(tensor, fp8_dtype):
    # Calculate scale to fit tensor in FP8 range
    abs_max = tensor.abs().max()

    if fp8_dtype == 'e4m3':
        fp8_max = 448.0
    else:  # e5m2
        fp8_max = 57344.0

    scale = fp8_max / abs_max

    # Scale and quantize
    scaled_tensor = tensor * scale
    fp8_tensor = scaled_tensor.to(torch.float8_e4m3fn)

    return fp8_tensor, scale

def dequantize_from_fp8(fp8_tensor, scale):
    return fp8_tensor.float() / scale
```

### Delayed Scaling

Compute scale from previous iteration's statistics:

```python
class DelayedScaling:
    def __init__(self):
        self.amax_history = []
        self.scale = 1.0

    def update(self, tensor):
        amax = tensor.abs().max().item()
        self.amax_history.append(amax)

        if len(self.amax_history) > 1:
            # Use previous max for current scale
            prev_amax = self.amax_history[-2]
            self.scale = 448.0 / prev_amax  # E4M3 max

    def get_scale(self):
        return self.scale
```

## Training with FP8

### Hybrid Precision Strategy

```
Forward:  E4M3 weights, E4M3 activations
Backward: E5M2 gradients
Master:   FP32 weights (for optimizer)
```

### Training Loop

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# FP8 recipe
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
    amax_history_len=16,
    amax_compute_algo="max",
)

# Training loop
for batch in dataloader:
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = model(batch)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
```

## Memory Savings

### Comparison

| Format | Memory per Parameter | Reduction from FP32 |
|--------|---------------------|---------------------|
| FP32 | 4 bytes | 1x |
| FP16/BF16 | 2 bytes | 2x |
| FP8 | 1 byte | 4x |

### Practical Impact

For a 70B parameter model:
- FP32: 280 GB
- BF16: 140 GB
- FP8: 70 GB

## Current Limitations

### Accuracy Sensitivity

FP8's low precision can affect accuracy:

```python
# Some models may need careful tuning
# Monitor validation loss closely when using FP8
```

### Software Maturity

- Still experimental in most frameworks
- Limited model support
- Scaling recipes need tuning

### Operations Not Supporting FP8

Some operations require higher precision:
- Normalization layers
- Softmax
- Loss computation
- Optimizer states (always FP32)

## Best Practices

### Start Conservative

```python
# Start with FP16/BF16, validate accuracy
# Then try FP8 on specific layers

# Good candidates for FP8:
# - Linear layers
# - Convolutions
# - Attention QKV projections

# Keep in higher precision:
# - Normalization
# - Loss computation
# - First and last layers
```

### Monitor Carefully

```python
# Track metrics during FP8 training
# - Training loss curve
# - Validation accuracy
# - Gradient statistics
# - Scale factors over time
```

### Fallback Strategy

```python
# If FP8 causes issues
if validation_loss > threshold:
    # Fall back to BF16/FP16
    fp8_enabled = False
```

## Future Outlook

### Framework Integration

- PyTorch: Active development of FP8 support
- JAX: FP8 support in progress
- TensorFlow: Planned support

### Hardware Roadmap

- Next-gen NVIDIA GPUs: Enhanced FP8
- AMD MI400: FP8 support expected
- Intel GPUs: FP8 consideration

### Standardization

- IEEE working on FP8 standardization
- Multiple formats being evaluated
- Industry convergence expected

## When to Use FP8

### Good Candidates

1. H100 GPU deployment
2. Inference optimization
3. Very large models where memory is critical
4. Well-established architectures (transformers)

### Not Recommended Yet

1. Research/novel architectures
2. Small models (benefits minimal)
3. Accuracy-critical applications
4. Non-H100 hardware

## Comparison Summary

| Aspect | FP32 | BF16 | FP16 | FP8 |
|--------|------|------|------|-----|
| Memory | 4B | 2B | 2B | 1B |
| H100 TFLOPS | 67 | 1000 | 1000 | 2000 |
| Scaling needed | No | No | Yes | Yes |
| Maturity | Mature | Mature | Mature | Early |
| Accuracy | Best | Good | Good | Varies |
