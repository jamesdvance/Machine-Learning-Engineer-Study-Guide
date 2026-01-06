# FP16 (Half Precision)

## Summary

FP16 (half precision floating point) uses 16 bits to represent numbers, providing 2x memory savings compared to FP32. FP16 training requires careful handling through loss scaling due to its limited dynamic range. Despite being largely superseded by BF16 on modern hardware, FP16 remains important for older GPUs (V100) and inference optimization.

Key points to remember:

- 16-bit format: 1 sign, 5 exponent, 10 mantissa bits
- Range: approximately 6e-8 to 65504
- Higher precision than BF16 (10 vs 7 mantissa bits)
- Requires loss scaling to prevent gradient underflow
- Supported on all modern NVIDIA GPUs
- Tensor cores provide significant speedup
- Good for inference where dynamic range is known
- Use BF16 instead when available (A100+)

## FP16 Format

### Bit Layout

```
FP16: [S][EEEEE][MMMMMMMMMM]
       1    5        10

S: Sign bit (1 bit)
E: Exponent (5 bits, bias 15)
M: Mantissa (10 bits)
```

### Value Range

| Value | FP16 Representation |
|-------|---------------------|
| Maximum | 65504 |
| Minimum positive | 6.10e-5 (normal) |
| Minimum subnormal | 5.96e-8 |
| Epsilon | 0.00097 |

### Comparison with FP32

| Property | FP32 | FP16 |
|----------|------|------|
| Total bits | 32 | 16 |
| Exponent bits | 8 | 5 |
| Mantissa bits | 23 | 10 |
| Range | ~1e-38 to 3e38 | ~6e-8 to 65504 |
| Precision | ~7 digits | ~3 digits |

## The Dynamic Range Problem

### Gradient Underflow

Many gradients in deep learning are small (1e-5 to 1e-8):

```python
gradient = 1e-6

# In FP32: Representable
fp32_grad = torch.tensor(1e-6, dtype=torch.float32)  # Works

# In FP16: May underflow to zero
fp16_grad = torch.tensor(1e-6, dtype=torch.float16)  # May be 0
```

### Loss Scaling Solution

Scale loss to shift gradients into representable range:

```python
# Without scaling
loss = 0.001
gradient = 1e-6  # May underflow in FP16

# With scaling (scale = 1024)
scaled_loss = loss * 1024  # = 1.024
scaled_gradient = 1e-6 * 1024  # = 0.001024 (representable)

# After backward, unscale
gradient = scaled_gradient / 1024
```

## Implementation

### Basic FP16 Training

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in FP16
    with autocast(dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)

    # Backward with scaled loss
    scaler.scale(loss).backward()

    # Unscale and step
    scaler.step(optimizer)
    scaler.update()
```

### Manual Loss Scaling

```python
scale = 1024.0

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in FP16
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)

    # Scale and backward
    scaled_loss = loss * scale
    scaled_loss.backward()

    # Unscale gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data /= scale

    # Check for overflow
    has_overflow = False
    for param in model.parameters():
        if param.grad is not None:
            if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                has_overflow = True
                break

    if not has_overflow:
        optimizer.step()
```

### Dynamic Loss Scaling

```python
scaler = GradScaler(
    init_scale=2**16,         # Start at 65536
    growth_factor=2.0,        # Double on success
    backoff_factor=0.5,       # Halve on overflow
    growth_interval=2000,     # Wait 2000 steps before growing
)

# The scaler automatically:
# 1. Scales loss during backward
# 2. Checks for inf/nan in gradients
# 3. Skips optimizer step on overflow
# 4. Adjusts scale factor dynamically
```

## Operations Requiring FP32

PyTorch autocast automatically keeps these in FP32:

```python
# Always FP32 in autocast
torch.nn.functional.softmax
torch.nn.functional.log_softmax
torch.nn.functional.cross_entropy
torch.nn.functional.nll_loss
torch.nn.functional.binary_cross_entropy
torch.nn.functional.binary_cross_entropy_with_logits
torch.nn.functional.layer_norm
torch.nn.functional.batch_norm
torch.nn.functional.group_norm
```

### Custom FP32 Sections

```python
with autocast(dtype=torch.float16):
    x = model.encoder(input)

    # Force FP32
    with autocast(enabled=False):
        x = x.float()
        x = numerically_sensitive_op(x)
        x = x.half()

    output = model.decoder(x)
```

## Memory Layout

### Tensor Storage

```python
# Memory per element
fp32_tensor = torch.randn(1000, 1000, dtype=torch.float32)
print(f"FP32: {fp32_tensor.element_size()} bytes")  # 4 bytes

fp16_tensor = torch.randn(1000, 1000, dtype=torch.float16)
print(f"FP16: {fp16_tensor.element_size()} bytes")  # 2 bytes
```

### Training Memory

| Component | FP32 Training | FP16 Mixed |
|-----------|---------------|------------|
| Parameters | 4P | 2P + 4P (master) |
| Gradients | 4P | 2P |
| Adam state | 8P | 8P (kept FP32) |
| Activations | 4A | 2A |

Note: Total parameter memory similar, but activation savings significant.

## Hardware Support

### Tensor Cores

FP16 operations use tensor cores when available:

| GPU | Tensor Core FP16 | Memory Bandwidth |
|-----|------------------|------------------|
| V100 | 125 TFLOPS | 900 GB/s |
| A100 | 312 TFLOPS | 2 TB/s |
| H100 | 1000 TFLOPS | 3.35 TB/s |

### Checking Tensor Core Usage

```python
import torch

# Ensure tensor core-friendly dimensions
# Dimensions should be multiples of 8 for best performance
hidden_size = 1024  # Good (multiple of 8)
hidden_size = 1000  # Works but not optimal
```

## Debugging FP16

### Monitor Scale Factor

```python
for step, batch in enumerate(dataloader):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        print(f"Step {step}: scale = {scaler.get_scale()}")
        # Decreasing scale indicates frequent overflow
```

### Detect Overflow

```python
def check_overflow(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if torch.isnan(grad).any():
                return True, f"NaN in {name}"
            if torch.isinf(grad).any():
                return True, f"Inf in {name}"
    return False, None
```

### Compare with FP32

```python
# Run same batch in both precisions
with autocast(enabled=False):
    fp32_output = model(batch)
    fp32_loss = criterion(fp32_output, target)

with autocast(dtype=torch.float16):
    fp16_output = model(batch)
    fp16_loss = criterion(fp16_output, target)

print(f"Loss diff: {abs(fp32_loss - fp16_loss)}")
```

## Best Practices

1. **Use GradScaler**: Let PyTorch handle loss scaling
2. **Monitor scale factor**: Investigate if consistently decreasing
3. **Check convergence**: Compare with FP32 baseline early
4. **Use BF16 on A100+**: More stable, no loss scaling needed
5. **Align dimensions to 8**: Better tensor core utilization
6. **Gradient clipping after unscale**: Prevents optimizer issues

## Common Issues

### Scale Factor Drops to Minimum

Cause: Frequent gradient overflow
Solution:
- Lower learning rate
- Check for numerical instabilities
- Consider BF16 instead

### Slower Than Expected

Cause: Operations falling back to FP32
Solution:
- Profile to find FP32 operations
- Check tensor core utilization
- Ensure dimensions are multiples of 8

### Accuracy Degradation

Cause: Precision loss in sensitive operations
Solution:
- Keep normalization in FP32
- Use higher precision for loss computation
- Consider BF16 for better stability
