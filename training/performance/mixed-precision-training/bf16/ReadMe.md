# BF16 (Brain Floating Point)

## Summary

BF16 (Brain Float 16) is a 16-bit floating point format that maintains the same exponent range as FP32 while reducing mantissa precision. Developed by Google Brain for deep learning, BF16 eliminates the need for loss scaling while providing the memory benefits of 16-bit precision. BF16 has become the preferred training precision on modern hardware (A100, H100).

Key points to remember:

- 16-bit format: 1 sign, 8 exponent, 7 mantissa bits
- Same dynamic range as FP32 (no loss scaling needed)
- Lower precision than FP16 (7 vs 10 mantissa bits)
- Native support on A100, H100, TPUs
- Not supported on V100 or older GPUs
- Simpler training code without GradScaler
- Preferred choice for training when available
- Slight accuracy reduction from precision, usually acceptable

## BF16 Format

### Bit Layout

```
BF16: [S][EEEEEEEE][MMMMMMM]
       1     8         7

S: Sign bit (1 bit)
E: Exponent (8 bits, same as FP32)
M: Mantissa (7 bits, truncated from FP32's 23)
```

### Comparison with Other Formats

| Property | FP32 | BF16 | FP16 |
|----------|------|------|------|
| Total bits | 32 | 16 | 16 |
| Exponent bits | 8 | 8 | 5 |
| Mantissa bits | 23 | 7 | 10 |
| Range | ~1e-38 to 3e38 | ~1e-38 to 3e38 | ~6e-8 to 65504 |
| Precision | ~7 digits | ~2 digits | ~3 digits |

### Why BF16 Works

BF16 is essentially truncated FP32:

```python
# Conceptual conversion
def fp32_to_bf16(x):
    # Keep sign and exponent, truncate mantissa
    bits = x.view(torch.int32)
    bf16_bits = bits >> 16  # Drop lower 16 bits
    return bf16_bits.view(torch.bfloat16)
```

The preserved range means:
- No gradient underflow issues
- No loss scaling required
- Simpler training code

## Implementation

### Basic BF16 Training

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in BF16
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, target)

    # Backward (no scaler needed!)
    loss.backward()

    # Optimizer step
    optimizer.step()
```

### Comparison with FP16

```python
# FP16 (requires loss scaling)
scaler = GradScaler()
with autocast(dtype=torch.float16):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# BF16 (no loss scaling)
with autocast(dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()
optimizer.step()
```

### Checking BF16 Support

```python
import torch

# Check hardware support
if torch.cuda.is_bf16_supported():
    print("BF16 supported!")
else:
    print("BF16 not supported, use FP16")

# Check specific GPU
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}")
# A100, H100: BF16 supported
# V100: BF16 NOT supported
```

## Framework Integration

### PyTorch

```python
# Automatic mixed precision with BF16
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)

# Or set default dtype
torch.set_default_dtype(torch.bfloat16)
```

### Hugging Face Transformers

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    bf16=True,  # Enable BF16 training
    # bf16_full_eval=True,  # Optional: BF16 for eval
)
```

### PyTorch Lightning

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    precision="bf16-mixed",  # BF16 mixed precision
)
```

### DeepSpeed

```json
{
    "bf16": {
        "enabled": true
    }
}
```

### FSDP

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(model, mixed_precision=bf16_policy)
```

## Hardware Support

### NVIDIA GPUs

| GPU | BF16 Support | Tensor Core BF16 |
|-----|--------------|------------------|
| V100 | No | No |
| A100 | Yes | 312 TFLOPS |
| H100 | Yes | 1000 TFLOPS |

### TPUs

All TPU generations support BF16 natively:

```python
import torch_xla.core.xla_model as xm

# BF16 on TPU
with torch.autocast(device_type='xla', dtype=torch.bfloat16):
    output = model(input)
```

### AMD GPUs

MI250X and newer support BF16.

## Precision Trade-offs

### Accuracy Comparison

BF16 has lower precision than FP16:

```python
import torch

# Precision comparison
x = torch.tensor(1.0001)

fp16 = x.half()
bf16 = x.bfloat16()

print(f"Original: {x}")
print(f"FP16: {fp16}")  # 1.0
print(f"BF16: {bf16}")  # 1.0 (less precise)
```

### When Precision Matters

BF16 precision is usually sufficient for:
- Weight updates (gradients are averaged)
- Forward computation (errors don't accumulate badly)
- Attention scores (softmax normalizes)

May need FP32 for:
- Loss computation (often automatic)
- Normalization layers (often automatic)
- Metrics calculation

### Automatic FP32 Upcasting

PyTorch autocast handles sensitive operations:

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    # These automatically run in FP32:
    # - Softmax
    # - Cross-entropy loss
    # - Layer normalization
    # - Batch normalization
    pass
```

## Memory and Performance

### Memory Savings

Same as FP16: 2 bytes per element vs 4 bytes.

```python
# Memory comparison
fp32_model = model.float()
bf16_model = model.bfloat16()

fp32_mem = sum(p.numel() * 4 for p in fp32_model.parameters())
bf16_mem = sum(p.numel() * 2 for p in bf16_model.parameters())

print(f"FP32: {fp32_mem / 1e9:.2f} GB")
print(f"BF16: {bf16_mem / 1e9:.2f} GB")
```

### Performance

BF16 tensor core performance on A100:
- BF16: 312 TFLOPS
- FP16: 312 TFLOPS
- FP32: 19.5 TFLOPS

Both BF16 and FP16 provide ~16x speedup over FP32.

## Migration from FP16

### Simplifying Code

```python
# Before (FP16)
scaler = GradScaler()
for batch in dataloader:
    with autocast(dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# After (BF16)
for batch in dataloader:
    with autocast(dtype=torch.bfloat16):
        loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Removing Loss Scaling

```python
# FP16 config
training_args = TrainingArguments(
    fp16=True,
    fp16_full_eval=True,
)

# BF16 config (simpler)
training_args = TrainingArguments(
    bf16=True,
)
```

## Best Practices

1. **Use BF16 on A100+**: Default choice for modern GPUs
2. **No loss scaling**: Remove GradScaler when switching from FP16
3. **Verify accuracy**: Compare with FP32 baseline for critical models
4. **Use autocast**: Let PyTorch handle precision casting
5. **Check hardware**: Fall back to FP16 on V100 and older

## Common Issues

### BF16 Not Available

```python
# Fallback pattern
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    use_scaler = False
else:
    dtype = torch.float16
    use_scaler = True
```

### Slight Accuracy Difference

BF16 has lower precision than FP16. Usually acceptable, but:
- Monitor validation metrics
- Compare with FP32 baseline
- Consider FP16 for precision-critical applications

### Incompatible Operations

Some custom CUDA kernels may not support BF16:

```python
# Cast to float for incompatible operations
with autocast(dtype=torch.bfloat16):
    x = model.encoder(input)

    # Unsupported op
    x = x.float()
    x = custom_cuda_op(x)
    x = x.bfloat16()

    output = model.decoder(x)
```
