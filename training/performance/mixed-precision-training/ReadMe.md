# Mixed Precision Training

## Summary

Mixed precision training uses lower precision numerical formats (FP16 or BF16) for most computations while maintaining FP32 for critical operations. This approach nearly doubles memory efficiency and throughput on modern GPUs with tensor cores, while maintaining model accuracy through careful precision management. Mixed precision has become standard practice for deep learning training.

Key points to remember:

- Uses FP16/BF16 for forward and backward passes, FP32 for master weights
- Nearly 2x memory reduction for parameters and activations
- Significant speed improvement from tensor cores
- BF16 preferred on modern GPUs (A100, H100) for stability
- FP16 requires loss scaling to handle gradient underflow
- Automatic mixed precision (AMP) handles precision casting
- Some operations must stay in FP32 for numerical stability
- Virtually no accuracy degradation when implemented correctly

## Precision Formats

### Comparison

| Format | Exponent | Mantissa | Range | Use Case |
|--------|----------|----------|-------|----------|
| FP32 | 8 bits | 23 bits | 1e-38 to 3e38 | Master weights |
| FP16 | 5 bits | 10 bits | 6e-8 to 65504 | Training compute |
| BF16 | 8 bits | 7 bits | 1e-38 to 3e38 | Training compute |
| FP8 | 4/5 bits | 3/2 bits | Varies | Emerging |

### FP16 vs BF16

**FP16 (Half precision)**:
- Higher precision (10-bit mantissa)
- Narrower range (requires loss scaling)
- Supported on V100, A100, H100

**BF16 (Brain float)**:
- Lower precision (7-bit mantissa)
- Same range as FP32 (no loss scaling needed)
- Supported on A100, H100 (not V100)

**Recommendation**: Use BF16 when available (A100+). Use FP16 with loss scaling on older hardware.

## How Mixed Precision Works

### Training Loop

```
Forward pass:  FP16/BF16 compute with FP16/BF16 weights
               |
Backward pass: FP16/BF16 gradients
               |
Loss scaling:  Scale gradients to prevent underflow (FP16 only)
               |
Optimizer:     FP32 master weights updated
               |
Copy:          FP16/BF16 weights = FP32 master weights
```

### Why Master Weights in FP32

Small gradient updates can underflow in FP16:
```
FP16 weight: 1.0
FP16 update: 0.00001
1.0 + 0.00001 = 1.0 (update lost due to precision)

FP32 weight: 1.0
FP32 update: 0.00001
1.0 + 0.00001 = 1.00001 (update preserved)
```

## PyTorch Implementation

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # For FP16; not needed for BF16

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Backward pass with scaling
    scaler.scale(loss).backward()

    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
```

### BF16 (No Loss Scaling)

```python
# BF16 on A100/H100
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
# No scaler needed for BF16
```

### Specifying Precision

```python
# FP16
with torch.autocast(device_type='cuda', dtype=torch.float16):
    ...

# BF16
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    ...
```

## Loss Scaling

### Why Loss Scaling

FP16 gradient range: 6e-8 to 65504

Small gradients can underflow to zero. Loss scaling shifts gradients into representable range.

### How Loss Scaling Works

```python
# Forward
loss = model(x)

# Scale loss to shift gradients up
scaled_loss = loss * scale_factor

# Backward
scaled_loss.backward()

# Unscale gradients before optimizer step
for param in model.parameters():
    param.grad /= scale_factor

# Check for inf/nan and adjust scale
if has_inf_or_nan(gradients):
    scale_factor /= 2
    skip_update = True
else:
    scale_factor *= 2
    skip_update = False
```

### GradScaler Details

```python
scaler = GradScaler(
    init_scale=65536.0,      # Initial scale factor
    growth_factor=2.0,        # Scale up by 2x
    backoff_factor=0.5,       # Scale down by 0.5x
    growth_interval=2000,     # Steps between growth
    enabled=True              # Can disable for debugging
)
```

## Operations That Must Stay FP32

Some operations are numerically unstable in FP16:

```python
# PyTorch autocast handles this automatically
# These ops run in FP32 even inside autocast:

# Softmax (in loss functions)
# Layer normalization
# Batch normalization statistics
# Exp, log, pow
# Precision-sensitive reductions
```

### Manual FP32 Casting

```python
with autocast():
    x = model.encoder(input)

    # Force FP32 for sensitive operation
    with autocast(enabled=False):
        x = x.float()  # Cast to FP32
        x = sensitive_operation(x)

    x = x.half()  # Back to FP16
    output = model.decoder(x)
```

## Memory Savings

### Analysis

For model with P parameters:

| Component | FP32 | Mixed Precision |
|-----------|------|-----------------|
| Parameters | 4P | 2P (FP16) + 4P (master) |
| Gradients | 4P | 2P |
| Optimizer | 8P (Adam) | 8P (FP32 states) |
| Activations | 4 x A | 2 x A |

Total parameter memory: 16P (FP32) vs 16P (mixed)

But activations are halved, which is often the larger component.

### Practical Savings

Mixed precision typically enables:
- 1.5-2x larger batch sizes
- Training models that otherwise OOM

## Framework Integration

### Hugging Face Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    fp16=True,              # Enable FP16
    # Or
    bf16=True,              # Enable BF16
    fp16_full_eval=False,   # FP16 for eval too
)
```

### PyTorch Lightning

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    precision="16-mixed",   # FP16 mixed precision
    # Or
    precision="bf16-mixed", # BF16 mixed precision
)
```

### DeepSpeed

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

Or BF16:
```json
{
    "bf16": {
        "enabled": true
    }
}
```

## Debugging Mixed Precision

### Check for Overflow

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
```

### Monitor Loss Scale

```python
for step, batch in enumerate(dataloader):
    # Training step...

    if step % 100 == 0:
        print(f"Step {step}, Loss scale: {scaler.get_scale()}")
```

### Common Issues

**Loss scale keeps decreasing**:
- Learning rate too high
- Model has numerical instability
- Try BF16 instead

**NaN loss**:
- Check for division by zero
- Check for log of zero/negative
- Ensure softmax inputs not too large

## Best Practices

1. **Use BF16 on A100+**: More stable than FP16, no loss scaling
2. **Start with defaults**: GradScaler defaults work well
3. **Monitor loss scale**: Decreasing scale indicates issues
4. **Clip gradients after unscaling**: Prevent optimizer issues
5. **Use framework support**: Let autocast handle precision
6. **Test accuracy**: Verify mixed precision matches FP32 baseline

## Further Reading

- [FP16](fp16/ReadMe.md): Half precision details
- [BF16](bf16/ReadMe.md): Brain float details
- [FP8](fp8/ReadMe.md): Next-generation precision
