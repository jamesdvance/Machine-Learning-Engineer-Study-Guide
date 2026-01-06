# Gradient Accumulation for Memory Optimization

## Summary

Gradient accumulation allows training with larger effective batch sizes without increasing memory usage. By accumulating gradients over multiple forward-backward passes before performing an optimizer step, you can simulate training with batch sizes that would otherwise cause out-of-memory errors. This chapter focuses on the memory optimization aspect of gradient accumulation.

Key points to remember:

- Enables larger effective batches without more memory
- Memory usage stays constant regardless of accumulation steps
- Effective batch = micro_batch x accumulation_steps x num_gpus
- Must scale loss by accumulation steps for correct gradients
- Use model.no_sync() in DDP to avoid redundant communication
- Combines well with other memory optimizations
- Important for training stability with large models
- See distributed training concepts for full implementation details

## Memory Perspective

### Why Accumulation Helps

Activation memory scales with batch size:
```
Activation memory proportional to batch_size x sequence_length x hidden_size
```

Without accumulation: Must fit entire batch in memory.
With accumulation: Only micro-batch activations needed.

### Memory Comparison

For effective batch 256, sequence 2048, hidden 4096:

| Approach | Micro-batch | Activation Memory |
|----------|-------------|-------------------|
| Direct | 256 | ~200 GB |
| Accumulate 8x | 32 | ~25 GB |
| Accumulate 32x | 8 | ~6 GB |

## Implementation

### Basic Pattern

```python
accumulation_steps = 8
effective_batch_size = micro_batch_size * accumulation_steps

optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward and backward
    loss = model(batch) / accumulation_steps  # Scale loss!
    loss.backward()  # Gradients accumulate

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### With Distributed Training

```python
from contextlib import nullcontext

for i, batch in enumerate(dataloader):
    is_accumulating = (i + 1) % accumulation_steps != 0

    # Skip gradient sync during accumulation (saves communication)
    context = model.no_sync() if is_accumulating else nullcontext()

    with context:
        loss = model(batch) / accumulation_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

### With Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for i, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Choosing Accumulation Steps

### Memory Calculation

```python
def find_accumulation_steps(target_batch, available_memory, model, input_shape):
    """Find max micro-batch that fits in memory."""
    for micro_batch in range(target_batch, 0, -1):
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(micro_batch, *input_shape[1:], device='cuda')
            loss = model(x).sum()
            loss.backward()

            if torch.cuda.max_memory_allocated() < available_memory * 0.9:
                accumulation_steps = target_batch // micro_batch
                return micro_batch, accumulation_steps
        except RuntimeError:  # OOM
            continue

    raise ValueError("Cannot fit even batch size 1")
```

### Practical Guidelines

| GPU Memory | Typical Micro-batch | Accumulation for 256 |
|------------|---------------------|---------------------|
| 16 GB | 4-8 | 32-64 |
| 40 GB | 16-32 | 8-16 |
| 80 GB | 32-64 | 4-8 |

## Interactions with Other Techniques

### With Gradient Checkpointing

Both reduce memory; combined effect is multiplicative:

```python
# Enable gradient checkpointing to reduce activations further
model.gradient_checkpointing_enable()

# Then use gradient accumulation for even larger effective batch
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### With FSDP/ZeRO

Accumulation works seamlessly with sharding:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)

# Same accumulation pattern
for i, batch in enumerate(dataloader):
    is_accumulating = (i + 1) % accumulation_steps != 0

    with model.no_sync() if is_accumulating else nullcontext():
        loss = model(batch) / accumulation_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

## Framework Support

### Hugging Face Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    # Effective batch = 8 * 4 * num_gpus
)
```

### PyTorch Lightning

```python
trainer = pl.Trainer(
    accumulate_grad_batches=4
)
```

### DeepSpeed

```json
{
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4
}
```

## Performance Considerations

### Overhead

Gradient accumulation has minimal overhead:
- Same total compute
- Less frequent optimizer steps (slight benefit)
- Less frequent gradient sync in distributed (benefit)

### Comparison with Larger Batch

| Approach | Memory | Speed | Result |
|----------|--------|-------|--------|
| Large batch | High | Fastest | May OOM |
| Accumulation | Low | Slightly slower | Always works |

The slight slowdown from more forward/backward passes is usually negligible.

## Debugging

### Verify Correct Scaling

```python
# Check gradient magnitude is similar to without accumulation
def check_gradient_scaling():
    # Without accumulation (reference)
    model.zero_grad()
    loss = model(large_batch)
    loss.backward()
    ref_grad_norm = compute_grad_norm(model)

    # With accumulation
    model.zero_grad()
    for micro_batch in split_batch(large_batch, accumulation_steps):
        loss = model(micro_batch) / accumulation_steps
        loss.backward()
    acc_grad_norm = compute_grad_norm(model)

    # Should be approximately equal
    assert abs(ref_grad_norm - acc_grad_norm) / ref_grad_norm < 0.01
```

### Common Issues

**Gradients too large**:
- Forgot to divide loss by accumulation_steps

**Gradients too small**:
- Divided loss multiple times
- Check loss scaling logic

**Training slower than expected**:
- no_sync() not used in distributed training
- Check for unnecessary synchronization

## Best Practices

1. **Always scale loss**: Divide by accumulation_steps
2. **Use no_sync()**: In distributed training
3. **Handle remainders**: Process final batches correctly
4. **Monitor gradient norms**: Verify scaling is correct
5. **Combine with checkpointing**: For maximum memory efficiency
6. **Profile both approaches**: Large batch vs accumulation

## Related Topics

For comprehensive gradient accumulation coverage including:
- Detailed distributed training integration
- Learning rate scaling
- Mathematical equivalence proofs

See: [Gradient Accumulation (Distributed Training)](../../../distributed-training/concepts/gradient-accumulation/ReadMe.md)
