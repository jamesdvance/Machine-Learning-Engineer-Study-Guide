# Gradient Accumulation

## Summary

Gradient accumulation is a technique that simulates training with larger batch sizes by accumulating gradients over multiple forward-backward passes before performing an optimizer step. This allows training with effectively larger batches without increasing memory usage, making it possible to train with batch sizes that would otherwise exceed GPU memory.

Key points to remember:

- Accumulates gradients across multiple micro-batches before updating
- Effective batch size = micro_batch_size x accumulation_steps x num_devices
- Memory usage stays constant regardless of accumulation steps
- Mathematically equivalent to training with larger batches
- Essential for large model training where batch size is memory-constrained
- Must scale loss by accumulation steps to maintain correct gradient magnitude
- Interacts with learning rate, batch norm, and distributed training

## Why Gradient Accumulation

### The Batch Size Dilemma

Large batch training often improves:
- Training stability
- Convergence speed
- GPU utilization (better parallelism)

But GPU memory limits batch size:
- Activations scale with batch size
- Cannot fit desired batch in memory

Gradient accumulation solves this by decoupling effective batch size from memory.

### Memory Analysis

Memory during training:
```
Total = Parameters + Gradients + Optimizer States + Activations

Activations approximately proportional to batch_size x sequence_length x hidden_size
```

With gradient accumulation:
- Activations: Based on micro_batch_size (small)
- Effective batch: micro_batch_size x accumulation_steps (large)

## Basic Implementation

### PyTorch Implementation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch)
    loss = criterion(outputs, batch.labels)

    # Scale loss to account for accumulation
    loss = loss / accumulation_steps

    # Backward pass (accumulates gradients)
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Handle remaining batches
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Why Scale the Loss

Without scaling:
```
accumulated_grad = grad_1 + grad_2 + grad_3 + grad_4
```

This is 4x the gradient of a single batch, not the average.

With scaling by 1/N:
```
accumulated_grad = (grad_1 + grad_2 + grad_3 + grad_4) / 4
```

This equals the gradient of processing all 4 batches together.

## With Distributed Training

### DDP and Gradient Accumulation

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[rank])
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # Disable gradient sync except on accumulation boundary
    is_accumulating = (i + 1) % accumulation_steps != 0

    with model.no_sync() if is_accumulating else contextlib.nullcontext():
        loss = model(batch) / accumulation_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

### Why no_sync() Matters

Without no_sync():
- DDP all-reduces gradients after every backward
- Wastes communication bandwidth during accumulation

With no_sync():
- Gradients accumulate locally
- All-reduce only on accumulation boundary
- Significantly faster

### Effective Batch Size Calculation

```
effective_batch_size = micro_batch x accumulation_steps x world_size

Example:
  micro_batch = 8
  accumulation_steps = 4
  world_size = 8 GPUs
  effective_batch = 8 x 4 x 8 = 256
```

## Framework Support

### Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    # Effective batch = 8 * 4 * num_gpus
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### PyTorch Lightning

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    accumulate_grad_batches=4,
    # Other arguments
)
```

### DeepSpeed

```python
# deepspeed_config.json
{
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    // effective_batch = 8 * 4 * num_gpus
}
```

## Interactions and Edge Cases

### Learning Rate Scaling

When increasing effective batch size via accumulation:

```python
# Linear scaling (common rule)
base_lr = 1e-4
base_batch = 32
effective_batch = micro_batch * accumulation_steps * world_size
scaled_lr = base_lr * (effective_batch / base_batch)

# With warmup for stability
warmup_steps = 1000
```

Linear scaling works well up to moderate batch sizes but may need adjustment for very large batches.

### Batch Normalization

BatchNorm statistics are computed per micro-batch:

```python
# With accumulation_steps=4, batch_size=8:
# BatchNorm sees batches of 8, not 32

# Solutions:
# 1. Use larger micro-batch if possible
# 2. Use SyncBatchNorm for distributed
# 3. Use LayerNorm or GroupNorm instead
```

For large model training, LayerNorm is standard and does not have this issue.

### Dropout

Dropout is applied per micro-batch, which is correct:

```python
# Each micro-batch has independent dropout masks
# This is equivalent to a single large batch with dropout
# No special handling needed
```

### Gradient Clipping

Clip gradients after accumulation:

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        # Clip after accumulation, before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for i, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch) / accumulation_steps

    # Scale loss and backward
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Memory-Throughput Trade-off

### Finding Optimal Configuration

```
Memory fixed -> Throughput depends on accumulation_steps

Low accumulation_steps (e.g., 1-2):
  + Lower latency per step
  + More frequent weight updates
  - May need smaller micro-batch

High accumulation_steps (e.g., 8-16):
  + Can use larger micro-batch (better GPU utilization)
  + Fewer gradient syncs in distributed training
  - Higher latency per effective batch
  - Less frequent weight updates
```

### Practical Guidelines

1. **Maximize micro-batch first**: Fill GPU memory with largest micro-batch
2. **Then accumulate**: Add accumulation steps to reach target effective batch
3. **Balance**: Very high accumulation can slow convergence (infrequent updates)

Example configuration process:
```python
# Goal: effective_batch = 1024, 8 GPUs

# Step 1: Find max micro_batch that fits in memory
# Testing shows micro_batch=16 fits

# Step 2: Calculate accumulation steps
# 1024 = 16 * accumulation_steps * 8
# accumulation_steps = 1024 / (16 * 8) = 8
```

## Debugging

### Verify Gradient Accumulation

```python
# Check gradient magnitude after accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        # Should be similar magnitude to single large batch
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm}")

        optimizer.step()
        optimizer.zero_grad()
```

### Verify Effective Batch Size

```python
# Count samples processed per optimizer step
samples_per_step = 0
for i, batch in enumerate(dataloader):
    samples_per_step += len(batch)
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        print(f"Samples this step: {samples_per_step}")
        samples_per_step = 0  # Reset counter
        optimizer.step()
        optimizer.zero_grad()
```

### Common Issues

**Loss not decreasing**:
- Check loss scaling (division by accumulation_steps)
- Verify gradient clipping is after accumulation

**Different results than large batch**:
- BatchNorm statistics differ (use LayerNorm)
- Random state differs (usually acceptable)

**Slower than expected**:
- Use model.no_sync() in distributed training
- Check if accumulation_steps is too high

## Best Practices

1. **Always scale loss**: Divide by accumulation_steps

2. **Use no_sync()**: In DDP, skip synchronization during accumulation

3. **Handle remainders**: Process remaining batches at end of epoch

4. **Clip after accumulation**: Gradient clipping before optimizer.step()

5. **Log effective batch**: Track and log the actual effective batch size

6. **Test equivalence**: Verify results match single large batch (when possible)

7. **Consider learning rate**: Adjust for effective batch size change
