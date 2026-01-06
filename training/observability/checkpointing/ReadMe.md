# Checkpointing

## Summary

Checkpointing is the practice of periodically saving training state to enable resumption after interruptions. In an era of expensive, long-running training jobs, checkpointing is essential for fault tolerance, cost management, and experiment flexibility. Effective checkpointing requires understanding what state to save, when to save, and how to manage checkpoints efficiently.

Key points to remember:

- Save model, optimizer, scheduler, and epoch/step state
- Checkpoint frequency balances safety vs overhead
- Distributed training requires coordinated checkpointing
- Use torch.save with state_dict() for portability
- Implement checkpoint rotation to manage disk space
- Save best models separately from regular checkpoints
- Include random states for exact reproducibility
- Test checkpoint loading before long training runs

## What to Checkpoint

### Essential Components

```python
import torch

checkpoint = {
    # Model weights
    'model_state_dict': model.state_dict(),

    # Optimizer state (momentum, etc.)
    'optimizer_state_dict': optimizer.state_dict(),

    # Learning rate scheduler
    'scheduler_state_dict': scheduler.state_dict(),

    # Training progress
    'epoch': epoch,
    'global_step': global_step,

    # Best metrics (for model selection)
    'best_val_loss': best_val_loss,
}

torch.save(checkpoint, 'checkpoint.pt')
```

### For Reproducibility

```python
import random
import numpy as np

checkpoint = {
    # ... essential components ...

    # Random states for exact reproducibility
    'random_state': random.getstate(),
    'numpy_random_state': np.random.get_state(),
    'torch_random_state': torch.get_rng_state(),
    'cuda_random_state': torch.cuda.get_rng_state_all(),
}
```

### For Analysis

```python
checkpoint = {
    # ... essential components ...

    # Training configuration
    'config': config_dict,

    # Final metrics
    'train_loss': train_loss,
    'val_loss': val_loss,

    # Gradient scaler for mixed precision
    'scaler_state_dict': scaler.state_dict(),
}
```

## Basic Checkpointing

### Save Checkpoint

```python
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, path)

# Usage
save_checkpoint(model, optimizer, scheduler, epoch, f'checkpoint_{epoch}.pt')
```

### Load Checkpoint

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']

# Resume training
start_epoch = 0
if resume_path:
    start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)

for epoch in range(start_epoch, num_epochs):
    train(...)
```

## Checkpoint Frequency

### Time-Based

```python
import time

last_checkpoint_time = time.time()
checkpoint_interval = 3600  # 1 hour

for step, batch in enumerate(dataloader):
    train_step(batch)

    if time.time() - last_checkpoint_time > checkpoint_interval:
        save_checkpoint(...)
        last_checkpoint_time = time.time()
```

### Step-Based

```python
checkpoint_steps = 1000

for step, batch in enumerate(dataloader):
    train_step(batch)

    if step % checkpoint_steps == 0:
        save_checkpoint(...)
```

### Epoch-Based

```python
for epoch in range(num_epochs):
    train_epoch(...)
    validate(...)
    save_checkpoint(model, optimizer, scheduler, epoch, f'checkpoint_{epoch}.pt')
```

### On Improvement

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(..., 'best_model.pt')

    # Also save regular checkpoint
    save_checkpoint(..., f'checkpoint_{epoch}.pt')
```

## Checkpoint Management

### Rotation (Keep Last N)

```python
import os
from pathlib import Path

def save_with_rotation(checkpoint, path, max_checkpoints=5):
    # Save new checkpoint
    torch.save(checkpoint, path)

    # Get all checkpoints
    checkpoint_dir = Path(path).parent
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))

    # Remove old checkpoints
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        oldest.unlink()
```

### Best Model Tracking

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, max_to_keep=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.best_metric = float('inf')
        self.checkpoints = []

    def save(self, state, step, metric=None):
        # Save checkpoint
        path = self.checkpoint_dir / f'checkpoint_{step}.pt'
        torch.save(state, path)
        self.checkpoints.append(path)

        # Update best if improved
        if metric is not None and metric < self.best_metric:
            self.best_metric = metric
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(state, best_path)

        # Rotate old checkpoints
        while len(self.checkpoints) > self.max_to_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
```

## Mixed Precision Checkpointing

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# Save
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
```

## Safe Checkpointing

### Atomic Writes

```python
import os
import tempfile
import shutil

def safe_save(state, path):
    """Save checkpoint atomically to prevent corruption."""
    # Save to temporary file
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp_path = tmp.name

    # Atomic rename
    shutil.move(tmp_path, path)
```

### Verification

```python
def save_and_verify(state, path):
    """Save checkpoint and verify it can be loaded."""
    torch.save(state, path)

    # Verify by loading
    try:
        loaded = torch.load(path, map_location='cpu')
        assert 'model_state_dict' in loaded
    except Exception as e:
        os.remove(path)
        raise RuntimeError(f"Checkpoint verification failed: {e}")
```

## Framework Integration

### PyTorch Lightning

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min',
    save_last=True
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])
```

### Hugging Face Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Save
accelerator.save_state('checkpoint/')

# Load
accelerator.load_state('checkpoint/')
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='./checkpoints',
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,
)
```

## Best Practices

1. **Test resumption early**: Verify checkpointing works before long runs
2. **Use atomic saves**: Prevent corruption from interruptions
3. **Include metadata**: Hyperparameters, git commit, timestamp
4. **Checkpoint to fast storage**: Network storage can be slow
5. **Rotate checkpoints**: Manage disk space
6. **Save best separately**: Don't lose best model to rotation
7. **Verify checksums**: For critical checkpoints

## Further Reading

- [State Dict Management](state-dict-management/ReadMe.md): Managing model state
- [Distributed Checkpointing](distributed-checkpointing/ReadMe.md): Multi-GPU/node checkpointing
- [Checkpoint Sharding](checkpoint-sharding/ReadMe.md): Handling large model checkpoints
