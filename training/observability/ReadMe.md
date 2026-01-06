# Training Observability

## Summary

Training observability encompasses the tools and practices that provide visibility into the training process. This includes checkpointing for fault tolerance and resumption, logging for experiment tracking and analysis, and debugging for identifying and resolving training issues. Effective observability is essential for managing long-running training jobs and ensuring reproducibility.

Key points to remember:

- Checkpointing enables training resumption and fault tolerance
- Logging tracks metrics, hyperparameters, and artifacts
- Debugging identifies NaN/Inf, gradient issues, and memory problems
- Observability tools integrate with distributed training
- Experiment tracking enables comparison and reproducibility
- Profiling identifies performance bottlenecks
- Early detection of issues saves compute costs
- Comprehensive logging is essential for production ML

## Observability Components

### Overview

| Component | Purpose | Key Tools |
|-----------|---------|-----------|
| Checkpointing | Save/resume training state | PyTorch, Orbax, DeepSpeed |
| Logging | Track experiments | W&B, MLflow, TensorBoard |
| Debugging | Identify training issues | Profilers, gradient monitors |

### Integration Points

```
Training Loop
    |
    +-- Checkpoint Manager
    |       - Save model state
    |       - Save optimizer state
    |       - Save training progress
    |
    +-- Metric Logger
    |       - Loss curves
    |       - Learning rates
    |       - Custom metrics
    |
    +-- Debug Monitors
            - Gradient statistics
            - Memory usage
            - NaN detection
```

## Quick Setup

### Basic Observability Stack

```python
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

# Initialize logging
wandb.init(project="my-project")
writer = SummaryWriter("runs/experiment")

# Training loop with observability
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Log metrics
        wandb.log({"loss": loss, "step": global_step})
        writer.add_scalar("loss", loss, global_step)

        # Gradient monitoring
        grad_norm = compute_grad_norm(model)
        wandb.log({"grad_norm": grad_norm})

    # Checkpoint at epoch end
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_{epoch}.pt')

    # Validation metrics
    val_loss = validate(model)
    wandb.log({"val_loss": val_loss, "epoch": epoch})
```

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(project="my-project")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    save_top_k=3
)

trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=50
)
```

## Key Practices

### 1. Checkpoint Strategy

```python
# Save frequency
- Every N steps for long training
- Every epoch for shorter training
- On validation improvement

# What to save
- Model state dict
- Optimizer state dict
- Learning rate scheduler
- Epoch/step number
- Best metric value
- Random states (for reproducibility)
```

### 2. Metric Logging

```python
# Essential metrics
- Training loss (per step and smoothed)
- Validation loss and metrics
- Learning rate
- Gradient norm
- GPU memory usage
- Throughput (samples/second)

# Periodic metrics
- Weight histograms
- Gradient histograms
- Attention visualizations
```

### 3. Debug Monitoring

```python
# Always monitor
- NaN/Inf in loss
- Gradient explosion/vanishing
- Memory usage trends

# On issues
- Per-layer gradient norms
- Activation statistics
- Parameter update magnitudes
```

## Common Issues and Detection

| Issue | Symptom | Detection Method |
|-------|---------|------------------|
| Gradient explosion | NaN loss | Gradient norm monitoring |
| Gradient vanishing | No learning | Per-layer gradient stats |
| Memory leak | OOM after hours | Memory profiling |
| Loss plateau | No improvement | Loss curve analysis |
| Overfitting | Train/val gap | Validation monitoring |

## Distributed Considerations

### Multi-GPU Checkpointing

```python
# Only save on rank 0
if torch.distributed.get_rank() == 0:
    torch.save(state, 'checkpoint.pt')

# Synchronize before loading
torch.distributed.barrier()
```

### Distributed Logging

```python
# Log only on main process
if accelerator.is_main_process:
    wandb.log(metrics)

# Or use framework support
accelerator.log(metrics)  # Handles distributed
```

## Tooling Ecosystem

### Experiment Tracking

| Tool | Strengths | Best For |
|------|-----------|----------|
| W&B | Collaboration, UI | Teams, production |
| MLflow | Self-hosted, MLOps | Enterprise |
| TensorBoard | Simple, lightweight | Quick experiments |
| Aim | Open-source, fast | Large-scale research |

### Profiling

| Tool | Focus | Usage |
|------|-------|-------|
| torch.profiler | PyTorch ops | Training optimization |
| NVIDIA Nsight | GPU details | Kernel optimization |
| memory_profiler | Python memory | Memory debugging |

## Best Practices

1. **Log early, log often**: Start with comprehensive logging
2. **Checkpoint frequently**: GPU time is expensive
3. **Monitor gradients**: Catch issues before they cascade
4. **Track hyperparameters**: Enable reproducibility
5. **Save artifacts**: Model weights, configs, code
6. **Use dashboards**: Visual monitoring catches issues
7. **Set alerts**: Automated issue detection
8. **Version control configs**: Reproducibility

## Further Reading

Checkpointing:
- [Checkpointing Overview](checkpointing/ReadMe.md)
- [State Dict Management](checkpointing/state-dict-management/ReadMe.md)
- [Distributed Checkpointing](checkpointing/distributed-checkpointing/ReadMe.md)
- [Checkpoint Sharding](checkpointing/checkpoint-sharding/ReadMe.md)

Logging:
- [Logging Overview](logging/ReadMe.md)
- [TensorBoard](logging/tensorboard/ReadMe.md)
- [Weights & Biases](logging/weights-and-biases/ReadMe.md)
- [MLflow Tracking](logging/mlflow-tracking/ReadMe.md)

Debugging:
- [Debugging Overview](debugging/ReadMe.md)
- [Memory Profiling](debugging/memory-profiling/ReadMe.md)
- [NaN Detection](debugging/nan-detection/ReadMe.md)
- [Gradient Monitoring](debugging/gradient-monitoring/ReadMe.md)
