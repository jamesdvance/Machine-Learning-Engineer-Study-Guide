# Learning Rate Scheduling

## Summary

Learning rate scheduling adjusts the learning rate during training to improve convergence speed and final performance. Proper scheduling can significantly reduce training time (and thus cost) by enabling faster initial learning and fine-grained optimization near convergence. Understanding different schedules and their tradeoffs is essential for efficient training.

Key points to remember:

- Warmup prevents early instability with large learning rates
- Decay schedules reduce LR as training progresses
- Cosine annealing is widely used for transformers
- OneCycleLR can reduce training time
- Learning rate affects training efficiency directly
- Too high: instability; Too low: slow convergence
- Schedule should match training duration
- Warmup is critical for large batch training

## Common Schedules

### Constant Learning Rate

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# No scheduler - LR stays constant
```

### Linear Warmup + Linear Decay

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000
)

# LR: 0 -> max_lr (warmup) -> 0 (decay)
```

### Cosine Annealing

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100000,  # Total steps
    eta_min=1e-6   # Minimum LR
)

# LR follows cosine curve from max to min
```

### Cosine with Warmup

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000
)

# Linear warmup + cosine decay
```

### OneCycleLR

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=100000,
    pct_start=0.1,      # 10% warmup
    anneal_strategy='cos'
)

# Warmup -> peak -> decay (can train faster)
```

## Schedule Comparison

### Visual Comparison

```python
import matplotlib.pyplot as plt
import torch

def plot_schedules(total_steps=10000, warmup_steps=1000):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    schedules = {
        'Linear Warmup + Decay': lambda opt: get_linear_schedule_with_warmup(
            opt, warmup_steps, total_steps
        ),
        'Cosine Annealing': lambda opt: CosineAnnealingLR(
            opt, T_max=total_steps
        ),
        'OneCycleLR': lambda opt: OneCycleLR(
            opt, max_lr=1e-3, total_steps=total_steps
        ),
        'Step Decay': lambda opt: StepLR(opt, step_size=2500, gamma=0.5)
    }

    for ax, (name, scheduler_fn) in zip(axes.flat, schedules.items()):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = scheduler_fn(opt)

        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]['lr'])
            sched.step()

        ax.plot(lrs)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig('schedules.png')
```

### Characteristics

| Schedule | Warmup | Decay | Use Case |
|----------|--------|-------|----------|
| Constant | No | No | Baseline, simple |
| Linear | Yes | Linear | Standard training |
| Cosine | Optional | Smooth | Transformers |
| OneCycle | Yes | Aggressive | Fast training |
| Step | No | Sudden | Fine-tuning |

## Warmup Strategies

### Why Warmup

```
Without warmup:
- Large initial gradients cause instability
- Optimizer momentum builds incorrectly
- Loss may spike or diverge

With warmup:
- Gradual LR increase stabilizes training
- Allows optimizer to calibrate
- Critical for large batch training
```

### Warmup Duration

```python
# Rule of thumb: 1-10% of training
num_training_steps = 100000

# Short warmup
warmup_steps = int(0.01 * num_training_steps)  # 1%

# Standard warmup
warmup_steps = int(0.05 * num_training_steps)  # 5%

# Long warmup (large batch, unstable)
warmup_steps = int(0.10 * num_training_steps)  # 10%
```

### Warmup Types

```python
# Linear warmup (most common)
def linear_warmup(step, warmup_steps, max_lr):
    return max_lr * (step / warmup_steps)

# Exponential warmup
def exponential_warmup(step, warmup_steps, max_lr):
    return max_lr * (1 - math.exp(-5 * step / warmup_steps))

# Gradual warmup (Transformer paper)
def transformer_warmup(step, warmup_steps, d_model):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
```

## Training Efficiency

### Impact on Training Time

```python
def compare_training_efficiency():
    """Compare schedules on training efficiency."""

    results = {}

    for schedule_name, scheduler in schedulers.items():
        start_time = time.time()
        final_loss = train_with_scheduler(model, scheduler)
        training_time = time.time() - start_time

        results[schedule_name] = {
            'time': training_time,
            'final_loss': final_loss,
            'efficiency': final_loss / training_time
        }

    return results
```

### Optimal Schedule Selection

```
Goal: Fastest convergence
  -> OneCycleLR or aggressive cosine

Goal: Best final performance
  -> Cosine with long training

Goal: Fine-tuning
  -> Low constant LR or linear decay

Goal: Large batch training
  -> Linear warmup + any decay
```

## Framework Integration

### PyTorch Lightning

```python
class LitModel(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    warmup_steps=500,
    # or warmup_ratio=0.1,
    lr_scheduler_type='cosine',  # 'linear', 'cosine', 'polynomial'
    num_train_epochs=3,
)
```

## Custom Schedulers

### Warmup + Cosine + Cooldown

```python
class WarmupCosineWithCooldown:
    def __init__(self, optimizer, warmup_steps, total_steps, cooldown_steps,
                 max_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cooldown_steps = cooldown_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Warmup
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        elif self.step_count <= self.total_steps - self.cooldown_steps:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps - self.cooldown_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        else:
            # Cooldown
            lr = self.min_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### Learning Rate Finder

```python
def find_lr(model, train_loader, optimizer, start_lr=1e-7, end_lr=10, num_steps=100):
    """Find optimal learning rate range."""
    lrs = []
    losses = []

    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    lr = start_lr

    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break

        # Set LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train step
        loss = train_step(model, batch, optimizer)

        lrs.append(lr)
        losses.append(loss)

        # Increase LR
        lr *= lr_mult

    # Plot
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.savefig('lr_finder.png')

    # Suggest LR (where loss drops fastest)
    suggested_lr = lrs[losses.index(min(losses))] / 10
    return suggested_lr
```

## Best Practices

1. **Always use warmup**: Especially for large batches
2. **Match schedule to duration**: Cosine needs known steps
3. **Log learning rate**: Track in experiment logs
4. **Use warmup ratio**: More portable than steps
5. **Consider OneCycleLR**: Can reduce training time
6. **Run LR finder**: For new architectures/datasets
7. **Don't change mid-training**: Unless resuming
8. **Monitor loss curves**: Adjust if plateauing
