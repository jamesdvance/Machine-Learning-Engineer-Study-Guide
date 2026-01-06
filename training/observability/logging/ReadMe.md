# Logging and Experiment Tracking

## Summary

Experiment tracking systematically records training runs, including hyperparameters, metrics, artifacts, and code versions. Effective logging enables reproducibility, comparison between experiments, and debugging of training issues. Modern ML teams rely on logging platforms to manage the complexity of iterating on models at scale.

Key points to remember:

- Log hyperparameters, metrics, and artifacts consistently
- Enable comparison between experiments
- Track code and environment for reproducibility
- Use structured logging for programmatic access
- Log on main process only in distributed training
- Store model checkpoints and predictions
- Visualize training curves in real-time
- Tag and organize experiments for findability

## What to Log

### Essential Metrics

```python
# Per-step logging
metrics = {
    'train/loss': loss.item(),
    'train/learning_rate': scheduler.get_last_lr()[0],
    'train/grad_norm': grad_norm,
    'train/throughput': samples_per_second,
}

# Per-epoch logging
epoch_metrics = {
    'train/epoch_loss': epoch_loss,
    'val/loss': val_loss,
    'val/accuracy': val_accuracy,
}
```

### Hyperparameters

```python
config = {
    # Model
    'model_name': 'transformer',
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 1000,
    'max_steps': 100000,

    # Data
    'dataset': 'wikipedia',
    'seq_length': 512,
}
```

### Artifacts

```python
artifacts = [
    'model_checkpoint.pt',      # Model weights
    'config.yaml',              # Configuration
    'train_metrics.csv',        # Metrics history
    'predictions.json',         # Sample predictions
    'confusion_matrix.png',     # Visualizations
]
```

## Logging Frameworks Comparison

| Feature | TensorBoard | W&B | MLflow |
|---------|-------------|-----|--------|
| Hosting | Local/TensorBoard.dev | Cloud | Self-hosted/Cloud |
| Collaboration | Limited | Excellent | Good |
| Experiment comparison | Basic | Advanced | Good |
| Artifact storage | Limited | Built-in | Built-in |
| Model registry | No | Yes | Yes |
| Integration | PyTorch native | Extensive | Extensive |
| Cost | Free | Freemium | Open source |

## Basic Logging Setup

### Manual Logging

```python
import json
from datetime import datetime

class SimpleLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = []
        self.start_time = datetime.now()

    def log(self, metrics, step):
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)

    def save(self):
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f)

# Usage
logger = SimpleLogger('./logs')
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    logger.log({'loss': loss}, step)
logger.save()
```

### CSV Logging

```python
import csv

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.file = open(filename, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, metrics):
        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        self.file.close()

# Usage
logger = CSVLogger('metrics.csv', ['step', 'loss', 'accuracy'])
logger.log({'step': 1, 'loss': 0.5, 'accuracy': 0.8})
```

## Distributed Logging

### Log on Main Process Only

```python
import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# Only log on rank 0
if is_main_process():
    wandb.log(metrics)
    logger.info(f"Step {step}: loss={loss:.4f}")
```

### With Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with='wandb')
accelerator.init_trackers('my_project', config=config)

# Logs only on main process
accelerator.log(metrics, step=step)

accelerator.end_training()
```

### Gather Metrics

```python
import torch.distributed as dist

def gather_mean(tensor):
    """Gather tensor from all ranks and compute mean."""
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()

# Aggregate metrics before logging
loss = gather_mean(loss)
if is_main_process():
    wandb.log({'loss': loss.item()})
```

## Structured Logging

### Python Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info(f"Starting training with config: {config}")
logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
logger.warning(f"Gradient norm {grad_norm:.2f} exceeds threshold")
logger.error(f"NaN detected in loss at step {step}")
```

### JSON Structured Logging

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
        }
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
        return json.dumps(log_data)

# Usage
handler = logging.FileHandler('training.jsonl')
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Log with extra data
logger.info("Training step", extra={'metrics': {'loss': 0.5}})
```

## Best Practices

### 1. Log Frequency

```python
# Log training metrics every N steps
if step % log_every_n_steps == 0:
    wandb.log({'train/loss': loss}, step=step)

# Log validation at each epoch
for epoch in range(num_epochs):
    train(...)
    val_metrics = validate(...)
    wandb.log(val_metrics, step=epoch)
```

### 2. Organize Experiments

```python
# Use meaningful run names
wandb.init(
    project='language-model',
    name=f'llama-7b-lr{lr}-bs{batch_size}',
    tags=['baseline', 'llama', 'fp16'],
    group='hyperparameter-sweep',
)
```

### 3. Log Code and Config

```python
# Save configuration
wandb.config.update(config)

# Log code
wandb.run.log_code(".")

# Or save config file
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
wandb.save('config.yaml')
```

### 4. Checkpoint Logging

```python
# Log model artifacts
model_artifact = wandb.Artifact('model', type='model')
model_artifact.add_file('checkpoint.pt')
wandb.log_artifact(model_artifact)
```

## Real-time Monitoring

### Training Dashboard

```python
# Log for real-time visualization
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    metrics = {
        'loss': loss,
        'learning_rate': scheduler.get_last_lr()[0],
        'gpu_memory_used': torch.cuda.memory_allocated() / 1e9,
    }

    wandb.log(metrics, step=step)
```

### Alerts

```python
# Weights & Biases alerts
wandb.alert(
    title="Training complete",
    text=f"Final validation loss: {val_loss:.4f}",
    level=wandb.AlertLevel.INFO
)

# Alert on issues
if loss > threshold:
    wandb.alert(
        title="Loss spike detected",
        text=f"Loss {loss:.4f} at step {step}",
        level=wandb.AlertLevel.WARN
    )
```

## Framework Integration

### PyTorch Lightning

```python
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

loggers = [
    WandbLogger(project='my-project'),
    TensorBoardLogger('logs/')
]

trainer = pl.Trainer(logger=loggers)
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=100,
    report_to=['wandb', 'tensorboard'],
)
```

## Further Reading

- [TensorBoard](tensorboard/ReadMe.md): PyTorch-native visualization
- [Weights & Biases](weights-and-biases/ReadMe.md): Cloud experiment tracking
- [MLflow Tracking](mlflow-tracking/ReadMe.md): Open-source MLOps platform
