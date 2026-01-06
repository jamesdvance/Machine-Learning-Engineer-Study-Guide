# Weights & Biases

## Summary

Weights & Biases (W&B) is a cloud-based experiment tracking platform that provides comprehensive tooling for ML teams. It offers experiment tracking, hyperparameter optimization, model versioning, and team collaboration features. W&B has become a standard tool in ML workflows due to its rich visualization, ease of use, and extensive framework integration.

Key points to remember:

- Cloud-hosted with free tier for individuals
- Real-time experiment tracking and visualization
- Hyperparameter sweeps with W&B Sweeps
- Artifact versioning for models and datasets
- Team collaboration and reporting
- Extensive framework integrations
- Model and dataset registry
- Automatic environment and code logging

## Setup

### Installation

```bash
pip install wandb

# Login (one-time)
wandb login
```

### Basic Usage

```python
import wandb

# Initialize run
wandb.init(
    project="my-project",
    name="experiment-1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
)

# Training loop
for epoch in range(epochs):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})

# Finish run
wandb.finish()
```

## Configuration

### Run Initialization

```python
wandb.init(
    # Required
    project="language-model",

    # Optional but recommended
    name="llama-7b-finetune",
    notes="Fine-tuning on custom dataset",
    tags=["llama", "finetune", "v2"],
    group="hyperparameter-sweep",

    # Configuration
    config={
        "model": "llama-7b",
        "learning_rate": 1e-5,
        "batch_size": 8,
    },

    # Advanced options
    save_code=True,
    resume="allow",  # "must", "never", "allow"
)
```

### Config Management

```python
# Access config
lr = wandb.config.learning_rate

# Update config
wandb.config.update({"new_param": value})

# Config from argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

wandb.init(config=args)
```

## Logging Metrics

### Basic Logging

```python
# Log single metric
wandb.log({"loss": 0.5})

# Log multiple metrics
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
})

# Log with step
wandb.log({"loss": 0.5}, step=global_step)

# Commit controls whether step increments
wandb.log({"loss": 0.5}, commit=False)
wandb.log({"accuracy": 0.9}, commit=True)  # Same step
```

### Summary Metrics

```python
# Summary appears in run overview
wandb.run.summary["best_val_loss"] = best_val_loss
wandb.run.summary["best_epoch"] = best_epoch

# Or use define_metric for automatic tracking
wandb.define_metric("val/loss", summary="min")
wandb.define_metric("val/accuracy", summary="max")
```

### Custom X-Axis

```python
# Use epoch as x-axis instead of step
wandb.define_metric("epoch")
wandb.define_metric("val/*", step_metric="epoch")

wandb.log({"epoch": epoch, "val/loss": val_loss})
```

## Visualizations

### Images

```python
import wandb
import numpy as np

# Log single image
wandb.log({"image": wandb.Image(image_array)})

# With caption
wandb.log({"image": wandb.Image(image_array, caption="Sample")})

# Multiple images
wandb.log({"images": [wandb.Image(img) for img in images]})

# From PIL
from PIL import Image
wandb.log({"pil_image": wandb.Image(pil_image)})

# From matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
wandb.log({"plot": wandb.Image(fig)})
plt.close()
```

### Tables

```python
# Create table
table = wandb.Table(columns=["input", "prediction", "target"])
for inp, pred, target in zip(inputs, predictions, targets):
    table.add_data(inp, pred, target)

wandb.log({"predictions": table})
```

### Histograms

```python
# Log histogram
wandb.log({"gradients": wandb.Histogram(gradient_values)})

# Or use numpy histogram
wandb.log({"activations": wandb.Histogram(np_array=activations)})
```

### Custom Charts

```python
# Use plotly
import plotly.express as px

fig = px.scatter(df, x="x", y="y", color="label")
wandb.log({"scatter": fig})

# Confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    y_true=labels,
    preds=predictions,
    class_names=class_names
)})
```

## Artifacts

### Saving Artifacts

```python
# Save model checkpoint
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pt')
wandb.log_artifact(artifact)

# Save directory
artifact = wandb.Artifact('dataset', type='dataset')
artifact.add_dir('data/')
wandb.log_artifact(artifact)

# With metadata
artifact = wandb.Artifact(
    'model',
    type='model',
    metadata={"accuracy": 0.95, "epochs": 100}
)
```

### Loading Artifacts

```python
# Download artifact
artifact = wandb.use_artifact('user/project/model:latest')
artifact_dir = artifact.download()

# Load specific version
artifact = wandb.use_artifact('user/project/model:v3')
```

### Model Registry

```python
# Link to model registry
run = wandb.init()
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pt')
run.log_artifact(artifact)

# Link to registry
run.link_artifact(artifact, 'model-registry/production-model')
```

## Hyperparameter Sweeps

### Define Sweep

```python
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'hidden_size': {
            'distribution': 'int_uniform',
            'min': 128,
            'max': 512
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='my-project')
```

### Run Sweep

```python
def train():
    wandb.init()

    # Access sweep parameters
    config = wandb.config

    model = build_model(config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(10):
        loss = train_epoch(model, config.batch_size)
        wandb.log({"val_loss": loss})

# Run sweep agent
wandb.agent(sweep_id, train, count=50)
```

## Framework Integrations

### PyTorch Lightning

```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project='my-project',
    log_model='all'  # Log checkpoints
)

trainer = pl.Trainer(logger=wandb_logger)
```

### Hugging Face

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    report_to='wandb',
    run_name='my-run',
)

# With Trainer
from transformers import Trainer
trainer = Trainer(args=training_args, ...)
```

### Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with='wandb')
accelerator.init_trackers('my-project', config=config)

accelerator.log({"loss": loss}, step=step)
accelerator.end_training()
```

## Distributed Training

### Multi-GPU

```python
import torch.distributed as dist

# Only init on rank 0
if not dist.is_initialized() or dist.get_rank() == 0:
    wandb.init(project='my-project')

# Log only on rank 0
if wandb.run is not None:
    wandb.log(metrics)
```

### With Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with='wandb')

# Automatically handles distributed logging
accelerator.log(metrics, step=step)
```

## Alerts and Notifications

```python
# Send alert
wandb.alert(
    title="Training Complete",
    text=f"Final accuracy: {accuracy:.2%}",
    level=wandb.AlertLevel.INFO  # INFO, WARN, ERROR
)

# Alert on condition
if loss > threshold:
    wandb.alert(
        title="Loss Spike",
        text=f"Loss {loss:.4f} at step {step}",
        level=wandb.AlertLevel.WARN
    )
```

## Reports

```python
# Reports are created in the W&B UI
# But can reference runs programmatically

# Add notes to run
wandb.run.notes = "Experiment with new architecture"

# Tag runs for filtering
wandb.run.tags = ["production", "v2"]
```

## Best Practices

1. **Use meaningful names**: Include key config in run name
2. **Group related runs**: Use `group` parameter
3. **Tag consistently**: Filter runs by tags
4. **Log config**: Always log hyperparameters
5. **Use summaries**: Track best metrics automatically
6. **Version artifacts**: Use artifact versioning
7. **Save code**: Enable `save_code=True`
8. **Clean up**: Use `wandb.finish()` to close runs

## Offline Mode

```python
# Run offline (no internet)
import os
os.environ['WANDB_MODE'] = 'offline'

wandb.init(project='my-project')
# ... training ...

# Sync later
# wandb sync wandb/offline-run-*
```

## Troubleshooting

```python
# Debug mode
import os
os.environ['WANDB_DEBUG'] = 'true'

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Silent mode (no stdout)
os.environ['WANDB_SILENT'] = 'true'
```
