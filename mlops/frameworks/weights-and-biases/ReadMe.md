# Weights and Biases

## Summary

Weights and Biases (W&B) is a managed platform for experiment tracking, model management, and ML collaboration. It provides rich visualization dashboards, hyperparameter sweep orchestration, and team collaboration features. W&B has become popular for deep learning projects due to its strong integration with PyTorch, TensorFlow, and Hugging Face.

Key points to remember:

- W&B Experiments tracks parameters, metrics, and artifacts with automatic visualizations
- Sweeps orchestrates hyperparameter tuning with multiple search strategies
- Artifacts provides dataset and model versioning with lineage tracking
- Reports enables sharing analysis and results with collaborators
- W&B is a managed service (cloud or enterprise); MLflow is the self-hosted alternative

## Core Features

### Experiment Tracking

Log experiments with minimal code:

```python
import wandb

# Initialize run
wandb.init(
    project="fraud-detection",
    config={
        "learning_rate": 0.01,
        "epochs": 100,
        "model": "transformer"
    }
)

# Training loop
for epoch in range(epochs):
    loss = train_epoch()
    val_accuracy = validate()

    # Log metrics
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "val_accuracy": val_accuracy
    })

# Finish run
wandb.finish()
```

W&B automatically captures:
- Git commit and diff
- System metrics (GPU usage, memory)
- Environment details
- Console output

### Rich Visualizations

W&B provides automatic visualizations:

- Training curves with smoothing and comparison
- Histograms for weight distributions
- Custom plots and charts
- Media logging (images, audio, video, 3D)

Log rich media:

```python
# Log images
images = [wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(samples)]
wandb.log({"samples": images})

# Log confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=labels,
    preds=predictions,
    class_names=class_names
)})

# Log table
table = wandb.Table(columns=["input", "prediction", "ground_truth"])
for inp, pred, gt in zip(inputs, predictions, ground_truths):
    table.add_data(inp, pred, gt)
wandb.log({"predictions": table})
```

### Framework Integrations

Deep integration with ML frameworks:

PyTorch Lightning:
```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="my-project")
trainer = Trainer(logger=logger)
```

Hugging Face:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    report_to="wandb",
    ...
)
```

Keras:
```python
from wandb.keras import WandbCallback

model.fit(X, y, callbacks=[WandbCallback()])
```

## Sweeps

Hyperparameter tuning orchestration:

```python
# Define sweep configuration
sweep_config = {
    "method": "bayes",  # or "grid", "random"
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "min": 0.0001,
            "max": 0.1,
            "distribution": "log_uniform_values"
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "dropout": {
            "min": 0.1,
            "max": 0.5
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Define training function
def train():
    wandb.init()
    config = wandb.config

    model = build_model(
        lr=config.learning_rate,
        dropout=config.dropout
    )

    for epoch in range(100):
        train_epoch(model, config.batch_size)
        accuracy = validate(model)
        wandb.log({"val_accuracy": accuracy})

# Run sweep agent
wandb.agent(sweep_id, train, count=50)
```

Sweep strategies:
- Grid search: Exhaustive parameter combinations
- Random search: Random sampling
- Bayesian optimization: Model-guided search
- Early termination: Stop underperforming runs

## Artifacts

Dataset and model versioning:

```python
# Log dataset artifact
artifact = wandb.Artifact('training-data', type='dataset')
artifact.add_dir('data/train')
wandb.log_artifact(artifact)

# Log model artifact
model_artifact = wandb.Artifact('fraud-model', type='model')
model_artifact.add_file('model.pt')
wandb.log_artifact(model_artifact)

# Use artifact in another run
artifact = wandb.use_artifact('training-data:latest')
data_dir = artifact.download()
```

Artifact features:
- Automatic versioning (v0, v1, v2...)
- Aliases (latest, production, staging)
- Lineage tracking (which run produced which artifact)
- Deduplication (unchanged files not re-uploaded)

## Model Registry

Production model management:

```python
# Link model to registry
run = wandb.init()
artifact = wandb.Artifact('fraud-model', type='model')
artifact.add_file('model.pt')
run.log_artifact(artifact)

# Link to model registry
run.link_artifact(artifact, 'model-registry/fraud-model')
```

Registry features:
- Version history
- Stage transitions (development, staging, production)
- Model cards and documentation
- Webhook integrations

## Reports

Share analysis and results:

Reports combine:
- Text and markdown documentation
- Embedded run comparisons
- Interactive visualizations
- Code snippets

Create reports from the W&B UI or programmatically:

```python
import wandb.apis.reports as wr

report = wr.Report(
    project="my-project",
    title="Training Results",
    description="Analysis of latest training runs"
)

report.blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="epoch", y="loss"),
            wr.LinePlot(x="epoch", y="accuracy")
        ]
    )
]

report.save()
```

## Team Collaboration

W&B is designed for teams:

Projects:
- Organize runs by project
- Project-level settings and defaults

Teams:
- Shared workspaces
- Access control
- Shared artifacts and reports

Service accounts:
- Automated logging from CI/CD
- Production monitoring

## Integration Patterns

### CI/CD Integration

Log from CI/CD pipelines:

```yaml
# GitHub Actions
- name: Train and log
  env:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  run: python train.py
```

### Production Monitoring

Log inference metrics:

```python
# Inference monitoring
wandb.init(project="production-monitoring", job_type="inference")

for batch in production_batches:
    predictions = model.predict(batch)
    wandb.log({
        "latency": measure_latency(),
        "prediction_distribution": wandb.Histogram(predictions)
    })
```

### MLflow Integration

Use W&B with MLflow:

```python
import mlflow
import wandb

# Log to both
wandb.init()
mlflow.start_run()

# W&B for visualization
wandb.log({"accuracy": 0.95})

# MLflow for model registry
mlflow.log_metric("accuracy", 0.95)
mlflow.pytorch.log_model(model, "model")
```

## Best Practices

### Project Organization

Structure projects consistently:

```
team/
  project-fraud-detection/
    experiment-runs
  project-recommender/
    experiment-runs
```

Use tags for filtering:

```python
wandb.init(
    tags=["baseline", "transformer", "v2"]
)
```

### Configuration Management

Use config files:

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

wandb.init(config=config)
```

Update config during runs:

```python
wandb.config.update({"learning_rate": new_lr})
```

### Efficient Logging

Log at appropriate frequency:

```python
# Log every N steps, not every step
if step % 100 == 0:
    wandb.log({"loss": loss}, step=step)

# Use summary for final metrics
wandb.run.summary["best_accuracy"] = best_accuracy
```

### Artifact Best Practices

Version datasets meaningfully:

```python
# Include metadata
artifact = wandb.Artifact(
    'training-data',
    type='dataset',
    description="Training data for Q1 2024",
    metadata={
        "source": "production_db",
        "date_range": "2024-01-01 to 2024-03-31",
        "num_samples": 1000000
    }
)
```

## Comparison with Alternatives

### vs MLflow

W&B advantages:
- Richer visualizations
- Better sweep orchestration
- Team collaboration features
- Managed service (no ops)

MLflow advantages:
- Open source, self-hostable
- Standard model format
- No vendor lock-in

### vs Neptune

Both are managed services with similar capabilities. W&B has stronger deep learning integration; Neptune has more flexible metadata structure.

### vs TensorBoard

TensorBoard is lighter weight and free. W&B provides collaboration, sweeps, and artifacts that TensorBoard lacks.

## Pricing and Deployment

### Pricing Tiers

- Free: Individual use, limited features
- Team: Team collaboration, more storage
- Enterprise: On-premises, SSO, advanced security

### Enterprise Deployment

W&B Server for on-premises:
- Full W&B features self-hosted
- Integration with internal infrastructure
- Data stays within organization

## Common Pitfalls

### Over-Logging

Logging too frequently or too much data increases costs and slows runs. Be selective about what and how often to log.

### Ignoring Artifacts

Not versioning datasets leads to reproducibility issues. Artifacts should be as carefully managed as code.

### Poor Project Structure

Dumping all runs into one project makes analysis difficult. Use projects and tags for organization.

### Not Using Sweeps

Manual hyperparameter tuning is inefficient. Sweeps systematize the search process.
