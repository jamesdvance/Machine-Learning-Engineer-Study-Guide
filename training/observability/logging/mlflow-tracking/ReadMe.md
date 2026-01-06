# MLflow Tracking

## Summary

MLflow is an open-source platform for managing the ML lifecycle, with MLflow Tracking being its experiment tracking component. MLflow can be self-hosted or used with managed services, making it popular in enterprise environments. It provides experiment tracking, model registry, and deployment capabilities in a single platform.

Key points to remember:

- Open-source and self-hostable
- Tracks experiments, parameters, metrics, artifacts
- Model registry for versioning and staging
- REST API for programmatic access
- Backend storage options (file, SQL, cloud)
- Integrates with major ML frameworks
- Supports custom model flavors
- Enterprise-friendly with access controls

## Setup

### Installation

```bash
pip install mlflow

# Start tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Or use local file storage
mlflow ui
```

### Basic Usage

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my-experiment")

# Start run
with mlflow.start_run(run_name="my-run"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    # Training
    for epoch in range(epochs):
        loss = train_epoch()
        mlflow.log_metric("loss", loss, step=epoch)

    # Log artifacts
    mlflow.log_artifact("model.pt")
```

## Logging

### Parameters

```python
import mlflow

with mlflow.start_run():
    # Single parameter
    mlflow.log_param("learning_rate", 0.001)

    # Multiple parameters
    mlflow.log_params({
        "model": "transformer",
        "layers": 12,
        "hidden_size": 768
    })
```

### Metrics

```python
with mlflow.start_run():
    # Single metric
    mlflow.log_metric("loss", 0.5)

    # With step
    for step in range(1000):
        mlflow.log_metric("loss", loss, step=step)

    # Multiple metrics
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy
    })
```

### Artifacts

```python
with mlflow.start_run():
    # Log single file
    mlflow.log_artifact("model.pt")

    # Log directory
    mlflow.log_artifacts("output_dir/")

    # Log with subdirectory
    mlflow.log_artifact("model.pt", artifact_path="models")

    # Log dict as JSON
    mlflow.log_dict(config_dict, "config.json")

    # Log figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    mlflow.log_figure(fig, "plot.png")
```

### Tags

```python
with mlflow.start_run():
    # Set tags
    mlflow.set_tag("model_type", "transformer")
    mlflow.set_tags({
        "team": "nlp",
        "priority": "high"
    })
```

## Model Logging

### PyTorch Models

```python
import mlflow.pytorch

with mlflow.start_run():
    # Train model
    model = train_model()

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # With signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(input_data, model(input_data))
    mlflow.pytorch.log_model(model, "model", signature=signature)
```

### Sklearn Models

```python
import mlflow.sklearn

with mlflow.start_run():
    model = train_sklearn_model()
    mlflow.sklearn.log_model(model, "model")
```

### Custom Models

```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        self.model = torch.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model(model_input)

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModel(),
        artifacts={"model": "model.pt"},
        conda_env="conda.yaml"
    )
```

## Model Registry

### Register Model

```python
# During logging
with mlflow.start_run():
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="my-model"
    )

# After logging
result = mlflow.register_model(
    "runs:/run_id/model",
    "my-model"
)
```

### Model Stages

```python
from mlflow import MlflowClient

client = MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)
```

### Load Registered Model

```python
import mlflow

# Load by stage
model = mlflow.pyfunc.load_model("models:/my-model/Production")

# Load by version
model = mlflow.pyfunc.load_model("models:/my-model/1")
```

## Autologging

### PyTorch Lightning

```python
import mlflow

mlflow.pytorch.autolog()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model)

# Automatically logs:
# - Parameters from model and trainer
# - Training and validation metrics
# - Model checkpoints
```

### Sklearn

```python
import mlflow

mlflow.sklearn.autolog()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Automatically logs:
# - Hyperparameters
# - Metrics (accuracy, etc.)
# - Feature importances
# - Model artifact
```

### Transformers

```python
import mlflow

mlflow.transformers.autolog()

trainer = Trainer(model=model, args=training_args)
trainer.train()
```

## Experiment Organization

### Create Experiment

```python
import mlflow

# Create experiment
experiment_id = mlflow.create_experiment(
    "my-experiment",
    artifact_location="s3://bucket/artifacts",
    tags={"team": "ml"}
)

# Set active experiment
mlflow.set_experiment("my-experiment")
```

### Search Runs

```python
from mlflow import MlflowClient

client = MlflowClient()

# Search by metric
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

# Search by parameter
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="params.learning_rate = '0.001'"
)
```

### Compare Runs

```python
# Using pandas
import mlflow

runs = mlflow.search_runs(experiment_names=["my-experiment"])
print(runs[["params.lr", "metrics.accuracy"]])
```

## Server Configuration

### Backend Stores

```bash
# SQLite (local)
mlflow server --backend-store-uri sqlite:///mlflow.db

# PostgreSQL
mlflow server --backend-store-uri postgresql://user:pass@host/db

# MySQL
mlflow server --backend-store-uri mysql://user:pass@host/db
```

### Artifact Stores

```bash
# Local
mlflow server --default-artifact-root ./artifacts

# S3
mlflow server --default-artifact-root s3://bucket/artifacts

# GCS
mlflow server --default-artifact-root gs://bucket/artifacts

# Azure Blob
mlflow server --default-artifact-root wasbs://container@account.blob.core.windows.net/artifacts
```

## Distributed Training

### Multi-GPU Logging

```python
import torch.distributed as dist

# Only log on rank 0
if not dist.is_initialized() or dist.get_rank() == 0:
    with mlflow.start_run():
        mlflow.log_params(params)

        for step, loss in enumerate(losses):
            mlflow.log_metric("loss", loss, step=step)
```

### With Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()

if accelerator.is_main_process:
    mlflow.start_run()

# Training...

if accelerator.is_main_process:
    mlflow.log_metrics(metrics)
    mlflow.end_run()
```

## API Usage

### Python API

```python
from mlflow import MlflowClient

client = MlflowClient()

# Get run info
run = client.get_run(run_id)
print(run.info.status)
print(run.data.metrics)
print(run.data.params)

# Update run
client.set_tag(run_id, "status", "reviewed")

# Download artifacts
client.download_artifacts(run_id, "model", ".")
```

### REST API

```bash
# Get experiment
curl http://localhost:5000/api/2.0/mlflow/experiments/get?experiment_id=0

# Search runs
curl -X POST http://localhost:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}'
```

## Framework Integration

### PyTorch Lightning

```python
from pytorch_lightning.loggers import MLFlowLogger

mlflow_logger = MLFlowLogger(
    experiment_name="my-experiment",
    tracking_uri="http://localhost:5000"
)

trainer = pl.Trainer(logger=mlflow_logger)
```

### Hugging Face

```python
# Set environment variable
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    report_to="mlflow"
)
```

## Best Practices

1. **Use experiments**: Organize related runs
2. **Set tracking URI**: Configure server location
3. **Log parameters first**: Before training starts
4. **Use autologging**: When available
5. **Register production models**: Use model registry
6. **Version artifacts**: Track model lineage
7. **Tag runs**: For filtering and organization
8. **Use context manager**: `with mlflow.start_run()`

## Comparison with W&B

| Feature | MLflow | W&B |
|---------|--------|-----|
| Hosting | Self-hosted/managed | Cloud |
| Cost | Free (self-hosted) | Freemium |
| Model registry | Built-in | Yes |
| Collaboration | Basic | Advanced |
| Visualization | Basic | Advanced |
| Sweeps | No built-in | Yes |
| Artifacts | Yes | Yes |
| Enterprise | Databricks | Enterprise tier |
