# MLflow

## Summary

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four core components: Tracking for experiments, Projects for reproducible runs, Models for packaging and deployment, and Registry for model versioning. MLflow has become the de facto standard for experiment tracking in the Python ML ecosystem.

Key points to remember:

- MLflow Tracking logs parameters, metrics, and artifacts from training runs
- The Model format standardizes packaging across frameworks (PyTorch, TensorFlow, sklearn, etc.)
- Model Registry provides versioning and stage management for production deployment
- MLflow can be self-hosted or used as a managed service (Databricks, cloud providers)
- Autologging reduces boilerplate for common frameworks

## Core Components

### MLflow Tracking

Tracking records experiment data during training:

```python
import mlflow

mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)

    # Train model
    model = train(...)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1", 0.93)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

Tracking captures:
- Parameters: Hyperparameters and configuration values
- Metrics: Performance measurements, can be logged over time (steps)
- Artifacts: Files like models, plots, data samples
- Tags: Metadata for organization and filtering
- Source: Git commit, entry point, user

The Tracking UI provides visualization, comparison, and search across experiments.

### MLflow Projects

Projects package code for reproducible execution:

```yaml
# MLproject
name: fraud-detection

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.01}
      epochs: {type: int, default: 100}
    command: "python train.py --lr {learning_rate} --epochs {epochs}"

  validate:
    parameters:
      model_path: path
    command: "python validate.py --model {model_path}"
```

Run projects from Git repositories or local directories:

```bash
mlflow run git@github.com:org/project.git -P learning_rate=0.001

mlflow run . -P epochs=200
```

Projects ensure reproducibility by capturing:
- Code version (Git commit)
- Environment specification (conda, Docker)
- Entry points and parameters

### MLflow Models

Models standardize packaging across frameworks:

```python
# Save model
mlflow.pytorch.save_model(model, "model")

# Load model
loaded = mlflow.pytorch.load_model("model")

# Serve model
mlflow models serve -m model -p 5000
```

Model flavors support multiple frameworks:
- `mlflow.pytorch`
- `mlflow.tensorflow`
- `mlflow.sklearn`
- `mlflow.xgboost`
- `mlflow.transformers`
- `mlflow.pyfunc` (generic Python functions)

Each flavor includes framework-specific loading logic while maintaining a common interface.

Model signature documents input/output schema:

```python
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### MLflow Model Registry

Registry provides centralized model management:

```python
# Register model
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="fraud-detector"
)

# Transition stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud-detector",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
```

Registry features:
- Version management with automatic versioning
- Stage transitions (Staging, Production, Archived)
- Descriptions and annotations
- Lineage tracking back to experiments
- Webhooks for automation

## Autologging

Reduce boilerplate with framework autologging:

```python
import mlflow

mlflow.autolog()

# Training code - MLflow automatically logs:
# - Hyperparameters
# - Metrics
# - Model artifacts
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

Supported frameworks:
- scikit-learn
- PyTorch (via Lightning)
- TensorFlow/Keras
- XGBoost, LightGBM, CatBoost
- Spark MLlib
- Transformers (Hugging Face)

Autologging captures framework-specific information automatically but can be supplemented with manual logging.

## Tracking Server

### Local Tracking

By default, MLflow tracks to local files:

```
mlruns/
  0/  # Default experiment
    abc123/  # Run ID
      params/
      metrics/
      artifacts/
      meta.yaml
```

Suitable for individual experimentation but not team collaboration.

### Remote Tracking Server

Deploy a central tracking server for teams:

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@host/mlflow \
    --default-artifact-root s3://bucket/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Backend stores:
- SQLite (simple, not recommended for teams)
- PostgreSQL, MySQL (production-ready)

Artifact stores:
- Local filesystem
- S3, GCS, Azure Blob
- HDFS

Configure client to use remote server:

```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Managed Services

Cloud-managed MLflow options:

Databricks:
- Native MLflow integration
- Managed tracking and registry
- Unity Catalog integration

Azure ML:
- MLflow-compatible tracking
- Integrated with Azure ecosystem

Amazon SageMaker:
- MLflow tracking server
- Integration with SageMaker training

Managed services reduce operational burden but may have higher costs.

## Model Deployment

### MLflow Model Serving

Built-in serving for quick deployment:

```bash
# Serve locally
mlflow models serve -m runs:/abc123/model -p 5000

# Build Docker container
mlflow models build-docker -m runs:/abc123/model -n mymodel

# Generate predictions
curl -X POST http://localhost:5000/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1, 2, 3]]}'
```

### Integration with Serving Platforms

MLflow models export to various formats:

```python
# Export to ONNX
mlflow.onnx.save_model(onnx_model, "model")

# Export to TensorFlow SavedModel
mlflow.tensorflow.save_model(tf_model, "model")

# Use with SageMaker
mlflow.sagemaker.deploy(
    model_uri="models:/mymodel/Production",
    app_name="mymodel",
    region_name="us-east-1"
)
```

## Best Practices

### Experiment Organization

Structure experiments logically:

```python
# Use meaningful experiment names
mlflow.set_experiment("fraud-detection/v2-transformer")

# Use tags for filtering
with mlflow.start_run(tags={"team": "fraud", "type": "experiment"}):
    ...

# Name runs for important experiments
with mlflow.start_run(run_name="baseline-model"):
    ...
```

### Parameter Logging

Log all relevant parameters:

```python
# Log configuration dictionaries
mlflow.log_params({
    "model_type": "transformer",
    "hidden_size": 256,
    "num_layers": 4,
    "dropout": 0.1
})

# Log from config files
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
mlflow.log_params(config)
```

### Metric Logging

Log metrics at appropriate granularity:

```python
# Log final metrics
mlflow.log_metric("test_accuracy", 0.95)

# Log training curves
for epoch in range(epochs):
    loss = train_epoch(...)
    mlflow.log_metric("train_loss", loss, step=epoch)
```

### Artifact Management

Log meaningful artifacts:

```python
# Plots and visualizations
mlflow.log_artifact("plots/confusion_matrix.png")

# Model cards and documentation
mlflow.log_artifact("model_card.md")

# Evaluation results
mlflow.log_artifact("evaluation_results.json")

# Log directory
mlflow.log_artifacts("outputs/", artifact_path="outputs")
```

### Registry Workflow

Formalize promotion process:

1. Train and log model to tracking
2. Register promising models to registry
3. Validate in staging environment
4. Promote to production after validation
5. Archive old production versions

Use webhooks to trigger validation pipelines on stage transitions.

## Comparison with Alternatives

### vs Weights and Biases

W&B advantages:
- Richer visualization and dashboards
- Better team collaboration features
- Automatic hyperparameter importance

MLflow advantages:
- Open source, self-hostable
- Standard model format
- Integrated registry

### vs Neptune

Neptune advantages:
- Flexible metadata structure
- Strong notebook integration

MLflow advantages:
- Broader adoption
- Model deployment features

### vs Kubeflow

Kubeflow is an orchestration platform; MLflow is tracking and packaging. They are complementary: use MLflow for tracking within Kubeflow pipelines.

## Common Pitfalls

### Tracking Everything

Logging too much creates noise. Focus on:
- Key hyperparameters that vary
- Primary evaluation metrics
- Artifacts needed for reproduction

### Ignoring Model Signatures

Models without signatures may fail at serving time. Always log signatures:

```python
signature = infer_signature(X, model.predict(X))
mlflow.log_model(model, "model", signature=signature)
```

### Manual Stage Management

Rely on automation, not manual stage transitions:

```python
# Automate promotion based on validation
if validation_passed:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
```

### Not Versioning Tracking Server

Tracking server configurations should be version-controlled and reproducible, just like application code.
