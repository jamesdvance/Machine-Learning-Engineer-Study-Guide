# ZenML

## Summary

ZenML is an open-source MLOps framework that provides a unified interface for building portable ML pipelines. It decouples pipeline logic from infrastructure, allowing the same pipeline code to run on different orchestrators (Airflow, Kubeflow, local) and integrate with various MLOps tools (MLflow, W&B, Seldon) through a plugin-based stack system.

Key points to remember:

- Stacks abstract infrastructure: switch between local, cloud, and enterprise environments without code changes
- Pipelines are defined with Python decorators, similar to other modern ML frameworks
- Integrations connect to existing MLOps tools rather than replacing them
- Artifact stores and model registries are pluggable components
- Focus on reproducibility and portability across environments

## Core Concepts

### Stacks

Stacks define the infrastructure components for running pipelines:

```python
from zenml import stack

# A stack combines components
# - Orchestrator: How pipelines run (local, Airflow, Kubeflow)
# - Artifact store: Where data is stored (local, S3, GCS)
# - Container registry: Where images are stored
# - Model registry: Where models are registered
# - Experiment tracker: Where metrics are logged
```

Stack configuration via CLI:

```bash
# Register components
zenml artifact-store register s3_store \
    --flavor=s3 \
    --path=s3://my-bucket/artifacts

zenml orchestrator register kubeflow_orch \
    --flavor=kubeflow \
    --kubernetes_context=my-cluster

# Create stack
zenml stack register production \
    --artifact-store s3_store \
    --orchestrator kubeflow_orch

# Set active stack
zenml stack set production
```

The same pipeline runs differently based on active stack.

### Pipelines and Steps

Define pipelines with decorators:

```python
from zenml import pipeline, step
from zenml.client import Client

@step
def load_data() -> pd.DataFrame:
    """Load and return training data."""
    return pd.read_csv("data/train.csv")

@step
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    return preprocess_features(data)

@step
def train_model(data: pd.DataFrame) -> sklearn.base.BaseEstimator:
    """Train a model."""
    X, y = data.drop('target', axis=1), data['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

@step
def evaluate(model: sklearn.base.BaseEstimator, data: pd.DataFrame) -> float:
    """Evaluate the model."""
    X, y = data.drop('target', axis=1), data['target']
    return model.score(X, y)

@pipeline
def training_pipeline():
    data = load_data()
    processed = preprocess(data)
    model = train_model(processed)
    score = evaluate(model, processed)
    return score
```

Run the pipeline:

```python
# Run locally (default stack)
training_pipeline()

# Or from CLI
# zenml pipeline run training_pipeline.py
```

### Artifacts

ZenML tracks all pipeline inputs and outputs:

```python
from zenml.client import Client

client = Client()

# Get latest pipeline run
run = client.get_pipeline("training_pipeline").last_run

# Access artifacts
model = run.steps["train_model"].output.load()
score = run.steps["evaluate"].output.load()
```

Artifacts are:
- Automatically serialized and stored in artifact store
- Versioned with each pipeline run
- Tracked with lineage information

### Materializers

Materializers handle artifact serialization:

```python
from zenml.materializers import BaseMaterializer

class MyCustomMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (MyCustomType,)

    def load(self, data_type):
        # Load from artifact store
        with self.artifact_store.open(self.uri, 'r') as f:
            return MyCustomType.load(f)

    def save(self, data):
        # Save to artifact store
        with self.artifact_store.open(self.uri, 'w') as f:
            data.save(f)
```

Built-in materializers handle common types (pandas, numpy, sklearn, pytorch).

## Integrations

### Experiment Tracking

Integrate with MLflow or W&B:

```python
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

@step(
    experiment_tracker="mlflow_tracker",
    settings={"experiment_tracker.mlflow": MLFlowExperimentTrackerSettings(nested=True)}
)
def train_model(data: pd.DataFrame) -> Model:
    import mlflow
    mlflow.log_param("n_estimators", 100)
    model = train(data)
    mlflow.log_metric("accuracy", evaluate(model))
    return model
```

### Model Registry

Register models with MLflow or custom registries:

```python
from zenml import pipeline, step, get_step_context
from zenml.client import Client

@step
def register_model(model: Model) -> None:
    """Register model in registry."""
    client = Client()
    model_registry = client.active_stack.model_registry

    model_registry.register_model(
        name="fraud-detector",
        model=model,
        metadata={"accuracy": 0.95}
    )
```

### Model Deployers

Deploy models through integrations:

```python
from zenml.integrations.seldon.steps import seldon_model_deployer_step

@pipeline
def deployment_pipeline():
    model = load_model()
    seldon_model_deployer_step(
        model=model,
        service_config=SeldonDeploymentConfig(replicas=3)
    )
```

Supported deployers:
- Seldon Core
- KServe
- BentoML
- MLflow

## Stack Components

### Orchestrators

Control how pipelines execute:

- **Local**: Default, runs in current process
- **Airflow**: DAG-based scheduling
- **Kubeflow**: Kubernetes-native
- **Vertex AI**: Google Cloud managed
- **SageMaker**: AWS managed
- **GitHub Actions**: CI/CD integration

Switch orchestrator:

```bash
zenml orchestrator register airflow_orch --flavor=airflow
zenml stack update --orchestrator airflow_orch
# Same pipeline now runs on Airflow
```

### Artifact Stores

Where artifacts are persisted:

- Local filesystem (development)
- S3 (AWS)
- GCS (GCP)
- Azure Blob Storage

### Container Registries

For containerized execution:

- Docker Hub
- ECR (AWS)
- GCR/Artifact Registry (GCP)
- ACR (Azure)

### Feature Stores

Integrate with feature stores:

```python
from zenml.integrations.feast import FeastFeatureStore

@step
def get_features(entity_ids: List[str]) -> pd.DataFrame:
    fs = FeastFeatureStore()
    return fs.get_online_features(
        entity_ids=entity_ids,
        features=["user:age", "user:income"]
    )
```

## Pipeline Patterns

### Caching

Steps are cached by default:

```python
@step
def expensive_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # Only runs if inputs change
    return process(data)

# Disable caching for specific step
@step(enable_cache=False)
def always_run_step():
    ...
```

### Parameterization

Configure pipelines with parameters:

```python
@pipeline
def training_pipeline(epochs: int = 100, lr: float = 0.01):
    data = load_data()
    model = train_model(data, epochs=epochs, lr=lr)

# Run with different parameters
training_pipeline(epochs=200, lr=0.001)
```

### Conditional Execution

Control flow in pipelines:

```python
@pipeline
def conditional_pipeline():
    data = load_data()
    model = train_model(data)
    score = evaluate(model)

    # Conditional deployment
    if score > 0.9:
        deploy_model(model)
```

### Parallel Execution

Parallelize independent steps:

```python
@pipeline
def parallel_pipeline():
    data = load_data()

    # These run in parallel
    model_a = train_model_a(data)
    model_b = train_model_b(data)

    # Wait for both
    ensemble = combine_models(model_a, model_b)
```

## Model Control Plane

ZenML provides model management:

```python
from zenml import Model

# Define model
model = Model(
    name="fraud-detector",
    version="1.0.0",
    description="Fraud detection model"
)

@pipeline(model=model)
def training_pipeline():
    ...

# Access model versions
from zenml.client import Client
client = Client()
model = client.get_model("fraud-detector")
for version in model.versions:
    print(version.name, version.metrics)
```

## Best Practices

### Stack Design

Design stacks for different environments:

```bash
# Development stack (local)
zenml stack register dev \
    --orchestrator local \
    --artifact-store local

# Staging stack (cloud)
zenml stack register staging \
    --orchestrator kubeflow \
    --artifact-store s3_staging

# Production stack
zenml stack register production \
    --orchestrator kubeflow \
    --artifact-store s3_prod \
    --model-registry mlflow
```

### Step Granularity

Balance step size:

```python
# Too granular: excessive overhead
@step
def step1(): ...
@step
def step2(): ...
# ... 50 steps

# Too coarse: no visibility
@step
def do_everything():
    load()
    preprocess()
    train()
    evaluate()

# Right balance: logical units
@step
def load_and_validate(): ...
@step
def preprocess(): ...
@step
def train(): ...
@step
def evaluate_and_report(): ...
```

### Testing

Test pipelines locally:

```python
def test_training_pipeline():
    # Use test stack
    from zenml.client import Client
    Client().activate_stack("test")

    # Run pipeline
    result = training_pipeline()

    # Assert on outputs
    assert result.steps["evaluate"].output.load() > 0.8
```

## Comparison with Alternatives

### vs Kubeflow Pipelines

ZenML advantages:
- Simpler local development
- Stack abstraction for portability
- Easier integration with existing tools

Kubeflow advantages:
- More mature Kubernetes integration
- Katib for hyperparameter tuning
- KServe for serving

### vs MLflow

Different focus:
- MLflow: Experiment tracking and model packaging
- ZenML: Pipeline orchestration and infrastructure abstraction

They integrate: use MLflow for tracking within ZenML pipelines.

### vs Kedro

ZenML advantages:
- Cloud-native orchestration
- Stack-based infrastructure abstraction

Kedro advantages:
- Stronger data engineering focus
- Data catalog concept
- Better for data pipelines

## Common Pitfalls

### Ignoring Stack Design

Poor stack design leads to environment inconsistencies. Invest in proper stack configuration for each environment.

### Over-Engineering Steps

Too many small steps create overhead. Group related operations into cohesive steps.

### Not Versioning Stacks

Stack configurations should be version-controlled:

```bash
zenml stack export production > stacks/production.yaml
```

### Forgetting Local Testing

Always test on local stack before cloud deployment:

```bash
zenml stack set local
python pipeline.py  # Test locally first
```
