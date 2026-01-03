# Kubeflow

## Summary

Kubeflow is a Kubernetes-native platform for developing, deploying, and managing machine learning workflows at scale. It provides components for notebook development, pipeline orchestration, model training, hyperparameter tuning, and model serving, all running on Kubernetes infrastructure.

Key points to remember:

- Kubeflow runs entirely on Kubernetes, leveraging its scaling and orchestration capabilities
- Kubeflow Pipelines define reproducible ML workflows as directed acyclic graphs
- Katib provides automated hyperparameter tuning with multiple search algorithms
- KServe (formerly KFServing) handles model serving with autoscaling and canary deployments
- Kubeflow is powerful but complex; consider whether your scale justifies the operational overhead

## Core Components

### Kubeflow Notebooks

Jupyter notebooks running on Kubernetes:

- Spawner for creating notebook servers on demand
- GPU access through Kubernetes device plugins
- Persistent storage for notebooks and data
- Pre-configured images for common frameworks

Notebooks provide an interactive development environment within the Kubeflow ecosystem, with direct access to cluster resources.

### Kubeflow Pipelines

Pipeline orchestration for ML workflows:

```python
from kfp import dsl

@dsl.component
def preprocess_data(input_path: str) -> str:
    # Preprocessing logic
    return output_path

@dsl.component
def train_model(data_path: str, epochs: int) -> str:
    # Training logic
    return model_path

@dsl.pipeline(name="training-pipeline")
def training_pipeline(input_path: str, epochs: int = 100):
    preprocess_task = preprocess_data(input_path=input_path)
    train_task = train_model(
        data_path=preprocess_task.output,
        epochs=epochs
    )
```

Pipeline features:
- Visual pipeline editor and monitoring
- Artifact tracking and lineage
- Caching of component outputs
- Scheduling and triggers
- Experiment organization

Pipelines compile to Argo Workflows, running as Kubernetes pods.

### Katib

Hyperparameter tuning and neural architecture search:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: random-search
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: random
  parallelTrialCount: 3
  maxTrialCount: 12
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.001"
        max: "0.1"
    - name: batch_size
      parameterType: int
      feasibleSpace:
        min: "16"
        max: "128"
  trialTemplate:
    primaryContainerName: training-container
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training-container
                image: mytraining:latest
                command:
                  - python
                  - train.py
                  - --lr=${trialParameters.learning_rate}
                  - --batch=${trialParameters.batch_size}
```

Supported algorithms:
- Random search
- Grid search
- Bayesian optimization
- Hyperband
- Neural architecture search (NAS)

Katib runs trials as Kubernetes jobs, enabling parallel execution on cluster resources.

### Training Operators

Distributed training operators:

- TFJob: TensorFlow distributed training
- PyTorchJob: PyTorch distributed training
- MPIJob: MPI-based training (Horovod)
- XGBoostJob: XGBoost distributed training

Example PyTorchJob:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch-training:latest
              resources:
                limits:
                  nvidia.com/gpu: 1
    Worker:
      replicas: 3
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch-training:latest
              resources:
                limits:
                  nvidia.com/gpu: 1
```

Training operators handle:
- Pod scheduling and coordination
- Service discovery between workers
- Failure recovery
- Resource management

### KServe

Model serving infrastructure:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detector
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: s3://models/fraud-detector/v1
```

KServe features:
- Autoscaling based on traffic (including scale to zero)
- Canary deployments with traffic splitting
- Pre/post processing transformers
- Model explanation
- Multiple serving runtimes (TensorFlow, PyTorch, sklearn, XGBoost)

KServe provides production-grade serving with minimal configuration.

## Architecture

### Kubernetes Foundation

Kubeflow relies on Kubernetes primitives:

- Pods for compute
- Services for networking
- Persistent Volumes for storage
- ConfigMaps and Secrets for configuration
- Custom Resources for ML-specific objects

Understanding Kubernetes is prerequisite for operating Kubeflow.

### Multi-tenancy

Kubeflow supports multi-tenant deployments:

- Namespace isolation per team or project
- RBAC for access control
- Resource quotas per namespace
- Istio for network policies

Multi-tenancy enables shared clusters for multiple teams while maintaining isolation.

### Storage Integration

Kubeflow integrates with various storage backends:

- Persistent volumes for notebooks and artifacts
- Object storage (S3, GCS, MinIO) for datasets and models
- NFS for shared storage

Storage configuration depends on the deployment environment.

## Deployment Options

### Managed Offerings

Cloud-managed Kubeflow:

AWS:
- Amazon SageMaker Pipelines (partial compatibility)
- Self-managed on EKS

GCP:
- Vertex AI Pipelines (Kubeflow Pipelines compatible)
- Full Kubeflow on GKE

Azure:
- Self-managed on AKS
- Integration with Azure ML

Managed offerings reduce operational burden but may limit flexibility.

### Self-Managed Deployment

Install Kubeflow on your Kubernetes cluster:

```bash
# Using manifests
kustomize build deployment/overlays/production | kubectl apply -f -

# Using installer
./kfctl apply -V -f kfctl_config.yaml
```

Self-managed deployment requires:
- Kubernetes cluster (1.21+)
- Istio for service mesh
- Cert-manager for TLS
- Object storage
- Database for metadata

Significant operational expertise is required.

## Pipeline Development

### Component Development

Components are containerized functions:

```python
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(base_image="python:3.11")
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    epochs: int = 100
):
    import pickle

    # Load data
    with open(training_data.path, 'rb') as f:
        X, y = pickle.load(f)

    # Train
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Save model
    with open(model.path, 'wb') as f:
        pickle.dump(clf, f)
```

Components can also be defined from container specifications:

```python
train_op = dsl.ContainerOp(
    name='train',
    image='mytraining:latest',
    command=['python', 'train.py'],
    arguments=['--epochs', epochs],
    file_outputs={'model': '/output/model.pkl'}
)
```

### Pipeline Patterns

Common pipeline patterns:

Sequential execution:
```python
@dsl.pipeline
def sequential_pipeline():
    step1 = component_a()
    step2 = component_b(step1.output)
    step3 = component_c(step2.output)
```

Parallel execution:
```python
@dsl.pipeline
def parallel_pipeline():
    prep = preprocess()
    model_a = train_model_a(prep.output)
    model_b = train_model_b(prep.output)
    ensemble = combine_models(model_a.output, model_b.output)
```

Conditional execution:
```python
@dsl.pipeline
def conditional_pipeline():
    eval_result = evaluate()
    with dsl.Condition(eval_result.output > 0.9):
        deploy_model()
```

### Caching

Component caching avoids redundant computation:

```python
@dsl.component
def expensive_preprocessing(data: Input[Dataset]) -> Output[Dataset]:
    # Result is cached based on inputs
    ...
```

Caching is based on component inputs; identical inputs reuse cached outputs.

## Best Practices

### Start Small

Begin with simple pipelines:
1. Start with linear pipelines
2. Add parallelism as needed
3. Introduce conditions and loops later

### Resource Management

Specify resource requirements:

```python
@dsl.component
def gpu_training(...):
    ...

gpu_training().set_gpu_limit(1)
gpu_training().set_memory_limit('16G')
gpu_training().set_cpu_limit('4')
```

Resource specifications enable Kubernetes scheduling and prevent resource contention.

### Error Handling

Handle failures gracefully:

```python
@dsl.component
def robust_component():
    try:
        # Main logic
        ...
    except Exception as e:
        # Log error
        # Clean up resources
        raise
```

Configure retry policies for transient failures.

### Testing

Test components and pipelines:

- Unit test component logic independently
- Test pipelines in development namespace
- Validate outputs before production promotion

## Comparison with Alternatives

### vs Airflow

Airflow is a general workflow orchestrator; Kubeflow is ML-specific:

Kubeflow advantages:
- Native Kubernetes integration
- ML-specific components (training operators, serving)
- Hyperparameter tuning

Airflow advantages:
- Broader ecosystem
- Simpler operations (no Kubernetes required)
- Better for ETL-heavy workflows

### vs MLflow

MLflow is complementary to Kubeflow:
- Use MLflow for experiment tracking within Kubeflow pipelines
- MLflow for model packaging, Kubeflow for orchestration

### vs Vertex AI / SageMaker

Managed services are simpler but less flexible:
- Use managed services for faster time-to-production
- Use Kubeflow for maximum control and multi-cloud

## Common Pitfalls

### Underestimating Complexity

Kubeflow requires Kubernetes expertise. Teams without Kubernetes experience face steep learning curves.

### Over-Engineering Pipelines

Complex pipelines become hard to maintain. Start simple and add complexity only when justified.

### Ignoring Resource Costs

Kubernetes clusters are expensive. Monitor resource usage and clean up unused experiments.

### Poor Component Design

Components should be focused and reusable. Avoid monolithic components that do too much.
