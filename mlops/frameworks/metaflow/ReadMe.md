# Metaflow

## Summary

Metaflow is a Python framework for building and managing data science workflows, originally developed at Netflix. It emphasizes developer productivity by making it easy to scale code from laptop to production without significant refactoring. Metaflow handles data management, versioning, and orchestration while keeping the programming model simple.

Key points to remember:

- Metaflow uses Python decorators to define flows, keeping code Pythonic and readable
- Automatic data versioning captures all artifacts and enables time-travel to any previous run
- Scaling from local to cloud requires minimal code changes
- Native AWS integration (with Kubernetes support) handles infrastructure automatically
- Focus on data scientist productivity rather than infrastructure concerns

## Core Concepts

### Flows and Steps

Metaflow organizes code into Flows containing Steps:

```python
from metaflow import FlowSpec, step

class TrainingFlow(FlowSpec):

    @step
    def start(self):
        self.data = load_data()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        self.features = preprocess(self.data)
        self.next(self.train)

    @step
    def train(self):
        self.model = train_model(self.features)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        self.metrics = evaluate(self.model, self.features)
        print(f"Accuracy: {self.metrics['accuracy']}")
        self.next(self.end)

    @step
    def end(self):
        print("Training complete")

if __name__ == '__main__':
    TrainingFlow()
```

Key concepts:
- `@step` decorator marks methods as flow steps
- `self.next()` defines step transitions
- Instance variables (self.x) are automatically persisted
- Flows run like normal Python scripts

### Parallelism and Branching

Metaflow supports parallel execution:

```python
class ParallelFlow(FlowSpec):

    @step
    def start(self):
        self.models = ['rf', 'xgb', 'lgb']
        self.next(self.train, foreach='models')

    @step
    def train(self):
        self.model_type = self.input
        self.model = train_model(self.model_type)
        self.score = evaluate(self.model)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = {inp.model_type: inp.score for inp in inputs}
        self.best = max(inputs, key=lambda x: x.score)
        self.next(self.end)

    @step
    def end(self):
        print(f"Best model: {self.best.model_type}")
```

`foreach` creates parallel branches; `join` collects results.

### Branching

Conditional branching:

```python
@step
def start(self):
    self.next(self.branch_a, self.branch_b)

@step
def branch_a(self):
    # One branch
    self.next(self.join)

@step
def branch_b(self):
    # Another branch
    self.next(self.join)

@step
def join(self, inputs):
    # Merge branches
    self.next(self.end)
```

## Data Management

### Automatic Versioning

Metaflow automatically versions all data:

```python
@step
def train(self):
    self.model = train_model(self.data)  # Automatically versioned
    self.metrics = {'accuracy': 0.95}     # Automatically versioned
    self.next(self.end)
```

Every run creates an immutable snapshot. Access historical data:

```python
from metaflow import Flow

# Get latest run
run = Flow('TrainingFlow').latest_run
print(run.data.metrics)

# Get specific run
run = Flow('TrainingFlow')['1234567890']
print(run.data.model)

# Iterate over all runs
for run in Flow('TrainingFlow').runs():
    print(run.id, run.data.metrics)
```

### Data Artifacts

Large data artifacts are handled efficiently:

```python
@step
def load_data(self):
    # Large DataFrame automatically stored
    self.df = pd.read_parquet('large_data.parquet')
    self.next(self.process)
```

Metaflow stores artifacts in S3 (or local filesystem) with content-addressed deduplication.

### Include and Exclude

Control what gets versioned:

```python
from metaflow import IncludeFile

class MyFlow(FlowSpec):
    config_file = IncludeFile('config', default='config.yaml')

    @step
    def start(self):
        config = yaml.safe_load(self.config_file)
        # config_file content is versioned with the flow
```

## Scaling

### Local to Cloud

Scale with decorators:

```python
from metaflow import FlowSpec, step, resources, batch

class ScalableFlow(FlowSpec):

    @resources(memory=16000, cpu=4)
    @step
    def preprocess(self):
        # Runs with 16GB RAM, 4 CPUs
        self.data = heavy_preprocessing()
        self.next(self.train)

    @batch(gpu=1, memory=32000)
    @step
    def train(self):
        # Runs on AWS Batch with GPU
        self.model = train_on_gpu(self.data)
        self.next(self.end)
```

No code changes required; decorators handle scaling.

### AWS Batch Integration

Run steps on AWS Batch:

```python
@batch(cpu=8, memory=64000, queue='gpu-queue', image='myimage:latest')
@step
def distributed_train(self):
    # Runs on AWS Batch
    ...
```

Configuration:

```bash
# Configure AWS Batch
metaflow configure aws

# Run on Batch
python flow.py run --with batch
```

### Kubernetes Integration

Run on Kubernetes:

```python
from metaflow import kubernetes

@kubernetes(cpu=4, memory=16000, image='myimage:latest')
@step
def train(self):
    ...
```

```bash
python flow.py run --with kubernetes
```

## Production Deployment

### Scheduling with Argo/Step Functions

Deploy flows as scheduled jobs:

```bash
# Deploy to AWS Step Functions
python flow.py step-functions create

# Deploy to Argo Workflows
python flow.py argo-workflows create
```

Scheduled execution:

```python
from metaflow import schedule

@schedule(daily=True)
class DailyRetrainingFlow(FlowSpec):
    ...
```

### Resume and Retry

Resume failed flows:

```bash
# Resume from failed step
python flow.py resume

# Resume specific run
python flow.py resume 1234567890
```

Failed steps can be fixed and resumed without re-running successful steps.

### Tagging and Namespaces

Organize runs with tags:

```bash
# Tag a run
python flow.py run --tag experiment:v2 --tag team:fraud

# Filter by tags
from metaflow import Flow
runs = Flow('MyFlow').runs('experiment:v2')
```

Namespaces isolate production from development:

```python
from metaflow import namespace

namespace('production')
# Only sees production runs
```

## Best Practices

### Parameter Management

Use Parameters for configurability:

```python
from metaflow import FlowSpec, step, Parameter

class ConfigurableFlow(FlowSpec):

    learning_rate = Parameter('lr', default=0.01)
    epochs = Parameter('epochs', default=100)
    data_path = Parameter('data', required=True)

    @step
    def start(self):
        print(f"Training with lr={self.learning_rate}")
        self.next(self.train)
```

Run with parameters:

```bash
python flow.py run --lr 0.001 --epochs 200 --data s3://bucket/data
```

### Environment Management

Manage dependencies with decorators:

```python
from metaflow import conda, conda_base

@conda_base(python='3.11')
class MyFlow(FlowSpec):

    @conda(libraries={'scikit-learn': '1.3.0', 'pandas': '2.0.0'})
    @step
    def train(self):
        import sklearn  # Correct version guaranteed
```

Or use Docker:

```python
@batch(image='myimage:v1.2.3')
@step
def train(self):
    ...
```

### Testing

Test flows locally:

```python
# Unit test steps
def test_preprocessing():
    flow = PreprocessingFlow()
    flow.start()
    assert flow.features is not None

# Integration test
# python flow.py run --no-pylint
```

Run subset of flow:

```bash
# Run specific steps
python flow.py run --start-at preprocess --end-at train
```

### Monitoring

Access run metadata:

```python
from metaflow import Flow, get_metadata

# Get run information
run = Flow('MyFlow').latest_run
print(run.created_at)
print(run.finished_at)
print(run.successful)

# Step-level information
for step in run.steps():
    for task in step.tasks():
        print(task.stdout)
        print(task.stderr)
```

## Comparison with Alternatives

### vs Kubeflow Pipelines

Metaflow advantages:
- Simpler programming model
- Lower operational overhead
- Better local development experience

Kubeflow advantages:
- Kubernetes-native
- Richer ecosystem (Katib, KServe)
- Better for large Kubernetes-first organizations

### vs Airflow

Metaflow advantages:
- Data versioning built-in
- Python-native (not DAG DSL)
- Better for ML-specific workflows

Airflow advantages:
- Broader adoption
- More scheduling flexibility
- Better for diverse pipeline types

### vs Prefect/Dagster

Similar philosophy with different trade-offs:
- Metaflow: AWS-native, Netflix-proven scale
- Prefect: Modern Pythonic, good cloud offering
- Dagster: Strong software engineering focus, asset-based

## Common Pitfalls

### Large Artifact Overhead

Storing very large artifacts in every step adds overhead. For huge datasets, consider external storage with references:

```python
@step
def load_data(self):
    self.data_path = 's3://bucket/data'  # Store path, not data
    self.next(self.process)

@step
def process(self):
    data = load_from_s3(self.data_path)  # Load when needed
```

### Ignoring Local Development

Metaflow shines for local development. Use local runs for debugging before scaling:

```bash
# Local run for debugging
python flow.py run

# Then scale
python flow.py run --with batch
```

### Over-Parallelization

Too many parallel branches can overwhelm resources:

```python
# Bad: 1000 parallel tasks
self.items = list(range(1000))
self.next(self.process, foreach='items')

# Better: batch items
self.batches = [items[i:i+100] for i in range(0, 1000, 100)]
self.next(self.process, foreach='batches')
```

### Not Leveraging Resume

Failed runs can be resumed. Design steps to be idempotent:

```python
@step
def train(self):
    # Idempotent: can resume safely
    if not os.path.exists(self.checkpoint_path):
        self.model = train_from_scratch()
    else:
        self.model = load_checkpoint()
```
