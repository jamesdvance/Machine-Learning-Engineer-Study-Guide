# Prefect

## Summary

Prefect is a modern workflow orchestration framework that treats Python functions as first-class workflow primitives. Unlike older orchestration tools that require developers to learn a domain-specific language or define rigid DAG structures upfront, Prefect lets you turn any Python function into a managed, observable workflow with a simple decorator. The framework originated as an alternative to Apache Airflow, and its second and third major versions (Prefect 2 and Prefect 3) represent a significant philosophical shift toward a code-first, infrastructure-flexible approach to pipeline orchestration.

For ML engineers, Prefect is particularly relevant because ML pipelines are inherently dynamic: feature engineering branches change based on data characteristics, hyperparameter sweeps generate variable numbers of training runs, and model evaluation gates determine whether downstream deployment steps execute at all. Prefect handles these patterns naturally because it does not require you to declare the execution graph before runtime.

Key points to remember:

- The `@flow` and `@task` decorators convert ordinary Python functions into orchestrated workflows with automatic state tracking, retries, and observability
- No DAG definition boilerplate: the execution graph is determined at runtime from your actual Python control flow
- Deployments decouple flow code from infrastructure, enabling the same pipeline to run locally, on Docker, on Kubernetes, or on serverless cloud infrastructure
- Work pools and workers route scheduled runs to the appropriate infrastructure, supporting hybrid, push, and fully managed execution models
- Blocks provide typed, reusable configuration objects for credentials, storage, and infrastructure settings
- Prefect Cloud offers a managed orchestration server with RBAC, audit logs, and automations; the open-source Prefect server can be self-hosted for full control

## Design Philosophy: Code-First Orchestration

Prefect 2 (released 2022) and Prefect 3 (released 2024) share the same core philosophy that sets them apart from earlier orchestration frameworks. The guiding principle is that orchestration should adapt to your code, not the other way around.

In Airflow, you define a DAG object, instantiate operator objects for each task, and wire them together with bitshift operators or explicit dependency declarations. The DAG definition is separate from the business logic. In Prefect, your business logic is the workflow definition. You write standard Python with standard control flow (if/else, for loops, try/except), and Prefect observes the execution as it happens.

This has several practical consequences for ML work:

**Dynamic workflows are natural.** If your pipeline needs to train a variable number of models based on the output of a previous step, you write a for loop. There is no need to declare the number of parallel branches at parse time.

**Testing is straightforward.** Because flows and tasks are Python functions, you can call them directly in unit tests without standing up an orchestration server or mocking operator internals.

**Local development matches production.** A decorated function runs the same way on your laptop as it does in a Kubernetes pod. The orchestration layer adds observability and resilience without changing the execution semantics.

**Incremental adoption is possible.** You can decorate a single function with `@flow` and immediately gain retry logic, logging, and state tracking. You do not need to restructure your entire codebase.

## Core Concepts

### Flows

A flow is the top-level unit of orchestration in Prefect. You create one by applying the `@flow` decorator to a Python function:

```python
from prefect import flow

@flow(name="training-pipeline", log_prints=True)
def training_pipeline(learning_rate: float = 0.01, epochs: int = 100):
    data = load_and_validate_data()
    features = engineer_features(data)
    model = train_model(features, lr=learning_rate, epochs=epochs)
    metrics = evaluate_model(model, features)
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    return metrics
```

Flows accept the following configuration parameters:

- `name`: a human-readable identifier (defaults to the function name)
- `description`: documentation that appears in the UI (defaults to the docstring)
- `retries` and `retry_delay_seconds`: automatic retry behavior on failure
- `timeout_seconds`: maximum execution duration before the flow is marked as failed
- `task_runner`: the execution backend for submitted tasks (defaults to `ThreadPoolTaskRunner`)
- `log_prints`: when True, captures `print()` output in the Prefect log stream
- `validate_parameters`: enables Pydantic-based parameter validation (on by default)

Flows can call other flows (subflows), enabling hierarchical composition. Each subflow run is tracked independently in the UI, which is useful for separating concerns like data preparation from model training.

```python
@flow
def data_pipeline():
    raw = extract_data()
    return transform_data(raw)

@flow
def ml_pipeline():
    data = data_pipeline()  # subflow call
    model = train_model(data)
    deploy_model(model)
```

### Tasks

Tasks are the individual units of work within a flow. They are created with the `@task` decorator and represent discrete, observable operations:

```python
from prefect import task
import pandas as pd

@task(retries=3, retry_delay_seconds=10, log_prints=True)
def load_and_validate_data() -> pd.DataFrame:
    df = pd.read_parquet("s3://bucket/training-data/latest.parquet")
    assert len(df) > 0, "Dataset is empty"
    assert df.isnull().sum().sum() == 0, "Dataset contains nulls"
    return df

@task(tags=["training", "gpu"])
def train_model(features: pd.DataFrame, lr: float, epochs: int):
    model = SomeModel(learning_rate=lr)
    model.fit(features, epochs=epochs)
    return model
```

Tasks support the same retry and timeout parameters as flows, plus additional capabilities:

- `cache_key_fn` and `cache_policy`: control when Prefect reuses a previous result instead of re-executing the task
- `cache_expiration`: how long a cached result remains valid
- `tags`: labels that appear in the UI and can be used to apply concurrency limits

The distinction between flows and tasks matters for observability. Each task run appears as a separate node in the Prefect UI's flow run graph, with its own state history, duration, and logs. This granularity is valuable when debugging a failed training pipeline: you can see exactly which task failed, inspect its inputs, and determine whether to retry or fix the underlying issue.

### Caching

Caching prevents redundant computation by storing and reusing task results. Prefect provides several built-in cache policies that determine what information is used to generate cache keys:

- `INPUTS`: caches based on the hash of the task's input arguments
- `TASK_SOURCE`: includes the task's source code in the cache key, so code changes invalidate the cache
- `FLOW_PARAMETERS`: includes the parent flow's parameters
- `RUN_ID`: scopes caching to the current flow run
- `NO_CACHE`: disables caching entirely

Policies can be combined with the `+` operator:

```python
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from datetime import timedelta

@task(
    cache_policy=INPUTS + TASK_SOURCE,
    cache_expiration=timedelta(hours=24)
)
def engineer_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Expensive feature computation
    # Result is cached for 24 hours if inputs and code are unchanged
    features = compute_expensive_features(raw_data)
    return features
```

For ML pipelines, caching is especially useful for feature engineering and data loading steps that are expensive and deterministic. If you re-run a pipeline with the same data but different hyperparameters, the feature engineering task can return its cached result immediately.

### Retries and Error Handling

Prefect provides built-in retry logic at both the flow and task level:

```python
@task(retries=3, retry_delay_seconds=[10, 30, 60])
def fetch_training_data_from_api():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

The `retry_delay_seconds` parameter accepts either a single number or a list for exponential/custom backoff patterns. Retries are particularly important for ML pipelines that interact with external services (data warehouses, model registries, cloud storage) where transient failures are common.

Prefect also tracks states throughout a run's lifecycle: Pending, Running, Completed, Failed, Cancelled, Crashed, and others. You can write logic that responds to these states:

```python
from prefect import flow, task
from prefect.states import Completed, Failed

@flow
def training_with_fallback():
    result = train_primary_model(return_state=True)
    if result.is_failed():
        print("Primary model failed, falling back to simpler model")
        train_fallback_model()
```

### Concurrency

Prefect provides concurrency controls at multiple levels. Global concurrency limits restrict how many tasks with a given tag can run simultaneously across all flows:

```python
@task(tags=["gpu"])
def train_on_gpu(config):
    ...
```

You configure the limit on the "gpu" tag to, say, 4, and Prefect ensures no more than four GPU training tasks execute at once, regardless of how many flow runs are active. This prevents resource contention on shared GPU clusters.

At the deployment level, you can set `concurrency_limit` to control how many runs of a particular deployment execute concurrently, with a `collision_strategy` of either `ENQUEUE` (queue new runs) or `CANCEL_NEW` (reject new runs when the limit is reached).

### Deployments

A deployment is a server-side object that packages a flow with the metadata needed for remote, scheduled, or event-driven execution. Deployments decouple "what to run" from "where and when to run it."

The simplest way to create a deployment is with the `serve` method:

```python
if __name__ == "__main__":
    training_pipeline.serve(
        name="nightly-retraining",
        cron="0 2 * * *",
        parameters={"learning_rate": 0.001, "epochs": 200}
    )
```

This starts a long-running process that listens for scheduled or API-triggered runs. For production use with dynamic infrastructure, you create deployments that target a work pool:

```python
from prefect import flow

if __name__ == "__main__":
    training_pipeline.deploy(
        name="gpu-training",
        work_pool_name="k8s-gpu-pool",
        image="myregistry/training:latest",
        cron="0 2 * * 1",  # Weekly on Monday
        parameters={"learning_rate": 0.001, "epochs": 500},
        job_variables={"gpu": 1, "memory": "32Gi"}
    )
```

Deployments are referenced as `{flow_name}/{deployment_name}`, allowing multiple deployment configurations for the same flow (for example, a nightly retraining deployment and an on-demand retraining deployment with different parameters).

### Work Pools and Workers

Work pools are the bridge between Prefect's orchestration layer and the infrastructure that executes flows. They serve two purposes: routing scheduled runs to the right infrastructure, and providing a template for infrastructure configuration that individual deployments can override.

Prefect supports three work pool architectures:

**Hybrid work pools** require you to run a worker process in your infrastructure. The worker polls the Prefect server for scheduled runs and submits them to the target infrastructure. Supported types include Process (local subprocesses), Docker, Kubernetes, AWS ECS, Azure Container Instances, and Google Cloud Run.

**Push work pools** submit runs directly to serverless infrastructure (AWS ECS Fargate, Azure Container Instances, Google Cloud Run, Modal) without requiring a worker process. This reduces operational overhead at the cost of sharing more connection information with Prefect.

**Managed work pools** run flows on Prefect-managed infrastructure. This is the lowest-friction option but provides less control over the execution environment.

Work pools contain queues with priority levels and concurrency limits, enabling sophisticated routing. A platform team can create a work pool with a default queue for standard jobs and a high-priority queue for production-critical retraining runs.

### Blocks

Blocks are typed configuration objects that store credentials, connection details, and infrastructure settings. They inherit from Pydantic's `BaseModel`, providing validation, serialization, and secret handling:

```python
from prefect_aws import S3Bucket, AwsCredentials

# Create and save a block (typically done once via UI or script)
aws_creds = AwsCredentials(
    aws_access_key_id="AKIA...",
    aws_secret_access_key="..."  # Stored encrypted
)
aws_creds.save("ml-team-aws")

s3_block = S3Bucket(
    bucket_name="ml-artifacts",
    credentials=aws_creds
)
s3_block.save("ml-artifacts-bucket")

# Use in a flow
@task
def save_model(model):
    s3 = S3Bucket.load("ml-artifacts-bucket")
    s3.upload_from_path("model.pkl", "models/latest/model.pkl")
```

Prefect encrypts all block values before storage. Fields typed as `SecretStr` are additionally masked in logs and the UI. The block system is extensible: Prefect's integration libraries (prefect-aws, prefect-gcp, prefect-dbt, and others) ship pre-built block types for common services.

## Prefect Cloud vs Self-Hosted Server

Prefect's architecture separates the orchestration API server from the execution infrastructure. You can run the API server in two ways.

**Prefect Cloud** is Prefect's managed SaaS offering. It provides the API server, UI, and additional enterprise features including role-based access control (RBAC), audit logs, service accounts, automations (event-driven triggers), and push notifications. Teams that do not want to manage orchestration infrastructure typically choose Cloud.

**Self-hosted Prefect server** runs the same open-source API server on your own infrastructure via `prefect server start`. You get the full API and UI but are responsible for availability, database management (PostgreSQL or SQLite), and do not get enterprise features like RBAC or automations. This option suits organizations with strict data sovereignty requirements or those that want to avoid vendor lock-in.

In both cases, the flow execution happens in your infrastructure (unless using managed work pools). Prefect Cloud never sees your data or code; it only manages orchestration state, schedules, and metadata.

## ML Pipeline Examples

### End-to-End Training Pipeline

```python
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from datetime import timedelta
import pandas as pd

@task(
    retries=2,
    retry_delay_seconds=30,
    cache_policy=INPUTS + TASK_SOURCE,
    cache_expiration=timedelta(hours=12)
)
def load_training_data(source_table: str) -> pd.DataFrame:
    """Load and validate training data from warehouse."""
    df = query_warehouse(f"SELECT * FROM {source_table}")
    assert len(df) > 1000, f"Expected >1000 rows, got {len(df)}"
    return df

@task(cache_policy=INPUTS + TASK_SOURCE, cache_expiration=timedelta(hours=6))
def engineer_features(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Apply feature transformations."""
    for col, transform in feature_config.items():
        df[f"{col}_{transform}"] = apply_transform(df[col], transform)
    return df

@task(tags=["gpu"], timeout_seconds=3600)
def train_model(features: pd.DataFrame, hyperparams: dict):
    """Train model with given hyperparameters."""
    X_train, X_val, y_train, y_val = split_data(features)
    model = XGBClassifier(**hyperparams)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model

@task
def evaluate_model(model, features: pd.DataFrame) -> dict:
    """Evaluate model and return metrics."""
    X_test, y_test = get_test_split(features)
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

@task
def register_model(model, metrics: dict, threshold: float = 0.85):
    """Register model if it meets quality threshold."""
    if metrics["auc"] < threshold:
        raise ValueError(f"Model AUC {metrics['auc']:.4f} below threshold {threshold}")
    save_to_registry(model, metrics)
    return True

@flow(name="model-training", log_prints=True)
def training_pipeline(
    source_table: str = "features.training_v3",
    feature_config: dict = None,
    hyperparams: dict = None,
    quality_threshold: float = 0.85
):
    feature_config = feature_config or {"amount": "log", "age": "bucket"}
    hyperparams = hyperparams or {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500}

    data = load_training_data(source_table)
    features = engineer_features(data, feature_config)
    model = train_model(features, hyperparams)
    metrics = evaluate_model(model, features)

    print(f"Model metrics: {metrics}")
    register_model(model, metrics, quality_threshold)
    return metrics
```

### Hyperparameter Sweep with Dynamic Tasks

```python
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner

@task(tags=["gpu"])
def train_single_config(config: dict, data_path: str) -> dict:
    model = train_model(config, data_path)
    score = evaluate_model(model)
    return {"config": config, "score": score, "model_path": save_model(model)}

@flow(task_runner=ThreadPoolTaskRunner(max_workers=4))
def hyperparameter_sweep(data_path: str):
    configs = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [4, 6, 8]
        for lr in [0.01, 0.05, 0.1]
        for n in [100, 300, 500]
    ]

    # Submit all configs as concurrent tasks
    futures = [
        train_single_config.submit(config, data_path)
        for config in configs
    ]

    # Collect results
    results = [f.result() for f in futures]
    best = max(results, key=lambda r: r["score"])
    print(f"Best config: {best['config']} with score {best['score']:.4f}")
    return best
```

### Scheduled Retraining with Drift Detection

```python
@flow(log_prints=True)
def monitored_retraining():
    """Check for drift, retrain if detected."""
    drift_detected = check_data_drift()

    if drift_detected:
        print("Drift detected, triggering retraining")
        metrics = training_pipeline(source_table="features.latest")
        notify_team(f"Model retrained. New metrics: {metrics}")
    else:
        print("No drift detected, skipping retraining")

if __name__ == "__main__":
    monitored_retraining.serve(
        name="drift-monitor",
        cron="0 */6 * * *"  # Every 6 hours
    )
```

## Comparison with Sibling Orchestrators

### Prefect vs Airflow

Airflow is the most widely adopted orchestration tool and the one Prefect was explicitly designed to improve upon.

**Workflow definition.** Airflow requires DAG objects with operators wired together in a specific DSL. Prefect uses decorated Python functions with native control flow. Airflow's `@dag` and `@task` decorators (TaskFlow API, introduced in Airflow 2.0) narrowed this gap, but dynamic workflows in Airflow still require workarounds like dynamic task mapping.

**Scheduler architecture.** Airflow relies on a centralized scheduler that parses DAG files on a fixed interval. Prefect uses an API-first architecture where flows register with the server and execution is triggered via API calls, schedules, or events.

**Infrastructure coupling.** Airflow tightly couples task execution to the Airflow worker infrastructure. Prefect separates orchestration from execution through work pools, allowing the same flow to run on different infrastructure without code changes.

**When to prefer Airflow.** If your organization already runs Airflow with hundreds of DAGs and has invested in custom operators and plugins, the migration cost may not be justified. Airflow also has a larger ecosystem of pre-built operators for data engineering tasks and broader community support.

**When to prefer Prefect.** If you are building ML pipelines that need dynamic behavior, want a better local development experience, or are starting a new orchestration stack without legacy constraints.

### Prefect vs Dagster

Dagster and Prefect share the goal of improving on Airflow but take different approaches.

**Core abstraction.** Dagster is asset-centric: the primary abstraction is a Software-Defined Asset (a data artifact that your pipeline produces), and the computation is defined in terms of what it produces. Prefect is task-centric: the primary abstraction is the function call, and data flows implicitly through return values and parameters.

**Type system.** Dagster has a rich type system with IO managers that handle serialization and storage of assets. Prefect relies on standard Python types and Pydantic validation, leaving serialization to the user.

**Development environment.** Dagster ships with Dagit, a development UI that visualizes assets, their lineage, and materialization history. Prefect's UI focuses on flow runs, task states, and logs rather than data assets.

**When to prefer Dagster.** If your team thinks primarily in terms of data assets and wants strong data lineage guarantees, or if you want opinionated patterns for IO management and testing.

**When to prefer Prefect.** If you want minimal boilerplate, need to orchestrate logic that does not map cleanly to the asset abstraction (API calls, notifications, deployments), or want the flexibility of a thinner orchestration layer.

### Prefect vs Argo Workflows

Argo Workflows is a Kubernetes-native workflow engine that defines pipelines as Kubernetes custom resources in YAML.

**Language and portability.** Argo is language-agnostic: each step runs in a container, and the workflow is defined in YAML. Prefect is Python-native and relies on the Prefect SDK for orchestration. Argo suits polyglot environments; Prefect suits Python-heavy ML teams.

**Infrastructure requirements.** Argo requires a Kubernetes cluster. Prefect can run on bare metal, Docker, Kubernetes, or serverless infrastructure. If your organization does not run Kubernetes, Argo is not a viable option.

**Workflow complexity.** Argo supports advanced patterns like DAG-based and step-based workflows, loops, conditionals, and artifact passing, but all in YAML. Prefect expresses the same patterns in Python, which most ML engineers find more natural to write and debug.

**When to prefer Argo.** If you are Kubernetes-native, need language-agnostic orchestration, or want tight integration with the Kubernetes ecosystem (RBAC, service mesh, observability).

**When to prefer Prefect.** If your team works primarily in Python, wants a faster local development loop, or does not want to manage Kubernetes for orchestration.

## When to Choose Prefect

Prefect is a strong choice when:

- Your team works primarily in Python and wants orchestration that feels like writing Python, not configuring YAML or a DAG DSL
- Your ML pipelines are dynamic: the number of tasks, the branching logic, or the infrastructure requirements change based on runtime data
- You want a fast iteration loop where you can run, test, and debug flows locally before deploying to production
- You need flexible infrastructure: the ability to run the same flow on your laptop, in Docker, on Kubernetes, or on serverless cloud infrastructure without code changes
- You want a managed orchestration option (Prefect Cloud) with the ability to self-host if requirements change
- You are starting a new orchestration stack and want to avoid Airflow's operational complexity

Prefect may not be the best choice when:

- Your organization is heavily invested in Airflow with established patterns, plugins, and operational knowledge
- You need language-agnostic orchestration (consider Argo Workflows)
- Your primary concern is data asset lineage and materialization tracking (consider Dagster)
- You need orchestration deeply integrated with a specific cloud ML platform (consider the platform's native orchestration, such as SageMaker Pipelines or Vertex AI Pipelines)

## Common Pitfalls

### Overusing Tasks

Not every function call needs to be a task. Each task adds orchestration overhead (state tracking, API calls to the Prefect server, result serialization). Wrap functions as tasks when you need observability, caching, retries, or concurrency control on that specific operation. Leave helper functions as plain Python.

```python
# Unnecessary: wrapping a trivial helper as a task
@task
def add_one(x):
    return x + 1

# Better: only wrap meaningful units of work
@task(retries=2)
def fetch_data():
    return requests.get(API_URL).json()
```

### Ignoring Result Serialization

When caching is enabled, Prefect serializes task results. Large objects like trained models or DataFrames need to be serializable. If your task returns an object that cannot be pickled, caching will fail. Consider returning a path or identifier instead:

```python
@task(cache_policy=INPUTS)
def train_model(data_path: str) -> str:
    model = train(data_path)
    model_path = f"s3://models/{uuid4()}.pkl"
    save_model(model, model_path)
    return model_path  # Return path, not the model object
```

### Not Setting Timeouts

ML training tasks can hang due to deadlocks, infinite loops, or unresponsive external services. Always set `timeout_seconds` on long-running tasks to prevent zombie runs from consuming resources indefinitely.

### Neglecting Concurrency Limits

Without concurrency limits, a hyperparameter sweep can spawn dozens of GPU training tasks simultaneously, overwhelming your cluster. Use tag-based concurrency limits to match your actual infrastructure capacity.
