# Apache Airflow

## Summary

Apache Airflow is the most widely adopted open-source workflow orchestration platform. Originally developed at Airbnb in 2014 and later donated to the Apache Software Foundation, it defines workflows as directed acyclic graphs (DAGs) written in Python. Airflow excels at scheduling, monitoring, and managing complex data and ML pipelines, and its massive ecosystem of providers makes it the default choice for teams that need to coordinate tasks across dozens of systems.

Key points to remember:

- Workflows are defined as Python code (DAGs), not YAML or configuration files, giving full programmatic flexibility
- The architecture consists of a scheduler, webserver, executor, metadata database, and a DAG directory
- Operators encapsulate units of work; the most relevant for ML are PythonOperator, KubernetesPodOperator, and cloud-specific operators
- The TaskFlow API (Airflow 2.x) simplifies DAG authoring with decorated Python functions and implicit XCom passing
- Managed offerings like Amazon MWAA and Astronomer eliminate operational overhead for running Airflow in production
- Airflow is best suited for batch-oriented, schedule-driven pipelines; it is not designed for streaming or event-driven workloads

## Architecture

### Core Components

Airflow follows a modular architecture with several collaborating services:

Scheduler: The scheduler is the heart of Airflow. It continuously monitors DAG definitions, determines which tasks are ready to run based on their dependencies and schedule, and submits them to the executor. In Airflow 2.x, the scheduler can run in high-availability mode with multiple instances sharing the workload.

Webserver: The web UI provides visibility into DAG structure, task status, run history, logs, and Gantt charts for timing analysis. It is built on Flask and communicates with the metadata database. The webserver is strictly a monitoring and management interface; it does not execute tasks.

Executor: The executor determines how tasks are actually run. Options include:

- LocalExecutor: Runs tasks as subprocesses on the same machine. Suitable for development and small-scale production.
- CeleryExecutor: Distributes tasks across a pool of worker machines using Celery and a message broker (Redis or RabbitMQ). This is the traditional production executor.
- KubernetesExecutor: Launches each task in its own Kubernetes pod. Provides strong isolation and dynamic resource allocation. Each task can have its own Docker image and resource requests.
- CeleryKubernetesExecutor: A hybrid that routes most tasks through Celery but allows specific tasks to run on Kubernetes pods.

Metadata Database: A relational database (typically PostgreSQL in production, SQLite for development) that stores DAG definitions, task instances, run history, variables, connections, XCom values, and scheduler state. This is the single source of truth for the entire system.

DAG Directory: A filesystem directory (or synced from a Git repository, S3, or GCS) where Airflow discovers DAG definition files. The scheduler periodically parses these files to detect new or modified DAGs.

### How a Task Executes

The lifecycle of a task execution follows this path:

1. The scheduler parses DAG files and creates DagRun entries for DAGs whose schedule interval has elapsed
2. For each DagRun, the scheduler evaluates task dependencies and marks eligible tasks as "scheduled"
3. The executor picks up scheduled tasks and runs them (as subprocesses, Celery messages, or Kubernetes pods)
4. The task runs its operator logic, logs output, and reports success or failure back to the metadata database
5. The scheduler re-evaluates downstream dependencies and continues until the DAG completes or fails

### Connections and Variables

Airflow manages credentials and configuration through two mechanisms:

Connections store external system credentials (database URIs, cloud credentials, API keys). They are stored in the metadata database and can be encrypted at rest. Access them in code via hooks:

```python
from airflow.hooks.base import BaseHook

conn = BaseHook.get_connection("my_postgres")
print(conn.host, conn.port, conn.schema)
```

Variables store arbitrary key-value configuration:

```python
from airflow.models import Variable

model_version = Variable.get("current_model_version")
config = Variable.get("training_config", deserialize_json=True)
```

Use connections for credentials and variables for runtime configuration. Avoid putting secrets in DAG code or environment variables that are visible in the UI.

## DAGs as Python Code

### Basic DAG Structure

A DAG is a Python file that defines tasks and their dependencies:

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-team@company.com"],
}

with DAG(
    dag_id="daily_training_pipeline",
    default_args=default_args,
    description="Daily model retraining pipeline",
    schedule_interval="0 6 * * *",  # 6 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as dag:

    def extract_features(**kwargs):
        # Feature extraction logic
        return {"feature_path": "s3://bucket/features/2025-01-01.parquet"}

    def train_model(**kwargs):
        ti = kwargs["ti"]
        feature_info = ti.xcom_pull(task_ids="extract_features")
        # Training logic using feature_info["feature_path"]

    extract = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    extract >> train  # Set dependency
```

### Operators

Operators define what a task does. Each operator type encapsulates a specific kind of work:

PythonOperator: Runs a Python callable. The most flexible operator but couples your code to the Airflow worker environment:

```python
from airflow.operators.python import PythonOperator

def preprocess(ds, **kwargs):
    # ds is the execution date string
    print(f"Processing data for {ds}")

task = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess,
)
```

BashOperator: Executes a shell command. Useful for invoking CLI tools, scripts, or existing batch jobs:

```python
from airflow.operators.bash import BashOperator

train = BashOperator(
    task_id="train_model",
    bash_command="python /opt/ml/train.py --date {{ ds }} --config {{ var.json.training_config }}",
)
```

KubernetesPodOperator: Runs a task inside a Kubernetes pod. This is often the best choice for ML workloads because each task gets its own isolated environment with specific resource requests:

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

gpu_resources = k8s.V1ResourceRequirements(
    requests={"nvidia.com/gpu": "1", "memory": "16Gi", "cpu": "4"},
    limits={"nvidia.com/gpu": "1", "memory": "32Gi", "cpu": "8"},
)

train = KubernetesPodOperator(
    task_id="train_model",
    name="model-training",
    namespace="ml-pipelines",
    image="company-registry/ml-training:v2.3",
    arguments=["--date", "{{ ds }}", "--experiment", "v2"],
    container_resources=gpu_resources,
    is_delete_operator_pod=True,
    get_logs=True,
    startup_timeout_seconds=600,
)
```

### Cloud-Specific Operators

Airflow ships with provider packages for major cloud platforms. These operators invoke managed services directly:

AWS:
```python
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator
```

GCP:
```python
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.operators.vertex_ai.auto_ml import CreateAutoMLTabularTrainingJobOperator
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
```

Azure:
```python
from airflow.providers.microsoft.azure.operators.data_factory import AzureDataFactoryRunPipelineOperator
from airflow.providers.microsoft.azure.operators.container_instances import AzureContainerInstancesOperator
```

Cloud operators are preferable to PythonOperator with SDK calls because they handle authentication, polling, and error handling consistently.

### Dependencies

Define task execution order with bitshift operators or explicit methods:

```python
# Linear chain
extract >> transform >> train >> evaluate >> deploy

# Fan-out / fan-in
extract >> [transform_a, transform_b] >> merge >> train

# Explicit method
train.set_upstream(transform)
deploy.set_downstream(notify)

# Cross-DAG dependencies (Airflow 2.x)
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_features = ExternalTaskSensor(
    task_id="wait_for_feature_pipeline",
    external_dag_id="feature_engineering",
    external_task_id="write_features",
    timeout=3600,
)
```

### XComs (Cross-Communication)

XComs allow tasks to pass small pieces of data to downstream tasks. They are stored in the metadata database:

```python
# Push explicitly
def producer(**kwargs):
    kwargs["ti"].xcom_push(key="model_path", value="s3://bucket/models/v3.pt")

# Pull explicitly
def consumer(**kwargs):
    model_path = kwargs["ti"].xcom_pull(task_ids="producer", key="model_path")
```

XCom limitations and guidelines:

- XComs are stored in the metadata database, so they should be small (paths, metrics, configuration). Do not pass large datasets through XComs.
- Return values from PythonOperator callables are automatically pushed as XComs.
- The TaskFlow API makes XCom passing implicit and cleaner.
- For large data, pass file paths (S3, GCS, local filesystem) rather than the data itself.

## TaskFlow API

Airflow 2.x introduced the TaskFlow API, which uses Python decorators to define tasks. This is now the recommended way to write DAGs:

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(
    dag_id="taskflow_training_pipeline",
    schedule="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "training"],
)
def training_pipeline():

    @task
    def extract_features(date: str) -> dict:
        # Feature extraction logic
        return {
            "feature_path": f"s3://bucket/features/{date}.parquet",
            "row_count": 50000,
        }

    @task
    def validate_features(feature_info: dict) -> dict:
        row_count = feature_info["row_count"]
        if row_count < 1000:
            raise ValueError(f"Too few rows: {row_count}")
        return feature_info

    @task
    def train_model(feature_info: dict) -> dict:
        # Training logic
        return {
            "model_path": "s3://bucket/models/v3.pt",
            "accuracy": 0.94,
        }

    @task
    def evaluate_model(model_info: dict) -> bool:
        return model_info["accuracy"] > 0.90

    @task.branch
    def decide_deployment(should_deploy: bool) -> str:
        if should_deploy:
            return "deploy_model"
        return "notify_failure"

    @task
    def deploy_model():
        # Deployment logic
        pass

    @task
    def notify_failure():
        # Alert team
        pass

    # Define flow - XCom passing is implicit
    features = extract_features(date="{{ ds }}")
    validated = validate_features(features)
    model_info = train_model(validated)
    should_deploy = evaluate_model(model_info)
    branch = decide_deployment(should_deploy)
    branch >> [deploy_model(), notify_failure()]

training_pipeline()
```

TaskFlow advantages over the classic API:

- Return values automatically become XComs; no manual push/pull
- Type hints document the data contract between tasks
- Branching logic is cleaner with `@task.branch`
- Less boilerplate; the DAG reads more like a Python program
- Supports mixing TaskFlow tasks with traditional operators in the same DAG

## ML Pipeline Patterns

### Training DAG

A production training DAG typically follows this structure:

```python
@dag(schedule="0 2 * * 1", start_date=datetime(2025, 1, 1), catchup=False)
def weekly_model_training():

    @task
    def pull_training_data():
        # Query data warehouse, write to object storage
        return {"path": "s3://data/training/2025-01-06.parquet", "rows": 100000}

    @task
    def compute_features(data_info: dict):
        # Feature engineering
        return {"path": "s3://features/2025-01-06.parquet"}

    @task
    def split_data(feature_info: dict):
        return {
            "train_path": "s3://features/train.parquet",
            "val_path": "s3://features/val.parquet",
            "test_path": "s3://features/test.parquet",
        }

    @task
    def train(split_info: dict):
        # Train model, log to MLflow
        return {"run_id": "abc123", "model_uri": "runs:/abc123/model"}

    @task
    def evaluate(model_info: dict, split_info: dict):
        # Evaluate on test set
        return {"accuracy": 0.95, "f1": 0.93, "model_uri": model_info["model_uri"]}

    @task.branch
    def promotion_gate(metrics: dict):
        if metrics["accuracy"] > 0.90 and metrics["f1"] > 0.88:
            return "register_model"
        return "send_alert"

    @task
    def register_model(metrics: dict):
        # Register in model registry, promote to staging
        pass

    @task
    def send_alert():
        # Notify team of failed quality gate
        pass

    data = pull_training_data()
    features = compute_features(data)
    splits = split_data(features)
    model = train(splits)
    metrics = evaluate(model, splits)
    gate = promotion_gate(metrics)
    gate >> [register_model(metrics), send_alert()]

weekly_model_training()
```

### Feature Engineering DAG

Feature pipelines often run on a different schedule from training and are consumed by multiple models:

```python
@dag(schedule="0 4 * * *", start_date=datetime(2025, 1, 1), catchup=False)
def daily_feature_engineering():

    @task
    def extract_transactions():
        return {"path": "s3://raw/transactions/{{ ds }}.parquet"}

    @task
    def extract_user_profiles():
        return {"path": "s3://raw/profiles/{{ ds }}.parquet"}

    @task
    def compute_transaction_features(txn_info: dict):
        # Aggregations, rolling windows
        return {"path": "s3://features/transaction/{{ ds }}.parquet"}

    @task
    def compute_user_features(profile_info: dict, txn_info: dict):
        # User-level features
        return {"path": "s3://features/user/{{ ds }}.parquet"}

    @task
    def write_to_feature_store(txn_features: dict, user_features: dict):
        # Write to Feast, Tecton, or custom store
        pass

    @task
    def validate_features(txn_features: dict, user_features: dict):
        # Run Great Expectations or similar
        pass

    txn = extract_transactions()
    profiles = extract_user_profiles()
    txn_feat = compute_transaction_features(txn)
    user_feat = compute_user_features(profiles, txn)
    write_to_feature_store(txn_feat, user_feat)
    validate_features(txn_feat, user_feat)

daily_feature_engineering()
```

### Model Deployment DAG

Deployment DAGs handle the rollout of a newly promoted model:

```python
@dag(schedule=None, start_date=datetime(2025, 1, 1))  # Triggered externally
def deploy_model():

    @task
    def fetch_model_artifact(model_name: str, version: int):
        # Download from registry
        return {"local_path": "/tmp/model.pt", "version": version}

    @task
    def build_container(artifact_info: dict):
        # Build and push serving container
        return {"image": f"registry/model:{artifact_info['version']}"}

    @task
    def run_canary(container_info: dict):
        # Deploy to canary, run smoke tests
        return {"passed": True}

    @task.branch
    def canary_gate(result: dict):
        return "promote_to_production" if result["passed"] else "rollback"

    @task
    def promote_to_production():
        pass

    @task
    def rollback():
        pass

    artifact = fetch_model_artifact(model_name="fraud_detector", version=3)
    container = build_container(artifact)
    canary = run_canary(container)
    gate = canary_gate(canary)
    gate >> [promote_to_production(), rollback()]

deploy_model()
```

### Triggering DAGs Programmatically

ML workflows often need to chain DAGs. Use TriggerDagRunOperator or the Airflow API:

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_deployment = TriggerDagRunOperator(
    task_id="trigger_deployment",
    trigger_dag_id="deploy_model",
    conf={"model_name": "fraud_detector", "version": 3},
    wait_for_completion=True,
)
```

## Managed Airflow Services

### Amazon MWAA (Managed Workflows for Apache Airflow)

MWAA is AWS's managed Airflow offering. AWS handles the scheduler, webserver, workers, and metadata database. You provide DAG files in S3 and a requirements.txt for additional Python packages.

Key characteristics:

- DAGs are synced from an S3 bucket
- Supports Airflow 2.x versions
- Workers auto-scale based on queued tasks
- Integrated with IAM for authentication and authorization
- Native access to AWS services (S3, SageMaker, EMR, Glue)
- Pricing is based on environment size, not per task

MWAA is well suited for AWS-centric organizations. Limitations include slower environment updates (plugin and requirement changes require environment restarts that can take 20-30 minutes) and less flexibility in executor configuration compared to self-managed deployments.

### Astronomer

Astronomer is a commercial platform built around Airflow. It provides:

- Astro CLI for local development and testing of DAGs
- Managed cloud deployment (Astro) with push-button upgrades
- Fine-grained resource isolation per DAG or task
- Built-in observability with alerting and SLA monitoring
- The Astro SDK, which extends Airflow with data-aware scheduling and simplified data transformations

Astronomer is the strongest option for organizations that want a managed Airflow experience with more flexibility than MWAA. The Astro CLI is useful even if you do not use their cloud service.

### Google Cloud Composer

Cloud Composer is Google's managed Airflow service, running on GKE (Google Kubernetes Engine):

- Composer 2 uses an autopilot architecture with improved scheduling and autoscaling
- Deep integration with BigQuery, Dataflow, Vertex AI, and GCS
- Environment creation and configuration via Terraform or the GCP console

### Self-Managed Deployment

Running Airflow yourself (on Kubernetes with the official Helm chart, or on VMs) gives full control over executor configuration, plugin management, and resource allocation. The tradeoff is operational burden: managing upgrades, scaling workers, maintaining the metadata database, and ensuring scheduler high availability.

The official Helm chart supports:

- KubernetesExecutor and CeleryExecutor configurations
- Git-sync for DAG deployment
- Flower monitoring for Celery workers
- PgBouncer for database connection pooling

## Comparison with Other Orchestrators

### Airflow vs Prefect

Prefect positions itself as a modern alternative to Airflow with a simpler developer experience. Key differences:

- Prefect uses native Python functions and decorators without requiring a DAG context manager
- Prefect supports dynamic task generation at runtime more naturally than Airflow
- Prefect's event-driven architecture handles reactive workflows better
- Airflow has a vastly larger ecosystem of providers and operators (1000+ integrations)
- Airflow has more battle-tested production deployments at scale
- Prefect 2.x (Orion) is a significant rewrite; the ecosystem is still maturing

Choose Prefect when you need strong dynamic workflows, event-driven triggers, or a simpler Python-native experience. Choose Airflow when you need breadth of integrations, proven scale, or your team already knows it.

### Airflow vs Dagster

Dagster takes a software-defined asset approach where pipelines are organized around the data assets they produce rather than the tasks they execute:

- Dagster's asset-centric model makes lineage and data cataloging first-class
- Dagster has stronger type checking and testing support for pipeline code
- Dagster integrates a UI that shows data assets rather than task graphs
- Airflow's task-centric model is more intuitive for ETL and job orchestration
- Airflow has far more third-party operator integrations

Choose Dagster when data lineage and asset management are primary concerns, especially for analytics-heavy workloads. Choose Airflow for general-purpose orchestration where the breadth of integrations matters.

### Airflow vs Argo Workflows

Argo Workflows is a Kubernetes-native orchestrator that defines workflows as Kubernetes custom resources in YAML:

- Argo is container-native; every step runs in its own pod by default
- Argo workflows are defined in YAML, not Python
- Argo has no built-in scheduler for periodic runs (pair with Argo Events or CronWorkflows)
- Argo is lighter weight and has fewer dependencies than Airflow
- Airflow is better for heterogeneous workloads that span multiple systems

Choose Argo when your entire stack runs on Kubernetes and you want minimal overhead. Choose Airflow when you need to orchestrate across cloud services, databases, and external APIs beyond just Kubernetes pods.

### Summary Table

| Criterion                  | Airflow           | Prefect           | Dagster          | Argo              |
|----------------------------|-------------------|-------------------|------------------|--------------------|
| Definition language        | Python            | Python            | Python           | YAML               |
| Scheduling                 | Built-in cron     | Built-in          | Built-in         | CronWorkflow       |
| Dynamic tasks              | Limited           | Strong            | Moderate         | Moderate           |
| Data lineage               | Basic             | Basic             | First-class      | None               |
| Provider ecosystem         | 1000+ operators   | Growing           | Growing          | Container-based    |
| Managed offerings          | MWAA, Astronomer  | Prefect Cloud     | Dagster Cloud    | None (self-hosted) |
| Learning curve             | Moderate          | Low               | Moderate         | Low-moderate       |

## When to Choose Airflow

Airflow is the right choice when:

- You need to orchestrate tasks across many systems (databases, cloud services, APIs, on-premise tools)
- Your pipelines are batch-oriented and schedule-driven
- You need a large ecosystem of pre-built integrations
- Your team is already familiar with Airflow or you are hiring from a talent pool that knows it
- You want managed service options on any major cloud provider
- You need a mature, battle-tested platform with extensive community support

Airflow is not the right choice when:

- You need real-time or streaming orchestration (consider Flink, Kafka Streams, or event-driven architectures)
- Your workflows are highly dynamic with runtime-determined task graphs (Prefect handles this better)
- You need sub-second task scheduling latency (Airflow's scheduler loop introduces seconds of delay)
- Your workloads are purely Kubernetes-native with no external system integration (Argo is lighter weight)
- You want a data-asset-centric paradigm rather than a task-centric one (Dagster is a better fit)
- You have simple, linear pipelines that do not need a full orchestration framework (a cron job or a CI/CD pipeline may suffice)

## Practical Tips

### DAG Design

Keep DAGs modular and idempotent. Every task should be safe to retry without side effects:

```python
# Idempotent write: overwrite partition, do not append
def write_features(ds, **kwargs):
    output_path = f"s3://features/dt={ds}/features.parquet"
    df.to_parquet(output_path, mode="overwrite")
```

Avoid putting expensive computation at DAG parse time. The scheduler parses all DAG files every few seconds. Anything at the module level runs during parsing:

```python
# Bad: runs on every scheduler parse
df = pd.read_csv("s3://bucket/large_file.csv")

# Good: runs only when task executes
@task
def load_data():
    df = pd.read_csv("s3://bucket/large_file.csv")
    return {"path": "s3://bucket/processed.parquet"}
```

### Testing

Test DAGs before deploying:

```python
# Test that DAG loads without errors
def test_dag_loads():
    from my_dags.training_pipeline import training_pipeline
    dag = training_pipeline()
    assert dag is not None
    assert len(dag.tasks) > 0

# Test individual task logic
def test_feature_extraction():
    result = extract_features(date="2025-01-01")
    assert "feature_path" in result

# Validate DAG integrity
def test_no_import_errors():
    from airflow.models import DagBag
    dag_bag = DagBag(include_examples=False)
    assert len(dag_bag.import_errors) == 0
```

### Performance Tuning

Key scheduler settings to tune for ML workloads:

```ini
# airflow.cfg or environment variables
[scheduler]
min_file_process_interval = 30  # Seconds between DAG file parses
dag_dir_list_interval = 300     # Seconds between scanning for new DAG files
max_tis_per_query = 512         # Task instances per scheduler loop
parsing_processes = 4           # Parallel DAG parsing processes

[core]
parallelism = 32               # Max concurrent tasks across all DAGs
max_active_tasks_per_dag = 16  # Max concurrent tasks within one DAG
max_active_runs_per_dag = 3    # Max concurrent runs of one DAG
```

For ML workloads with long-running training tasks, ensure your executor timeout is sufficiently long and your worker pool is sized for the expected parallelism.

### Resource Isolation with KubernetesPodOperator

For ML workloads, KubernetesPodOperator is often the best approach because it provides:

- Per-task Docker images: each task can have its own dependency set
- GPU scheduling: request GPUs through Kubernetes resource limits
- Memory isolation: a training task that uses 64 GB of memory will not affect other tasks
- Node affinity: schedule GPU tasks on GPU nodes, CPU tasks on CPU nodes

```python
from kubernetes.client import models as k8s

affinity = k8s.V1Affinity(
    node_affinity=k8s.V1NodeAffinity(
        required_during_scheduling_ignored_during_execution=k8s.V1NodeSelector(
            node_selector_terms=[
                k8s.V1NodeSelectorTerm(
                    match_expressions=[
                        k8s.V1NodeSelectorRequirement(
                            key="nvidia.com/gpu.product",
                            operator="In",
                            values=["NVIDIA-A100-SXM4-80GB"],
                        )
                    ]
                )
            ]
        )
    )
)

train = KubernetesPodOperator(
    task_id="train_model",
    image="ml-training:latest",
    container_resources=k8s.V1ResourceRequirements(
        requests={"nvidia.com/gpu": "1", "memory": "64Gi"},
        limits={"nvidia.com/gpu": "1", "memory": "80Gi"},
    ),
    affinity=affinity,
    namespace="ml-pipelines",
    get_logs=True,
    is_delete_operator_pod=True,
)
```

### Common Pitfalls

Storing large data in XComs: XComs are stored in the metadata database. Pushing large dataframes or model artifacts into XComs will bloat the database and slow the scheduler. Always pass paths to data stored in object storage.

Top-level code in DAG files: Any code at the module level runs every time the scheduler parses the file. Database queries, API calls, or heavy imports at parse time will degrade scheduler performance.

Not setting catchup=False: By default, Airflow will backfill all missed runs since start_date. For ML pipelines that should only run going forward, always set catchup=False.

Ignoring SLAs and timeouts: Long-running training jobs can silently hang. Set execution_timeout on tasks and use SLA misses to alert when pipelines run longer than expected:

```python
@task(execution_timeout=timedelta(hours=4))
def train_model():
    # Training logic
    pass
```

Over-engineering DAGs: Not every pipeline needs Airflow. If you have a single script that runs on a cron schedule with no dependencies, a simple cron job or a CI/CD scheduled pipeline is adequate. Airflow adds value when you have multiple interdependent tasks, need visibility, or require retry logic across complex workflows.
