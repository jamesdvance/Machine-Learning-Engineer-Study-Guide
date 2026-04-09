# Dagster

## Summary

Dagster is a data-aware orchestrator built around the concept of software-defined assets. Unlike task-centric orchestrators that model pipelines as sequences of operations, Dagster treats data objects themselves as the primary abstraction. You declare what data should exist, how it is computed, and what it depends on. Dagster then handles materialization, dependency resolution, lineage tracking, and observability automatically. This asset-oriented model aligns naturally with ML workflows where the focus is on producing and validating datasets, feature tables, trained models, and evaluation reports.

Key points to remember:

- Software-defined assets are Dagster's core abstraction, representing data objects rather than tasks
- The asset graph replaces the task DAG, providing automatic lineage and staleness detection
- Resources and IO managers decouple business logic from infrastructure, making pipelines testable and portable
- Dagster provides first-class testing support with in-process execution and resource mocking
- Sensors and schedules automate materialization based on time or external events
- Dagster+ (Cloud) adds managed hosting, branch deployments, RBAC, and an asset catalog on top of the open-source core
- The type system and asset checks enable data validation as a built-in concern rather than an afterthought

## Core Concepts

### Assets

Assets are the fundamental building block in Dagster. An asset represents a persistent data object, such as a database table, a Parquet file, a trained model artifact, or a feature store entry. You define an asset by writing a Python function decorated with `@dg.asset` that computes the object and optionally declares its upstream dependencies.

```python
import dagster as dg

@dg.asset
def raw_transactions(context: dg.AssetExecutionContext):
    """Ingest raw transaction data from source system."""
    df = fetch_from_source()
    context.log.info(f"Fetched {len(df)} rows")
    return df

@dg.asset(deps=[raw_transactions])
def cleaned_transactions(raw_transactions):
    """Apply cleaning and deduplication."""
    return raw_transactions.drop_duplicates().dropna(subset=["amount"])
```

When you define assets this way, Dagster constructs an asset graph that represents data lineage. The UI displays this graph, tracks which assets are materialized, and identifies stale assets when upstream data or code changes.

Dagster provides four asset decorators for different scenarios:

- `@asset` produces a single asset from a single function
- `@multi_asset` produces multiple assets from one function, useful when a single API call or computation yields several outputs
- `@graph_asset` produces a single asset from a graph of ops, encapsulating multi-step computation behind a single asset interface
- `@graph_multi_asset` combines both, producing multiple assets from a graph of ops

Assets support `code_version` annotations that let Dagster detect when the code that produced an asset has changed, marking downstream assets as stale. Multi-part asset keys created with `key_prefix` allow hierarchical namespacing, such as `["warehouse", "schema", "table_name"]`.

### Ops and Graphs

Ops are the lower-level computational units in Dagster. They predate assets in Dagster's design and represent individual steps of computation. Ops connect through graphs, which define execution order through explicit dependencies.

```python
@dg.op
def fetch_data(context):
    return load_data()

@dg.op
def transform_data(context, data):
    return data.transform()

@dg.graph
def etl_pipeline():
    data = fetch_data()
    transform_data(data)
```

In current Dagster practice, assets have largely replaced ops for most use cases. Ops remain useful when you need fine-grained control over execution steps within an asset or when modeling computations that do not produce persistent data objects. The relationship is straightforward: assets describe what data exists, while ops describe how computation happens.

### Jobs

Jobs are the primary execution unit in Dagster. A job can target a selection of assets for materialization or wrap a graph of ops for execution. Jobs define what gets executed when a schedule fires or a sensor triggers.

```python
# Asset-based job: materialize a selection of assets
training_job = dg.define_asset_job(
    name="training_job",
    selection=["feature_table", "trained_model", "evaluation_report"]
)

# Op-based job
@dg.job
def etl_job():
    etl_pipeline()
```

Asset-based jobs are the recommended approach. You specify which assets to materialize, and Dagster resolves the full dependency graph to determine what needs to run.

### Resources

Resources represent external dependencies that your assets and ops interact with: database connections, API clients, cloud storage handles, ML experiment trackers, and similar infrastructure. Resources are configured separately from business logic and injected at runtime.

```python
class MLFlowResource(dg.ConfigurableResource):
    tracking_uri: str

    def log_metrics(self, metrics: dict):
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.log_metrics(metrics)

@dg.asset
def trained_model(context, mlflow: MLFlowResource):
    model = train()
    mlflow.log_metrics({"accuracy": model.accuracy})
    return model

defs = dg.Definitions(
    assets=[trained_model],
    resources={
        "mlflow": MLFlowResource(tracking_uri="http://mlflow:5000")
    }
)
```

This separation is critical for testability. In tests, you swap production resources with mock or in-memory implementations without changing asset code.

### IO Managers

IO managers handle the serialization and deserialization of data passed between assets. They abstract the storage layer so that asset functions return Python objects while the IO manager decides where and how to persist them.

```python
class ParquetIOManager(dg.IOManager):
    def __init__(self, base_path: str):
        self.base_path = base_path

    def handle_output(self, context, obj):
        path = f"{self.base_path}/{context.asset_key.path[-1]}.parquet"
        obj.to_parquet(path)

    def load_input(self, context):
        path = f"{self.base_path}/{context.asset_key.path[-1]}.parquet"
        return pd.read_parquet(path)
```

Dagster ships with built-in IO managers for common storage backends including local filesystems, S3, GCS, and BigQuery. You can assign different IO managers to different assets, allowing a pipeline to write intermediate data to local Parquet during development and to cloud storage in production with no code changes to the assets themselves.

### Sensors and Schedules

Schedules trigger jobs at fixed time intervals using cron expressions:

```python
training_schedule = dg.ScheduleDefinition(
    job=training_job,
    cron_schedule="0 2 * * 1",  # Every Monday at 2 AM
)
```

Sensors trigger jobs in response to external events. A sensor function runs periodically, checks for conditions, and yields run requests when conditions are met:

```python
@dg.sensor(job=training_job)
def new_data_sensor(context):
    last_modified = check_data_source_timestamp()
    if last_modified > context.cursor:
        yield dg.RunRequest(run_key=str(last_modified))
        context.update_cursor(str(last_modified))
```

Sensors are particularly useful in ML pipelines for triggering retraining when new data arrives, when model performance drops below a threshold, or when upstream feature tables are refreshed.

## Software-Defined Assets: The Declarative Approach

The software-defined asset model represents Dagster's core philosophical difference from other orchestrators. In a task-centric orchestrator like Airflow, you define a DAG of operations: extract, then transform, then load, then train, then evaluate. The focus is on what to do and in what order. In Dagster, you define what data should exist and how each piece of data is derived from other data. The focus is on what exists.

This distinction has practical consequences.

Lineage is automatic. Because assets declare their dependencies, Dagster knows the full provenance of every piece of data without you building or maintaining separate lineage documentation. The asset graph in the Dagster UI is not a task execution graph but a data lineage graph.

Staleness detection is built in. When you change the code that produces an asset (tracked via `code_version`) or when an upstream asset is re-materialized, Dagster marks downstream assets as stale. This lets you know exactly which assets need to be refreshed after a code change or data update.

Partial materialization is straightforward. You can materialize any subset of the asset graph. Need to rebuild just the feature table and everything downstream? Select those assets and run. Dagster resolves the dependency order automatically.

Observability assets let you track external data you do not produce. If your pipeline depends on a table maintained by another team, you can define an observable source asset that checks freshness without materializing anything:

```python
@dg.observable_source_asset
def external_user_table(context):
    last_updated = query_metadata("SELECT MAX(updated_at) FROM users")
    return dg.DataVersion(str(last_updated))
```

### Partitions

Partitions split an asset into logical segments, typically by time. A daily-partitioned asset produces one partition per day, and Dagster tracks materialization status per partition:

```python
daily_partitions = dg.DailyPartitionsDefinition(start_date="2024-01-01")

@dg.asset(partitions_def=daily_partitions)
def daily_features(context):
    date = context.partition_key
    return compute_features_for_date(date)
```

Partitions enable incremental processing. When new data arrives, you materialize only the affected partitions rather than reprocessing everything. This pattern is essential for ML pipelines operating on time-series data, where recomputing the full history on every run would be prohibitively expensive.

## Type System and Data Validation

Dagster includes a type system that allows you to annotate asset and op inputs and outputs with expected types. Beyond standard Python type hints, Dagster types can include runtime validation logic that checks data as it flows through the pipeline.

```python
from dagster import DagsterType

def validate_dataframe(_, value):
    if value.empty:
        return False
    if "amount" not in value.columns:
        return False
    return True

ValidatedTransactions = DagsterType(
    name="ValidatedTransactions",
    type_check_fn=validate_dataframe,
    description="DataFrame with non-empty rows and an amount column"
)

@dg.asset(dagster_type=ValidatedTransactions)
def cleaned_transactions(raw_transactions):
    return raw_transactions.dropna(subset=["amount"])
```

Asset checks provide a more modern and flexible validation mechanism. They run after materialization and can verify data quality, schema conformance, freshness, and business rules:

```python
@dg.asset_check(asset=cleaned_transactions)
def no_negative_amounts(cleaned_transactions):
    has_negatives = (cleaned_transactions["amount"] < 0).any()
    return dg.AssetCheckResult(
        passed=not has_negatives,
        metadata={"negative_count": int((cleaned_transactions["amount"] < 0).sum())}
    )
```

Asset checks integrate with the Dagster UI, providing a dashboard of data quality status across your entire asset graph. Failed checks can trigger alerts and block downstream materialization when configured to do so. This is Dagster's approach to data contracts: validation rules live alongside the assets they validate and are tracked as first-class objects.

## ML Pipeline Patterns with Assets

### Feature Engineering as Assets

Feature tables map naturally to assets. Each feature set is a persistent data object with clear upstream dependencies:

```python
@dg.asset
def user_features(transactions, user_profiles):
    features = transactions.groupby("user_id").agg(
        total_spend=("amount", "sum"),
        avg_transaction=("amount", "mean"),
        transaction_count=("amount", "count")
    )
    return features.join(user_profiles[["user_id", "tenure_days"]])

@dg.asset
def training_dataset(user_features, labels):
    return user_features.join(labels, on="user_id")
```

When the upstream transaction data changes, Dagster marks `user_features` and `training_dataset` as stale. You can see at a glance which features need recomputation.

### Model Training and Evaluation

Trained models and evaluation reports are assets too. This brings model artifacts into the same lineage graph as the data that produced them:

```python
@dg.asset
def trained_model(context, training_dataset, mlflow: MLFlowResource):
    X = training_dataset.drop("label", axis=1)
    y = training_dataset["label"]
    model = XGBClassifier().fit(X, y)
    mlflow.log_model(model, "model")
    context.add_output_metadata({
        "num_features": X.shape[1],
        "training_rows": X.shape[0]
    })
    return model

@dg.asset
def evaluation_report(trained_model, test_dataset):
    predictions = trained_model.predict(test_dataset.drop("label", axis=1))
    accuracy = accuracy_score(test_dataset["label"], predictions)
    return {"accuracy": accuracy, "predictions": predictions}
```

### Multi-Model Patterns

Use `@multi_asset` when a single training run produces multiple outputs:

```python
@dg.multi_asset(
    outs={
        "champion_model": dg.AssetOut(),
        "challenger_model": dg.AssetOut(),
        "comparison_report": dg.AssetOut(),
    }
)
def model_comparison(training_dataset):
    champion = train_xgboost(training_dataset)
    challenger = train_lightgbm(training_dataset)
    report = compare_models(champion, challenger, training_dataset)
    return champion, challenger, report
```

### Retraining Automation

Combine sensors with asset-based jobs to automate retraining:

```python
retraining_job = dg.define_asset_job(
    name="retraining_job",
    selection=["training_dataset", "trained_model", "evaluation_report"]
)

@dg.sensor(job=retraining_job)
def drift_sensor(context, monitoring: MonitoringResource):
    drift_score = monitoring.get_drift_score("production_model")
    if drift_score > 0.15:
        yield dg.RunRequest(
            run_key=f"drift-{datetime.now().isoformat()}",
            run_config={"reason": "data_drift_detected"}
        )
```

This pattern creates a closed loop: monitor performance, detect drift, trigger retraining, and produce updated assets, all tracked in the same lineage graph.

## Testing

Dagster was designed with testability as a core concern. The framework supports several testing strategies that are particularly valuable for ML pipelines where correctness is difficult to verify in production.

### In-Process Execution

Assets and ops can be invoked directly as Python functions in tests. Dagster does not require a running instance or external services for testing:

```python
def test_cleaned_transactions():
    raw = pd.DataFrame({
        "amount": [100, None, 200, 200],
        "user_id": ["a", "b", "c", "c"]
    })
    result = cleaned_transactions(raw)
    assert len(result) == 2
    assert result["amount"].notna().all()
```

For assets that use context or resources, Dagster provides `build_asset_context` to construct a test context:

```python
def test_trained_model():
    context = dg.build_asset_context()
    mock_dataset = create_test_dataset()
    result = trained_model(context, mock_dataset, mlflow=MockMLFlowResource())
    assert result is not None
```

### Mocking Resources

Because resources are injected dependencies, swapping them for test doubles is straightforward. Define a mock resource that implements the same interface without side effects:

```python
class MockMLFlowResource(dg.ConfigurableResource):
    tracking_uri: str = "mock://localhost"
    logged_metrics: dict = {}

    def log_metrics(self, metrics: dict):
        self.logged_metrics.update(metrics)

def test_training_logs_metrics():
    mock_mlflow = MockMLFlowResource()
    context = dg.build_asset_context()
    trained_model(context, test_data, mlflow=mock_mlflow)
    assert "accuracy" in mock_mlflow.logged_metrics
```

### Materializing Assets in Tests

For integration tests, you can materialize assets within a test process using `materialize`:

```python
def test_full_pipeline():
    result = dg.materialize(
        assets=[raw_transactions, cleaned_transactions, user_features],
        resources={"io_manager": dg.mem_io_manager}
    )
    assert result.success
    assert result.output_for_node("user_features") is not None
```

The `mem_io_manager` keeps all intermediate data in memory, avoiding filesystem or network dependencies during tests. This enables fast, isolated integration tests that exercise the full asset graph.

### Testing Asset Checks

Asset checks can be tested independently:

```python
def test_no_negative_amounts_passes():
    valid_data = pd.DataFrame({"amount": [10, 20, 30]})
    result = no_negative_amounts(valid_data)
    assert result.passed

def test_no_negative_amounts_fails():
    invalid_data = pd.DataFrame({"amount": [10, -5, 30]})
    result = no_negative_amounts(invalid_data)
    assert not result.passed
```

## Dagster+ (Cloud) vs Open Source

Dagster is available in two forms: the open-source project and Dagster+ (formerly Dagster Cloud), a managed offering.

The open-source distribution includes the full orchestration engine, the Dagit web UI, all asset and op abstractions, IO managers, sensors, schedules, and the type system. You deploy and manage the infrastructure yourself, typically running the Dagster webserver, daemon, and database (PostgreSQL) on your own Kubernetes cluster or virtual machines.

Dagster+ adds a managed control plane and several features not available in OSS:

**Deployment options.** Dagster+ offers Serverless mode, where your code runs in Dagster's managed infrastructure, and Hybrid mode, where you run your own execution environment connected to the Dagster+ control plane. Serverless reduces operational overhead to near zero. Hybrid gives you control over compute and network boundaries while offloading UI hosting, metadata storage, and scheduling to Dagster+.

**Branch deployments.** Dagster+ can spin up isolated staging environments for each Git branch, allowing you to preview asset graph changes, test materializations, and validate before merging to production.

**Asset catalog.** A search-first interface for discovering assets across your organization, with column-level lineage, metadata browsing, and ownership tracking. This is aimed at data mesh architectures where multiple teams produce and consume assets.

**RBAC and audit logs.** Role-based access control for managing who can materialize, view, or configure assets. Audit logs provide compliance tracking for regulated environments.

**Insights and alerting.** Cost tracking, performance trend analysis, and integrations with Slack, PagerDuty, and email for failure and SLA violation alerts.

**Compliance.** Dagster+ is SOC 2 Type II certified, HIPAA compliant, and GDPR compliant.

For ML teams, the choice often comes down to operational maturity. Small teams or early-stage projects can start with OSS and Docker Compose. Production teams with multiple pipelines, compliance requirements, or cross-team asset sharing benefit from Dagster+.

## Comparison with Sibling Orchestrators

### Dagster vs Airflow

Airflow is a task-centric orchestrator. You define a DAG of tasks, each performing an operation. Dependencies are between tasks, not between data. Airflow excels at scheduling and has a massive ecosystem of operators, but it was designed for ETL workflows, not data-aware orchestration.

Key differences:

- Airflow models task dependencies. Dagster models data dependencies. In Airflow, a downstream task runs after an upstream task completes. In Dagster, a downstream asset materializes when its upstream assets are available.
- Airflow has no native concept of data lineage or staleness. You build lineage separately. Dagster tracks lineage automatically through the asset graph.
- Airflow testing requires mocking the Airflow context and often a running metadata database. Dagster assets can be tested as plain Python functions.
- Airflow's dynamic task mapping and task groups add flexibility but increase complexity. Dagster's partitions provide a cleaner model for processing data segments.
- Airflow has a larger ecosystem of pre-built operators and integrations. Dagster's ecosystem is growing but smaller.

Choose Airflow when you have heavy existing investment in Airflow DAGs, need specific operators from its ecosystem, or your team is already proficient with it. Choose Dagster when you want data lineage as a first-class concern, need strong testability, or are building new pipelines from scratch.

### Dagster vs Prefect

Prefect is a Python-native orchestrator that focuses on developer experience. It uses a flow-and-task model with decorators similar to Dagster's ops. Prefect emphasizes simplicity, dynamic workflows, and a managed cloud offering.

Key differences:

- Prefect is task-centric like Airflow, though with a more Pythonic API. Dagster is asset-centric.
- Prefect supports dynamic task creation and imperative control flow more naturally. Dagster's asset graph is more declarative and static.
- Prefect Cloud provides managed execution and observability. Dagster+ provides similar capabilities plus the asset catalog and branch deployments.
- Prefect's testing story is straightforward (flows are Python functions) but does not include built-in data validation. Dagster adds asset checks and type validation.

Choose Prefect when you need lightweight orchestration with dynamic workflows and prefer an imperative Python style. Choose Dagster when your primary concern is tracking and managing data assets across their lifecycle.

### Dagster vs Argo Workflows

Argo Workflows is a Kubernetes-native orchestrator that defines workflows as sequences of containers. It is language-agnostic, running any containerized workload, and integrates tightly with the Kubernetes ecosystem.

Key differences:

- Argo defines workflows in YAML. Dagster defines pipelines in Python. Argo is infrastructure-centric; Dagster is code-centric.
- Argo has no concept of data assets or lineage. It orchestrates containers. Dagster orchestrates data.
- Argo excels at heterogeneous workloads (Spark jobs, training on GPU nodes, serving containers) within Kubernetes. Dagster can run on Kubernetes but is not tied to it.
- Argo Workflows is often paired with Argo Events for event-driven triggers, Argo CD for deployment, and Argo Rollouts for progressive delivery. This composability is powerful but requires Kubernetes expertise.

Choose Argo when your ML platform is built on Kubernetes and you need to orchestrate containers running different languages and frameworks. Choose Dagster when you want a Python-first development experience with built-in data awareness.

## When to Choose Dagster

Dagster is a strong choice in the following situations:

**Data-centric teams.** If your team thinks in terms of data objects (tables, features, models, reports) rather than tasks, Dagster's asset model will match your mental model.

**ML pipelines with complex lineage.** When you need to trace how a model prediction connects back to raw data through feature engineering, training, and evaluation steps, the automatic lineage graph is invaluable.

**Strong testing requirements.** If your organization requires thorough testing of data pipelines, Dagster's in-process execution, resource mocking, and asset checks provide a testing story that other orchestrators lack.

**Gradual adoption.** Dagster does not require you to go all-in on assets. You can start with ops and jobs, then progressively migrate to assets as you identify stable data objects. The framework supports both models simultaneously.

**Multi-environment deployment.** The resource abstraction makes it straightforward to run the same pipeline against local files in development, staging databases in CI, and production warehouses in deployment, with configuration changes only.

**Data quality as a first-class concern.** Asset checks integrate validation into the orchestration layer rather than requiring separate data quality tools.

Dagster is less ideal when you need to orchestrate non-Python workloads across a Kubernetes cluster (consider Argo), when you have heavy existing Airflow infrastructure with no appetite for migration, or when your workflows are primarily imperative scripts with limited data dependencies (consider Prefect or a simple task runner).

## Definitions Object and Project Structure

All Dagster entities are collected into a top-level `Definitions` object that serves as the entry point for the Dagster runtime:

```python
defs = dg.Definitions(
    assets=[
        raw_transactions,
        cleaned_transactions,
        user_features,
        training_dataset,
        trained_model,
        evaluation_report
    ],
    asset_checks=[no_negative_amounts],
    resources={
        "mlflow": MLFlowResource(tracking_uri=os.environ["MLFLOW_URI"]),
        "io_manager": ParquetIOManager(base_path="/data/assets")
    },
    schedules=[training_schedule],
    sensors=[drift_sensor],
    jobs=[retraining_job]
)
```

A typical project structure for an ML pipeline in Dagster:

```
my_dagster_project/
    my_dagster_project/
        __init__.py
        definitions.py        # Top-level Definitions object
        assets/
            __init__.py
            ingestion.py      # Raw data assets
            features.py       # Feature engineering assets
            training.py       # Model training assets
            evaluation.py     # Evaluation and reporting assets
        resources/
            __init__.py
            mlflow.py
            storage.py
        asset_checks/
            __init__.py
            data_quality.py
    tests/
        test_assets.py
        test_resources.py
        test_integration.py
    pyproject.toml
```

Code locations allow you to deploy multiple independent projects to the same Dagster instance, each with its own Python environment. This supports team-based ownership where different teams maintain different code locations while sharing a unified asset catalog.

## Deployment Architecture

A Dagster deployment consists of several components:

- **Webserver (Dagit):** The web UI for viewing the asset graph, launching runs, monitoring execution, and browsing metadata.
- **Daemon:** A background process that evaluates schedules and sensors, launches runs, and manages the run queue.
- **Run workers:** Processes that execute the actual asset materializations and op computations. These can run in-process, as separate processes, or as Kubernetes jobs.
- **Storage:** A PostgreSQL database for run history, event logs, and schedule state. SQLite is supported for local development.

For production deployments, the common pattern is to run the webserver and daemon as long-lived services on Kubernetes, with run workers launched as Kubernetes jobs. This provides isolation between runs and horizontal scaling. The Dagster Helm chart provides a standard deployment template for this architecture.
