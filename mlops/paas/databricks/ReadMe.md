# Databricks

## Summary

Databricks is a unified analytics platform built on Apache Spark that has evolved into one of the most comprehensive ML Platform-as-a-Service offerings available today. Originally founded by the creators of Spark, Delta Lake, and MLflow, Databricks positions itself around the "lakehouse" architecture, which merges data lake flexibility with data warehouse reliability. For ML engineers, this means a single platform where data engineering, feature engineering, model training, experiment tracking, and model serving all coexist without the friction of moving data between disparate systems.

The platform's ML capabilities are now branded under Mosaic AI (following the 2023 acquisition of MosaicML), which encompasses everything from classical ML workflows to large language model training and serving. Unlike competitors that bolt ML tooling onto an existing cloud console, Databricks treats the notebook-driven data science workflow as a first-class citizen, with deep integrations between Spark, MLflow, and the underlying Delta Lake storage layer.

Key points to remember:

- Databricks provides managed MLflow as its experiment tracking and model registry backbone, tightly integrated with Unity Catalog for governance
- The Feature Store uses Delta Lake tables as the storage layer, with automatic lineage tracking and both online and offline serving
- Model Serving offers serverless REST endpoints with autoscaling, GPU support, and sub-50ms overhead latency at over 25K QPS
- Databricks Runtime ML ships pre-configured environments with PyTorch, TensorFlow, and distributed training libraries
- The lakehouse architecture eliminates ETL between data engineering and ML workflows because training data and features live in the same Delta Lake tables that data engineers already maintain
- Mosaic AI brings LLM pre-training, fine-tuning, and optimized inference to the platform
- Databricks runs on AWS, Azure, and GCP, making it a cloud-agnostic choice compared to native offerings like SageMaker or Vertex AI

## Lakehouse Architecture and Why It Matters for ML

The lakehouse is the architectural foundation that differentiates Databricks from other ML platforms. Understanding it is essential to understanding why teams choose Databricks for ML workloads.

### Delta Lake

Delta Lake is an open-source storage layer that brings ACID transactions, schema enforcement, and time travel to data lakes. For ML engineers, the practical implications are significant:

- Training datasets are versioned automatically. You can reproduce a training run from six months ago by querying the Delta table as of a specific timestamp.
- Schema enforcement prevents silent data corruption. If an upstream pipeline changes a column type, your feature pipeline fails loudly rather than producing garbage features.
- MERGE operations enable incremental feature computation. Instead of recomputing an entire feature table, you upsert only changed rows.

```python
# Time travel: read training data as it existed at a specific point
df = spark.read.format("delta").option("timestampAsOf", "2025-01-15").table("ml.features.user_signals")

# Or by version number
df = spark.read.format("delta").option("versionAsOf", 42).table("ml.features.user_signals")
```

### Unity Catalog

Unity Catalog is Databricks' governance layer for all data and AI assets. For ML specifically, it governs:

- Feature tables and their lineage (which upstream tables feed which features)
- Registered models and model versions
- Functions (including user-defined functions used in feature engineering)
- Volumes (for storing unstructured data like images or model artifacts)

Unity Catalog enforces access control at the column and row level, which matters when ML teams need access to sensitive data for training but should not have unrestricted access to PII.

```sql
-- Grant an ML team access to feature tables but not raw PII
GRANT SELECT ON TABLE ml.features.user_signals TO `ml-team`;
GRANT USE SCHEMA ON SCHEMA ml.features TO `ml-team`;
```

### Why This Matters

On most cloud ML platforms, data lives in a warehouse or lake, features get copied into a feature store, models reference artifacts in yet another storage system, and governance is an afterthought. Databricks collapses these layers. A feature table is just a Delta table with metadata. A registered model references artifacts stored in Unity Catalog volumes. Lineage flows from raw data through feature engineering through model training without crossing system boundaries.

For organizations where the data engineering team already uses Databricks (or Spark), this convergence eliminates an entire class of integration work.

## Databricks Runtime ML

Databricks Runtime ML is a variant of the standard Databricks Runtime that comes pre-installed with popular ML and deep learning libraries. Each runtime version is pinned to a specific set of library versions, providing reproducible environments without Dockerfile management.

A typical Runtime ML installation includes:

- PyTorch and TensorFlow (with GPU support when running on GPU instances)
- scikit-learn, XGBoost, LightGBM
- Hugging Face Transformers
- MLflow (pre-configured to log to the workspace tracking server)
- Horovod and PyTorch Distributed for multi-node training
- Ray (in newer runtimes) for distributed compute
- Hyperopt for hyperparameter tuning

The runtime handles CUDA driver compatibility, NCCL configuration for multi-GPU communication, and library version conflicts, which are some of the most time-consuming aspects of setting up ML training environments from scratch.

```python
# On a Databricks cluster with Runtime ML, these are all available immediately
import torch
import tensorflow as tf
import mlflow
import horovod.torch as hvd
from transformers import AutoModelForSequenceClassification
```

When you need libraries beyond what the runtime provides, you can install them at the cluster level, the notebook level, or via cluster init scripts. For production jobs, cluster policies can lock down which libraries are permitted.

## Training on Databricks

### Notebook-Driven Development

The primary development interface on Databricks is the notebook. Notebooks support Python, Scala, SQL, and R, and they connect to Spark clusters that scale from a single node to hundreds. For ML, the typical workflow is:

1. Use SQL or PySpark to explore and prepare training data in Delta Lake
2. Pull a DataFrame into Pandas (or use Spark MLlib directly) for model training
3. Log experiments to MLflow automatically or manually
4. Register promising models to the Model Registry via Unity Catalog

```python
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Read features from Delta Lake
df = spark.table("ml.features.fraud_signals").toPandas()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("label", axis=1), df["label"], test_size=0.2
)

mlflow.autolog()

with mlflow.start_run():
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    # autolog captures parameters, metrics, model artifact, and signature
```

### Distributed Training

For models that exceed single-node capacity, Databricks supports several distributed training paradigms:

**Spark MLlib** is native to the platform and works well for traditional ML at scale (logistic regression, random forests, gradient boosting on large datasets):

```python
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100)

pipeline = Pipeline(stages=[assembler, gbt])
model = pipeline.fit(train_df)
```

**Horovod** enables data-parallel distributed deep learning across multiple GPUs and nodes. Databricks provides the HorovodRunner wrapper to simplify launching distributed training from a notebook:

```python
from sparkdl import HorovodRunner

def train_fn():
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(10):
        train_epoch(model, optimizer, train_loader)

hr = HorovodRunner(np=4)  # 4 GPUs
hr.run(train_fn)
```

**PyTorch Distributed (via TorchDistributor)** is the newer and now preferred approach for distributed PyTorch training on Databricks:

```python
from pyspark.ml.torch.distributor import TorchDistributor

def train_fn():
    import torch.distributed as dist
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    model = MyModel().cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # training loop
    return model

distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True
)
model = distributor.run(train_fn)
```

**Ray on Databricks** integrates Ray clusters with Spark clusters, providing access to Ray Train, Ray Tune, and other Ray libraries for distributed training and hyperparameter search.

### AutoML

Databricks AutoML automates the process of building baseline models. Given a training dataset and a target column, it generates a set of notebooks, each implementing a different approach (XGBoost, LightGBM, sklearn, etc.) with hyperparameter tuning via Hyperopt. The generated notebooks are fully editable, which distinguishes this from black-box AutoML services.

```python
from databricks import automl

summary = automl.classify(
    dataset=spark.table("ml.features.churn_signals"),
    target_col="churned",
    timeout_minutes=30
)

# View best trial
print(summary.best_trial)
```

AutoML is useful for establishing baselines quickly and for teams that want a starting point they can customize rather than a fully opaque model.

## Feature Store

The Databricks Feature Store provides centralized management of features used in ML models. Feature tables are Delta tables registered in Unity Catalog, which means they benefit from all the governance, lineage, and versioning capabilities of both systems.

### Creating Feature Tables

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create a feature table from a Spark DataFrame
fe.create_table(
    name="ml.features.user_signals",
    primary_keys=["user_id"],
    timestamp_keys=["event_date"],
    df=feature_df,
    description="User behavioral signals for fraud detection"
)

# Update features incrementally
fe.write_table(
    name="ml.features.user_signals",
    df=new_features_df,
    mode="merge"
)
```

### Training with Feature Lookups

When training a model, you specify which features to look up from which tables. The Feature Store records this lineage so that at inference time, it knows which features the model needs:

```python
from databricks.feature_engineering import FeatureLookup

training_set = fe.create_training_set(
    df=labels_df,  # DataFrame with primary keys and labels
    feature_lookups=[
        FeatureLookup(
            table_name="ml.features.user_signals",
            lookup_key=["user_id"]
        ),
        FeatureLookup(
            table_name="ml.features.transaction_signals",
            lookup_key=["user_id"],
            feature_names=["avg_transaction_amount", "transaction_count_7d"]
        )
    ],
    label="is_fraud"
)

training_df = training_set.load_df()
```

### Online Serving

For real-time inference, features need to be served with low latency. Databricks provides online tables that sync from the offline Delta tables and serve features via REST API with millisecond latency. This eliminates the training-serving skew problem because the same feature definitions are used in both contexts.

### On-Demand Features

For features that must be computed at request time (such as the time elapsed since a user's last login), Databricks supports on-demand feature computation using Python functions registered as Unity Catalog functions. These are executed during inference alongside the lookup of pre-computed features.

## MLflow on Databricks

While MLflow is available as an open-source tool anywhere, the Databricks-managed version adds several capabilities:

- Automatic experiment tracking tied to notebooks (each notebook gets a default experiment)
- Unity Catalog integration for the Model Registry, providing fine-grained access control on registered models
- Managed artifact storage without needing to configure S3 buckets or GCS
- Model lineage from data source through features through training run through registered model
- Integrated model evaluation tools for comparing runs

The Model Registry in Unity Catalog replaces the older workspace-level registry. Models are registered with three-level namespacing (catalog.schema.model_name) and inherit the governance policies of their catalog:

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

# Register a model to Unity Catalog
model_info = mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="ml.production.fraud_detector"
)

# Set an alias for deployment
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("ml.production.fraud_detector", "champion", model_info.version)
```

Note that Unity Catalog's model registry uses aliases (like "champion" and "challenger") instead of the older stage-based transitions (Staging, Production, Archived). This is a deliberate design change that supports more flexible promotion workflows.

## Model Serving

Databricks Model Serving deploys models as serverless REST endpoints with automatic scaling. The service handles provisioning, load balancing, and scaling to zero when endpoints are idle.

### Deploying a Custom Model

```python
import requests

# Create a serving endpoint via the REST API
endpoint_config = {
    "name": "fraud-detector",
    "config": {
        "served_entities": [
            {
                "entity_name": "ml.production.fraud_detector",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": "fraud_detector-1", "traffic_percentage": 100}
            ]
        }
    }
}
```

### Key Capabilities

- Serverless compute with automatic scaling from zero to thousands of QPS
- GPU-backed endpoints for deep learning models and LLMs
- Traffic splitting for A/B testing and canary deployments
- Sub-50ms overhead latency at production scale
- Automatic feature lookup at inference time when the model was trained with the Feature Store
- Pay-per-token pricing for hosted foundation models
- Provisioned throughput for performance-guaranteed LLM serving

### Querying an Endpoint

```python
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint="fraud-detector",
    inputs={"dataframe_records": [{"user_id": "abc123", "transaction_amount": 500.0}]}
)
```

When the model was trained with Feature Store lookups, the endpoint automatically fetches the required features from online tables at inference time. You only need to pass the primary keys and any real-time inputs.

## Mosaic AI

Databricks acquired MosaicML in 2023, bringing large language model training and serving capabilities into the platform. The resulting Mosaic AI suite includes:

### Foundation Model Training

Mosaic AI provides infrastructure for pre-training and fine-tuning LLMs on Databricks. This includes the Composer training library (optimized for efficient LLM training), support for multi-node GPU training across hundreds of GPUs, and integration with the lakehouse for training data management.

Fine-tuning is available as a managed API:

```python
from databricks.model_training import foundation_model as fm

run = fm.create(
    model="meta-llama/Meta-Llama-3.1-8B",
    train_data_path="dbfs:/datasets/fine_tune/train.jsonl",
    eval_data_path="dbfs:/datasets/fine_tune/eval.jsonl",
    register_to="ml.llms.custom_llama",
    training_duration="1ep",
    learning_rate="2e-5"
)
```

### Foundation Model APIs

Databricks hosts popular open-source models (Llama, Mistral, DBRX, and others) as pay-per-token endpoints. These are accessible via an OpenAI-compatible API, making it straightforward to swap providers:

```python
from openai import OpenAI

client = OpenAI(
    api_key=db_token,
    base_url="https://<workspace>.databricks.com/serving-endpoints"
)

response = client.chat.completions.create(
    model="databricks-meta-llama-3-1-70b-instruct",
    messages=[{"role": "user", "content": "Explain gradient boosting."}],
    max_tokens=256
)
```

### AI Gateway

The AI Gateway provides a unified interface for routing requests to different model providers (Databricks-hosted, OpenAI, Anthropic, etc.) with centralized rate limiting, cost tracking, and guardrails. This is particularly useful for organizations that want to standardize their LLM access patterns while retaining the flexibility to switch providers.

## Comparison with Other ML PaaS Offerings

### Databricks vs. Amazon SageMaker

SageMaker is deeply integrated with the AWS ecosystem and provides more granular control over training infrastructure (instance types, spot training, distributed strategies). SageMaker Pipelines offers a more structured, production-oriented workflow compared to Databricks notebooks. However, SageMaker's model development experience is more fragmented: Studio, Canvas, notebooks, and processing jobs are distinct concepts that do not share a unified data layer.

Databricks advantages over SageMaker:
- Unified data layer via Delta Lake eliminates data movement between engineering and ML
- Cloud-agnostic deployment (AWS, Azure, GCP) versus SageMaker's AWS lock-in
- Native Spark integration for teams already using Spark for data processing
- More cohesive experience from data exploration through deployment

SageMaker advantages over Databricks:
- Deeper AWS integration (IAM, VPC, PrivateLink, more instance type options)
- More mature MLOps pipeline tooling with SageMaker Pipelines
- Built-in model monitoring and bias detection (Clarify)
- Better support for edge deployment with Neo

### Databricks vs. Google Vertex AI

Vertex AI excels at AutoML and provides strong integration with BigQuery and the Google Cloud AI ecosystem. Its model garden and generative AI studio are polished offerings for LLM workflows.

Databricks advantages over Vertex AI:
- Superior Spark and big data integration
- Cloud-agnostic deployment
- More transparent AutoML (generates editable notebooks)
- Stronger data engineering convergence

Vertex AI advantages over Databricks:
- Better AutoML for structured data tasks with minimal configuration
- Deeper BigQuery integration for GCP-native analytics
- First-party access to Google models (Gemini)
- Stronger managed pipeline tooling with Vertex AI Pipelines

### Databricks vs. Azure Machine Learning

Azure ML and Databricks both run on Azure, and Microsoft is a major Databricks investor, which creates an interesting dynamic. Azure ML provides a more traditional ML platform experience with designer, pipelines, and managed endpoints.

Databricks advantages over Azure ML:
- Unified lakehouse architecture (Azure ML requires moving data between storage systems)
- Better Spark integration and distributed compute story
- MLflow is native rather than a compatibility layer
- Available on all three major clouds

Azure ML advantages over Databricks:
- Tighter Azure DevOps and Azure Pipelines integration
- Responsible AI dashboard and model interpretability tooling
- More flexible compute options (including AKS for serving)
- Better integration with Microsoft Fabric for organizations in the Microsoft ecosystem

### Summary Table

| Capability | Databricks | SageMaker | Vertex AI | Azure ML |
|---|---|---|---|---|
| Cloud | AWS, Azure, GCP | AWS only | GCP only | Azure only |
| Data Layer | Delta Lake (unified) | S3 (separate) | BigQuery/GCS | Azure Storage |
| Experiment Tracking | MLflow (native) | Experiments | Vertex Experiments | MLflow (compat) |
| Feature Store | Unity Catalog | Feature Store | Feature Store | Feature Store |
| Distributed Training | Spark, Horovod, Ray | Built-in distributed | Built-in distributed | Built-in distributed |
| LLM Training | Mosaic AI | JumpStart, Bedrock | Model Garden, Gemini | Azure OpenAI |
| Notebook Experience | Primary interface | Studio notebooks | Workbench | Studio notebooks |

## When to Choose Databricks

Databricks is the strongest choice when:

- Your organization already uses Spark or Databricks for data engineering. The marginal cost of adopting ML on Databricks is low when the data is already in Delta Lake and the team already knows the platform.
- You need cloud portability. Databricks runs on AWS, Azure, and GCP with a consistent API, which matters for multi-cloud organizations or those that want to avoid vendor lock-in.
- Data engineering and ML teams need to converge. The lakehouse architecture means feature engineering is just data engineering, reducing organizational friction.
- You are working with large-scale tabular data. Spark-native feature engineering and distributed training are core strengths.
- You want a managed MLflow experience. If you are already using MLflow (or planning to), Databricks provides the most integrated managed version.

Databricks is a weaker choice when:

- You need deep integration with a single cloud provider's ecosystem (prefer SageMaker on AWS, Vertex on GCP, Azure ML on Azure for maximum native integration)
- Your ML workloads are primarily computer vision or NLP on unstructured data without a significant data engineering component
- You need advanced MLOps pipeline orchestration (Databricks Workflows exists but is less mature than SageMaker Pipelines or Vertex Pipelines for complex DAGs)
- Cost sensitivity is paramount and your workloads are small enough that Spark overhead is not justified

## Practical Tips

### Cost Management

Databricks billing is based on Databricks Units (DBUs) consumed, which vary by instance type and runtime. ML workloads, especially GPU training, can become expensive quickly.

- Use autoscaling clusters with aggressive scale-down for interactive development. Set min workers to 0 when possible.
- For training jobs, use Databricks Jobs with job clusters that terminate after the run completes, rather than long-running interactive clusters.
- Consider spot instances (called "Spot" on AWS, "Low Priority" on Azure, "Preemptible" on GCP) for fault-tolerant training. Checkpointing is essential when using spot.
- Use single-node clusters for development and small experiments. Not everything needs Spark.
- Monitor costs via the account console and set budget alerts. GPU clusters cost 5-10x more per hour than CPU clusters.

### Environment Management

- Pin your Runtime ML version in production jobs. Do not use "latest" because library version changes can break training pipelines.
- Use cluster-scoped libraries for dependencies needed by all notebooks on a cluster, and notebook-scoped libraries (via %pip install) for one-off experiments.
- For complex dependency trees, build a custom Docker container based on the Databricks Runtime ML base image and use it as a custom container runtime.

### Workflow Organization

- Use Databricks Repos to sync notebooks with Git. This enables code review, branching, and CI/CD for notebook-based ML code.
- Structure projects with a clear separation: one notebook/script for feature engineering, one for training, one for evaluation, orchestrated by a Databricks Workflow.
- Use MLflow experiments with consistent naming conventions. A good pattern is `/{team}/{project}/{experiment_type}`.

### Production Readiness

- Move from interactive notebooks to Databricks Jobs for production training. Jobs provide scheduling, retries, alerting, and cost controls that interactive clusters do not.
- Use Unity Catalog model aliases ("champion", "challenger") to manage deployment without hardcoding model versions.
- Enable Model Serving endpoint autoscaling and set scale-to-zero for non-latency-critical endpoints to reduce cost.
- Set up monitoring for model serving endpoints using Databricks system tables, which capture request logs, latency, and error rates.
- Use Databricks Workflows to build retraining pipelines that run on a schedule, retrain models, evaluate against the current champion, and promote automatically if the new model wins.

### Common Pitfalls

- Running all development on GPU clusters when CPU is sufficient for data preparation and feature engineering. Separate your compute by workload type.
- Not using Delta Lake time travel for dataset versioning. If you are manually snapshotting CSVs for training reproducibility, you are doing unnecessary work.
- Ignoring Unity Catalog governance. It is tempting to dump everything into a "default" catalog, but proper catalog/schema organization pays dividends when the team grows.
- Using the workspace-level MLflow model registry instead of the Unity Catalog registry. The workspace registry is legacy and lacks cross-workspace sharing and fine-grained access control.
- Over-distributing training. Not every model needs Spark or Horovod. If your dataset fits in memory on a single node, single-node training with a GPU is simpler and often faster.
