# Hopsworks

## Summary

Hopsworks is an open-source ML platform that provides a complete feature store alongside model training, serving, and monitoring capabilities. It offers a unified architecture for batch, real-time, and LLM-based AI systems through its FTI (Feature/Training/Inference) pipeline framework. Hopsworks can be self-hosted or run as a managed service, with support for deployment on AWS, Azure, GCP, and on-premises environments.

Key points to remember:

- Comprehensive ML platform beyond just feature storage
- FTI pipeline architecture: feature pipelines, training pipelines, inference pipelines
- RonDB provides ultra-low-latency online feature serving
- Supports Spark, PySpark, Flink, and Pandas for feature engineering
- Kafka integration for streaming features and model monitoring
- External feature groups connect to Snowflake, Databricks, BigQuery, Redshift
- Built-in vector database via OpenSearch for embedding storage
- Available as open-source, managed cloud, or on-premises deployment
- Compared to Feast, Hopsworks is a full ML platform vs focused feature store
- Compared to Tecton, Hopsworks offers more flexibility and self-hosting options

## Architecture

### Platform Components

**Feature Store**
Core feature management layer:
- Feature groups for organizing features
- Offline store for training data
- Online store (RonDB) for low-latency serving
- Point-in-time correct joins

**HopsFS**
Distributed file system:
- Based on HDFS with enhanced metadata
- Stores feature data, models, and datasets
- Supports object storage backends (S3, GCS, Azure Blob)

**RonDB**
Online feature store:
- Developed by Hopsworks for ultra-low latency
- Highest throughput feature serving
- High availability with automatic failover

**Kafka**
Streaming infrastructure:
- Feature pipeline streaming
- Model prediction logging
- Monitoring data collection
- Supports bring-your-own Kafka (BYOK)

**OpenSearch**
Vector database capabilities:
- Embedding storage and similarity search
- kNN search via FAISS and nmslib
- Integrated access control and filtering

### FTI Pipeline Architecture

Hopsworks structures ML systems as three independent pipelines:

```
Raw Data ’ Feature Pipeline ’ Feature Store
                                    “
          Training Pipeline ’ Model Registry
                                    “
                  Model ’ Inference Pipeline ’ Predictions
                              ‘
                        Feature Store
```

**Feature Pipelines**
Transform raw data into features:
- Batch: Spark, PySpark, Pandas
- Streaming: Flink, Spark Streaming
- Store results in feature groups

**Training Pipelines**
Create models from features:
- Read from feature store
- Train with any framework
- Register models in model registry

**Inference Pipelines**
Serve predictions:
- Fetch features from online store
- Run model inference
- Log predictions to Kafka

## Feature Groups

### Creating Feature Groups

```python
import hopsworks

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group
customer_fg = fs.create_feature_group(
    name="customer_features",
    version=1,
    description="Customer transaction aggregates",
    primary_key=["customer_id"],
    event_time="event_timestamp",
    online_enabled=True,
)

# Insert data
customer_fg.insert(customer_df)
```

### Feature Group Types

**Internal Feature Groups**
Stored in Hopsworks (HopsFS + RonDB):

```python
fg = fs.create_feature_group(
    name="transactions",
    version=1,
    primary_key=["transaction_id"],
    event_time="timestamp",
    online_enabled=True,
)
```

**External Feature Groups**
Connect to external data sources:

```python
# Snowflake external feature group
external_fg = fs.create_external_feature_group(
    name="snowflake_features",
    version=1,
    query="SELECT * FROM analytics.customer_features",
    storage_connector=snowflake_connector,
    primary_key=["customer_id"],
    event_time="updated_at",
)
```

Supported external sources:
- Snowflake
- Databricks
- BigQuery
- Redshift
- Any JDBC-enabled database

### Streaming Feature Groups

```python
# Create streaming feature group
streaming_fg = fs.create_feature_group(
    name="realtime_features",
    version=1,
    primary_key=["user_id"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
)

# Insert from streaming pipeline
streaming_fg.insert_stream(
    kafka_topic="user_events",
    kafka_config=kafka_config,
)
```

## Feature Engineering

### Spark/PySpark

```python
from pyspark.sql import functions as F

# Feature engineering with PySpark
features_df = (
    transactions_df
    .groupBy("customer_id")
    .agg(
        F.sum("amount").alias("total_spend"),
        F.count("*").alias("transaction_count"),
        F.avg("amount").alias("avg_transaction"),
        F.max("timestamp").alias("last_transaction"),
    )
)

# Write to feature group
customer_fg.insert(features_df)
```

### Pandas

```python
import pandas as pd

# Feature engineering with Pandas
features_df = (
    transactions_df
    .groupby("customer_id")
    .agg({
        "amount": ["sum", "mean", "count"],
        "timestamp": "max"
    })
)
features_df.columns = [
    "total_spend", "avg_transaction",
    "transaction_count", "last_transaction"
]

# Write to feature group
customer_fg.insert(features_df)
```

### Pandas UDFs (Recommended for Scale)

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def calculate_velocity(amounts: pd.Series, times: pd.Series) -> pd.Series:
    return amounts / (times.max() - times.min()).total_seconds()

# Apply in Spark pipeline
df = df.withColumn("spend_velocity", calculate_velocity("amount", "timestamp"))
```

## Training Datasets

### Creating Training Data

```python
# Select features from multiple feature groups
query = fs.select_all(customer_fg) \
    .join(transaction_fg, on=["customer_id"]) \
    .filter(customer_fg.active == True)

# Create training dataset with point-in-time correctness
td = fs.create_training_dataset(
    name="churn_training",
    version=1,
    description="Churn prediction training data",
    label=["churned"],
    splits={
        "train": 0.8,
        "test": 0.2
    },
    data_format="parquet",
    statistics_config={"enabled": True}
)

# Save training data
td.save(query)
```

### Point-in-Time Joins

```python
# Entity dataframe with labels and timestamps
entity_df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "event_timestamp": ["2024-01-15", "2024-01-16", "2024-01-17"],
    "churned": [0, 1, 0]
})

# Get features as of each timestamp
training_data = fs.create_training_data(
    entity_df=entity_df,
    feature_groups=[customer_fg, transaction_fg],
    event_time="event_timestamp",
)
```

### Training Dataset Formats

Supported formats:
- Parquet
- TFRecords
- CSV/TSV
- HDF5
- NumPy (.npy)

## Online Feature Serving

### Low-Latency Retrieval

```python
# Get online features
features = customer_fg.get_online_features(
    entry={"customer_id": "cust_123"}
)

# Batch retrieval
features = customer_fg.get_online_features(
    entries=[
        {"customer_id": "cust_123"},
        {"customer_id": "cust_456"},
    ]
)
```

### Feature Vectors

```python
# Create feature view for serving
feature_view = fs.create_feature_view(
    name="churn_features",
    version=1,
    query=query,
    labels=["churned"],
)

# Get feature vector for inference
vector = feature_view.get_feature_vector(
    entry={"customer_id": "cust_123"}
)

# Use in model
prediction = model.predict([vector])
```

## Model Serving

### Deploy Models

```python
from hsml import connection

# Connect to model registry
conn = connection()
mr = conn.get_model_registry()

# Get model
model = mr.get_model("churn_model", version=1)

# Deploy
deployment = model.deploy(
    name="churn_predictor",
    serving_tool="kserve",
    resources={
        "requests": {"cpu": "500m", "memory": "1Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"}
    }
)
```

### Inference with Features

```python
# Get features and predict
features = feature_view.get_feature_vector(
    entry={"customer_id": "cust_123"}
)

prediction = deployment.predict({"instances": [features]})
```

### Prediction Logging

Predictions automatically logged to Kafka for monitoring:

```python
# Enable logging
deployment = model.deploy(
    name="churn_predictor",
    logging_enabled=True,
    kafka_topic="model_predictions",
)
```

## Vector Database

### Embedding Storage

```python
# Create embedding feature group
embedding_fg = fs.create_feature_group(
    name="document_embeddings",
    version=1,
    primary_key=["doc_id"],
    embedding_index=True,
    features=[
        Feature("doc_id", str),
        Feature("embedding", list, online_type="embedding(768)"),
        Feature("text", str),
        Feature("category", str),
    ]
)

# Insert embeddings
embedding_fg.insert(documents_with_embeddings)
```

### Similarity Search

```python
# Query similar documents
results = embedding_fg.find_neighbors(
    col="embedding",
    vector=query_embedding,
    k=10,
    filter=("category", "==", "technology")
)
```

Powered by:
- OpenSearch kNN
- FAISS and nmslib backends
- Supports filtering alongside similarity search

## Monitoring

### Feature Monitoring

```python
# Enable statistics
fg = fs.create_feature_group(
    name="customer_features",
    statistics_config={
        "enabled": True,
        "histograms": True,
        "correlations": True
    }
)

# Get statistics
stats = fg.get_statistics()
```

### Data Validation

```python
from great_expectations import ExpectationSuite

# Define expectations
suite = ExpectationSuite("customer_expectations")
suite.add_expectation(
    expect_column_values_to_be_between,
    column="age",
    min_value=0,
    max_value=120
)

# Attach to feature group
fg.attach_expectation_suite(suite)
```

### Model Monitoring

- Prediction logging to Kafka
- Grafana/Prometheus integration
- Drift detection
- Performance metrics

## Deployment Options

### Managed Cloud

```bash
# Hopsworks.ai - managed service
# Available on AWS, Azure, GCP
```

### Self-Hosted Kubernetes

```bash
# Install via Helm
helm repo add hopsworks https://charts.hopsworks.ai
helm install hopsworks hopsworks/hopsworks
```

### On-Premises

- Ubuntu/RedHat compatible
- Air-gapped deployment supported
- Full enterprise features

## Comparison with Alternatives

### Hopsworks vs Feast

| Aspect | Hopsworks | Feast |
|--------|-----------|-------|
| Scope | Full ML platform | Feature store only |
| Model Serving | Included (KServe) | External |
| Vector DB | Included | External |
| Streaming | Native Kafka | Limited |
| Complexity | Higher | Lower |
| Self-hosting | Kubernetes | Simpler |

### Hopsworks vs Tecton

| Aspect | Hopsworks | Tecton |
|--------|-----------|--------|
| Hosting | Self or managed | Managed only |
| Open Source | Yes | No |
| ML Platform | Full | Features only |
| Customization | Full control | Limited |
| Cost | Self-host free | Enterprise |

## When to Use Hopsworks

Hopsworks is well-suited for:
- Organizations wanting complete ML platform
- Teams needing self-hosted deployment
- Use cases requiring vector database integration
- Complex streaming feature pipelines
- Enterprise requiring on-premises deployment

Consider alternatives when:
- Only feature serving needed (Feast, Tecton)
- Minimal infrastructure desired (Tecton)
- Quick prototyping focus (Feast)
- Tight budget without engineering resources
