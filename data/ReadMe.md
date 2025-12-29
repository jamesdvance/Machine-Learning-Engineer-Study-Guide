# Data Engineering for Machine Learning

## Summary

Data engineering forms the foundation of every ML system. The quality, accessibility, and reliability of data directly determine model performance and system reliability. This section covers the four pillars of ML data infrastructure: storage systems for persisting data at scale, formats for efficient serialization and interchange, processing for transforming data into features, and versioning for reproducibility and governance.

Key points to remember:

- Storage: Choose specialized systems for each access pattern (object storage, warehouses, feature stores, vector databases)
- Formats: Binary formats (Parquet) for analytics, text formats (JSON) for interchange, ML-specific formats (Safetensors) for models
- Processing: Batch for throughput, streaming for latency, ELT for modern data integration
- Versioning: Version data alongside code for reproducible experiments
- The modern ML stack combines multiple specialized tools rather than one-size-fits-all solutions
- Data quality and accessibility often matter more than model architecture

## The ML Data Lifecycle

```
Data Sources                     Data Platform                      ML System
+-------------+                +------------------+               +------------------+
| Databases   |                |                  |               |                  |
| APIs        |---> Ingestion -+-> Raw Storage    |               |                  |
| Events      |                |   (Object Store) |               |                  |
| Files       |                |        |         |               |                  |
+-------------+                |        v         |               |                  |
                               |   Processing     |               |                  |
                               |   (Batch/Stream) |               |                  |
                               |        |         |               |                  |
                               |        v         |               |                  |
                               |   Curated Storage|---> Features -+-> Training       |
                               |   (Warehouse)    |               |                  |
                               |        |         |               |                  |
                               |        v         |               |                  |
                               |   Feature Store  |---> Serving --+-> Inference      |
                               |   (Online/Offline)|              |                  |
                               +------------------+               +------------------+
```

## Storage Decisions

### Storage Taxonomy

| Storage Type | Access Pattern | Use Case |
|--------------|----------------|----------|
| Object Storage | Batch read/write | Raw data, model artifacts |
| Data Warehouse | Analytical queries | Feature engineering, SQL analytics |
| Data Lake | Mixed workloads | Large-scale processing |
| Feature Store | Training + serving | ML feature management |
| Vector Database | Similarity search | RAG, embeddings, recommendations |

### Decision Framework

```
What are you storing?

Raw Data (logs, events, files)
  -> Object Storage (S3, GCS, Azure Blob)

Structured Data for Analytics
  -> Data Warehouse (Snowflake, BigQuery, Redshift)

Large-Scale Processing Data
  -> Data Lake with Table Format (Delta Lake, Iceberg)

ML Features
  -> Feature Store (Feast, Tecton, Hopsworks)

Embeddings for Search
  -> Vector Database (Pinecone, Milvus, Qdrant)

Model Weights and Artifacts
  -> Object Storage + Versioning (DVC, MLflow)
```

### Cost vs Performance

| Priority | Storage Choice |
|----------|---------------|
| Lowest cost | Object storage (cold tier) |
| Best query performance | Data warehouse |
| Lowest latency | Feature store online store |
| Best flexibility | Data lake with table format |

## Format Decisions

### Format Categories

| Category | Formats | Use Case |
|----------|---------|----------|
| Text | CSV, JSON, JSONL | Configuration, interchange, small data |
| Binary Tabular | Parquet, ORC, Avro | Large datasets, analytics |
| ML-Specific | TFRecord, Safetensors, ONNX | Training data, model weights |
| Media | JPEG, PNG, H.264, VP9 | Vision and video ML |

### Decision Framework

```
What type of data?

Configuration/API Interchange
  -> JSON (hierarchical) or CSV (tabular)

Large Tabular Dataset
  -> Parquet (columnar analytics) or Avro (streaming)

Training Data (TensorFlow)
  -> TFRecord

Model Weights
  -> Safetensors (secure, fast)

Model Deployment
  -> ONNX (cross-framework)

Images (training)
  -> WebP (balanced) or PNG (lossless labels)

Video (training)
  -> H.264 (compatibility) or VP9 (efficiency)
```

### Format Trade-offs

| Trade-off | Choose Left | Choose Right |
|-----------|-------------|--------------|
| Human readable vs Efficient | JSON/CSV | Parquet/Avro |
| Lossy vs Lossless | JPEG/H.264 | PNG/TIFF |
| Compatibility vs Security | Pickle | Safetensors |
| Flexibility vs Performance | JSON | Binary formats |

## Processing Decisions

### Processing Paradigms

| Paradigm | Latency | Best For |
|----------|---------|----------|
| Batch | Minutes to hours | Historical features, training data |
| Micro-batch | 100ms+ | Near-real-time with Spark |
| Streaming | Sub-second | Real-time features |
| ELT | Variable | Data integration |

### Decision Framework

```
What are the latency requirements?

Daily/Hourly Updates OK
  -> Batch Processing (Spark, Dask)

Minutes Acceptable
  -> Micro-batch (Spark Streaming)

Sub-second Required
  -> True Streaming (Flink, Kafka)

Data Integration from SaaS/Databases
  -> ELT (Fivetran/Airbyte + dbt)
```

### Framework Selection

| Framework | Best For |
|-----------|----------|
| Apache Spark | Large-scale ETL, SQL analytics |
| Dask | Scaling pandas, Python-native |
| Ray Data | ML preprocessing, GPU workloads |
| Apache Flink | Stateful streaming, low latency |
| Kafka | Event streaming backbone |
| dbt | SQL transformations in warehouse |

## Versioning Decisions

### Versioning Scope

| Tool | Scope | Best For |
|------|-------|----------|
| DVC | Project/files | ML project versioning |
| LakeFS | Data lake | Large-scale data versioning |
| Delta Lake Time Travel | Tables | Point-in-time queries |

### Decision Framework

```
What needs versioning?

ML Project (code + data + models)
  -> DVC (Git-like for data)

Data Lake (many files, teams)
  -> LakeFS (branching, isolation)

Individual Tables
  -> Delta Lake Time Travel (SQL access)

Combined Approach
  -> LakeFS for lake + Delta Lake for tables + DVC for artifacts
```

### Reproducibility Checklist

1. Version training data with experiment
2. Record data version in experiment tracker
3. Enable point-in-time queries for debugging
4. Retain versions aligned with model lifecycle
5. Document data lineage and transformations

## Architecture Patterns

### Startup/Simple ML

```
Data Sources -> Object Storage -> pandas -> Model -> Serving
                     |
                     +-> DVC (versioning)
```

Tools: S3, pandas, scikit-learn, DVC

### Growth Stage

```
Data Sources -> Object Storage -> Spark -> Warehouse -> Feature Store -> Serving
                     |                         |
                     +-> Delta Lake        dbt transforms
```

Tools: S3, Spark, Snowflake, dbt, Feast

### Enterprise Scale

```
Data Sources -> Kafka -> Flink/Spark -> Data Lake -> Feature Store -> Serving
                  |           |              |             |
               Events    Batch + Stream   Iceberg      Tecton
                  |           |              |             |
                  +---> Vector DB ----> RAG System
```

Tools: Kafka, Flink, Spark, Iceberg, Tecton, Pinecone

## Common Patterns

### Feature Engineering Pipeline

```python
# 1. Extract from warehouse
df = spark.sql("SELECT * FROM raw.events WHERE date > '2024-01-01'")

# 2. Transform
features = df.groupBy("user_id").agg(
    count("*").alias("event_count"),
    avg("amount").alias("avg_amount"),
    max("timestamp").alias("last_seen")
)

# 3. Load to feature store
feature_store.materialize(features, "user_features")
```

### Training Data Preparation

```python
# 1. Get historical features
training_data = feature_store.get_historical_features(
    entity_df=labels_df,
    features=["user_features:event_count", "user_features:avg_amount"],
    timestamp_key="label_timestamp"
)

# 2. Version with experiment
dvc.add("training_data.parquet")
mlflow.log_param("data_version", get_dvc_hash("training_data.parquet"))

# 3. Train model
model.fit(training_data)
```

### Serving Pipeline

```python
# 1. Get online features
features = feature_store.get_online_features(
    entity_ids={"user_id": request.user_id},
    features=["user_features:event_count", "user_features:avg_amount"]
)

# 2. Get embeddings for RAG
similar_docs = vector_db.search(
    query_embedding=embed(request.query),
    top_k=5
)

# 3. Predict
prediction = model.predict(features)
```

## Best Practices

### Data Quality

| Stage | Validation |
|-------|------------|
| Ingestion | Schema validation, null checks |
| Processing | Business rules, range checks |
| Feature Store | Freshness monitoring, drift detection |
| Serving | Input validation, fallback values |

### Cost Management

| Strategy | Impact |
|----------|--------|
| Lifecycle policies | Reduce storage 50-70% |
| Columnar formats | Reduce I/O 5-10x |
| Incremental processing | Process only new data |
| Right-size compute | Match resources to workload |

### Governance

| Concern | Solution |
|---------|----------|
| Data lineage | dbt docs, metadata catalogs |
| Access control | Warehouse RBAC, encryption |
| Audit trail | Version control, logging |
| Compliance | Retention policies, PII handling |

## Further Reading

For detailed information on each data engineering area, see:

- [Storage](storage/ReadMe.md) - Object storage, warehouses, feature stores, vector databases
- [Formats](formats/ReadMe.md) - Text, binary, ML-specific, media formats
- [Processing](processing/ReadMe.md) - Batch, streaming, ETL/ELT
- [Versioning](versioning/ReadMe.md) - DVC, LakeFS, Delta Lake time travel
