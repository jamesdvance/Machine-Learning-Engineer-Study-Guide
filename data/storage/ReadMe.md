# Data Storage for Machine Learning

## Summary

Data storage is the foundation of any ML system. The choice of storage technology directly impacts training data access speed, feature serving latency, model reproducibility, and infrastructure costs. Modern ML architectures typically use multiple specialized storage systems, each optimized for different access patterns within the ML lifecycle.

Key points to remember:

- Object storage (S3, GCS, Azure Blob) is the default for raw data and model artifacts due to cost and scalability
- Data warehouses (Snowflake, BigQuery, Redshift) enable SQL-based analytics and feature engineering
- Data lakes with table formats (Delta Lake, Hudi, Iceberg) add ACID transactions to object storage
- Feature stores bridge training and serving by ensuring consistent feature computation
- Vector databases enable similarity search for RAG, recommendations, and semantic retrieval
- Storage decisions should align with access patterns: batch vs streaming, read-heavy vs write-heavy

## Storage Taxonomy

ML workflows require different storage types at different stages:

| Stage | Primary Storage | Access Pattern | Key Requirement |
|-------|-----------------|----------------|-----------------|
| Raw Data Ingestion | Object Storage | Write-heavy, batch | Scalability, cost |
| Data Processing | Data Lake/Warehouse | Read-heavy, batch | Query performance |
| Feature Engineering | Data Warehouse | Read-heavy, analytical | SQL support, joins |
| Feature Serving | Feature Store (Online) | Read-heavy, low-latency | Sub-10ms reads |
| Model Training | Object Storage / Data Lake | Sequential reads | Throughput |
| Embedding Storage | Vector Database | Read-heavy, similarity | ANN search |
| Model Serving | Object Storage | Read-once per deployment | Availability |

## Object Storage

Object storage is the foundational layer for ML data infrastructure. It provides virtually unlimited capacity, high durability, and cost-effective storage for datasets, model checkpoints, and artifacts.

### When to Use Object Storage

- Training datasets (images, text, tabular files)
- Model checkpoints and final weights
- Experiment artifacts (logs, metrics, visualizations)
- Data lake storage layer (underneath table formats)
- Long-term archival of training runs

### Key Characteristics

| Aspect | Typical Values |
|--------|---------------|
| Durability | 11+ nines (99.999999999%) |
| Latency | 10-100ms first byte |
| Throughput | Scales with parallelism |
| Cost | $0.02-0.03/GB/month (standard tier) |
| Max Object Size | 5 TB (GCS), 50 TB (S3) |

### Provider Comparison

| Feature | Amazon S3 | Google Cloud Storage | Azure Blob Storage |
|---------|-----------|---------------------|-------------------|
| Low-Latency Option | Express One Zone | None | Premium Block Blob |
| Hierarchical Namespace | No | HNS available | ADLS Gen2 |
| Archive Retrieval | Hours | Milliseconds | Hours |
| ML Framework Integration | Broad | TensorFlow/JAX optimized | Azure ML integrated |

Choose based on your cloud platform. For checkpointing-heavy distributed training, GCS with HNS or Azure ADLS Gen2 provides atomic directory operations.

For cost-sensitive workloads with predictable access patterns, S3-compatible alternatives (Cloudflare R2, Backblaze B2, Wasabi) offer 50-80% savings with zero egress fees.

## Data Warehouses

Data warehouses provide managed, scalable infrastructure for analytical workloads. They use columnar storage and massively parallel processing to efficiently aggregate and transform large datasets.

### When to Use Data Warehouses

- Feature engineering with complex SQL queries
- Aggregating raw data into training features
- Ad-hoc analysis during model development
- Storing structured, curated datasets
- SQL-based ML (BQML, Snowpark ML, Redshift ML)

### Key Characteristics

| Aspect | Typical Values |
|--------|---------------|
| Query Latency | Seconds to minutes |
| Data Volume | Petabytes |
| Compute Model | Serverless or provisioned |
| Cost | Per-query or per-compute-hour |

### Platform Comparison

| Feature | Snowflake | BigQuery | Redshift |
|---------|-----------|----------|----------|
| Compute Model | Virtual Warehouses | Serverless slots | Nodes or Serverless |
| Multi-Cloud | Yes | GCP only | AWS only |
| ML Integration | Snowpark ML | BigQuery ML | Redshift ML (SageMaker) |
| Streaming | Snowpipe | Pub/Sub, Streaming API | Kinesis |

Snowflake offers operational simplicity and multi-cloud portability. BigQuery provides zero-admin serverless with pay-per-query pricing. Redshift offers deep AWS integration and fine-grained tuning.

## Data Lakes and Table Formats

Data lakes store data in open formats on object storage. Modern table formats (Delta Lake, Apache Hudi, Apache Iceberg) add ACID transactions, schema evolution, and time travel to raw data lakes.

### When to Use Table Formats

- Large-scale data processing with Spark
- Datasets requiring transactional updates
- ML pipelines needing point-in-time snapshots
- Multi-engine environments
- Replacing or augmenting data warehouse storage

### Format Comparison

| Feature | Delta Lake | Apache Hudi | Apache Iceberg |
|---------|------------|-------------|----------------|
| Best For | Spark/Databricks | Streaming upserts | Multi-engine |
| Strength | Ecosystem maturity | Record-level updates | Engine compatibility |
| Origin | Databricks | Uber | Netflix |

For general-purpose lakehouse workloads on Spark, Delta Lake provides the smoothest experience. For high-frequency CDC and streaming upserts, Hudi's indexing optimizes record-level operations. For heterogeneous query engines (Trino, Flink, Snowflake), Iceberg offers the broadest compatibility.

## Feature Stores

Feature stores are centralized repositories for ML features. They solve training-serving skew by providing consistent feature definitions and values across training pipelines and serving endpoints.

### When to Use Feature Stores

- Production ML systems with online serving
- Multiple models sharing common features
- Teams collaborating on feature engineering
- Features requiring point-in-time correctness
- Real-time features with streaming updates

### Architecture

Feature stores maintain dual storage:

- **Offline Store**: Historical features for training (data warehouse, lake, or object storage)
- **Online Store**: Latest feature values for serving (Redis, DynamoDB, or managed)

Materialization syncs data from offline to online stores.

### Platform Comparison

| Feature | Feast | Tecton | Hopsworks |
|---------|-------|--------|-----------|
| Hosting | Self-managed | Managed | Both |
| Open Source | Yes | No | Yes |
| Real-time Features | Basic | Sub-second | Kafka-based |
| Complexity | Medium | Low | High |

Feast provides open-source flexibility with infrastructure responsibility. Tecton offers managed simplicity with native streaming. Hopsworks includes a complete ML platform beyond just features.

## Vector Databases

Vector databases enable efficient similarity search over high-dimensional embeddings. They are essential for RAG systems, recommendation engines, and semantic search.

### When to Use Vector Databases

- RAG (Retrieval-Augmented Generation) systems
- Semantic search over documents
- Image/video similarity search
- Recommendation systems
- Anomaly detection via nearest neighbors

### Key Concepts

| Concept | Description |
|---------|-------------|
| Embeddings | Dense vector representations of data |
| ANN (Approximate Nearest Neighbor) | Sub-linear similarity search |
| HNSW | Most common index algorithm |
| Quantization | Vector compression (4-32x memory reduction) |
| Hybrid Search | Combining vector + keyword matching |

### Platform Comparison

| Feature | Pinecone | Milvus | Qdrant | Weaviate | Chroma |
|---------|----------|--------|--------|----------|--------|
| Hosting | Managed | Both | Both | Both | Self-hosted |
| Scale | Billions | Billions | Billions | Millions | Thousands |
| Best For | Simplicity | GPU acceleration | Performance, filtering | Built-in ML | Prototyping |

Pinecone offers operational simplicity for production RAG. Milvus and Qdrant provide high-performance self-hosted options. Chroma enables rapid prototyping and learning.

## Decision Framework

### Start with Access Patterns

1. **Batch training data**: Object storage with table format (Delta/Iceberg) for versioning
2. **Feature engineering**: Data warehouse for SQL transformations
3. **Real-time feature serving**: Feature store with online store
4. **Semantic retrieval**: Vector database
5. **Model artifacts**: Object storage with versioning

### Consider Scale

| Scale | Recommended Approach |
|-------|---------------------|
| Prototype | Object storage + Chroma + pandas |
| Production (small) | Data warehouse + Feast + Qdrant |
| Production (large) | Data lake + Tecton/Hopsworks + Milvus |
| Enterprise | Multiple specialized systems with orchestration |

### Evaluate Operational Complexity

| Preference | Storage Choices |
|------------|-----------------|
| Managed-first | Snowflake, Pinecone, Tecton |
| Open-source | Iceberg, Feast, Milvus/Qdrant |
| Hybrid | Mix based on criticality |

## Common Patterns

### ML Platform Storage Stack

A typical production ML platform combines:

```
Raw Data -> Object Storage (S3/GCS)
    |
    v
Processing -> Data Lake (Iceberg on S3)
    |
    v
Curated Data -> Data Warehouse (Snowflake)
    |
    v
Features -> Feature Store (Feast/Tecton)
    |
    +-> Online Store (Redis) -> Serving
    |
    +-> Offline Store (S3) -> Training
    |
    v
Embeddings -> Vector Database (Qdrant)
```

### Avoiding Common Pitfalls

1. **Object storage for everything**: Works for prototypes, but production needs specialized stores
2. **Ignoring training-serving skew**: Feature stores exist to solve this problem
3. **Over-engineering early**: Start simple, add complexity as needed
4. **Vendor lock-in concerns**: Open formats (Parquet, Iceberg) provide portability
5. **Underestimating egress costs**: Co-locate storage with compute

### Cost Optimization

| Strategy | Impact |
|----------|--------|
| Lifecycle policies | 50-70% storage savings |
| Tiered storage | 30-50% for infrequently accessed data |
| Quantization (vectors) | 4-32x memory reduction |
| Columnar formats | 5-10x compression vs CSV |
| Regional placement | Eliminates egress costs |

## Further Reading

For detailed information on each storage type, see:

- [Object Storage](object-storage/ReadMe.md)
- [Data Warehouses](data-warehouses/ReadMe.md)
- [Data Lakes and Table Formats](data-lakes/ReadMe.md)
- [Feature Stores](feature-stores/ReadMe.md)
- [Vector Databases](vector-databases/ReadMe.md)
