# Milvus

## Summary

Milvus is an open-source, distributed vector database designed for billion-scale similarity search. Written in Go and C++, it features a cloud-native architecture that separates compute and storage, enabling independent scaling of query and data nodes. Milvus offers extensive indexing options, GPU acceleration, and hybrid search capabilities, making it suitable for production deployments requiring both flexibility and performance.

Key points to remember:

- Open source with fully managed option (Zilliz Cloud)
- Distributed architecture separates compute and storage
- Multiple index types: HNSW, IVF variants, DiskANN, GPU-accelerated
- GPU indexing via NVIDIA CAGRA delivers significant performance gains
- Supports hybrid search combining dense, sparse, and scalar filtering
- Collections support dynamic schemas and partition keys
- 30-70% better performance than FAISS/HNSWLib in benchmarks
- Compared to Pinecone, Milvus offers more control and index options
- Compared to Weaviate, Milvus focuses more on performance at scale

## Architecture

### Distributed Design

Milvus uses a microservice architecture with four primary component types:

**Coordinators (Control Plane)**
- Root Coordinator: Manages metadata and DDL operations
- Query Coordinator: Manages query node topology and load balancing
- Data Coordinator: Manages data node assignments and segment allocation
- Index Coordinator: Manages index building tasks

**Workers (Data Plane)**
- Query Nodes: Execute search queries on loaded segments
- Data Nodes: Process inserts and persist data to storage
- Index Nodes: Build indexes asynchronously

**Storage Layer**
- Log Broker: Pulsar or Kafka for streaming log storage
- Object Storage: S3, MinIO, or Azure Blob for segment storage
- Meta Storage: etcd for metadata persistence

**Message Queue**
All data mutations flow through the log broker, enabling:
- Decoupled read and write paths
- Fault tolerance and replay capability
- Real-time data availability

### Scaling Model

```
Query-heavy workload:  Add Query Nodes
Write-heavy workload:  Add Data Nodes
Storage growth:        Horizontal scaling via sharding
```

Each component type scales independently based on workload characteristics.

### Deployment Options

**Standalone**
Single process for development and small datasets:
```bash
docker run -d --name milvus_standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest standalone
```

**Cluster**
Distributed deployment for production:
- Kubernetes with Helm charts
- Milvus Operator for lifecycle management
- Scales to billions of vectors

**Zilliz Cloud (Managed)**
Fully managed Milvus with:
- Automatic scaling
- Built-in backup and recovery
- SLA guarantees
- Multi-cloud support (AWS, GCP, Azure)

## Index Types

Milvus supports multiple index algorithms optimized for different scenarios:

### Graph-Based Indexes

**HNSW (Hierarchical Navigable Small World)**
```python
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,           # Max connections per node
        "efConstruction": 256  # Build-time search depth
    }
}
```

Characteristics:
- Excellent recall and speed balance
- Memory-intensive (entire graph in RAM)
- Best for datasets > 2GB requiring high performance
- Build-time vs query-time tradeoff via parameters

**GPU_CAGRA (NVIDIA CAGRA)**
```python
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",
    "params": {
        "intermediate_graph_degree": 64,
        "graph_degree": 32
    }
}
```

Introduced in Milvus 2.4, CAGRA leverages GPU for:
- Faster index building than CPU HNSW
- High-throughput queries on GPU
- Optimal for large-scale deployments with GPU resources

### Quantization-Based Indexes

**IVF_FLAT (Inverted File with Flat Quantization)**
```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 1024      # Number of clusters
    }
}

search_params = {
    "nprobe": 16          # Clusters to search
}
```

Characteristics:
- Exact distance computation within clusters
- Good recall with reasonable speed
- Suitable for datasets < 2GB

**IVF_SQ8 (Scalar Quantization)**
```python
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
```

Characteristics:
- 75% memory reduction vs IVF_FLAT
- Minor recall loss acceptable
- Good for memory-constrained environments

**IVF_PQ (Product Quantization)**
```python
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 1024,
        "m": 8,            # Subvector count
        "nbits": 8         # Bits per subvector
    }
}
```

Characteristics:
- Highest compression ratio
- More recall loss than SQ8
- Suitable when storage is critical

### Disk-Based Indexes

**DiskANN**
```python
index_params = {
    "index_type": "DISKANN",
    "metric_type": "L2",
    "params": {}
}
```

Characteristics:
- SSD-based index for datasets exceeding RAM
- Low memory footprint
- Good performance with NVMe storage

### Index Selection Guide

| Scenario | Recommended Index |
|----------|------------------|
| High recall, sufficient RAM | HNSW |
| GPU available, large scale | GPU_CAGRA |
| Balanced performance | IVF_FLAT or IVF_SQ8 |
| Memory constrained | IVF_PQ |
| Dataset exceeds RAM | DiskANN |
| Exact search (small data) | FLAT |

## Collections and Schema

### Creating Collections

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="timestamp", dtype=DataType.INT64),
]

schema = CollectionSchema(fields, description="Document embeddings")
collection = Collection(name="documents", schema=schema)
```

### Supported Data Types

Vector types:
- FLOAT_VECTOR: Standard dense vectors
- BINARY_VECTOR: Binary embeddings
- FLOAT16_VECTOR: Half-precision (Milvus 2.4+)
- BFLOAT16_VECTOR: Brain float (Milvus 2.4+)
- SPARSE_FLOAT_VECTOR: Sparse vectors for hybrid search

Scalar types:
- INT8, INT16, INT32, INT64
- FLOAT, DOUBLE
- VARCHAR, JSON
- BOOL

### Partitions and Partition Keys

**Manual Partitions**
```python
# Create partition
collection.create_partition("category_tech")

# Insert to specific partition
collection.insert(data, partition_name="category_tech")

# Search specific partition
collection.search(
    data=query_vectors,
    partition_names=["category_tech"]
)
```

**Partition Keys (Automatic)**
```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=100,
                is_partition_key=True),  # Automatic partitioning
]
```

Partition keys:
- Automatically route data to partitions
- Enable efficient multi-tenant queries
- Default 16 partitions per collection

## Hybrid Search

### Multi-Vector Search

Combine multiple vector fields with reranking:

```python
from pymilvus import AnnSearchRequest, RRFRanker

# Define searches on different vector fields
search1 = AnnSearchRequest(
    data=text_embeddings,
    anns_field="text_vector",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=100
)

search2 = AnnSearchRequest(
    data=image_embeddings,
    anns_field="image_vector",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=100
)

# Combine with RRF ranking
results = collection.hybrid_search(
    reqs=[search1, search2],
    rerank=RRFRanker(),
    limit=10
)
```

### Dense + Sparse Hybrid

Combine semantic embeddings with keyword vectors:

```python
from pymilvus import WeightedRanker

# Dense vector search
dense_search = AnnSearchRequest(
    data=dense_embeddings,
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=100
)

# Sparse vector search (e.g., BM25, SPLADE)
sparse_search = AnnSearchRequest(
    data=sparse_embeddings,
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=100
)

# Weighted combination
results = collection.hybrid_search(
    reqs=[dense_search, sparse_search],
    rerank=WeightedRanker(0.7, 0.3),  # 70% dense, 30% sparse
    limit=10
)
```

### Scalar Filtering

Combine vector search with attribute filters:

```python
results = collection.search(
    data=query_embeddings,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10,
    expr="category == 'technology' and timestamp > 1704067200",
    output_fields=["text", "category", "timestamp"]
)
```

Supported filter operators:
- Comparison: ==, !=, >, <, >=, <=
- Logical: and, or, not
- Set: in, not in
- String: like (pattern matching)

## GPU Acceleration

### GPU Index Building

```python
# GPU-accelerated IVF
index_params = {
    "index_type": "GPU_IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}

# GPU CAGRA (Milvus 2.4+)
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",
    "params": {
        "intermediate_graph_degree": 64,
        "graph_degree": 32
    }
}
```

### Performance Benefits

GPU acceleration provides:
- 10-100x faster index building
- Higher query throughput
- Better latency for large batches
- Cost-effective for high-volume workloads

Requirements:
- NVIDIA GPU with CUDA support
- cuVS library (RAPIDS)
- Adequate GPU memory for index

## Performance Tuning

### Index Parameters

**HNSW Tuning**
```python
# Higher M = better recall, more memory
# Higher efConstruction = better quality, slower build
index_params = {
    "M": 32,              # Default: 16
    "efConstruction": 512  # Default: 256
}

# Higher ef = better recall, slower search
search_params = {"ef": 256}  # Default: 64
```

**IVF Tuning**
```python
# More clusters = finer granularity
index_params = {"nlist": 4096}  # sqrt(n) as starting point

# More probes = better recall, slower search
search_params = {"nprobe": 128}  # Start with nlist/16
```

### Query Optimization

1. Load only necessary collections
2. Use partition keys for multi-tenant filtering
3. Apply scalar filters to reduce search space
4. Tune consistency level for latency vs freshness

```python
# Consistency levels
from pymilvus import ConsistencyLevel

collection.search(
    consistency_level=ConsistencyLevel.Eventually,  # Fastest
    # ConsistencyLevel.Strong  # Most consistent
    # ConsistencyLevel.Bounded  # Balanced
    # ConsistencyLevel.Session  # Read-your-writes
)
```

### Batch Operations

```python
# Batch insert
batch_size = 10000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    collection.insert(batch)
    collection.flush()  # Persist after each batch

# Batch search
results = collection.search(
    data=query_vectors,  # Multiple queries
    batch_size=100
)
```

## Comparison with Alternatives

### Milvus vs Pinecone

| Aspect | Milvus | Pinecone |
|--------|--------|----------|
| Hosting | Self-hosted or Zilliz Cloud | Managed only |
| Open Source | Yes | No |
| Index Options | Many (HNSW, IVF, DiskANN, GPU) | Limited |
| GPU Support | Yes (CAGRA, IVF) | No |
| Operational Burden | Higher (self-hosted) | None |
| Customization | Full control | Limited |

### Milvus vs Weaviate

| Aspect | Milvus | Weaviate |
|--------|--------|----------|
| Language | Go/C++ | Go |
| Focus | Performance at scale | Developer experience |
| GPU Support | Yes | Limited |
| Built-in ML | No | Yes (modules) |
| Schema | Required | Optional (auto-schema) |

### Milvus vs Qdrant

| Aspect | Milvus | Qdrant |
|--------|--------|--------|
| Maturity | Older, more battle-tested | Newer, growing |
| Deployment | K8s-native | Simpler deployment |
| GPU Support | Yes | Partial |
| Memory Mode | On-disk options | Memory + disk hybrid |

## When to Use Milvus

Milvus is well-suited for:
- Large-scale deployments (billions of vectors)
- Workloads requiring GPU acceleration
- Organizations preferring open-source solutions
- Complex search requirements (multi-vector, hybrid)
- Kubernetes-native infrastructure

Consider alternatives when:
- Operational simplicity is paramount (Pinecone)
- Rapid prototyping without infrastructure (managed services)
- Tighter ML integration needed (Weaviate)
- Simpler deployment preferred (Qdrant)
