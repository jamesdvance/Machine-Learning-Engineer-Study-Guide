# Qdrant

## Summary

Qdrant (pronounced "quadrant") is an open-source vector database written in Rust, designed for high-performance similarity search with advanced filtering capabilities. It uses HNSW indexing with a unique approach to filtering that extends the graph structure rather than relying on pre- or post-filtering, enabling efficient filtered searches at scale.

Key points to remember:

- Written in Rust for memory safety and C++-like performance
- HNSW index with custom filtering integration in the graph structure
- Quantization options reduce memory by up to 97% with minimal accuracy loss
- GPU-accelerated indexing on NVIDIA, AMD, and Intel GPUs (Qdrant 1.13+)
- Rich payload filtering: keywords, full-text, numeric ranges, geo-locations
- Flexible deployment: local, Docker, Kubernetes, or Qdrant Cloud
- Achieves highest RPS and lowest latencies in public benchmarks
- Compared to Milvus, Qdrant offers simpler deployment and Rust performance
- Compared to Weaviate, Qdrant focuses more on performance than ML integration

## Architecture

### Core Design

Qdrant is built around:

**Storage Engine**
- Vectors stored alongside JSON payloads
- Memory-mapped files for efficient disk access
- Configurable memory vs disk tradeoffs
- Write-ahead log for durability

**HNSW Index**
- Graph-based approximate nearest neighbor search
- Extended with payload-based edges for filtering
- Configurable layer structure and connection limits
- Delta encoding for memory-efficient graph storage

**Payload Index**
- Indexes for structured data filtering
- Supports multiple index types per field
- Enables combined vector + filter queries

### Deployment Modes

**Local Development**
```bash
# Single binary
./qdrant

# Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Distributed Cluster**
```bash
# First node
./qdrant --uri http://node1:6335

# Additional nodes
./qdrant --uri http://node2:6335 \
  --bootstrap http://node1:6335
```

Features:
- Automatic sharding across nodes
- Replication for fault tolerance
- Consensus via Raft protocol

**Qdrant Cloud (Managed)**
- Fully managed service
- Free tier available
- Automatic scaling and backups
- Multi-region deployment

### Data Model

Collections contain points, where each point has:

```python
{
    "id": "unique-identifier",
    "vector": [0.1, 0.2, ...],  # or named vectors
    "payload": {
        "field1": "value",
        "field2": 123,
        ...
    }
}
```

## Collections and Vectors

### Creating Collections

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

# Single vector collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)
```

### Named Vectors

Multiple embedding spaces per point:

```python
from qdrant_client.models import VectorParams

client.create_collection(
    collection_name="products",
    vectors_config={
        "text": VectorParams(size=768, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE),
    }
)
```

### Distance Metrics

- COSINE: Normalized similarity (most common)
- EUCLID: L2 distance
- DOT: Inner product (unnormalized)
- MANHATTAN: L1 distance

### Sparse Vectors

For hybrid search with keyword vectors:

```python
from qdrant_client.models import SparseVectorParams

client.create_collection(
    collection_name="hybrid_search",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    sparse_vectors_config={
        "keywords": SparseVectorParams()
    }
)
```

## HNSW Configuration

### Index Parameters

```python
from qdrant_client.models import HnswConfigDiff

client.create_collection(
    collection_name="optimized",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,                    # Max connections per node
        ef_construct=100,        # Build-time search depth
        full_scan_threshold=10000,  # Use brute force below this
        max_indexing_threads=0,   # 0 = use all cores
        on_disk=False,           # Store index in RAM
    )
)
```

**Parameter Guidelines**

| Scenario | m | ef_construct | Notes |
|----------|---|--------------|-------|
| Balanced | 16 | 100 | Default, good for most cases |
| High Recall | 32 | 200 | Better accuracy, more memory |
| Low Memory | 8 | 64 | Reduced accuracy |
| Large Scale | 16 | 128 | On-disk recommended |

### Search Parameters

```python
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    search_params=SearchParams(
        hnsw_ef=128,  # Query-time search depth
        exact=False   # Set True for exact (slow) search
    )
)
```

Higher `hnsw_ef` improves recall at the cost of latency.

## Quantization

### Scalar Quantization

Compress vectors to 8-bit integers:

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.update_collection(
    collection_name="documents",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # Outlier handling
            always_ram=True       # Keep quantized in RAM
        )
    )
)
```

Memory reduction: ~4x
Accuracy impact: Minimal with rescoring

### Binary Quantization

Extreme compression to 1-bit:

```python
from qdrant_client.models import BinaryQuantization, BinaryQuantizationConfig

client.update_collection(
    collection_name="documents",
    quantization_config=BinaryQuantization(
        binary=BinaryQuantizationConfig(
            always_ram=True
        )
    )
)
```

Memory reduction: ~32x
Best for: High-dimensional embeddings, oversampling needed

### Product Quantization

Balanced compression:

```python
from qdrant_client.models import ProductQuantization, ProductQuantizationConfig

client.update_collection(
    collection_name="documents",
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=CompressionRatio.X16,
            always_ram=True
        )
    )
)
```

### Rescoring

Quantization works best with rescoring:

```python
results = client.search(
    collection_name="documents",
    query_vector=query,
    limit=10,
    search_params=SearchParams(
        quantization=QuantizationSearchParams(
            rescore=True,     # Re-rank with full vectors
            oversampling=2.0  # Fetch 2x candidates before rescoring
        )
    )
)
```

## Filtering

### Filter-Integrated Search

Qdrant's HNSW index is extended with payload-based edges:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=query,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="technology")
            )
        ]
    ),
    limit=10
)
```

This approach avoids the accuracy problems of pre/post-filtering.

### Filter Conditions

**Match Conditions**
```python
# Exact match
FieldCondition(key="status", match=MatchValue(value="active"))

# Match any
FieldCondition(key="tags", match=MatchAny(any=["ml", "ai", "data"]))

# Match except
FieldCondition(key="type", match=MatchExcept(except_=["draft"]))
```

**Range Conditions**
```python
FieldCondition(
    key="price",
    range=Range(gte=10.0, lte=100.0)
)

FieldCondition(
    key="timestamp",
    range=DatetimeRange(
        gte="2024-01-01T00:00:00Z",
        lte="2024-12-31T23:59:59Z"
    )
)
```

**Geo Conditions**
```python
FieldCondition(
    key="location",
    geo_bounding_box=GeoBoundingBox(
        top_left=GeoPoint(lat=40.8, lon=-74.1),
        bottom_right=GeoPoint(lat=40.6, lon=-73.9)
    )
)

FieldCondition(
    key="location",
    geo_radius=GeoRadius(
        center=GeoPoint(lat=40.7, lon=-74.0),
        radius=5000  # meters
    )
)
```

**Text Conditions**
```python
# Full-text search
FieldCondition(
    key="description",
    match=MatchText(text="vector database")
)
```

### Complex Filters

```python
Filter(
    must=[
        FieldCondition(key="active", match=MatchValue(value=True))
    ],
    should=[
        FieldCondition(key="priority", match=MatchValue(value="high")),
        FieldCondition(key="featured", match=MatchValue(value=True))
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="deleted"))
    ]
)
```

## Payload Indexes

### Creating Indexes

```python
from qdrant_client.models import PayloadSchemaType

# Keyword index
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

# Integer index
client.create_payload_index(
    collection_name="documents",
    field_name="views",
    field_schema=PayloadSchemaType.INTEGER
)

# Full-text index
client.create_payload_index(
    collection_name="documents",
    field_name="content",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=15,
        lowercase=True
    )
)
```

### Index Types

- KEYWORD: Exact match on strings
- INTEGER: Numeric range queries
- FLOAT: Floating-point ranges
- GEO: Geographic queries
- TEXT: Full-text search with tokenization
- BOOL: Boolean filtering
- DATETIME: Timestamp ranges

## Batch Operations

### Batch Insert

```python
from qdrant_client.models import PointStruct

points = [
    PointStruct(
        id=i,
        vector=[0.1, 0.2, ...],
        payload={"title": f"Document {i}"}
    )
    for i in range(1000)
]

client.upsert(
    collection_name="documents",
    points=points,
    wait=True  # Wait for indexing
)
```

### Batch Search

```python
from qdrant_client.models import SearchRequest

results = client.search_batch(
    collection_name="documents",
    requests=[
        SearchRequest(
            vector=[0.1, 0.2, ...],
            limit=10,
            with_payload=True
        ),
        SearchRequest(
            vector=[0.3, 0.4, ...],
            limit=10,
            with_payload=True
        ),
    ]
)
```

### Scroll (Pagination)

```python
# Paginate through all points
offset = None
while True:
    results, offset = client.scroll(
        collection_name="documents",
        scroll_filter=Filter(...),
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    if offset is None:
        break

    process(results)
```

## GPU Acceleration

### GPU Indexing (Qdrant 1.13+)

```yaml
# qdrant config
storage:
  hnsw_index:
    on_disk: false

  gpu:
    indexing:
      enabled: true
      device: "cuda:0"  # or "rocm:0" for AMD
```

Supported GPUs:
- NVIDIA (CUDA)
- AMD (ROCm)
- Intel (oneAPI)

Benefits:
- Significantly faster index building
- Supports all quantization options
- Automatic fallback to CPU

## Performance Tuning

### Memory vs Disk Tradeoffs

**All in RAM (Fastest)**
```python
hnsw_config=HnswConfigDiff(on_disk=False)
# Store vectors in RAM too
```

**Index in RAM, Vectors on Disk**
```python
hnsw_config=HnswConfigDiff(on_disk=False)
vectors_config=VectorParams(
    size=768,
    distance=Distance.COSINE,
    on_disk=True
)
```

**All on Disk (Most Memory Efficient)**
```python
hnsw_config=HnswConfigDiff(on_disk=True)
vectors_config=VectorParams(
    size=768,
    distance=Distance.COSINE,
    on_disk=True
)
```

### Segment Configuration

```python
from qdrant_client.models import OptimizersConfigDiff

client.update_collection(
    collection_name="documents",
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,     # Vectors before indexing
        memmap_threshold=50000,       # Vectors before mmap
        max_segment_size=200000,      # Points per segment
        default_segment_number=4      # Parallel search segments
    )
)
```

More segments = higher throughput, more memory overhead.

## Comparison with Alternatives

### Qdrant vs Milvus

| Aspect | Qdrant | Milvus |
|--------|--------|--------|
| Language | Rust | Go/C++ |
| Deployment | Simpler | More complex |
| GPU Support | Index only | Index + search |
| Filtering | Graph-integrated | Separate |
| Index Types | HNSW only | Many options |
| Maturity | Younger | More established |

### Qdrant vs Pinecone

| Aspect | Qdrant | Pinecone |
|--------|--------|----------|
| Hosting | Self-hosted or cloud | Managed only |
| Open Source | Yes | No |
| Performance | Benchmark leader | Good |
| Filtering | Advanced | Basic |
| Cost Control | Full | Usage-based |

### Qdrant vs Weaviate

| Aspect | Qdrant | Weaviate |
|--------|--------|----------|
| Focus | Performance | Developer experience |
| Vectorization | Bring your own | Built-in modules |
| API | REST/gRPC | GraphQL |
| Memory Efficiency | Better | More overhead |

## When to Use Qdrant

Qdrant is well-suited for:
- Performance-critical applications
- Workloads with complex filtering requirements
- Teams wanting Rust's reliability and performance
- Deployments requiring fine-grained memory control
- Projects needing GPU-accelerated indexing

Consider alternatives when:
- Built-in ML integration is needed (Weaviate)
- Managed service is required with no self-hosting (Pinecone)
- Multiple index algorithm options needed (Milvus)
- Minimal setup for prototyping (Chroma)
