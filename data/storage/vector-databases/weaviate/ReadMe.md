# Weaviate

## Summary

Weaviate is an open-source vector database that combines vector search with structured data storage, enabling hybrid queries that blend semantic similarity with traditional filtering. Built in Go, it emphasizes developer experience through a modular architecture, GraphQL API, and integrated ML model support that can generate embeddings automatically during ingestion.

Key points to remember:

- Open source with managed cloud option (Weaviate Cloud Services)
- Modular architecture allows plugging in vectorization, ML, and storage backends
- GraphQL-first API enables precise, graph-based querying
- Built-in vectorization modules for text, images, and multi-modal data
- Hybrid search combines BM25 keyword matching with vector similarity
- Named vectors allow multiple embedding spaces per object
- Schema-based with semantic interpretation of class definitions
- Compared to Milvus, Weaviate offers better developer experience and ML integration
- Compared to Pinecone, Weaviate provides more customization and self-hosting option

## Architecture

### Core Components

Weaviate's architecture consists of:

**Storage Engine**
- Objects stored with associated vectors
- LSM-tree based storage for scalability
- Supports both in-memory and disk-based operation

**Vector Index**
- HNSW for approximate nearest neighbor search
- Configurable index parameters per collection
- Flat index option for small datasets

**Inverted Index**
- BM25 for keyword search
- Enables hybrid search combining vectors and keywords
- Supports structured field filtering

**Module System**
- Pluggable components for vectorization, ML, and storage
- Runs as separate containers (microservice pattern)
- Configurable per collection

### Deployment Options

**Docker (Development)**
```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      ENABLE_MODULES: 'text2vec-openai,generative-openai'
      CLUSTER_HOSTNAME: 'node1'
```

**Kubernetes (Production)**
- Helm charts for cluster deployment
- Horizontal scaling via sharding
- Multi-node replication

**Weaviate Cloud Services (Managed)**
- Fully managed with automatic scaling
- Free sandbox tier for development
- Enterprise tier with SLA guarantees

## Schema and Collections

### Defining Collections

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()

# Create collection with schema
client.collections.create(
    name="Article",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="published", data_type=DataType.DATE),
        Property(name="views", data_type=DataType.INT),
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    generative_config=Configure.Generative.openai(),
)
```

### Data Types

- TEXT: String with full-text indexing
- TEXT_ARRAY: Array of strings
- INT, NUMBER: Numeric types
- BOOLEAN: True/false
- DATE: Timestamps
- UUID: Unique identifiers
- GEO_COORDINATES: Latitude/longitude
- BLOB: Binary data
- OBJECT: Nested objects
- CROSS_REFERENCE: Links between objects

### Auto-Schema

Weaviate can infer schema from data:

```python
# Auto-schema enabled by default
collection = client.collections.get("Article")
collection.data.insert({
    "title": "Vector Databases Explained",
    "content": "An introduction to...",
    "category": "Technology"
})
# Schema inferred from first object
```

### Cross-References

Link objects between collections:

```python
# Create reference
article.references.add(
    from_property="hasAuthor",
    to=author_uuid
)

# Query with references
articles = client.collections.get("Article")
response = articles.query.fetch_objects(
    include_vector=True,
    return_references=["hasAuthor"]
)
```

## Vectorization Modules

### Text Vectorization

**OpenAI Integration**
```python
client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    )
)
```

**Cohere Integration**
```python
vectorizer_config=Configure.Vectorizer.text2vec_cohere(
    model="embed-english-v3.0"
)
```

**Hugging Face (Self-hosted)**
```python
vectorizer_config=Configure.Vectorizer.text2vec_transformers()
# Requires transformer model container
```

**Available Text Modules**
- text2vec-openai: OpenAI embeddings
- text2vec-cohere: Cohere embeddings
- text2vec-huggingface: Hugging Face API
- text2vec-transformers: Self-hosted transformers
- text2vec-palm: Google PaLM embeddings
- text2vec-aws: AWS Bedrock embeddings

### Image Vectorization

```python
vectorizer_config=Configure.Vectorizer.img2vec_neural()
# Supports ResNet-based embeddings

# Multi-modal (CLIP)
vectorizer_config=Configure.Vectorizer.multi2vec_clip()
```

### Bring Your Own Vectors

```python
# Insert with pre-computed vectors
collection.data.insert(
    properties={"title": "My Document"},
    vector=[0.1, 0.2, 0.3, ...]  # Your embedding
)
```

### Named Vectors

Multiple embedding spaces per object:

```python
client.collections.create(
    name="Product",
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
        Property(name="image", data_type=DataType.BLOB),
    ],
    vectorizer_config=[
        Configure.NamedVectors.text2vec_openai(
            name="text_vector",
            source_properties=["name", "description"]
        ),
        Configure.NamedVectors.img2vec_neural(
            name="image_vector",
            source_properties=["image"]
        ),
    ]
)
```

Query specific vector space:

```python
response = products.query.near_text(
    query="comfortable running shoes",
    target_vector="text_vector",
    limit=5
)
```

## Search Capabilities

### Vector Search

```python
articles = client.collections.get("Article")

# Near text (semantic search)
response = articles.query.near_text(
    query="machine learning applications",
    limit=10,
    return_metadata=["distance", "certainty"]
)

# Near vector (raw embedding)
response = articles.query.near_vector(
    near_vector=[0.1, 0.2, ...],
    limit=10
)

# Near object (find similar to existing)
response = articles.query.near_object(
    near_object=uuid,
    limit=10
)
```

### Keyword Search (BM25)

```python
response = articles.query.bm25(
    query="vector database",
    limit=10,
    return_metadata=["score"]
)
```

### Hybrid Search

Combine vector and keyword search:

```python
response = articles.query.hybrid(
    query="vector database performance",
    alpha=0.5,  # 0=BM25 only, 1=vector only
    limit=10,
    return_metadata=["score", "explain_score"]
)
```

Fusion algorithms:
- Ranked Fusion: Combines result rankings
- Relative Score Fusion: Normalizes and combines scores

### Filtering

```python
from weaviate.classes.query import Filter

# Single filter
response = articles.query.near_text(
    query="AI news",
    filters=Filter.by_property("category").equal("Technology"),
    limit=10
)

# Complex filters
response = articles.query.near_text(
    query="AI news",
    filters=(
        Filter.by_property("category").equal("Technology") &
        Filter.by_property("views").greater_than(1000) &
        (
            Filter.by_property("published").greater_than("2024-01-01") |
            Filter.by_property("featured").equal(True)
        )
    ),
    limit=10
)
```

### Grouping

Aggregate results by field:

```python
response = articles.query.near_text(
    query="AI applications",
    group_by=GroupBy(
        prop="category",
        objects_per_group=3,
        number_of_groups=5
    )
)
```

## Generative Search (RAG)

Weaviate includes built-in RAG capabilities:

### Single Prompt

```python
response = articles.generate.near_text(
    query="renewable energy",
    single_prompt="Summarize this article: {content}",
    limit=5
)

for obj in response.objects:
    print(obj.generated)  # LLM-generated summary
```

### Grouped Task

```python
response = articles.generate.near_text(
    query="AI in healthcare",
    grouped_task="Write a comprehensive overview based on these articles",
    limit=5
)

print(response.generated)  # Single response from all results
```

### Generative Modules

- generative-openai: GPT models
- generative-cohere: Cohere Command
- generative-palm: Google PaLM
- generative-aws: AWS Bedrock
- generative-ollama: Local models via Ollama

## GraphQL API

### Query Structure

```graphql
{
  Get {
    Article(
      nearText: { concepts: ["machine learning"] }
      limit: 5
      where: {
        path: ["category"]
        operator: Equal
        valueText: "Technology"
      }
    ) {
      title
      content
      category
      _additional {
        certainty
        distance
        id
      }
    }
  }
}
```

### Aggregate Queries

```graphql
{
  Aggregate {
    Article {
      meta {
        count
      }
      views {
        sum
        mean
        maximum
      }
    }
  }
}
```

### Cross-References in Queries

```graphql
{
  Get {
    Article(limit: 5) {
      title
      hasAuthor {
        ... on Author {
          name
          bio
        }
      }
    }
  }
}
```

## Performance Tuning

### HNSW Configuration

```python
client.collections.create(
    name="LargeCollection",
    vector_index_config=Configure.VectorIndex.hnsw(
        ef=100,              # Query-time search depth
        ef_construction=128,  # Build-time search depth
        max_connections=64,   # Max edges per node
        dynamic_ef_min=100,
        dynamic_ef_max=500,
        dynamic_ef_factor=8,
    )
)
```

### Quantization

Reduce memory usage with compression:

```python
vector_index_config=Configure.VectorIndex.hnsw(
    quantizer=Configure.VectorIndex.Quantizer.pq(
        segments=128,        # Number of segments
        centroids=256,       # Centroids per segment
        training_limit=100000
    )
)

# Binary quantization (faster, more compression)
vector_index_config=Configure.VectorIndex.hnsw(
    quantizer=Configure.VectorIndex.Quantizer.bq(
        rescore_limit=200
    )
)
```

### Flat Index

For small collections (< 10,000 objects):

```python
vector_index_config=Configure.VectorIndex.flat(
    quantizer=Configure.VectorIndex.Quantizer.bq()
)
```

### Batch Operations

```python
# Efficient batch insert
with collection.batch.dynamic() as batch:
    for item in items:
        batch.add_object(
            properties=item["properties"],
            vector=item.get("vector")
        )
```

## Multi-Tenancy

Isolate data per tenant:

```python
# Enable multi-tenancy
client.collections.create(
    name="UserData",
    multi_tenancy_config=Configure.multi_tenancy(enabled=True)
)

# Create tenant
collection = client.collections.get("UserData")
collection.tenants.create([
    Tenant(name="tenant_A"),
    Tenant(name="tenant_B"),
])

# Operations scoped to tenant
tenant_collection = collection.with_tenant("tenant_A")
tenant_collection.data.insert({"content": "Tenant A's data"})
```

Benefits:
- Data isolation between tenants
- Independent scaling
- Efficient resource utilization

## Comparison with Alternatives

### Weaviate vs Pinecone

| Aspect | Weaviate | Pinecone |
|--------|----------|----------|
| Hosting | Self-hosted or managed | Managed only |
| Open Source | Yes | No |
| Built-in Vectorization | Yes (modules) | No |
| GraphQL API | Yes | No |
| Generative Search | Native | No |
| Operational Complexity | Higher (self-hosted) | None |

### Weaviate vs Milvus

| Aspect | Weaviate | Milvus |
|--------|----------|--------|
| Focus | Developer experience | Performance at scale |
| Vectorization | Built-in modules | Bring your own |
| Query Language | GraphQL + REST | gRPC + REST |
| GPU Support | Limited | Extensive |
| Cross-References | Native | Manual |

### Weaviate vs Qdrant

| Aspect | Weaviate | Qdrant |
|--------|----------|--------|
| API Style | GraphQL-first | REST/gRPC |
| Modules | Extensive ecosystem | Fewer integrations |
| Memory Management | Configurable | Mmap-based |
| RAG Support | Native generative | Via integration |

## When to Use Weaviate

Weaviate is well-suited for:
- Applications needing built-in vectorization
- RAG systems requiring generative search
- Projects where developer experience matters
- Multi-modal search (text, images, audio)
- Graph-like data with cross-references

Consider alternatives when:
- Maximum query throughput is critical (Milvus)
- Operational simplicity is paramount (Pinecone)
- GPU acceleration is required (Milvus)
- Minimal resource footprint needed (Qdrant, Chroma)
