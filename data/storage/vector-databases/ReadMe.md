# Vector Databases

## Summary

Vector databases are specialized storage systems designed for efficient similarity search over high-dimensional embeddings. Unlike traditional databases that match exact values, vector databases find the most similar items based on distance metrics like cosine similarity or Euclidean distance. They power modern AI applications including RAG systems, recommendation engines, image search, and anomaly detection.

Key points to remember:

- Store embeddings (dense vectors) alongside metadata for filtered similarity search
- Use approximate nearest neighbor (ANN) algorithms for sub-linear query time
- HNSW is the most common index algorithm, offering good recall/speed tradeoff
- Quantization reduces memory by compressing vectors with minimal accuracy loss
- Hybrid search combines semantic (vector) search with keyword (BM25) matching
- Managed services (Pinecone) offer simplicity; self-hosted (Milvus, Qdrant) offer control
- Choice depends on scale, performance requirements, and operational preferences

## Core Concepts

### Embeddings

Embeddings are dense vector representations of data (text, images, audio) that capture semantic meaning. Similar items have vectors that are close together in the embedding space.

```
"machine learning" -> [0.12, -0.34, 0.56, ...]  (768 dimensions)
"artificial intelligence" -> [0.11, -0.32, 0.58, ...]  (nearby in space)
"pizza recipe" -> [-0.45, 0.23, -0.12, ...]  (distant)
```

Common embedding sources:
- Text: OpenAI embeddings, Sentence Transformers, Cohere
- Images: CLIP, ResNet features
- Multi-modal: CLIP, ImageBind

### Similarity Search

Given a query embedding, find the k most similar vectors in the database:

1. Compute distance between query and stored vectors
2. Return k vectors with smallest distance (or highest similarity)

Distance metrics:
- **Cosine Similarity**: Measures angle between vectors (normalized)
- **Euclidean (L2)**: Straight-line distance
- **Dot Product**: Unnormalized similarity (magnitude matters)
- **Manhattan (L1)**: Sum of absolute differences

Cosine similarity is most common for text embeddings since they are typically normalized.

### Approximate Nearest Neighbors (ANN)

Exact nearest neighbor search requires comparing against every vector: O(n) complexity. For million/billion-scale datasets, this is impractical.

ANN algorithms trade perfect accuracy for speed, achieving sub-linear query time with 95-99% recall. Key insight: finding approximately correct answers quickly is more valuable than perfect answers slowly.

## Index Algorithms

### HNSW (Hierarchical Navigable Small World)

The dominant algorithm in production vector databases:

**How it works:**
1. Builds a multi-layer graph structure
2. Upper layers: sparse, for fast navigation
3. Lower layers: dense, for fine-grained search
4. Query traverses from top layer down, following edges to nearby nodes

**Parameters:**
- M: Maximum connections per node (more = better recall, more memory)
- efConstruction: Build-time search depth (more = better quality, slower build)
- efSearch: Query-time search depth (more = better recall, slower query)

**Characteristics:**
- Excellent query speed and recall balance
- Memory-intensive (entire graph in RAM for best performance)
- Incremental insertions supported
- Most databases use HNSW: Pinecone, Qdrant, Weaviate, Milvus

### IVF (Inverted File)

Partition-based approach:

**How it works:**
1. Cluster vectors into k partitions (using k-means)
2. At query time, identify most relevant partitions
3. Search only within those partitions

**Parameters:**
- nlist: Number of clusters
- nprobe: Clusters to search at query time

**Characteristics:**
- Lower memory than HNSW
- Faster building than HNSW
- Works well with quantization (IVF-PQ, IVF-SQ8)
- Good for very large datasets

### DiskANN

Disk-based algorithm for datasets exceeding RAM:

**How it works:**
1. Graph index stored on SSD
2. In-memory cache for frequently accessed nodes
3. Optimized for sequential disk reads

**Characteristics:**
- Enables billion-scale search without massive RAM
- Requires fast NVMe storage
- Slightly higher latency than pure in-memory

### GPU-Accelerated Indexes

GPU acceleration for index building and search:
- NVIDIA CAGRA (Milvus, Qdrant)
- GPU IVF variants
- 10-100x speedup for large-scale operations

## Quantization

Reduce memory and increase speed by compressing vectors:

### Scalar Quantization (SQ)

Compress each float to smaller representation:
- FP32 (4 bytes) -> INT8 (1 byte): 4x compression
- Minimal accuracy loss with rescoring
- Fastest to compute

### Product Quantization (PQ)

Divide vector into subvectors, quantize each:
- Higher compression ratios (8-32x)
- More accuracy loss than SQ
- Good for extremely large datasets

### Binary Quantization (BQ)

Compress to single bits:
- 32x compression (FP32 -> 1 bit per dimension)
- Requires oversampling and rescoring
- Works well for high-dimensional embeddings

### Rescoring Strategy

Quantized search returns approximate results. Rescoring with full-precision vectors improves accuracy:

1. Search quantized index for k * oversample candidates
2. Fetch full vectors for candidates
3. Re-rank using exact distance
4. Return top k results

## Filtering

Combine vector similarity with metadata constraints:

```python
# Find similar documents that are also recent and in tech category
results = search(
    vector=query_embedding,
    filter={
        "category": "technology",
        "date": {"$gte": "2024-01-01"}
    },
    top_k=10
)
```

### Pre-filtering vs Post-filtering

**Post-filtering:**
1. Find top N similar vectors
2. Filter by metadata
3. Return remaining results

Problem: May return fewer than k results if filter is restrictive.

**Pre-filtering:**
1. Identify vectors matching filter
2. Search only within that subset

Problem: Expensive for low-selectivity filters (most vectors match).

**Integrated filtering (Qdrant approach):**
Extend HNSW graph with filter-aware edges. Most sophisticated but implementation-dependent.

### Partition Keys / Namespaces

Partition data for efficient filtered access:
- Pinecone: Namespaces
- Milvus: Partition keys
- Qdrant: Payload-based filtering

Best for tenant isolation or categorical filtering.

## Hybrid Search

Combine vector similarity with keyword matching:

### Dense + Sparse Vectors

```
Query: "python machine learning tutorial"

Dense (semantic):  Find conceptually similar content
Sparse (keywords): Match exact terms like "python", "tutorial"
```

Sparse vectors from BM25 or SPLADE capture keyword signals that dense embeddings might miss.

### Fusion Strategies

**Reciprocal Rank Fusion (RRF):**
```
score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
```

**Weighted Combination:**
```
score = alpha * dense_score + (1 - alpha) * sparse_score
```

### Reranking

Use a cross-encoder or dedicated reranking model to re-score combined results:

1. Retrieve candidates from both dense and sparse search
2. Apply reranking model (e.g., Cohere Rerank, cross-encoder)
3. Return final sorted results

## Platform Comparison

### Feature Matrix

| Feature | Pinecone | Milvus | Weaviate | Qdrant | Chroma |
|---------|----------|--------|----------|--------|--------|
| Hosting | Managed | Both | Both | Both | Self-hosted |
| Open Source | No | Yes | Yes | Yes | Yes |
| GPU Support | No | Yes | Limited | Yes | No |
| Index Types | Proprietary | Many | HNSW | HNSW | HNSW |
| Hybrid Search | Yes | Yes | Yes | Yes | Limited |
| Built-in Embedding | No | No | Yes | No | Yes |
| Filtering | Good | Excellent | Good | Excellent | Basic |
| Scale | Billions | Billions | Millions | Billions | Thousands |

### Performance Characteristics

| Database | Query Latency | Build Speed | Memory Efficiency |
|----------|---------------|-------------|-------------------|
| Pinecone | Low | N/A (managed) | Good (serverless) |
| Milvus | Low | Fast (GPU) | Configurable |
| Qdrant | Lowest | Fast (GPU) | Excellent |
| Weaviate | Medium | Medium | Medium |
| Chroma | Medium | Fast | Low optimization |

### Operational Complexity

| Database | Self-Hosted Complexity | Managed Option |
|----------|------------------------|----------------|
| Pinecone | N/A | Native |
| Milvus | High (K8s required) | Zilliz Cloud |
| Weaviate | Medium | Weaviate Cloud |
| Qdrant | Low | Qdrant Cloud |
| Chroma | Low | No |

## Decision Framework

### Choose Pinecone when:
- Operational simplicity is paramount
- Team lacks infrastructure expertise
- Rapid time-to-production required
- Variable workload with serverless benefits

### Choose Milvus when:
- Billion-scale deployments needed
- GPU acceleration is required
- Multiple index algorithm options desired
- Kubernetes expertise available

### Choose Weaviate when:
- Built-in ML integration matters
- GraphQL API preferred
- Multi-modal search needed
- RAG with generative search

### Choose Qdrant when:
- Performance is critical
- Advanced filtering required
- Memory efficiency important
- Simpler self-hosting than Milvus

### Choose Chroma when:
- Rapid prototyping
- Learning/tutorials
- Small-scale applications
- Tight LangChain integration

## RAG Integration

Vector databases are the retrieval component in RAG (Retrieval-Augmented Generation):

```python
# 1. Index documents
for doc in documents:
    embedding = embed(doc.text)
    vector_db.upsert(
        id=doc.id,
        vector=embedding,
        metadata={"text": doc.text, "source": doc.source}
    )

# 2. Query
query_embedding = embed(user_question)
results = vector_db.query(vector=query_embedding, top_k=5)

# 3. Generate
context = "\n".join([r.metadata["text"] for r in results])
response = llm.generate(
    prompt=f"Context:\n{context}\n\nQuestion: {user_question}"
)
```

### Chunking Strategies

Before indexing, split documents into chunks:
- Fixed size (e.g., 512 tokens)
- Semantic (paragraph/section boundaries)
- Recursive (split on multiple delimiters)
- Document-specific (code blocks, tables separately)

Chunk size affects retrieval quality:
- Too small: loses context
- Too large: dilutes relevance, wastes tokens

### Embedding Model Selection

| Model | Dimensions | Quality | Speed | Cost |
|-------|------------|---------|-------|------|
| OpenAI text-embedding-3-small | 1536 | Good | Fast | Low |
| OpenAI text-embedding-3-large | 3072 | Excellent | Fast | Medium |
| Cohere embed-v3 | 1024 | Excellent | Fast | Medium |
| Sentence Transformers | 384-768 | Good | Local | Free |

Consider:
- Dimensionality affects storage cost
- Model must match between index and query
- Local models avoid API latency/costs

## Scaling Considerations

### Horizontal Scaling

- Sharding: Distribute data across nodes
- Replication: Copies for fault tolerance and read throughput
- Load balancing: Route queries across replicas

### Cost Optimization

- Quantization reduces storage 4-32x
- Disk-based indexes for cold data
- Auto-scaling for variable workloads
- Delete unused vectors regularly

### Monitoring

Key metrics:
- Query latency (p50, p95, p99)
- Recall/accuracy
- Index build time
- Memory utilization
- Query throughput (QPS)

## Further Reading

For detailed information on each platform, see:
- [Pinecone](pinecone/ReadMe.md)
- [Milvus](milvus/ReadMe.md)
- [Weaviate](weaviate/ReadMe.md)
- [Qdrant](qdrant/ReadMe.md)
- [Chroma](chroma/ReadMe.md)
