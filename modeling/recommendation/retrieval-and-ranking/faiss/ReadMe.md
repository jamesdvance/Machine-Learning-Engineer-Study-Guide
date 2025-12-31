# FAISS (Facebook AI Similarity Search)

## Summary

FAISS is Meta's open-source library for efficient similarity search and clustering of dense vectors. It provides highly optimized implementations of various ANN algorithms, with particular strength in GPU acceleration and billion-scale indexes. FAISS is the most widely used vector search library in production ML systems, supporting everything from simple brute-force search to sophisticated composite indexes that combine multiple techniques.

Key points to remember:

- Meta's production-grade library for vector similarity search
- Supports CPU and GPU (CUDA) acceleration
- Indexes: Flat, IVF, PQ, HNSW, and composites (IVF+PQ, IVFPQ+HNSW)
- Inner product and L2 distance metrics (use normalized vectors for cosine)
- GPU version 5-10x faster than CPU for large batches
- Index factory syntax for easy index creation: `"IVF1024,PQ32"`
- Memory-mapped indexes for larger-than-RAM datasets
- Used by: Spotify, Pinterest, Meta, Instacart, many others

## Installation

```bash
# CPU only
pip install faiss-cpu

# GPU (requires CUDA)
pip install faiss-gpu

# Conda (recommended for GPU)
conda install -c pytorch faiss-gpu
```

## Core Concepts

### Index Types Overview

```
+----------------+------------------+------------------------+
|    Index       |    Best For      |   Memory/Speed         |
+----------------+------------------+------------------------+
| IndexFlatL2    | <100K vectors    | Exact, slow for large  |
| IndexFlatIP    | Cosine (norm'd)  | Exact, slow for large  |
| IndexIVFFlat   | 100K-1M vectors  | Good recall, fast      |
| IndexIVFPQ     | 1M-100M vectors  | Compressed, very fast  |
| IndexHNSWFlat  | High recall need | Graph-based, high mem  |
| IndexIVFScalarQ| Memory limited   | 4x compression         |
+----------------+------------------+------------------------+
```

### Basic Usage

```python
import faiss
import numpy as np

# Create some vectors
d = 128  # Dimension
n = 100000  # Database size
nq = 1000  # Number of queries

# Database vectors
xb = np.random.random((n, d)).astype('float32')
# Query vectors
xq = np.random.random((nq, d)).astype('float32')

# Build exact (brute-force) index
index = faiss.IndexFlatL2(d)
index.add(xb)

# Search
k = 10  # Top-k neighbors
distances, indices = index.search(xq, k)

print(f"Nearest neighbors shape: {indices.shape}")  # (1000, 10)
print(f"Distances shape: {distances.shape}")  # (1000, 10)
```

## Index Types in Detail

### Flat Indexes (Exact Search)

```python
import faiss
import numpy as np

d = 256
vectors = np.random.randn(100000, d).astype('float32')

# L2 (Euclidean) distance
index_l2 = faiss.IndexFlatL2(d)
index_l2.add(vectors)

# Inner product (for cosine similarity, normalize first)
faiss.normalize_L2(vectors)  # In-place normalization
index_ip = faiss.IndexFlatIP(d)
index_ip.add(vectors)

# For cosine similarity queries
query = np.random.randn(1, d).astype('float32')
faiss.normalize_L2(query)
scores, indices = index_ip.search(query, k=10)
# scores are now cosine similarities
```

### IVF Indexes (Inverted File)

```python
import faiss
import numpy as np

d = 256
n = 1000000
vectors = np.random.randn(n, d).astype('float32')

# Number of clusters (Voronoi cells)
nlist = 1024  # Rule of thumb: sqrt(n) to 4*sqrt(n)

# Create IVF index with flat storage
quantizer = faiss.IndexFlatL2(d)  # Coarse quantizer
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train on subset of data (learns cluster centroids)
training_vectors = vectors[:100000]  # Use 10% or 100K vectors
index.train(training_vectors)

# Add all vectors
index.add(vectors)

# Search parameters
index.nprobe = 10  # Number of clusters to search (trade-off: speed vs recall)

query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=10)

# Tuning nprobe:
# nprobe=1:   Very fast, low recall (~0.6)
# nprobe=10:  Balanced (~0.9 recall)
# nprobe=100: Slow, high recall (~0.99)
# nprobe=nlist: Equivalent to exhaustive search
```

### Product Quantization (PQ)

```python
import faiss
import numpy as np

d = 256
n = 10000000  # 10M vectors
vectors = np.random.randn(n, d).astype('float32')

# PQ parameters
m = 32  # Number of subquantizers (d must be divisible by m)
nbits = 8  # Bits per subquantizer (256 centroids each)

# IVF + PQ index
nlist = 4096
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Train (learns both IVF centroids and PQ codebooks)
training_size = min(n, 1000000)
index.train(vectors[:training_size])

# Add vectors (stores compressed codes)
index.add(vectors)

# Memory comparison:
# Original: 10M * 256 * 4 bytes = 10 GB
# Compressed: 10M * 32 bytes = 320 MB (32x compression!)

print(f"Index size: {index.ntotal} vectors")

# Search
index.nprobe = 64
query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=10)
```

### HNSW Index

```python
import faiss
import numpy as np

d = 256
n = 1000000
vectors = np.random.randn(n, d).astype('float32')

# HNSW parameters
M = 32  # Number of connections per layer
ef_construction = 200  # Size of dynamic candidate list during construction

# Create HNSW index
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction

# No training needed for HNSW
index.add(vectors)

# Search parameter
index.hnsw.efSearch = 64  # Higher = better recall, slower

query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=10)
```

### Scalar Quantization

```python
import faiss
import numpy as np

d = 256
n = 1000000
vectors = np.random.randn(n, d).astype('float32')

# IVF with Scalar Quantizer (8-bit quantization)
nlist = 1024
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFScalarQuantizer(
    quantizer, d, nlist,
    faiss.ScalarQuantizer.QT_8bit  # 4x compression
)

index.train(vectors[:100000])
index.add(vectors)

# Memory: 1M * 256 bytes = 256 MB (vs 1GB for float32)
```

## Index Factory

The index factory provides a string-based interface for creating indexes:

```python
import faiss
import numpy as np

d = 256
n = 1000000
vectors = np.random.randn(n, d).astype('float32')

# Index factory syntax: "preprocessing,index_type,encoding"

# Common patterns:
index_strings = {
    # Flat (exact)
    "Flat": "Flat",

    # IVF with flat storage
    "IVF1024,Flat": faiss.index_factory(d, "IVF1024,Flat"),

    # IVF with PQ compression
    "IVF1024,PQ32": faiss.index_factory(d, "IVF1024,PQ32"),

    # IVF with Scalar Quantizer
    "IVF1024,SQ8": faiss.index_factory(d, "IVF1024,SQ8"),

    # HNSW with flat storage
    "HNSW32": faiss.index_factory(d, "HNSW32"),

    # HNSW + IVF + PQ (composite)
    "IVF1024_HNSW32,PQ32": faiss.index_factory(d, "IVF1024_HNSW32,PQ32"),

    # With preprocessing (OPQ rotation for better PQ)
    "OPQ32,IVF1024,PQ32": faiss.index_factory(d, "OPQ32,IVF1024,PQ32"),

    # PCA dimension reduction + index
    "PCA128,IVF1024,PQ16": faiss.index_factory(d, "PCA128,IVF1024,PQ16"),
}

# Example: Build an IVF+PQ index
index = faiss.index_factory(d, "IVF4096,PQ64")
index.train(vectors[:100000])
index.add(vectors)

# Set search parameters
faiss.ParameterSpace().set_index_parameter(index, "nprobe", 32)
```

## GPU Acceleration

```python
import faiss
import numpy as np

# Check GPU availability
ngpus = faiss.get_num_gpus()
print(f"Number of GPUs: {ngpus}")

d = 256
n = 10000000  # 10M vectors
vectors = np.random.randn(n, d).astype('float32')

# Option 1: Single GPU
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # GPU 0
gpu_index.add(vectors)

# Option 2: All GPUs (shards across GPUs)
cpu_index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(vectors)

# Option 3: GPU IVF index
res = faiss.StandardGpuResources()
config = faiss.GpuIndexIVFFlatConfig()
config.device = 0

gpu_index = faiss.GpuIndexIVFFlat(
    res, d, 4096,  # nlist
    faiss.METRIC_L2, config
)
gpu_index.train(vectors[:100000])
gpu_index.add(vectors)

# Search
query = np.random.randn(1000, d).astype('float32')
gpu_index.setNumProbes(32)
distances, indices = gpu_index.search(query, k=10)

# Transfer back to CPU if needed
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
```

### GPU Memory Management

```python
import faiss

# Configure GPU resources
res = faiss.StandardGpuResources()

# Limit GPU memory usage
res.setTempMemory(1024 * 1024 * 512)  # 512 MB temp memory

# Use pinned memory for faster CPU-GPU transfers
res.setPinnedMemory(256 * 1024 * 1024)  # 256 MB

# For large indexes that don't fit in GPU memory
# Use GPU for search only, keep data on CPU
config = faiss.GpuClonerOptions()
config.useFloat16 = True  # Use FP16 to save memory
config.usePrecomputed = False  # Don't precompute tables
```

## Advanced Features

### ID Mapping

```python
import faiss
import numpy as np

d = 128
n = 100000

# Vectors with custom IDs
vectors = np.random.randn(n, d).astype('float32')
ids = np.arange(1000000, 1000000 + n).astype('int64')  # Custom IDs

# Wrap index with ID map
base_index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap(base_index)

# Add with IDs
index.add_with_ids(vectors, ids)

# Search returns custom IDs
query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=10)
print(f"Returned IDs: {indices[0]}")  # [1000xxx, ...]
```

### Removing Vectors

```python
import faiss
import numpy as np

d = 128
vectors = np.random.randn(10000, d).astype('float32')
ids = np.arange(10000).astype('int64')

# Use IndexIDMap2 for removable index
base_index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap2(base_index)
index.add_with_ids(vectors, ids)

# Remove specific IDs
ids_to_remove = np.array([0, 1, 2, 3]).astype('int64')
index.remove_ids(ids_to_remove)

print(f"Remaining vectors: {index.ntotal}")  # 9996
```

### Range Search

```python
import faiss
import numpy as np

d = 128
n = 100000
vectors = np.random.randn(n, d).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(vectors)

# Find all neighbors within distance threshold
query = np.random.randn(1, d).astype('float32')
radius = 10.0  # L2 distance threshold

# Returns sparse results
lims, distances, indices = index.range_search(query, radius)

# Parse results (first query only)
neighbors = indices[lims[0]:lims[1]]
dists = distances[lims[0]:lims[1]]
print(f"Found {len(neighbors)} neighbors within radius {radius}")
```

### Save and Load

```python
import faiss
import numpy as np

d = 256
vectors = np.random.randn(1000000, d).astype('float32')

# Create and train index
index = faiss.index_factory(d, "IVF4096,PQ32")
index.train(vectors[:100000])
index.add(vectors)

# Save to disk
faiss.write_index(index, "my_index.faiss")

# Load from disk
loaded_index = faiss.read_index("my_index.faiss")

# Memory-mapped loading (for larger-than-RAM indexes)
loaded_index = faiss.read_index("my_index.faiss", faiss.IO_FLAG_MMAP)
```

### Sharding Across Multiple Indexes

```python
import faiss
import numpy as np

d = 256
n = 10000000  # 10M vectors
vectors = np.random.randn(n, d).astype('float32')

# Create sharded index
n_shards = 4
shard_size = n // n_shards

# IndexShards for merging results
merged_index = faiss.IndexShards(d, True, False)  # (dim, threaded, add_id)

for i in range(n_shards):
    start = i * shard_size
    end = start + shard_size

    shard = faiss.IndexFlatL2(d)
    shard.add(vectors[start:end])
    merged_index.add_shard(shard)

# Search across all shards
query = np.random.randn(1, d).astype('float32')
distances, indices = merged_index.search(query, k=10)

# Note: IndexReplicas for replicating across GPUs (for throughput)
```

## Production Patterns

### Online Index Updates

```python
import faiss
import numpy as np
from threading import Lock

class OnlineFaissIndex:
    """Thread-safe FAISS index with online updates."""

    def __init__(self, dim: int, index_string: str = "IVF1024,Flat"):
        self.dim = dim
        self.index = faiss.index_factory(dim, index_string)
        self.is_trained = False
        self.lock = Lock()

    def train(self, vectors: np.ndarray):
        """Train the index (required before adding for IVF)."""
        with self.lock:
            self.index.train(vectors)
            self.is_trained = True

    def add(self, vectors: np.ndarray, ids: np.ndarray = None):
        """Add vectors to the index."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        vectors = np.ascontiguousarray(vectors.astype('float32'))

        with self.lock:
            if ids is not None:
                # Wrap in IDMap if using custom IDs
                if not isinstance(self.index, faiss.IndexIDMap):
                    base_index = self.index
                    self.index = faiss.IndexIDMap(base_index)
                self.index.add_with_ids(vectors, ids)
            else:
                self.index.add(vectors)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple:
        """Search for nearest neighbors."""
        queries = np.ascontiguousarray(queries.astype('float32'))

        with self.lock:
            return self.index.search(queries, k)

    def save(self, path: str):
        """Save index to disk."""
        with self.lock:
            faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load index from disk."""
        with self.lock:
            self.index = faiss.read_index(path)
            self.is_trained = True
```

### Batched Query Processing

```python
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def batch_search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int = 10,
    batch_size: int = 1000,
) -> tuple:
    """Process queries in batches for better memory efficiency."""
    n_queries = len(queries)
    all_distances = []
    all_indices = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = queries[start:end]

        distances, indices = index.search(batch, k)
        all_distances.append(distances)
        all_indices.append(indices)

    return np.vstack(all_distances), np.vstack(all_indices)

# For GPU, larger batches are better (saturate GPU)
# For CPU, smaller batches may be needed for memory
```

### Combining with Vector Databases

```python
import faiss
import numpy as np
import sqlite3
from typing import List, Dict

class HybridVectorStore:
    """FAISS for vectors + SQLite for metadata."""

    def __init__(self, dim: int, db_path: str):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                metadata TEXT
            )
        """)
        self.db.commit()

    def add(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        metadata: List[Dict]
    ):
        """Add vectors with metadata."""
        # Add to FAISS
        self.index.add_with_ids(vectors, ids)

        # Add metadata to SQLite
        import json
        for id_, meta in zip(ids, metadata):
            self.db.execute(
                "INSERT OR REPLACE INTO items (id, metadata) VALUES (?, ?)",
                (int(id_), json.dumps(meta))
            )
        self.db.commit()

    def search(self, query: np.ndarray, k: int = 10) -> List[Dict]:
        """Search and return results with metadata."""
        import json

        distances, indices = self.index.search(query.reshape(1, -1), k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for missing
                cursor = self.db.execute(
                    "SELECT metadata FROM items WHERE id = ?",
                    (int(idx),)
                )
                row = cursor.fetchone()
                if row:
                    results.append({
                        'id': int(idx),
                        'distance': float(dist),
                        'metadata': json.loads(row[0])
                    })

        return results
```

## Choosing the Right Index

```python
def recommend_index(
    n_vectors: int,
    dim: int,
    memory_gb: float,
    recall_target: float,
    qps_target: int,
) -> str:
    """Recommend FAISS index based on requirements."""

    # Memory per vector (float32)
    bytes_per_vector = dim * 4

    # Available memory for index
    available_bytes = memory_gb * 1e9

    # Can we fit flat index?
    flat_memory = n_vectors * bytes_per_vector
    if flat_memory < available_bytes * 0.8:
        if n_vectors < 100000:
            return "Flat"  # Exact search is fast enough
        else:
            return "HNSW32"  # Better for larger datasets

    # Need compression
    compression_needed = flat_memory / available_bytes

    if compression_needed < 4:
        # Scalar quantization (4x compression)
        nlist = min(int(np.sqrt(n_vectors)), 4096)
        return f"IVF{nlist},SQ8"

    elif compression_needed < 32:
        # Product quantization
        nlist = min(int(np.sqrt(n_vectors)), 4096)
        # m = number of subquantizers, nbits = 8
        # Compression: dim / m ratio
        m = max(dim // 8, 8)
        return f"IVF{nlist},PQ{m}"

    else:
        # Heavy compression with OPQ preprocessing
        nlist = min(int(np.sqrt(n_vectors)), 8192)
        m = 16  # Aggressive compression
        return f"OPQ{m},IVF{nlist},PQ{m}"

# Example usage
print(recommend_index(
    n_vectors=10_000_000,
    dim=768,
    memory_gb=8,
    recall_target=0.95,
    qps_target=1000
))
# Output: "IVF4096,PQ64" or similar
```

## Benchmarking

```python
import faiss
import numpy as np
import time

def benchmark_index(
    index_string: str,
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
):
    """Benchmark a FAISS index configuration."""
    d = train_vectors.shape[1]

    # Build index
    index = faiss.index_factory(d, index_string)

    # Train
    start = time.time()
    if hasattr(index, 'train'):
        index.train(train_vectors)
    train_time = time.time() - start

    # Add
    start = time.time()
    index.add(test_vectors)
    add_time = time.time() - start

    # Search with different nprobe values
    results = []
    for nprobe in [1, 4, 16, 64, 256]:
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        except:
            pass  # Not all indexes support nprobe

        start = time.time()
        distances, indices = index.search(queries, k)
        search_time = time.time() - start

        # Compute recall
        recall = 0
        for i in range(len(queries)):
            true_set = set(ground_truth[i][:k])
            pred_set = set(indices[i][:k])
            recall += len(true_set & pred_set) / k
        recall /= len(queries)

        results.append({
            'nprobe': nprobe,
            'recall@10': recall,
            'qps': len(queries) / search_time,
            'latency_ms': search_time / len(queries) * 1000,
        })

    return {
        'index': index_string,
        'train_time': train_time,
        'add_time': add_time,
        'memory_mb': index.ntotal * index.d * 4 / 1e6,  # Approximate
        'results': results,
    }
```
