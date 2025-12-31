# Approximate Nearest Neighbors (ANN)

## Summary

Approximate Nearest Neighbors (ANN) algorithms find the closest vectors to a query in high-dimensional spaces, trading exact accuracy for massive speed improvements. Instead of computing distances to every vector (O(n)), ANN methods use clever indexing structures to achieve sub-linear search times, typically O(log n) or O(1). These algorithms are essential for real-time retrieval in recommendation systems, semantic search, and any application requiring fast similarity lookups over millions of embeddings.

Key points to remember:

- Trade-off: recall vs. latency vs. memory (can't optimize all three)
- Main algorithm families: tree-based, hash-based, graph-based, quantization-based
- Graph-based (HNSW) often best for high recall requirements
- Quantization (IVF, PQ) best for memory-constrained settings
- Always benchmark on your specific data distribution
- Index build time vs. query time trade-off varies by algorithm
- Common metrics: recall@k, queries per second (QPS), memory footprint
- Distance metrics: L2 (Euclidean), inner product, cosine similarity

## Why Approximate?

### The Curse of Dimensionality

Exact nearest neighbor search in high dimensions is impractical:

```python
import numpy as np
import time

def exact_knn(query, corpus, k=10):
    """Exact k-nearest neighbors - O(n*d) per query."""
    distances = np.linalg.norm(corpus - query, axis=1)
    return np.argsort(distances)[:k]

# Benchmark
n_vectors = 1_000_000
embedding_dim = 768  # BERT-sized
corpus = np.random.randn(n_vectors, embedding_dim).astype('float32')
query = np.random.randn(embedding_dim).astype('float32')

start = time.time()
result = exact_knn(query, corpus)
exact_time = time.time() - start
print(f"Exact search: {exact_time:.2f}s")  # ~2-5 seconds per query!

# At 100 QPS requirement, you'd need 200-500 CPU cores just for search
```

### Approximate Search Trade-offs

```
                    High Recall
                        ^
                        |
                        |  Graph-based (HNSW)
                        |     *
                        |
                        |      * IVF + fine quantizer
                        |
                        |  * IVF-PQ
                        |
                        |    * LSH
                        +-------------------------> Low Latency
                   High Latency

Memory usage: LSH < PQ < IVF-PQ < HNSW
Build time:   LSH < IVF-PQ < HNSW < IVF-flat
```

## Algorithm Families

### 1. Tree-Based Methods

#### KD-Trees

Partition space recursively along dimensions:

```python
from scipy.spatial import KDTree
import numpy as np

# Build tree
vectors = np.random.randn(100000, 32).astype('float32')
tree = KDTree(vectors)

# Query
query = np.random.randn(32).astype('float32')
distances, indices = tree.query(query, k=10)

# Problem: KD-trees degrade in high dimensions (>20)
# The number of nodes visited approaches n as d increases
```

#### Ball Trees

Better for high dimensions than KD-trees, but still limited:

```python
from sklearn.neighbors import BallTree
import numpy as np

vectors = np.random.randn(100000, 64).astype('float32')
tree = BallTree(vectors, leaf_size=40)

query = np.random.randn(1, 64).astype('float32')
distances, indices = tree.query(query, k=10)
```

### 2. Hash-Based Methods

#### Locality-Sensitive Hashing (LSH)

Hash similar vectors to the same bucket with high probability:

```python
import numpy as np
from typing import List, Dict, Set

class RandomProjectionLSH:
    """LSH using random hyperplane projections."""

    def __init__(
        self,
        dim: int,
        n_tables: int = 10,
        n_bits: int = 16,
    ):
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits

        # Random projection matrices (one per table)
        self.projections = [
            np.random.randn(n_bits, dim).astype('float32')
            for _ in range(n_tables)
        ]

        # Hash tables
        self.tables: List[Dict[tuple, Set[int]]] = [
            {} for _ in range(n_tables)
        ]
        self.vectors = None

    def _hash(self, vector: np.ndarray, table_idx: int) -> tuple:
        """Compute hash for a vector."""
        projection = self.projections[table_idx]
        # Sign of dot product with each random vector
        bits = (projection @ vector > 0).astype(int)
        return tuple(bits)

    def build(self, vectors: np.ndarray):
        """Index all vectors."""
        self.vectors = vectors

        for idx, vector in enumerate(vectors):
            for table_idx in range(self.n_tables):
                hash_key = self._hash(vector, table_idx)
                if hash_key not in self.tables[table_idx]:
                    self.tables[table_idx][hash_key] = set()
                self.tables[table_idx][hash_key].add(idx)

    def query(self, query: np.ndarray, k: int = 10) -> List[int]:
        """Find approximate nearest neighbors."""
        candidates = set()

        # Collect candidates from all tables
        for table_idx in range(self.n_tables):
            hash_key = self._hash(query, table_idx)
            if hash_key in self.tables[table_idx]:
                candidates.update(self.tables[table_idx][hash_key])

        if not candidates:
            return []

        # Compute exact distances for candidates only
        candidate_list = list(candidates)
        distances = np.linalg.norm(
            self.vectors[candidate_list] - query, axis=1
        )
        sorted_indices = np.argsort(distances)[:k]

        return [candidate_list[i] for i in sorted_indices]

# Usage
lsh = RandomProjectionLSH(dim=128, n_tables=20, n_bits=12)
lsh.build(vectors)
neighbors = lsh.query(query_vector, k=10)
```

### 3. Quantization-Based Methods

#### Product Quantization (PQ)

Compress vectors by splitting into subspaces and quantizing each:

```python
import numpy as np
from sklearn.cluster import KMeans
from typing import List

class ProductQuantizer:
    """Product Quantization for vector compression."""

    def __init__(
        self,
        dim: int,
        n_subvectors: int = 8,
        n_centroids: int = 256,  # Per subspace
    ):
        self.dim = dim
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.subvector_dim = dim // n_subvectors

        assert dim % n_subvectors == 0

        self.codebooks: List[np.ndarray] = []  # Centroids per subspace
        self.codes: np.ndarray = None  # Compressed vectors

    def train(self, vectors: np.ndarray):
        """Learn codebooks from training data."""
        for m in range(self.n_subvectors):
            # Extract subvector
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # Cluster to find centroids
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=1)
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_.astype('float32'))

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to PQ codes."""
        n = len(vectors)
        codes = np.zeros((n, self.n_subvectors), dtype=np.uint8)

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # Find nearest centroid for each subvector
            distances = np.linalg.norm(
                subvectors[:, np.newaxis] - self.codebooks[m],
                axis=2
            )
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def compute_distances(
        self,
        query: np.ndarray,
        codes: np.ndarray
    ) -> np.ndarray:
        """Compute approximate distances using precomputed tables."""
        # Precompute distance table: distance from query subvector to each centroid
        distance_table = np.zeros((self.n_subvectors, self.n_centroids))

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            query_subvector = query[start:end]

            distance_table[m] = np.linalg.norm(
                self.codebooks[m] - query_subvector, axis=1
            ) ** 2

        # Sum up distances using lookup
        distances = np.zeros(len(codes))
        for m in range(self.n_subvectors):
            distances += distance_table[m, codes[:, m]]

        return np.sqrt(distances)

# Memory savings: 768-dim float32 = 3072 bytes
# PQ with 8 subvectors, 256 centroids = 8 bytes (97% compression!)
```

#### Inverted File Index (IVF)

Cluster vectors and only search relevant clusters:

```python
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

class IVFIndex:
    """Inverted File Index for coarse quantization."""

    def __init__(self, n_clusters: int = 1000, n_probe: int = 10):
        self.n_clusters = n_clusters
        self.n_probe = n_probe  # Clusters to search
        self.kmeans = None
        self.inverted_lists: List[List[Tuple[int, np.ndarray]]] = []

    def train(self, vectors: np.ndarray):
        """Train coarse quantizer."""
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=1)
        self.kmeans.fit(vectors)
        self.inverted_lists = [[] for _ in range(self.n_clusters)]

    def add(self, vectors: np.ndarray, ids: np.ndarray = None):
        """Add vectors to index."""
        if ids is None:
            ids = np.arange(len(vectors))

        # Assign to clusters
        assignments = self.kmeans.predict(vectors)

        for idx, (vec, cluster_id) in enumerate(zip(vectors, assignments)):
            self.inverted_lists[cluster_id].append((ids[idx], vec))

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors."""
        # Find nearest clusters
        cluster_distances = np.linalg.norm(
            self.kmeans.cluster_centers_ - query, axis=1
        )
        nearest_clusters = np.argsort(cluster_distances)[:self.n_probe]

        # Search within selected clusters
        candidates = []
        for cluster_id in nearest_clusters:
            for vec_id, vec in self.inverted_lists[cluster_id]:
                distance = np.linalg.norm(vec - query)
                candidates.append((vec_id, distance))

        # Return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
```

### 4. Graph-Based Methods

#### Hierarchical Navigable Small World (HNSW)

Build a navigable graph with hierarchical layers:

```python
import numpy as np
from typing import List, Set, Dict, Tuple
import heapq
import random

class HNSWIndex:
    """
    Simplified HNSW implementation.
    Real implementations (like hnswlib) are much more optimized.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,           # Max connections per node
        ef_construction: int = 200,  # Size of dynamic candidate list
        ml: float = 1 / np.log(16),  # Level multiplier
    ):
        self.dim = dim
        self.M = M
        self.M0 = M * 2  # Connections at layer 0
        self.ef_construction = ef_construction
        self.ml = ml

        self.vectors: List[np.ndarray] = []
        self.graphs: List[Dict[int, List[int]]] = []  # Graph per layer
        self.max_layer = 0
        self.entry_point = None

    def _get_random_level(self) -> int:
        """Sample a random level for new node."""
        return int(-np.log(random.random()) * self.ml)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance."""
        return np.linalg.norm(a - b)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        """Search a single layer, return ef nearest neighbors."""
        visited = set(entry_points)
        candidates = []  # Min-heap
        results = []     # Max-heap (negative distance)

        for ep in entry_points:
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # If candidate is farther than worst result, stop
            if len(results) >= ef and dist_c > -results[0][0]:
                break

            # Explore neighbors
            for neighbor in self.graphs[layer].get(c, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_n = self._distance(query, self.vectors[neighbor])

                    if len(results) < ef or dist_n < -results[0][0]:
                        heapq.heappush(candidates, (dist_n, neighbor))
                        heapq.heappush(results, (-dist_n, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

        return [(-dist, idx) for dist, idx in results]

    def add(self, vector: np.ndarray) -> int:
        """Add a vector to the index."""
        node_id = len(self.vectors)
        self.vectors.append(vector)
        node_level = self._get_random_level()

        # Ensure enough layers exist
        while len(self.graphs) <= node_level:
            self.graphs.append({})

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = node_level
            for layer in range(node_level + 1):
                self.graphs[layer][node_id] = []
            return node_id

        # Search from top to insertion layer
        current_nodes = [self.entry_point]

        for layer in range(self.max_layer, node_level, -1):
            results = self._search_layer(vector, current_nodes, 1, layer)
            current_nodes = [results[0][1]]

        # Insert at each layer from node_level down to 0
        for layer in range(min(node_level, self.max_layer), -1, -1):
            results = self._search_layer(
                vector, current_nodes, self.ef_construction, layer
            )

            # Select neighbors (simplified: just take closest)
            M = self.M0 if layer == 0 else self.M
            neighbors = [idx for _, idx in sorted(results)[:M]]

            # Add bidirectional connections
            self.graphs[layer][node_id] = neighbors
            for neighbor in neighbors:
                self.graphs[layer][neighbor].append(node_id)
                # Prune if too many connections
                if len(self.graphs[layer][neighbor]) > M:
                    # Keep closest M
                    dists = [
                        (self._distance(self.vectors[neighbor], self.vectors[n]), n)
                        for n in self.graphs[layer][neighbor]
                    ]
                    self.graphs[layer][neighbor] = [
                        n for _, n in sorted(dists)[:M]
                    ]

            current_nodes = [idx for _, idx in results]

        # Update entry point if new node is at higher layer
        if node_level > self.max_layer:
            self.max_layer = node_level
            self.entry_point = node_id

        return node_id

    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> List[int]:
        """Search for k nearest neighbors."""
        if self.entry_point is None:
            return []

        current_nodes = [self.entry_point]

        # Traverse from top layer
        for layer in range(self.max_layer, 0, -1):
            results = self._search_layer(query, current_nodes, 1, layer)
            current_nodes = [results[0][1]]

        # Search bottom layer with ef candidates
        results = self._search_layer(query, current_nodes, ef, 0)

        # Return top k
        return [idx for _, idx in sorted(results)[:k]]
```

## Comparison of Algorithms

| Algorithm | Build Time | Query Time | Memory | Recall | Best For |
|-----------|------------|------------|--------|--------|----------|
| KD-Tree | O(n log n) | O(log n)* | O(n) | 100%* | Low dim (<20) |
| LSH | O(n) | O(1) | O(n·L) | Variable | Streaming data |
| IVF | O(n) | O(n/k·nprobe) | O(n) | ~95% | Memory constrained |
| PQ | O(n) | O(n) | O(n·m) | ~90% | Very large scale |
| IVF-PQ | O(n) | O(n/k·nprobe) | O(n·m) | ~90% | Billion-scale |
| HNSW | O(n log n) | O(log n) | O(n·M) | ~99% | High recall needed |

*Degrades in high dimensions

## Practical Implementation with Libraries

### Using hnswlib

```python
import hnswlib
import numpy as np

# Initialize index
dim = 768
num_elements = 1000000

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(
    max_elements=num_elements,
    ef_construction=200,  # Higher = better quality, slower build
    M=16,                 # Connections per node
)

# Add vectors
vectors = np.random.randn(num_elements, dim).astype('float32')
ids = np.arange(num_elements)
index.add_items(vectors, ids)

# Query
index.set_ef(50)  # Higher = better recall, slower search
query = np.random.randn(dim).astype('float32')
labels, distances = index.knn_query(query, k=10)

# Save/load
index.save_index("hnsw_index.bin")
# index.load_index("hnsw_index.bin", max_elements=num_elements)
```

### Using Annoy (Spotify)

```python
from annoy import AnnoyIndex
import numpy as np

dim = 768
n_trees = 100  # More trees = better accuracy, more memory

# Build index
index = AnnoyIndex(dim, 'angular')  # or 'euclidean'

for i in range(1000000):
    vector = np.random.randn(dim).astype('float32')
    index.add_item(i, vector)

index.build(n_trees)

# Query
query = np.random.randn(dim).astype('float32')
neighbors = index.get_nns_by_vector(query, 10, include_distances=True)

# Save/load
index.save('annoy_index.ann')
# index.load('annoy_index.ann')
```

## Benchmarking ANN Algorithms

```python
import numpy as np
import time
from typing import List, Tuple, Callable

def benchmark_ann(
    index_build_fn: Callable,
    search_fn: Callable,
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,  # Exact nearest neighbors
    k: int = 10,
) -> dict:
    """Benchmark an ANN algorithm."""

    # Build time
    start = time.time()
    index = index_build_fn(train_vectors)
    build_time = time.time() - start

    # Query time and recall
    total_time = 0
    total_recall = 0

    for i, query in enumerate(query_vectors):
        start = time.time()
        results = search_fn(index, query, k)
        total_time += time.time() - start

        # Recall
        true_neighbors = set(ground_truth[i][:k])
        found_neighbors = set(results[:k])
        recall = len(true_neighbors & found_neighbors) / k
        total_recall += recall

    n_queries = len(query_vectors)
    return {
        'build_time_s': build_time,
        'avg_query_time_ms': (total_time / n_queries) * 1000,
        'queries_per_second': n_queries / total_time,
        'recall_at_k': total_recall / n_queries,
    }

# Use ann-benchmarks.com for comprehensive comparisons
```

## Tuning Parameters

### HNSW Tuning

```python
# For high recall (>99%):
ef_construction = 400  # Build parameter
ef_search = 200        # Search parameter
M = 32                 # Connections

# For low latency:
ef_construction = 100
ef_search = 20
M = 12

# Memory vs quality trade-off
# Memory ~ O(n * M * 4 bytes)
# 1M vectors, M=16: ~64MB just for graph
```

### IVF Tuning

```python
# Rule of thumb: n_clusters ~ sqrt(n)
# For 1M vectors: ~1000 clusters

# Recall vs speed
n_probe = 1    # Fast, low recall
n_probe = 10   # Balanced
n_probe = 100  # Slow, high recall

# More clusters = faster search but lower recall (more vectors missed)
```

## When to Use Which Algorithm

- **HNSW**: Default choice for most use cases, best recall/speed trade-off
- **IVF-PQ**: Memory constrained, very large scale (billions)
- **Annoy**: Simple, good for static datasets, memory-mapped
- **LSH**: Streaming data, exact nearest neighbor probability bounds needed
- **ScaNN**: Google's optimized implementation, often fastest on benchmarks

## Common Pitfalls

1. **Not normalizing for cosine similarity**: Many indexes use inner product; normalize vectors first
2. **Wrong distance metric**: Ensure index metric matches your embedding space
3. **Over-tuning on benchmarks**: Real query distributions may differ
4. **Ignoring build time**: HNSW can take hours for large datasets
5. **Not testing update patterns**: Some indexes don't support efficient updates
