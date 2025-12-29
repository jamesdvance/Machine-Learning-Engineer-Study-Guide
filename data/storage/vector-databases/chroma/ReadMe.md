# Chroma

## Summary

Chroma is an open-source embedding database designed for simplicity and ease of use, making it the fastest way to add memory to AI applications. With minimal setup, developers can store documents with automatic embedding generation, query by semantic similarity, and filter by metadata. Chroma prioritizes developer experience over raw performance, making it ideal for prototyping and small-to-medium scale applications.

Key points to remember:

- Designed for simplicity: get started with a few lines of code
- Built-in embedding functions for automatic vector generation
- In-memory by default, with optional persistent storage
- Native integrations with LangChain, LlamaIndex, and other frameworks
- Supports OpenAI, Cohere, HuggingFace, and custom embeddings
- Collections store documents with arbitrary metadata for filtering
- Best for prototyping, development, and smaller-scale deployments
- Compared to Pinecone/Milvus/Qdrant, Chroma trades performance for simplicity
- Python and JavaScript clients available

## Getting Started

### Installation

```bash
# Python
pip install chromadb

# JavaScript
npm install chromadb
```

### Basic Usage

```python
import chromadb

# Create client (in-memory by default)
client = chromadb.Client()

# Create collection
collection = client.create_collection("my_documents")

# Add documents (auto-embedded)
collection.add(
    documents=["Document about AI", "Document about databases"],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["artificial intelligence"],
    n_results=2
)
```

This simplicity is Chroma's core value proposition: semantic search in five lines of code.

## Client Modes

### In-Memory (Ephemeral)

```python
import chromadb

# Data exists only during session
client = chromadb.Client()
```

Best for:
- Prototyping and experimentation
- Unit tests
- Temporary processing

### Persistent Client

```python
import chromadb

# Data persists to disk
client = chromadb.PersistentClient(path="/path/to/storage")
```

Data survives process restarts. Essential for:
- Development with real data
- Production deployments
- Applications requiring durability

### Client-Server Mode

```python
# Start server
# chroma run --host localhost --port 8000

# Connect from client
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
```

Benefits:
- Separate client and server processes
- Multiple clients can connect
- Better resource management

## Collections

### Creating Collections

```python
# Create new collection
collection = client.create_collection(
    name="documents",
    metadata={"description": "My document collection"}
)

# Get existing collection
collection = client.get_collection("documents")

# Get or create
collection = client.get_or_create_collection("documents")

# List all collections
collections = client.list_collections()

# Delete collection
client.delete_collection("documents")
```

### Collection Settings

```python
from chromadb.config import Settings

collection = client.create_collection(
    name="custom_collection",
    embedding_function=my_embedding_function,  # Custom embeddings
    metadata={"hnsw:space": "cosine"}  # Distance metric
)
```

Distance metrics:
- "l2" (default): Euclidean distance
- "cosine": Cosine similarity
- "ip": Inner product

## Embedding Functions

### Default (Sentence Transformers)

```python
# Uses all-MiniLM-L6-v2 by default
collection = client.create_collection("docs")
collection.add(
    documents=["Text to embed"],
    ids=["id1"]
)
```

### OpenAI Embeddings

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="openai_docs",
    embedding_function=openai_ef
)
```

### Cohere Embeddings

```python
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="your-api-key",
    model_name="embed-english-v3.0"
)

collection = client.create_collection(
    name="cohere_docs",
    embedding_function=cohere_ef
)
```

### HuggingFace Embeddings

```python
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="your-api-key",
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### Custom Embeddings

Bring your own embeddings:

```python
collection.add(
    embeddings=[[0.1, 0.2, 0.3, ...]],  # Pre-computed vectors
    documents=["Original text"],
    ids=["id1"]
)
```

Important: Each collection must use a consistent embedding function. Mixing embedding functions between add and query operations produces incorrect results.

## Adding Data

### With Automatic Embedding

```python
collection.add(
    documents=[
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "NLP processes human language"
    ],
    metadatas=[
        {"topic": "ml", "year": 2024},
        {"topic": "dl", "year": 2023},
        {"topic": "nlp", "year": 2024}
    ],
    ids=["doc1", "doc2", "doc3"]
)
```

### With Pre-computed Embeddings

```python
collection.add(
    embeddings=[
        [0.1, 0.2, ...],  # Vector for doc1
        [0.3, 0.4, ...],  # Vector for doc2
    ],
    documents=["Doc 1 text", "Doc 2 text"],
    ids=["doc1", "doc2"]
)
```

### Upserting Data

Update existing or insert new:

```python
collection.upsert(
    documents=["Updated document content"],
    metadatas=[{"updated": True}],
    ids=["doc1"]
)
```

### Updating Data

Modify existing records:

```python
collection.update(
    ids=["doc1"],
    documents=["New content"],
    metadatas=[{"modified": True}]
)
```

## Querying

### Basic Query

```python
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=5
)

# Results structure
# {
#     "ids": [["doc1", "doc2", ...]],
#     "documents": [["text1", "text2", ...]],
#     "metadatas": [[{...}, {...}, ...]],
#     "distances": [[0.1, 0.2, ...]]
# }
```

### Query with Embeddings

```python
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5
)
```

### Include Options

```python
results = collection.query(
    query_texts=["query"],
    n_results=5,
    include=["documents", "metadatas", "distances", "embeddings"]
)
```

## Metadata Filtering

### Where Filters

```python
# Exact match
results = collection.query(
    query_texts=["query"],
    where={"topic": "ml"}
)

# Comparison operators
results = collection.query(
    query_texts=["query"],
    where={"year": {"$gt": 2023}}
)

# Multiple conditions
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"topic": {"$eq": "ml"}},
            {"year": {"$gte": 2023}}
        ]
    }
)
```

### Operators

| Operator | Description |
|----------|-------------|
| $eq | Equal to |
| $ne | Not equal to |
| $gt | Greater than |
| $gte | Greater than or equal |
| $lt | Less than |
| $lte | Less than or equal |
| $in | In list |
| $nin | Not in list |
| $and | Logical AND |
| $or | Logical OR |

### Document Filtering

Filter on document content:

```python
results = collection.query(
    query_texts=["query"],
    where_document={"$contains": "neural network"}
)
```

## Retrieving Data

### Get by ID

```python
results = collection.get(
    ids=["doc1", "doc2"],
    include=["documents", "metadatas"]
)
```

### Get with Filters

```python
results = collection.get(
    where={"topic": "ml"},
    limit=10
)
```

### Get All

```python
# Get all documents (use with caution on large collections)
all_docs = collection.get()
```

## Deleting Data

```python
# Delete by ID
collection.delete(ids=["doc1", "doc2"])

# Delete by filter
collection.delete(where={"topic": "deprecated"})

# Delete all
collection.delete()  # Empties collection
```

## Framework Integrations

### LangChain

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Query
results = vectorstore.similarity_search("query", k=5)
```

### LlamaIndex

```python
from llama_index.vector_stores import ChromaVectorStore
from llama_index import VectorStoreIndex

vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("What is AI?")
```

## Configuration

### HNSW Parameters

```python
collection = client.create_collection(
    name="tuned_collection",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 128,
        "hnsw:search_ef": 64,
        "hnsw:M": 16
    }
)
```

Parameters:
- construction_ef: Build-time search depth
- search_ef: Query-time search depth
- M: Maximum connections per node

### Batch Size

```python
from chromadb.config import Settings

client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    chroma_db_impl="duckdb+parquet"  # Storage backend
))
```

## Limitations

Chroma prioritizes simplicity over performance:

- Not designed for billion-scale datasets
- Limited horizontal scaling options
- Fewer index customization options than Milvus/Qdrant
- No GPU acceleration
- Single distance metric per collection
- Deleted data cannot be recovered

## Comparison with Alternatives

### Chroma vs Pinecone

| Aspect | Chroma | Pinecone |
|--------|--------|----------|
| Hosting | Self-hosted | Managed only |
| Setup | Instant | Account required |
| Scale | Small-medium | Large scale |
| Cost | Free (OSS) | Usage-based |
| Embeddings | Built-in | Bring your own |

### Chroma vs Qdrant

| Aspect | Chroma | Qdrant |
|--------|--------|--------|
| Focus | Simplicity | Performance |
| Language | Python | Rust |
| Filtering | Basic | Advanced |
| Quantization | No | Yes |
| GPU | No | Yes |

### Chroma vs Weaviate

| Aspect | Chroma | Weaviate |
|--------|--------|----------|
| API | Simple Python | GraphQL |
| ML Modules | Basic | Extensive |
| Scale | Prototype | Production |
| Generative | Via frameworks | Built-in |

## When to Use Chroma

Chroma is well-suited for:
- Rapid prototyping and experimentation
- Learning and tutorials
- Small-to-medium datasets (< 1M vectors)
- Projects prioritizing development speed
- LangChain/LlamaIndex applications

Consider alternatives when:
- Production scale is required (Pinecone, Milvus)
- Performance is critical (Qdrant, Milvus)
- Advanced features needed (Weaviate)
- GPU acceleration required (Milvus, Qdrant)
- Billion-scale datasets (Milvus, Pinecone)
