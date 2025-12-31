# OpenAI Embeddings

## Summary

OpenAI Embeddings are API-accessible text embedding models that convert text into dense vector representations. These proprietary models offer state-of-the-art performance on semantic similarity and search tasks without requiring local infrastructure. The text-embedding-3 family (launched 2024) provides flexible dimensionality options and improved performance over the previous ada-002 model.

Key points to remember:

- API-based: No model hosting required, pay-per-token pricing
- text-embedding-3-large: 3072 dimensions (or configurable), best quality
- text-embedding-3-small: 1536 dimensions, cost-effective with strong performance
- text-embedding-ada-002: Legacy model, 1536 dimensions, still widely used
- Matryoshka Representation Learning: Embeddings can be truncated to smaller dimensions with minimal quality loss
- Context length: Up to 8191 tokens for text-embedding-3 models
- Normalized embeddings: Cosine similarity equals dot product
- Best for: Semantic search, clustering, classification, RAG applications
- Limitations: Proprietary, requires internet, data sent to OpenAI servers

## Models Overview

### Current Models

| Model | Dimensions | Max Tokens | Performance | Cost (per 1M tokens) |
|-------|------------|------------|-------------|---------------------|
| text-embedding-3-large | 3072 (default) | 8191 | Best | Higher |
| text-embedding-3-small | 1536 | 8191 | Good | Lower |
| text-embedding-ada-002 | 1536 | 8191 | Good | Medium |

### Dimension Flexibility

The text-embedding-3 models support Matryoshka Representation Learning, allowing dimension reduction without retraining:

```
text-embedding-3-large: 3072, 1536, 1024, 512, 256 dimensions
text-embedding-3-small: 1536, 1024, 512, 256 dimensions
```

Lower dimensions trade some accuracy for reduced storage and faster similarity computation.

## Implementation

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env variable

# Single text embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Machine learning is a subset of artificial intelligence"
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 1536
```

### Batch Embedding

```python
from openai import OpenAI

client = OpenAI()

texts = [
    "The cat sat on the mat",
    "Dogs are loyal companions",
    "Machine learning uses statistical methods",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [item.embedding for item in response.data]
print(f"Embedded {len(embeddings)} texts")
```

### Custom Dimensions

```python
from openai import OpenAI

client = OpenAI()

# Reduce dimensions for storage efficiency
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Text to embed",
    dimensions=256  # Truncate from 3072 to 256
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 256
```

### Semantic Search

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Corpus
documents = [
    "Python is a programming language known for its simplicity",
    "JavaScript is widely used for web development",
    "Machine learning models learn patterns from data",
    "Neural networks are inspired by biological neurons",
]

# Embed corpus
doc_embeddings = get_embeddings(documents)

# Query
query = "How do neural networks work?"
query_embedding = get_embeddings([query])[0]

# Find most similar
similarities = [
    cosine_similarity(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]

# Rank results
ranked = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)

for doc, score in ranked:
    print(f"{score:.4f}: {doc}")
```

### Async Batch Processing

```python
import asyncio
from openai import AsyncOpenAI

async def embed_batch(texts, model="text-embedding-3-small"):
    client = AsyncOpenAI()
    response = await client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]

async def embed_large_corpus(corpus, batch_size=100):
    """Embed large corpus in batches asynchronously."""
    all_embeddings = []

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        embeddings = await embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"Processed {min(i + batch_size, len(corpus))}/{len(corpus)}")

    return all_embeddings

# Run
corpus = ["doc " + str(i) for i in range(1000)]
embeddings = asyncio.run(embed_large_corpus(corpus))
```

### Rate Limiting and Retries

```python
import time
from openai import OpenAI, RateLimitError
from tenacity import retry, wait_exponential, retry_if_exception_type

client = OpenAI()

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def get_embedding_with_retry(text, model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def embed_with_rate_limit(texts, model="text-embedding-3-small",
                          batch_size=100, delay=0.1):
    """Embed texts with rate limiting."""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

        # Rate limit delay
        time.sleep(delay)

    return embeddings
```

## Cost Optimization

### Dimension Reduction Trade-offs

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Compare quality at different dimensions
dimensions_to_test = [256, 512, 1024, 3072]

texts = [
    "What is machine learning?",
    "Machine learning is a type of artificial intelligence",
    "The weather is nice today",
]

for dim in dimensions_to_test:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=dim
    )

    embeddings = [np.array(item.embedding) for item in response.data]

    # Similarity between first two (related) vs first and third (unrelated)
    sim_related = np.dot(embeddings[0], embeddings[1])
    sim_unrelated = np.dot(embeddings[0], embeddings[2])
    separation = sim_related - sim_unrelated

    print(f"Dim {dim}: Related={sim_related:.4f}, "
          f"Unrelated={sim_unrelated:.4f}, Gap={separation:.4f}")
```

### Token Estimation

```python
import tiktoken

def estimate_embedding_cost(texts, model="text-embedding-3-small"):
    """Estimate cost before making API calls."""
    # Pricing per 1M tokens (check current pricing)
    pricing = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }

    # Get tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")

    # Count tokens
    total_tokens = sum(len(encoding.encode(text)) for text in texts)

    cost = (total_tokens / 1_000_000) * pricing[model]

    return {
        "total_tokens": total_tokens,
        "estimated_cost": cost,
        "model": model
    }

# Estimate before embedding
texts = ["Document " + str(i) * 100 for i in range(1000)]
estimate = estimate_embedding_cost(texts)
print(f"Tokens: {estimate['total_tokens']:,}")
print(f"Estimated cost: ${estimate['estimated_cost']:.4f}")
```

### Caching Embeddings

```python
import hashlib
import json
import os
from openai import OpenAI

class EmbeddingCache:
    def __init__(self, cache_file="embedding_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.client = OpenAI()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def _get_key(self, text, model, dimensions):
        content = f"{model}:{dimensions}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_embedding(self, text, model="text-embedding-3-small",
                      dimensions=None):
        key = self._get_key(text, model, dimensions)

        if key in self.cache:
            return self.cache[key]

        kwargs = {"model": model, "input": text}
        if dimensions:
            kwargs["dimensions"] = dimensions

        response = self.client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding

        self.cache[key] = embedding
        self._save_cache()

        return embedding

# Usage
cache = EmbeddingCache()
emb = cache.get_embedding("Same text queried multiple times")
```

## Integration Patterns

### With LangChain

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Optional dimension reduction
)

# Create vector store
documents = ["Document 1 content", "Document 2 content"]
vectorstore = FAISS.from_texts(documents, embeddings)

# Search
results = vectorstore.similarity_search("search query", k=5)
```

### With Pinecone

```python
from openai import OpenAI
import pinecone

openai_client = OpenAI()

# Initialize Pinecone
pinecone.init(api_key="your-api-key")
index = pinecone.Index("embeddings-index")

# Embed and upsert
def embed_and_upsert(documents, ids):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=documents
    )

    vectors = [
        {
            "id": doc_id,
            "values": item.embedding,
            "metadata": {"text": doc}
        }
        for doc_id, doc, item in zip(ids, documents, response.data)
    ]

    index.upsert(vectors)

# Query
def search(query, top_k=5):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )

    results = index.query(
        vector=response.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results
```

### RAG Pipeline

```python
from openai import OpenAI
import numpy as np

class SimpleRAG:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.embedding_model = model
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents):
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=documents
        )

        self.documents.extend(documents)
        self.embeddings.extend([item.embedding for item in response.data])

    def retrieve(self, query, top_k=3):
        query_response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = query_response.data[0].embedding

        # Compute similarities
        similarities = [
            np.dot(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]

    def query(self, question, top_k=3):
        # Retrieve relevant context
        context = self.retrieve(question, top_k)
        context_str = "\n".join(context)

        # Generate answer
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {question}"
                }
            ]
        )

        return response.choices[0].message.content

# Usage
rag = SimpleRAG()
rag.add_documents([
    "Python was created by Guido van Rossum",
    "Machine learning uses statistical methods",
    "Neural networks are inspired by the brain",
])

answer = rag.query("Who created Python?")
```

## Comparison with Other Embedding Options

### OpenAI vs Open-Source

| Aspect | OpenAI Embeddings | Sentence Transformers |
|--------|-------------------|----------------------|
| Hosting | API (managed) | Self-hosted |
| Cost model | Per-token | Infrastructure |
| Latency | Network + processing | Local processing |
| Data privacy | Data sent to OpenAI | Data stays local |
| Customization | None | Fine-tuning possible |
| Offline | No | Yes |
| Quality | Excellent | Good to Excellent |

### When to Use OpenAI Embeddings

Choose OpenAI when:
- Rapid development without ML infrastructure
- Budget allows per-token pricing
- Data privacy policies permit external API calls
- Need consistently high-quality embeddings
- Embedding volume is moderate (not billions of texts)

Choose alternatives when:
- Data cannot leave your infrastructure
- Need to fine-tune for domain-specific tasks
- Very high volume makes API costs prohibitive
- Offline capability is required
- Latency requirements are strict

### Performance Comparison

On MTEB (Massive Text Embedding Benchmark):

| Model | Avg Score | Dimensions |
|-------|-----------|------------|
| text-embedding-3-large | ~64 | 3072 |
| text-embedding-3-small | ~62 | 1536 |
| all-mpnet-base-v2 | ~58 | 768 |
| text-embedding-ada-002 | ~61 | 1536 |

Scores vary by task type (retrieval, clustering, classification, etc.).

## Best Practices

### Text Preprocessing

```python
import re

def preprocess_for_embedding(text, max_tokens=8000):
    """Preprocess text before embedding."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove very long repeated patterns
    text = re.sub(r'(.{10,}?)\1{3,}', r'\1', text)

    # Truncate if too long (rough estimate: 4 chars per token)
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        text = text[:max_chars]

    return text
```

### Handling Long Documents

```python
import tiktoken
from openai import OpenAI

client = OpenAI()

def embed_long_document(text, model="text-embedding-3-small",
                        max_tokens=8000, overlap=200):
    """Embed long document by chunking."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    # Split into overlapping chunks
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap

    # Embed chunks
    response = client.embeddings.create(model=model, input=chunks)
    chunk_embeddings = [item.embedding for item in response.data]

    # Average embeddings (weighted by chunk length)
    import numpy as np
    weights = [len(c) for c in chunks]
    weighted_avg = np.average(chunk_embeddings, axis=0, weights=weights)

    # Normalize
    return (weighted_avg / np.linalg.norm(weighted_avg)).tolist()
```

### Error Handling

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import time

client = OpenAI()

def robust_embed(texts, model="text-embedding-3-small", max_retries=3):
    """Embed with robust error handling."""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIConnectionError:
            wait_time = 2 ** attempt
            print(f"Connection error. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except APIError as e:
            if e.status_code >= 500:
                wait_time = 2 ** attempt
                print(f"Server error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise Exception(f"Failed after {max_retries} attempts")
```

### Monitoring Usage

```python
from openai import OpenAI

client = OpenAI()

def embed_with_logging(texts, model="text-embedding-3-small"):
    """Embed with usage logging."""
    response = client.embeddings.create(model=model, input=texts)

    # Log usage
    usage = response.usage
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    return [item.embedding for item in response.data]
```

## Security Considerations

### API Key Management

```python
import os
from openai import OpenAI

# Load from environment (recommended)
client = OpenAI()  # Uses OPENAI_API_KEY

# Or explicitly (not for production code)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Never hardcode keys
# BAD: client = OpenAI(api_key="sk-...")
```

### Data Privacy

Consider before using:
- Text is sent to OpenAI servers for processing
- Check OpenAI data usage policies
- Consider PII/sensitive data implications
- Enterprise customers can request data processing agreements
- For sensitive data, consider self-hosted alternatives

### Input Validation

```python
def validate_embedding_input(text, max_length=100000):
    """Validate input before sending to API."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if len(text) == 0:
        raise ValueError("Input cannot be empty")

    if len(text) > max_length:
        raise ValueError(f"Input exceeds {max_length} characters")

    # Remove potential injection attempts
    text = text.replace("\x00", "")

    return text
```
