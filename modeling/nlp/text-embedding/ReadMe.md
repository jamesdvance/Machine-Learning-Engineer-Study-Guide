# Text Embedding

## Summary

Text embeddings are dense vector representations of text that capture semantic meaning in a continuous vector space. They transform words, sentences, or documents into fixed-length numerical vectors where similar meanings map to nearby points. Text embeddings are foundational to modern NLP, enabling semantic search, clustering, classification, and retrieval-augmented generation (RAG).

Key points to remember:

- Word embeddings (Word2Vec, GloVe): Map individual words to vectors, one vector per word
- Sentence embeddings (Sentence Transformers): Map entire sequences to single vectors
- API embeddings (OpenAI): Managed services for high-quality embeddings without infrastructure
- Static vs contextual: Word2Vec/GloVe produce fixed vectors; transformer-based methods are context-aware
- Similarity metrics: Cosine similarity is standard, dot product for normalized vectors
- Dimensionality: Typically 100-1536 dimensions depending on method and quality needs
- Choice depends on: granularity needed, infrastructure constraints, quality requirements, domain specificity

## Embedding Types

### Word-Level Embeddings

Word embeddings assign one vector per word in the vocabulary.

```
"king"   -> [0.2, -0.1, 0.8, ...]
"queen"  -> [0.3, -0.2, 0.7, ...]
"apple"  -> [-0.5, 0.4, 0.1, ...]
```

Methods: Word2Vec, GloVe, FastText

Limitations:
- No disambiguation: "bank" (financial) and "bank" (river) share one vector
- No OOV handling in Word2Vec/GloVe (FastText uses subwords)
- Sentence meaning requires aggregation (averaging loses word order)

### Sentence/Document Embeddings

Encode entire text sequences into single vectors, preserving word order and context.

```
"The cat sat on the mat"      -> [0.1, 0.2, -0.3, ...]
"A feline rested on the rug"  -> [0.1, 0.2, -0.3, ...]  # Similar meaning, similar vector
```

Methods: Sentence Transformers, OpenAI Embeddings, Cohere, Voyage

### Contextual Embeddings

Produce different vectors for the same word based on surrounding context.

```
"I deposited money in the bank"  -> "bank" = [financial vector]
"I sat by the river bank"        -> "bank" = [nature vector]
```

Methods: BERT, ELMo (for word-level); Sentence Transformers (for sentence-level)

## Method Comparison

### Overview

| Method | Level | Context | OOV | Self-host | Quality | Cost |
|--------|-------|---------|-----|-----------|---------|------|
| Word2Vec | Word | Static | None | Yes | Good | Free |
| GloVe | Word | Static | None | Yes | Good | Free |
| FastText | Word | Static | Subword | Yes | Good | Free |
| Sentence Transformers | Sentence | Dynamic | Subword | Yes | Excellent | Free |
| OpenAI Embeddings | Sentence | Dynamic | Subword | No | Excellent | Per-token |

### Training Approach

| Method | Training Objective |
|--------|-------------------|
| Word2Vec (Skip-gram) | Predict context words from center word |
| Word2Vec (CBOW) | Predict center word from context |
| GloVe | Factorize co-occurrence matrix |
| Sentence Transformers | Contrastive learning on sentence pairs |
| OpenAI | Proprietary (likely contrastive + large scale) |

### When to Use Each

**Word2Vec / GloVe**
- Need word-level embeddings for downstream models
- Memory and compute constraints
- Well-studied domain with available pre-trained vectors
- Input to traditional ML pipelines (feature engineering)

**Sentence Transformers**
- Semantic search and similarity at sentence level
- Need to self-host for privacy or cost reasons
- Want to fine-tune on domain-specific data
- Offline operation required

**OpenAI Embeddings**
- Rapid prototyping without infrastructure
- Best quality with minimal effort
- Moderate volume (API costs manageable)
- No data privacy restrictions

## Practical Considerations

### Dimensionality Selection

| Dimensions | Trade-off |
|------------|-----------|
| 50-100 | Fast, small storage, less semantic nuance |
| 200-384 | Balanced for most applications |
| 512-768 | High quality, standard for modern models |
| 1024-3072 | Maximum quality, higher storage/compute |

Higher dimensions capture more semantic nuance but have diminishing returns beyond 512-768 for most tasks.

### Similarity Computation

Cosine similarity is standard:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

For normalized vectors, dot product equals cosine similarity:

```python
# Normalize once
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)

# Then dot product = cosine
similarity = np.dot(a_norm, b_norm)
```

Use normalized embeddings with dot product for faster computation in production.

### Storage and Retrieval

Embedding storage scales with corpus size:

```
Storage = num_documents x dimensions x 4 bytes (float32)

1M documents x 384 dimensions = 1.5 GB
1M documents x 1536 dimensions = 6 GB
```

For large corpora, use:
- Vector databases: Pinecone, Milvus, Weaviate, Qdrant, Chroma
- Approximate nearest neighbor: FAISS, ScaNN, Annoy
- Dimensionality reduction: PCA, random projection

### Handling Out-of-Vocabulary

| Method | OOV Handling |
|--------|--------------|
| Word2Vec | None (error or unknown token) |
| GloVe | None (error or unknown token) |
| FastText | Subword composition |
| Sentence Transformers | Subword tokenization (WordPiece/BPE) |
| OpenAI | Subword tokenization |

For Word2Vec/GloVe, common strategies:
- Map to lowercase variant
- Map to [UNK] token embedding
- Use character n-gram approximation
- Fall back to zero vector (not recommended)

## Evaluation

### Intrinsic Evaluation

Measure embedding quality directly:

**Word Similarity**: Correlation with human similarity judgments
- WordSim-353
- SimLex-999

**Analogies**: Accuracy on "a is to b as c is to d" tasks
- Google Analogy Dataset
- BATS (Bigger Analogy Test Set)

**Semantic Textual Similarity (STS)**: Correlation with human sentence similarity scores
- STS Benchmark

### Extrinsic Evaluation

Measure performance on downstream tasks:
- Text classification accuracy
- Named entity recognition F1
- Question answering accuracy
- Retrieval precision/recall

### MTEB Benchmark

The Massive Text Embedding Benchmark evaluates across multiple task types:
- Retrieval
- Clustering
- Classification
- Pair Classification
- Reranking
- STS
- Summarization

Use MTEB leaderboard to compare embedding models: https://huggingface.co/spaces/mteb/leaderboard

## Common Pipelines

### Semantic Search

```python
# 1. Embed corpus offline
corpus_embeddings = model.encode(corpus)
save(corpus_embeddings)

# 2. At query time
query_embedding = model.encode(query)
similarities = cosine_similarity(query_embedding, corpus_embeddings)
top_k = np.argsort(similarities)[-k:][::-1]
results = [corpus[i] for i in top_k]
```

### Clustering

```python
from sklearn.cluster import KMeans

embeddings = model.encode(documents)
clusters = KMeans(n_clusters=10).fit_predict(embeddings)
```

### Classification

```python
from sklearn.linear_model import LogisticRegression

# Embed training data
X_train = model.encode(train_texts)
y_train = train_labels

# Train classifier on embeddings
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on new data
X_test = model.encode(test_texts)
predictions = classifier.predict(X_test)
```

### RAG (Retrieval-Augmented Generation)

```python
# Index documents
doc_embeddings = embed(documents)
index.add(doc_embeddings)

# At query time
query_embedding = embed(query)
relevant_docs = index.search(query_embedding, k=5)
context = "\n".join(relevant_docs)

# Generate with context
response = llm.generate(f"Context: {context}\n\nQuestion: {query}")
```

## Best Practices

### Preprocessing

```python
def preprocess(text):
    # Lowercase (for word embeddings, not always for sentence)
    text = text.lower()

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Handle encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    return text
```

### Batching

Always batch embed for efficiency:

```python
# Good: batch encoding
embeddings = model.encode(texts, batch_size=32)

# Bad: one at a time
embeddings = [model.encode(t) for t in texts]  # Much slower
```

### Caching

Cache embeddings for frequently accessed or static content:

```python
import hashlib
import pickle

def get_cached_embedding(text, cache_path="embeddings_cache.pkl"):
    key = hashlib.md5(text.encode()).hexdigest()

    cache = load_cache(cache_path)
    if key in cache:
        return cache[key]

    embedding = model.encode(text)
    cache[key] = embedding
    save_cache(cache, cache_path)

    return embedding
```

### Normalization

Normalize embeddings for consistent similarity computation:

```python
from sklearn.preprocessing import normalize

# Normalize for cosine similarity via dot product
embeddings = normalize(embeddings, norm='l2')
```

## Migration Between Methods

### Word Embeddings to Sentence Embeddings

When migrating from averaging word embeddings to sentence transformers:

- Quality improves significantly (word order preserved)
- Storage may increase (sentence vs word vectors)
- Update similarity computation (likely no changes if using cosine)
- Reindex entire corpus

### Self-hosted to API (or vice versa)

When switching between Sentence Transformers and OpenAI:

- Dimension changes may require reindexing
- API adds latency and cost but removes infrastructure
- Quality differences may affect downstream tasks
- Test thoroughly on representative samples before full migration

### Reducing Dimensions

To reduce storage or computation:

```python
from sklearn.decomposition import PCA

# Fit PCA on corpus embeddings
pca = PCA(n_components=256)
reduced = pca.fit_transform(embeddings)

# For new embeddings, use same PCA
new_reduced = pca.transform(new_embeddings)

# Check variance retained
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")
```

Alternatively, OpenAI text-embedding-3 models support native dimension reduction via the API.

## Choosing an Embedding Method

Decision tree:

1. **Need word-level embeddings for feature engineering?**
   - Yes: Word2Vec or GloVe (or FastText for OOV handling)

2. **Can data leave your infrastructure?**
   - No: Sentence Transformers (self-hosted)
   - Yes: Consider OpenAI or Sentence Transformers

3. **Need to fine-tune for domain-specific task?**
   - Yes: Sentence Transformers (supports fine-tuning)
   - No: Either works

4. **Prioritizing quality over cost/infrastructure?**
   - Quality: OpenAI Embeddings
   - Cost/control: Sentence Transformers

5. **Working with non-English text?**
   - Use multilingual models (multilingual Sentence Transformers or OpenAI)

6. **Need offline capability?**
   - Yes: Self-hosted (Sentence Transformers, Word2Vec, GloVe)
   - No: API options available
