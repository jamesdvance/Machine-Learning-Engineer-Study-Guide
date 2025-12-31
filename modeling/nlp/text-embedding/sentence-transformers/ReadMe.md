# Sentence Transformers

## Summary

Sentence Transformers is a Python framework for computing dense vector representations of sentences and paragraphs using transformer models fine-tuned for semantic similarity. Built on top of Hugging Face Transformers, it provides pre-trained models that encode entire text sequences into fixed-length embeddings, enabling efficient semantic search, clustering, and similarity comparison at the sentence or document level.

Key points to remember:

- Produces fixed-size embeddings for variable-length text (sentences, paragraphs, documents)
- Uses Siamese and triplet network structures for training on similarity tasks
- Pooling strategies: Mean pooling of token embeddings is most common
- Pre-trained models available for many languages and tasks
- Significantly faster than cross-encoders for large-scale similarity search
- Typical embedding dimensions: 384, 768, or 1024
- State-of-the-art on semantic textual similarity (STS) benchmarks
- Supports symmetric (sentence-to-sentence) and asymmetric (query-to-document) search
- Key models: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (quality), multilingual-e5 (multilingual)

## Core Concepts

### From Word to Sentence Embeddings

Word2Vec and GloVe produce word-level embeddings. Averaging word vectors loses word order and semantic nuance:

```
"The dog bit the man" vs "The man bit the dog"
Average word vectors would be identical, but meanings differ completely
```

Sentence Transformers solve this by encoding the full sequence through a transformer, capturing:
- Word order and syntax
- Long-range dependencies
- Contextual meaning

### Architecture

```
Input: "Machine learning is fascinating"
            |
            v
    +----------------+
    | Tokenizer      |  "Machine", "learning", "is", "fascinating"
    +----------------+
            |
            v
    +----------------+
    | Transformer    |  BERT, RoBERTa, MPNet, etc.
    | Encoder        |
    +----------------+
            |
            v
    +----------------+
    | Pooling Layer  |  Mean, Max, or CLS token
    +----------------+
            |
            v
    +----------------+
    | Dense Layer    |  Optional dimensionality reduction
    +----------------+
            |
            v
    768-dim embedding vector
```

### Pooling Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Mean pooling | Average of all token embeddings | Most common, robust |
| Max pooling | Element-wise max across tokens | Captures salient features |
| CLS token | Use [CLS] token embedding | BERT-style, less effective |
| Weighted mean | Attention-weighted average | Emphasizes important tokens |

Mean pooling typically outperforms CLS token pooling for sentence embeddings.

### Training Objectives

Sentence Transformers are fine-tuned using contrastive learning:

**Siamese Networks**: Two encoders share weights, minimize distance for similar pairs, maximize for dissimilar pairs.

```
Sentence A -----> Encoder -----> Embedding A
                                      |
                                      v  Similarity
                                      |
Sentence B -----> Encoder -----> Embedding B
                  (shared)

Loss: Encourage high similarity for paraphrases, low for unrelated
```

**Triplet Networks**: Learn to rank anchor closer to positive than negative.

```
Anchor:   "How to train a neural network"
Positive: "Neural network training tutorial"
Negative: "Best pizza recipes in NYC"

Loss: distance(anchor, positive) < distance(anchor, negative) + margin
```

**Contrastive Loss Functions**:
- Multiple Negatives Ranking Loss: In-batch negatives for efficiency
- Cosine Similarity Loss: Direct optimization of cosine similarity
- Triplet Loss: Margin-based ranking

## Implementation

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "Machine learning is a subset of artificial intelligence",
    "AI systems learn from data patterns",
    "The weather is nice today"
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(embeddings)
print(sim_matrix)
# First two sentences are similar, third is different
```

### Semantic Search

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus to search
corpus = [
    "Python is a programming language",
    "Java is used for enterprise applications",
    "Machine learning uses statistical methods",
    "Deep learning requires neural networks",
    "Natural language processing handles text",
]

# Encode corpus once
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Search query
query = "How do neural networks work?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Find most similar
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)

for hit in hits[0]:
    print(f"{corpus[hit['corpus_id']]} (Score: {hit['score']:.4f})")
```

### Batch Processing for Large Datasets

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

# Large corpus
corpus = [...]  # millions of sentences

# Encode in batches
embeddings = model.encode(
    corpus,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity via dot product
)

# Save embeddings
np.save("corpus_embeddings.npy", embeddings)
```

### Asymmetric Search (Query-Document)

Some models are trained for asymmetric semantic search where queries and documents have different characteristics:

```python
from sentence_transformers import SentenceTransformer

# Models trained for asymmetric search
model = SentenceTransformer('msmarco-distilbert-base-v4')

# Short query
query = "what is machine learning"

# Longer documents
documents = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
    "The weather forecast indicates rain tomorrow with temperatures around 15 degrees Celsius.",
]

query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# Compute similarities
from sentence_transformers import util
similarities = util.cos_sim(query_embedding, doc_embeddings)
print(similarities)  # First doc scores higher
```

### Clustering

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sits on the mat",
    "Dogs are loyal pets",
    "Python is great for ML",
    "JavaScript runs in browsers",
    "Cats and dogs are popular pets",
    "Machine learning uses Python often",
]

embeddings = model.encode(sentences)

# Cluster sentences
num_clusters = 2
clustering = KMeans(n_clusters=num_clusters)
labels = clustering.fit_predict(embeddings)

# Group sentences by cluster
for i in range(num_clusters):
    cluster_sentences = [s for s, l in zip(sentences, labels) if l == i]
    print(f"Cluster {i}: {cluster_sentences}")
```

## Model Selection

### Popular Pre-trained Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose, production |
| all-mpnet-base-v2 | 768 | Medium | Best | Quality-critical applications |
| multi-qa-MiniLM-L6-cos-v1 | 384 | Fast | Good | Question-answering |
| msmarco-distilbert-base-v4 | 768 | Medium | Good | Asymmetric search |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Fast | Good | Multilingual |
| all-MiniLM-L12-v2 | 384 | Medium | Better | Balanced speed/quality |

### Multilingual Models

```python
# For multilingual text
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Supports 50+ languages
sentences = [
    "How are you?",           # English
    "Comment allez-vous?",    # French
    "Wie geht es Ihnen?",     # German
]

embeddings = model.encode(sentences)
# Semantically similar sentences cluster together regardless of language
```

### Domain-Specific Models

```python
# Scientific papers
model = SentenceTransformer('allenai-specter')

# Legal documents
model = SentenceTransformer('legal-bert-base-uncased')

# Code search
model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')
```

## Fine-tuning

### Training on Custom Data

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=['Query 1', 'Relevant document 1'], label=1.0),
    InputExample(texts=['Query 1', 'Irrelevant document'], label=0.0),
    InputExample(texts=['Query 2', 'Relevant document 2'], label=1.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Choose loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./fine-tuned-model'
)
```

### Contrastive Learning with Multiple Negatives

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# Pairs of similar sentences (positives)
train_examples = [
    InputExample(texts=['How do I reset my password?', 'Password reset instructions']),
    InputExample(texts=['What are your business hours?', 'Store opening times']),
    InputExample(texts=['Return policy', 'How to return items']),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Multiple Negatives Ranking Loss uses in-batch negatives
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
)
```

### Triplet Training

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# Triplets: (anchor, positive, negative)
train_examples = [
    InputExample(texts=[
        'What is the capital of France?',
        'Paris is the capital of France',
        'The Eiffel Tower is in Paris'
    ]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
)
```

## Performance Optimization

### Faster Inference

```python
from sentence_transformers import SentenceTransformer

# Use smaller model for speed
model = SentenceTransformer('all-MiniLM-L6-v2')

# Enable GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# Batch encoding
embeddings = model.encode(
    sentences,
    batch_size=128,  # Larger batches on GPU
    show_progress_bar=True
)

# Normalize for faster similarity (dot product = cosine)
embeddings = model.encode(sentences, normalize_embeddings=True)
```

### Quantization

```python
# ONNX export for faster CPU inference
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Export to ONNX
model.save('model_onnx', create_onnx=True)
```

### Dimensionality Reduction

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer('all-mpnet-base-v2')  # 768d

# Encode corpus
embeddings = model.encode(corpus)  # (n, 768)

# Reduce dimensions
pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(embeddings)  # (n, 256)

# Explained variance
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")
```

## Comparison with Other Approaches

### Sentence Transformers vs Cross-Encoders

| Aspect | Sentence Transformers | Cross-Encoders |
|--------|----------------------|----------------|
| Architecture | Bi-encoder (independent) | Cross-encoder (joint) |
| Speed | Fast (encode once, compare many) | Slow (encode pairs together) |
| Quality | Good | Better for ranking |
| Use case | Large-scale retrieval | Re-ranking top results |

```python
from sentence_transformers import CrossEncoder

# Cross-encoder for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score pairs directly (slower but more accurate)
pairs = [
    ['Query', 'Document 1'],
    ['Query', 'Document 2'],
]
scores = cross_encoder.predict(pairs)
```

Typical pipeline: Use Sentence Transformers for fast retrieval, then re-rank top-k with Cross-Encoder.

### vs Word Embeddings (Word2Vec, GloVe)

| Aspect | Word Embeddings | Sentence Transformers |
|--------|-----------------|----------------------|
| Granularity | Word-level | Sentence/paragraph-level |
| Context | Static | Contextual |
| Word order | Lost in averaging | Preserved |
| Semantic similarity | Weak for sentences | Strong |
| Training data | Unsupervised | Supervised on NLI/STS |

### vs BERT [CLS] Token

| Aspect | BERT [CLS] | Sentence Transformers |
|--------|------------|----------------------|
| Training | MLM/NSP | Similarity objectives |
| Similarity performance | Poor | Excellent |
| Direct usability | Requires fine-tuning | Works out-of-box |

Raw BERT [CLS] embeddings are not suitable for semantic similarity without fine-tuning.

## Best Practices

### Model Selection Guidelines

1. Start with `all-MiniLM-L6-v2` for general English tasks
2. Use `all-mpnet-base-v2` when quality is paramount
3. Choose multilingual models for non-English or mixed-language text
4. Select domain-specific models when available (scientific, legal, etc.)
5. Consider asymmetric models for query-document retrieval

### Embedding Storage and Retrieval

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode and normalize
embeddings = model.encode(corpus, normalize_embeddings=True)

# Save embeddings
np.save('embeddings.npy', embeddings)

# For production: use vector databases
# - FAISS for millions of vectors
# - Pinecone, Milvus, Weaviate for managed solutions
```

### Handling Long Documents

```python
def encode_long_document(model, text, max_length=512, stride=256):
    """Encode long document by chunking and averaging."""
    # Tokenize to find length
    tokens = model.tokenizer.tokenize(text)

    if len(tokens) <= max_length:
        return model.encode(text)

    # Split into overlapping chunks
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = model.tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)

    # Encode chunks and average
    chunk_embeddings = model.encode(chunks)
    return np.mean(chunk_embeddings, axis=0)
```

### Caching Embeddings

```python
import hashlib
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingCache:
    def __init__(self, model_name, cache_dir='./embedding_cache'):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, text):
        hash_key = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.npy")

    def encode(self, text):
        cache_path = self._get_cache_path(text)

        if os.path.exists(cache_path):
            return np.load(cache_path)

        embedding = self.model.encode(text)
        np.save(cache_path, embedding)
        return embedding
```

## Integration with Vector Databases

### FAISS

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode corpus
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
dimension = corpus_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized
index.add(corpus_embeddings.astype('float32'))

# Search
query_embedding = model.encode(query, normalize_embeddings=True)
scores, indices = index.search(
    query_embedding.reshape(1, -1).astype('float32'),
    k=5
)
```

### Pinecone

```python
import pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key='your-api-key')
index = pinecone.Index('semantic-search')

# Upsert embeddings
embeddings = model.encode(documents)
vectors = [
    (f"doc_{i}", emb.tolist(), {"text": doc})
    for i, (emb, doc) in enumerate(zip(embeddings, documents))
]
index.upsert(vectors)

# Query
query_embedding = model.encode(query).tolist()
results = index.query(query_embedding, top_k=5, include_metadata=True)
```

## Evaluation

### Semantic Textual Similarity (STS)

```python
from sentence_transformers import SentenceTransformer, evaluation

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load STS benchmark
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    test_examples,
    name='sts-test'
)

# Evaluate (returns Spearman correlation)
score = evaluator(model)
print(f"STS Score: {score:.4f}")
```

### Information Retrieval Metrics

```python
from sentence_transformers import evaluation

# Evaluate retrieval performance
evaluator = evaluation.InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,  # Dict mapping query_id to doc_ids
    name='retrieval-test'
)

metrics = evaluator(model)
# Returns MRR, nDCG, Recall@k, Precision@k
```
