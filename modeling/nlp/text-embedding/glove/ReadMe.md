# GloVe

## Summary

GloVe (Global Vectors for Word Representation) is a word embedding method that combines the strengths of count-based and prediction-based approaches. Developed at Stanford in 2014, GloVe constructs word vectors by factorizing the global word-word co-occurrence matrix, capturing statistical information about how frequently words appear together across an entire corpus. Unlike Word2Vec which learns from local context windows, GloVe explicitly leverages global corpus statistics.

Key points to remember:

- Count-based method: Builds on word co-occurrence statistics from the entire corpus
- Weighted least squares objective: Optimizes embeddings to reflect log co-occurrence ratios
- Global context: Uses corpus-wide statistics rather than local sliding windows
- Training is deterministic given the co-occurrence matrix (unlike stochastic Word2Vec)
- Pre-trained vectors available in 50, 100, 200, and 300 dimensions
- Common corpora: Wikipedia + Gigaword (6B tokens), Common Crawl (42B and 840B tokens)
- Performs comparably to Word2Vec on analogy and similarity benchmarks
- Same limitations as Word2Vec: static embeddings, no OOV handling, single vector per word

## Core Concepts

### Co-occurrence Matrix

GloVe begins by constructing a word-word co-occurrence matrix X, where X_ij counts how often word j appears in the context of word i within a specified window.

```
Corpus: "the cat sat on the mat"
Window size: 1

        the  cat  sat  on  mat
the      0    1    0   1    1
cat      1    0    1   0    0
sat      0    1    0   1    0
on       1    0    1   0    1
mat      1    0    0   1    0
```

Key statistics derived from the matrix:
- X_i = sum of row i (total co-occurrences for word i)
- P_ij = X_ij / X_i (probability that word j appears in context of word i)

### Ratio of Co-occurrence Probabilities

The insight behind GloVe is that ratios of co-occurrence probabilities encode meaning:

```
Example: "ice" and "steam" with probe words

P(solid | ice) / P(solid | steam) = high (solid relates to ice)
P(gas | ice) / P(gas | steam) = low (gas relates to steam)
P(water | ice) / P(water | steam) = 1 (both relate to water equally)
P(fashion | ice) / P(fashion | steam) = 1 (neither relates to fashion)
```

These ratios distinguish relevant relationships from noise.

### Objective Function

GloVe aims to learn word vectors w and context vectors w_tilde such that their dot product approximates the log co-occurrence count:

```
w_i . w_tilde_j + b_i + b_tilde_j = log(X_ij)
```

The weighted least squares objective:

```
J = sum_{i,j=1}^{V} f(X_ij) * (w_i . w_tilde_j + b_i + b_tilde_j - log(X_ij))^2
```

### Weighting Function

The weighting function f(x) prevents common word pairs from dominating:

```
f(x) = (x / x_max)^alpha  if x < x_max
f(x) = 1                   if x >= x_max

Typical values: x_max = 100, alpha = 0.75
```

This gives less weight to very frequent co-occurrences while still considering them.

## Comparison with Word2Vec

| Aspect | GloVe | Word2Vec |
|--------|-------|----------|
| Approach | Count-based (co-occurrence matrix) | Prediction-based (context windows) |
| Training data | Global statistics | Local context windows |
| Optimization | Weighted least squares | Stochastic gradient descent |
| Reproducibility | Deterministic | Stochastic |
| Memory | Needs full co-occurrence matrix | Streams through corpus |
| Parallelization | Matrix operations parallelize well | Skip-gram is inherently sequential |
| Performance | Similar on benchmarks | Similar on benchmarks |

In practice, both methods produce comparable results. GloVe can be faster to train once the co-occurrence matrix is built, but building the matrix requires significant memory for large vocabularies.

## Implementation

### Using Pre-trained GloVe Vectors

```python
import numpy as np

def load_glove_vectors(filepath, embedding_dim=100):
    """Load GloVe vectors from text file."""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load vectors
glove = load_glove_vectors("glove.6B.100d.txt", embedding_dim=100)

# Access word vectors
vector = glove.get("machine")

# Compute similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim = cosine_similarity(glove["king"], glove["queen"])
```

### Building an Embedding Matrix for Neural Networks

```python
import torch
import torch.nn as nn
import numpy as np

def create_embedding_matrix(word_to_idx, glove_vectors, embedding_dim):
    """Create embedding matrix from GloVe vectors."""
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    for word, idx in word_to_idx.items():
        vector = glove_vectors.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found += 1
        else:
            # Initialize OOV words randomly
            embedding_matrix[idx] = np.random.normal(
                scale=0.6, size=(embedding_dim,)
            )

    print(f"Found {found}/{vocab_size} words in GloVe")
    return embedding_matrix

# Create PyTorch embedding layer
embedding_matrix = create_embedding_matrix(word_to_idx, glove, 100)
embedding_layer = nn.Embedding.from_pretrained(
    torch.FloatTensor(embedding_matrix),
    freeze=False  # Allow fine-tuning
)
```

### Using Gensim

```python
import gensim.downloader as api

# Download pre-trained GloVe vectors
glove_vectors = api.load("glove-wiki-gigaword-100")  # 100d
glove_vectors = api.load("glove-wiki-gigaword-200")  # 200d
glove_vectors = api.load("glove-wiki-gigaword-300")  # 300d

# Use like Word2Vec
similar = glove_vectors.most_similar("python", topn=5)
similarity = glove_vectors.similarity("cat", "dog")

# Analogies
result = glove_vectors.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
```

### Training GloVe from Scratch

Training GloVe requires building the co-occurrence matrix first:

```python
from collections import defaultdict
import numpy as np
from scipy.sparse import lil_matrix

def build_cooccurrence_matrix(corpus, vocab, window_size=10):
    """Build sparse co-occurrence matrix from corpus."""
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Use sparse matrix for memory efficiency
    cooccur = lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    for sentence in corpus:
        tokens = [w for w in sentence if w in word_to_idx]
        for i, word in enumerate(tokens):
            word_idx = word_to_idx[word]

            # Context window with distance weighting
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word = tokens[j]
                    context_idx = word_to_idx[context_word]
                    distance = abs(i - j)
                    # Weight by inverse distance
                    cooccur[word_idx, context_idx] += 1.0 / distance

    return cooccur.tocsr(), word_to_idx

def train_glove(cooccur_matrix, embedding_dim=100, epochs=50,
                learning_rate=0.05, x_max=100, alpha=0.75):
    """Train GloVe embeddings."""
    vocab_size = cooccur_matrix.shape[0]

    # Initialize embeddings and biases
    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.01
    b = np.zeros(vocab_size)
    b_tilde = np.zeros(vocab_size)

    # Get nonzero entries
    rows, cols = cooccur_matrix.nonzero()
    values = np.array(cooccur_matrix[rows, cols]).flatten()

    # Weighting function
    def weight(x):
        return np.minimum((x / x_max) ** alpha, 1.0)

    weights = weight(values)
    log_values = np.log(values + 1)

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle training pairs
        indices = np.random.permutation(len(rows))

        for idx in indices:
            i, j = rows[idx], cols[idx]
            x_ij = values[idx]
            f_ij = weights[idx]
            log_x_ij = log_values[idx]

            # Compute difference
            diff = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - log_x_ij

            # Weighted squared error
            loss = f_ij * diff ** 2
            total_loss += loss

            # Gradients
            grad_common = f_ij * diff

            # Update embeddings
            W[i] -= learning_rate * grad_common * W_tilde[j]
            W_tilde[j] -= learning_rate * grad_common * W[i]
            b[i] -= learning_rate * grad_common
            b_tilde[j] -= learning_rate * grad_common

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # Final embeddings: average of W and W_tilde
    embeddings = W + W_tilde
    return embeddings
```

## Pre-trained GloVe Vectors

Stanford provides pre-trained vectors trained on different corpora:

| Corpus | Tokens | Vocab Size | Dimensions | File Size |
|--------|--------|------------|------------|-----------|
| Wikipedia 2014 + Gigaword 5 | 6B | 400K | 50, 100, 200, 300 | 822MB |
| Common Crawl | 42B | 1.9M | 300 | 5.0GB |
| Common Crawl | 840B | 2.2M | 300 | 5.6GB |
| Twitter | 27B | 1.2M | 25, 50, 100, 200 | 1.4GB |

Download from: https://nlp.stanford.edu/projects/glove/

### Choosing Pre-trained Vectors

- Wikipedia + Gigaword (6B): Good general-purpose choice, reasonable size
- Common Crawl (42B): Broader vocabulary, better for diverse text
- Common Crawl (840B): Largest, best coverage but huge file
- Twitter: For social media text, handles hashtags and @mentions

## Evaluation

### Word Similarity Tasks

```python
def evaluate_similarity(embeddings, test_pairs):
    """Evaluate on word similarity benchmark."""
    human_scores = []
    model_scores = []

    for word1, word2, human_score in test_pairs:
        if word1 in embeddings and word2 in embeddings:
            v1, v2 = embeddings[word1], embeddings[word2]
            model_score = cosine_similarity(v1, v2)
            human_scores.append(human_score)
            model_scores.append(model_score)

    # Spearman correlation with human judgments
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(human_scores, model_scores)
    return correlation
```

### Analogy Tasks

```python
def solve_analogy(embeddings, a, b, c, topn=5):
    """Solve a:b :: c:? analogy."""
    if a not in embeddings or b not in embeddings or c not in embeddings:
        return None

    # target = b - a + c
    target = embeddings[b] - embeddings[a] + embeddings[c]
    target = target / np.linalg.norm(target)

    # Find nearest neighbors
    similarities = {}
    for word, vec in embeddings.items():
        if word not in [a, b, c]:
            sim = cosine_similarity(target, vec)
            similarities[word] = sim

    # Sort by similarity
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:topn]

# Example
result = solve_analogy(glove, "king", "man", "queen")
# Expected: "woman" as top result
```

## Best Practices

### Dimensionality Selection

| Dimension | Use Case |
|-----------|----------|
| 50 | Quick experiments, memory constrained |
| 100 | Good balance for most tasks |
| 200-300 | Maximum quality, downstream fine-tuning |

Higher dimensions capture more nuance but have diminishing returns above 300.

### Handling Out-of-Vocabulary Words

GloVe has no built-in OOV handling. Common strategies:

```python
def handle_oov(word, embeddings, embedding_dim):
    """Handle out-of-vocabulary words."""
    if word in embeddings:
        return embeddings[word]

    # Strategy 1: Zero vector
    # return np.zeros(embedding_dim)

    # Strategy 2: Random vector
    # return np.random.randn(embedding_dim) * 0.01

    # Strategy 3: Average of all vectors
    # return np.mean(list(embeddings.values()), axis=0)

    # Strategy 4: Lowercase fallback
    if word.lower() in embeddings:
        return embeddings[word.lower()]

    # Strategy 5: Character n-gram approximation
    return approximate_from_subwords(word, embeddings)

def approximate_from_subwords(word, embeddings, min_ngram=3, max_ngram=6):
    """Approximate OOV embedding using character n-grams."""
    vectors = []
    word_lower = word.lower()

    for n in range(min_ngram, min(max_ngram + 1, len(word_lower) + 1)):
        for i in range(len(word_lower) - n + 1):
            ngram = word_lower[i:i+n]
            if ngram in embeddings:
                vectors.append(embeddings[ngram])

    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(len(next(iter(embeddings.values()))))
```

### Normalization

```python
# L2 normalize for similarity tasks
def normalize_embeddings(embeddings):
    normalized = {}
    for word, vec in embeddings.items():
        normalized[word] = vec / np.linalg.norm(vec)
    return normalized

glove_normalized = normalize_embeddings(glove)
```

## Integration with Deep Learning

### TensorFlow/Keras

```python
from tensorflow.keras.layers import Embedding
import numpy as np

def create_keras_embedding(glove_vectors, word_index, embedding_dim=100):
    """Create Keras embedding layer from GloVe."""
    vocab_size = len(word_index) + 1  # +1 for padding
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        vector = glove_vectors.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True  # Fine-tune during training
    )
    return embedding_layer
```

### PyTorch

```python
import torch
import torch.nn as nn

class GloVeEmbedding(nn.Module):
    def __init__(self, glove_vectors, word_to_idx, embedding_dim=100,
                 freeze=False, padding_idx=0):
        super().__init__()

        vocab_size = len(word_to_idx)
        weights = np.zeros((vocab_size, embedding_dim))

        for word, idx in word_to_idx.items():
            if word in glove_vectors:
                weights[idx] = glove_vectors[word]
            else:
                weights[idx] = np.random.normal(0, 0.1, embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(weights),
            freeze=freeze,
            padding_idx=padding_idx
        )

    def forward(self, x):
        return self.embedding(x)
```

## Limitations

### Shared with Word2Vec

- Static embeddings: One vector per word regardless of context
- No OOV handling: Unknown words have no representation
- No subword information: Cannot generalize to morphological variants
- Captures training data biases

### GloVe-Specific

- Memory requirements: Co-occurrence matrix can be large for big vocabularies
- Preprocessing cost: Building the co-occurrence matrix takes time
- Less flexible: Harder to update incrementally with new text

## When to Use GloVe

Choose GloVe when:
- Pre-trained vectors fit your domain (Wikipedia, news, general text)
- You need deterministic, reproducible results
- Training from scratch on a static corpus
- Memory is available for co-occurrence matrix

Consider alternatives when:
- Handling morphologically rich languages (use FastText)
- Need context-dependent representations (use BERT, ELMo)
- Domain is very different from available pre-trained vectors
- Corpus is too large for co-occurrence matrix in memory
