# Word2Vec

## Summary

Word2Vec is a foundational word embedding technique that learns dense vector representations of words from large text corpora. Introduced by Mikolov et al. at Google in 2013, Word2Vec captures semantic and syntactic relationships between words by training shallow neural networks on word co-occurrence patterns. The resulting embeddings enable algebraic operations on word meanings, famously demonstrated by relationships like "king - man + woman = queen".

Key points to remember:

- Two architectures: Skip-gram (predicts context from word) and CBOW (predicts word from context)
- Skip-gram works better for rare words and smaller datasets; CBOW is faster and works well for frequent words
- Training optimizations: Negative sampling and hierarchical softmax reduce computational cost
- Embeddings capture both semantic similarity (cat/dog) and syntactic patterns (running/walked)
- Subsampling of frequent words improves quality by reducing noise from common words like "the"
- Static embeddings: Each word has exactly one vector regardless of context
- Standard dimensionality ranges from 100 to 300 dimensions
- Pre-trained embeddings (Google News, etc.) provide a strong starting point for many tasks
- Limitations: Cannot handle out-of-vocabulary words, no disambiguation of polysemous words

## Core Concepts

### Distributional Hypothesis

Word2Vec is built on the distributional hypothesis: words that appear in similar contexts have similar meanings. By analyzing which words frequently co-occur within a sliding window, Word2Vec learns to represent words with similar usage patterns as nearby vectors in the embedding space.

### Vector Space Properties

Word2Vec embeddings exhibit linear substructures that encode relationships:

```
Semantic relationships:
  vec("Paris") - vec("France") + vec("Germany") H vec("Berlin")
  vec("king") - vec("man") + vec("woman") H vec("queen")

Syntactic relationships:
  vec("walking") - vec("walk") + vec("swim") H vec("swimming")
```

Similarity between words is measured using cosine similarity:

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

## Architectures

### Skip-gram

Skip-gram predicts surrounding context words given a center word. For each word in the corpus, it maximizes the probability of observing the actual context words within a defined window.

```
Training example for "The cat sat on the mat" with window=2:

Center word: "sat"
Context words to predict: ["The", "cat", "on", "the"]

Objective: Maximize P(The|sat) * P(cat|sat) * P(on|sat) * P(the|sat)
```

The objective function:

```
J = (1/T) * sum_{t=1}^{T} sum_{-c <= j <= c, j != 0} log P(w_{t+j} | w_t)
```

Where T is the corpus size and c is the context window size.

### CBOW (Continuous Bag of Words)

CBOW predicts the center word from the average of context word vectors. It treats context as a "bag" without considering word order.

```
Training example for "The cat sat on the mat" with window=2:

Context words: ["The", "cat", "on", "the"]
Target word to predict: "sat"

Objective: Maximize P(sat | average(The, cat, on, the))
```

### Architecture Comparison

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Training speed | Slower | Faster |
| Rare words | Better representations | Weaker representations |
| Dataset size | Works with smaller corpora | Needs larger corpora |
| Memory | Higher (more training pairs) | Lower |
| Use case | When rare words matter | When speed is priority |

## Training Optimizations

### Negative Sampling

Computing softmax over the entire vocabulary is prohibitively expensive. Negative sampling approximates the full softmax by treating the problem as binary classification: distinguish the true context word from randomly sampled "negative" words.

```python
# For each positive (word, context) pair:
# Sample k negative words that don't appear in context
# Train binary classifier to distinguish positive from negative

# Typical k values: 5-20 for small datasets, 2-5 for large datasets
```

The negative sampling objective:

```
log(sigmoid(v_c . v_w)) + sum_{i=1}^{k} E[log(sigmoid(-v_{n_i} . v_w))]
```

Where v_c is the context word vector, v_w is the center word vector, and v_{n_i} are negative sample vectors.

### Hierarchical Softmax

An alternative to negative sampling that organizes the vocabulary as a binary tree. Prediction becomes a series of binary decisions along the path from root to leaf, reducing complexity from O(V) to O(log V).

```
Vocabulary organized as binary tree:
                [root]
               /      \
           [node]    [node]
           /    \    /    \
        "cat" "dog" "run" "walk"

Predicting "cat":
  P(cat) = P(left|root) * P(left|node1)
```

### Subsampling Frequent Words

Common words like "the", "a", "is" provide little information but dominate training. Subsampling randomly discards frequent words with probability:

```
P(discard) = 1 - sqrt(t / f(w))
```

Where f(w) is the word frequency and t is a threshold (typically 10^-5). This reduces noise and speeds up training.

## Implementation

### Using Gensim

```python
from gensim.models import Word2Vec

# Prepare corpus as list of tokenized sentences
sentences = [
    ["machine", "learning", "is", "fascinating"],
    ["deep", "learning", "requires", "data"],
    ["neural", "networks", "learn", "representations"],
]

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Embedding dimensionality
    window=5,             # Context window size
    min_count=1,          # Ignore words with frequency below this
    workers=4,            # Parallel training threads
    sg=1,                 # 1 for Skip-gram, 0 for CBOW
    negative=5,           # Number of negative samples
    epochs=5,             # Training epochs
)

# Access word vectors
vector = model.wv["machine"]

# Find similar words
similar = model.wv.most_similar("learning", topn=5)

# Compute similarity
similarity = model.wv.similarity("machine", "neural")

# Analogy: king - man + woman = ?
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
```

### Training on Large Corpora

```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Stream sentences from file (one sentence per line)
sentences = LineSentence("corpus.txt")

# Build vocabulary first for large corpora
model = Word2Vec(vector_size=300, window=5, min_count=5, workers=8)
model.build_vocab(sentences)

# Train in epochs
model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=10
)

# Save and load
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")

# Save only vectors (smaller file, read-only)
model.wv.save_word2vec_format("vectors.txt", binary=False)
```

### Using Pre-trained Embeddings

```python
import gensim.downloader as api

# Load pre-trained Google News vectors (3 million words, 300d)
model = api.load("word2vec-google-news-300")

# Smaller alternatives
model = api.load("glove-wiki-gigaword-100")  # GloVe, 100d

# Use the vectors
vector = model["computer"]
similar = model.most_similar("python", topn=10)
```

### From Scratch with NumPy

```python
import numpy as np
from collections import Counter

class Word2VecSkipGram:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # Initialize embeddings: center word and context word matrices
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.C = np.random.randn(vocab_size, embedding_dim) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx, context_idx, negative_indices):
        # Center word embedding
        v_center = self.W[center_idx]

        # Positive sample: context word
        v_context = self.C[context_idx]
        score = np.dot(v_center, v_context)
        grad = self.sigmoid(score) - 1  # Gradient for positive sample

        # Update embeddings
        self.W[center_idx] -= self.lr * grad * v_context
        self.C[context_idx] -= self.lr * grad * v_center

        # Negative samples
        for neg_idx in negative_indices:
            v_neg = self.C[neg_idx]
            score = np.dot(v_center, v_neg)
            grad = self.sigmoid(score)  # Gradient for negative sample

            self.W[center_idx] -= self.lr * grad * v_neg
            self.C[neg_idx] -= self.lr * grad * v_center

    def get_embedding(self, word_idx):
        return self.W[word_idx]
```

## Hyperparameter Selection

### Embedding Dimensionality

| Dimensionality | Use Case |
|----------------|----------|
| 50-100 | Small corpora, limited resources, simple tasks |
| 100-200 | General purpose, good balance |
| 300 | Standard for pre-trained models, captures more nuance |
| 500+ | Diminishing returns, risk of overfitting |

### Context Window Size

| Window Size | Captures |
|-------------|----------|
| 2-3 | Syntactic relationships (word forms, grammar) |
| 5-10 | Semantic relationships (meaning, topics) |
| 15+ | Broader topical similarity, less precise |

### Negative Samples

- Small datasets: 5-20 negative samples
- Large datasets: 2-5 negative samples
- More negatives improve precision but slow training

## Evaluation

### Intrinsic Evaluation

```python
from gensim.test.utils import datapath

# Word similarity benchmarks
model.wv.evaluate_word_pairs(datapath("wordsim353.tsv"))

# Analogy tasks
model.wv.evaluate_word_analogies(datapath("questions-words.txt"))
```

Common benchmarks:
- WordSim-353: Human similarity judgments for word pairs
- SimLex-999: Focuses on similarity (not relatedness)
- Google Analogy Dataset: Syntactic and semantic analogies

### Extrinsic Evaluation

Evaluate embeddings through downstream task performance:
- Text classification accuracy
- Named entity recognition F1 score
- Sentiment analysis accuracy

## Limitations

### Static Embeddings

Each word has exactly one vector, regardless of context:

```
"bank" (financial) and "bank" (river) share the same vector
"The bank approved the loan" vs "I sat by the river bank"
```

This limitation led to contextual embeddings like ELMo, BERT, and GPT.

### Out-of-Vocabulary Words

Words not seen during training have no representation:

```python
# Raises KeyError for unknown words
try:
    vector = model.wv["cryptocurrency"]  # If not in training data
except KeyError:
    # Need to handle OOV words
    pass
```

Solutions:
- Use subword models like FastText
- Map to nearest known word
- Use [UNK] token with average embedding

### Training Data Bias

Word2Vec captures biases present in training data:

```python
# Can reflect societal biases in embeddings
# "man" : "computer programmer" :: "woman" : "homemaker"
```

Debiasing techniques exist but are not perfect.

## Comparison with Other Embedding Methods

| Method | Type | OOV Handling | Context | Computational Cost |
|--------|------|--------------|---------|-------------------|
| Word2Vec | Prediction-based | None | Static | Low |
| GloVe | Count-based | None | Static | Low |
| FastText | Prediction + subword | Subword composition | Static | Low |
| ELMo | Contextual | Character-based | Dynamic | Medium |
| BERT | Contextual | Subword tokens | Dynamic | High |

## Practical Recommendations

1. Start with pre-trained embeddings for most tasks. Train custom embeddings only when domain vocabulary differs significantly (medical, legal, technical).

2. Choose Skip-gram for specialized vocabularies with rare terms. Choose CBOW for general text with common vocabulary.

3. Use 300 dimensions as a default. Reduce only if memory or computation is constrained.

4. Set window size based on task: smaller (2-3) for syntactic tasks, larger (5-10) for semantic tasks.

5. Apply min_count filtering (typically 5-10) to remove noisy rare words.

6. For production, save only the word vectors (not the full model) to reduce file size.

7. Normalize vectors for similarity tasks:

```python
from sklearn.preprocessing import normalize
vectors = normalize(model.wv.vectors)
```

8. Consider FastText if handling misspellings or morphologically rich languages.

## Integration with Deep Learning

### As Input Layer

```python
import torch
import torch.nn as nn
import numpy as np

# Load pre-trained vectors
pretrained_vectors = model.wv.vectors  # Shape: (vocab_size, embedding_dim)

# Create embedding layer with pre-trained weights
embedding = nn.Embedding.from_pretrained(
    torch.FloatTensor(pretrained_vectors),
    freeze=False  # Set True to keep embeddings fixed
)

# Use in model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_vectors),
            freeze=False
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # Simple average pooling
        return self.fc(pooled)
```

### Combining with Modern Architectures

Word2Vec embeddings can initialize the embedding layer in LSTMs, CNNs, or Transformers, though modern practice often uses learned embeddings or pretrained language models instead.
