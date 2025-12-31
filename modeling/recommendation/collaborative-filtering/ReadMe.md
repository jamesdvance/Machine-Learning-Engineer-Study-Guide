# Collaborative Filtering

## Summary

Collaborative Filtering (CF) is a recommendation approach that predicts user preferences based on the collective behavior of many users. Rather than relying on item attributes (content-based filtering), CF exploits the insight that users who agreed in the past will likely agree in the future. The approach has evolved from memory-based methods (finding similar users or items) to model-based methods like matrix factorization, and more recently to neural network approaches.

Key points to remember:

- Leverages collective user behavior, not item content
- Two paradigms: memory-based (neighborhood) and model-based (latent factors)
- Memory-based: user-user similarity or item-item similarity
- Model-based: matrix factorization learns latent representations
- Neural CF: replaces linear dot product with neural networks
- Handles explicit feedback (ratings) and implicit feedback (clicks, views)
- Cold start problem: cannot recommend for new users/items
- Sparsity challenge: most users rate few items
- Modern systems combine CF with content features (hybrid approaches)

## Memory-Based vs Model-Based

### Memory-Based (Neighborhood) Methods

Find similar users or items and aggregate their ratings:

**User-Based CF:**
```
pred(u, i) = avg_rating(u) +
             sum(sim(u, v) * (rating(v, i) - avg_rating(v))) / sum(sim(u, v))

for all users v who rated item i
```

**Item-Based CF:**
```
pred(u, i) = sum(sim(i, j) * rating(u, j)) / sum(sim(i, j))

for all items j rated by user u
```

Pros:
- Interpretable (similar users/items)
- No training required
- Can explain recommendations

Cons:
- Doesn't scale (O(n^2) similarity computation)
- Sensitive to sparsity
- Cannot capture complex patterns

### Model-Based Methods

Learn latent factor representations:

```
User u --> embedding vector u (k dimensions)
Item i --> embedding vector v (k dimensions)

pred(u, i) = u . v (dot product)
```

Pros:
- Scalable (O(k) per prediction)
- Handles sparsity through regularization
- Captures latent patterns

Cons:
- Requires training
- Less interpretable
- Cold start still problematic

## Similarity Metrics

For memory-based CF:

### Cosine Similarity

```python
def cosine_similarity(u, v):
    """Similarity between user vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
```

### Pearson Correlation

```python
def pearson_similarity(u, v):
    """Centered cosine similarity."""
    u_centered = u - np.mean(u[u > 0])  # Mean of non-zero
    v_centered = v - np.mean(v[v > 0])
    return cosine_similarity(u_centered, v_centered)
```

### Jaccard Similarity

For implicit feedback (binary):

```python
def jaccard_similarity(items_u, items_v):
    """Set overlap for implicit feedback."""
    intersection = len(items_u & items_v)
    union = len(items_u | items_v)
    return intersection / union if union > 0 else 0
```

## Explicit vs Implicit Feedback

### Explicit Feedback

User explicitly rates items (1-5 stars):

- Direct signal of preference
- Sparse (users rate few items)
- Rating scale may vary by user
- Optimize for rating prediction (RMSE)

```python
# Explicit feedback loss
loss = sum((actual_rating - predicted_rating)^2)
```

### Implicit Feedback

User behavior signals preference (clicks, purchases, time spent):

- Abundant data
- Only positive signals (no explicit dislikes)
- Noisy (click != interest)
- Optimize for ranking (not point prediction)

```python
# Implicit feedback: binary with confidence
# p_ui = 1 if interacted, 0 otherwise
# c_ui = 1 + alpha * count (confidence)

loss = sum(c_ui * (p_ui - predicted)^2)
```

## Matrix Factorization Methods

The dominant paradigm for model-based CF:

### Standard Matrix Factorization

Decompose rating matrix into user and item factors:

```
R (m x n) = U (m x k) * V^T (k x n)

Minimize: ||R - UV^T||^2 + lambda*(||U||^2 + ||V||^2)
```

Training options:
- **SGD**: Update after each rating, online updates possible
- **ALS**: Closed-form updates, highly parallelizable

See [Matrix Factorization](matrix-factorization/ReadMe.md) for details.

### SVD++

Extends MF with implicit feedback and user bias:

```
pred(u, i) = mu + b_u + b_i + (p_u + |N(u)|^-0.5 * sum(y_j)) . q_i

where N(u) = items user u has interacted with
```

### Factorization Machines

Generalize MF to arbitrary feature interactions:

```
y(x) = w_0 + sum(w_i * x_i) + sum_i sum_j <v_i, v_j> * x_i * x_j
```

Enables incorporating user/item features alongside IDs.

## Neural Collaborative Filtering

Replace linear dot product with neural networks:

### Generalized Matrix Factorization (GMF)

```
pred(u, i) = sigmoid(h^T * (p_u * q_i))
```

Element-wise product with learnable output.

### Multi-Layer Perceptron (MLP)

```
pred(u, i) = MLP([p_u; q_i])
```

Concatenate embeddings and process through hidden layers.

### NeuMF

Combine GMF and MLP with separate embeddings:

```
pred(u, i) = sigmoid(h^T * [GMF_output; MLP_output])
```

See [Neural Collaborative Filtering](neural-collaborative-filtering/ReadMe.md) for details.

## Handling Common Challenges

### Cold Start Problem

New users/items have no interaction history:

**For new users:**
- Ask for initial preferences (onboarding)
- Use demographic similarity
- Show popular items initially
- Use content-based until history builds

**For new items:**
- Use content features
- Explore-exploit strategies
- Boost new items in rankings

### Sparsity

Most users rate very few items:

**Regularization:**
```python
loss += lambda * (||U||^2 + ||V||^2)
```

**Dimensionality reduction:**
Use small k (20-100 latent factors)

**Side information:**
Incorporate user/item features

### Scalability

For millions of users and items:

**Approximate Nearest Neighbors:**
For memory-based methods, use ANN for similarity search (see FAISS, ScaNN).

**Distributed Training:**
ALS parallelizes naturally (see [Alternating Least Squares](alternating-least-squares/ReadMe.md)).

**Candidate Generation:**
Two-stage: retrieve candidates, then rank (see Retrieval and Ranking).

## Evaluation

### For Explicit Feedback

```python
def rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def mae(predictions, actuals):
    return np.mean(np.abs(predictions - actuals))
```

### For Implicit Feedback (Ranking)

```python
def hit_rate_at_k(recommendations, ground_truth, k=10):
    """Fraction of users where relevant item is in top-k."""
    hits = 0
    for user, recs in recommendations.items():
        if ground_truth[user] in recs[:k]:
            hits += 1
    return hits / len(recommendations)

def ndcg_at_k(recommendations, ground_truth, k=10):
    """Normalized Discounted Cumulative Gain."""
    ndcgs = []
    for user, recs in recommendations.items():
        true_item = ground_truth[user]
        if true_item in recs[:k]:
            rank = recs[:k].index(true_item)
            ndcg = 1 / np.log2(rank + 2)
        else:
            ndcg = 0
        ndcgs.append(ndcg)
    return np.mean(ndcgs)
```

### Offline Evaluation Protocol

**Leave-One-Out:**
Hold out last interaction per user for testing.

**Temporal Split:**
Train on data before time T, test on data after.

**Random Split:**
Randomly sample interactions for test (leaks future information).

## Implementation Example

Simple item-based CF:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ItemBasedCF:
    def __init__(self, k=50):
        self.k = k  # Number of similar items to consider

    def fit(self, ratings_matrix):
        """
        ratings_matrix: users x items sparse matrix
        """
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(ratings_matrix.T)
        np.fill_diagonal(self.item_similarity, 0)  # Remove self-similarity

        # Keep only top-k similar items per item
        for i in range(self.item_similarity.shape[0]):
            top_k_idx = np.argsort(self.item_similarity[i])[-self.k:]
            mask = np.ones(self.item_similarity.shape[0], dtype=bool)
            mask[top_k_idx] = False
            self.item_similarity[i, mask] = 0

        self.ratings = ratings_matrix

    def predict(self, user, item):
        # Get user's ratings
        user_ratings = self.ratings[user].toarray().flatten()

        # Items user has rated
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return 0

        # Weighted average of similar items
        similarities = self.item_similarity[item, rated_items]
        ratings = user_ratings[rated_items]

        if np.sum(np.abs(similarities)) == 0:
            return 0

        return np.dot(similarities, ratings) / np.sum(np.abs(similarities))

    def recommend(self, user, n=10, exclude_seen=True):
        scores = []
        user_ratings = self.ratings[user].toarray().flatten()
        seen = set(np.where(user_ratings > 0)[0])

        for item in range(self.ratings.shape[1]):
            if exclude_seen and item in seen:
                continue
            score = self.predict(user, item)
            scores.append((item, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
```

## Method Comparison

| Method | Scale | Cold Start | Features | Online Updates |
|--------|-------|------------|----------|----------------|
| User-based CF | Small | Poor | No | Yes |
| Item-based CF | Medium | Poor | No | Partial |
| Matrix Factorization | Large | Poor | Limited | SGD only |
| Neural CF | Large | Poor | Yes | Difficult |
| Factorization Machines | Large | Better | Yes | SGD only |

## When to Use Collaborative Filtering

CF is well-suited for:
- Sufficient user-item interaction data
- Personalization at scale
- When content features are limited or unhelpful
- Domains where user taste correlates (movies, music, products)

Consider alternatives when:
- New platform with little data (content-based)
- Items change rapidly (news, trending content)
- Rich item features available (hybrid)
- Need to explain recommendations (content-based)
- Extreme cold start (popularity-based)

## Further Reading

For detailed implementations:

- [Matrix Factorization](matrix-factorization/ReadMe.md) - Core latent factor model
- [Alternating Least Squares](alternating-least-squares/ReadMe.md) - Scalable training for MF
- [Neural Collaborative Filtering](neural-collaborative-filtering/ReadMe.md) - Deep learning approach
