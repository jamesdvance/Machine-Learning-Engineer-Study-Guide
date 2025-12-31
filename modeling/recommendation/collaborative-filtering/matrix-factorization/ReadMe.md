# Matrix Factorization

## Summary

Matrix Factorization (MF) is a fundamental technique for collaborative filtering that decomposes the sparse user-item interaction matrix into lower-dimensional latent factor matrices. By learning dense vector representations (embeddings) for both users and items, MF can predict missing entries in the interaction matrix, enabling recommendations for items users have not yet interacted with.

Key points to remember:

- Decomposes user-item matrix R into user factors U and item factors V: R = U x V^T
- Each user and item represented as k-dimensional latent vector (embedding)
- Prediction: dot product of user and item vectors gives predicted rating/score
- Handles sparse data well (most users rate few items)
- Regularization prevents overfitting on observed interactions
- Training via SGD (online) or ALS (batch, parallelizable)
- Implicit feedback variant uses binary interactions with confidence weighting
- Foundation for modern embedding-based recommenders
- Compared to memory-based CF: better scalability, handles sparsity
- Compared to deep learning: simpler, faster training, interpretable

## Core Concepts

### The Rating Matrix

User-item interactions form a sparse matrix:

```
           Item1  Item2  Item3  Item4  Item5
User1        5      3      ?      ?      4
User2        4      ?      ?      2      ?
User3        ?      2      4      ?      ?
User4        ?      ?      5      3      ?
User5        3      ?      ?      4      5
```

Goal: Predict missing entries (?) to recommend items.

### Latent Factors

MF assumes ratings arise from latent factors representing hidden characteristics:

```
User preferences (U):         Item attributes (V):

User1: [0.8, 0.2, 0.5]       Item1: [0.9, 0.1, 0.6]  (action, romance, comedy)
User2: [0.3, 0.9, 0.1]       Item2: [0.2, 0.8, 0.3]
...                           ...
```

Predicted rating: User1 x Item1 = 0.8*0.9 + 0.2*0.1 + 0.5*0.6 = 1.04

### Model Formulation

```
Given:
- R: m x n rating matrix (m users, n items)
- k: number of latent factors

Learn:
- U: m x k user factor matrix
- V: n x k item factor matrix

Objective:
minimize ||R - UV^T||^2 + regularization
         (over observed ratings only)
```

### Bias Terms

Ratings include systematic biases:

```
r_ui = mu + b_u + b_i + u_i . v_i

where:
- mu: global average rating
- b_u: user bias (some users rate higher)
- b_i: item bias (some items rated higher)
- u_i . v_i: user-item interaction
```

## Training Methods

### Stochastic Gradient Descent (SGD)

Updates factors after each observed rating:

```python
import numpy as np

class MatrixFactorizationSGD:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.02):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        # Initialize factors randomly
        self.U = np.random.normal(0, 0.1, (n_users, n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, n_factors))

        # Initialize biases
        self.mu = 0.0
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

    def predict(self, user, item):
        return (self.mu +
                self.b_u[user] +
                self.b_i[item] +
                np.dot(self.U[user], self.V[item]))

    def fit(self, ratings, n_epochs=20):
        # ratings: list of (user, item, rating) tuples
        self.mu = np.mean([r for _, _, r in ratings])

        for epoch in range(n_epochs):
            np.random.shuffle(ratings)
            total_loss = 0

            for user, item, rating in ratings:
                # Compute prediction error
                pred = self.predict(user, item)
                error = rating - pred

                # Update biases
                self.b_u[user] += self.lr * (error - self.reg * self.b_u[user])
                self.b_i[item] += self.lr * (error - self.reg * self.b_i[item])

                # Update factors
                U_old = self.U[user].copy()
                self.U[user] += self.lr * (error * self.V[item] - self.reg * self.U[user])
                self.V[item] += self.lr * (error * U_old - self.reg * self.V[item])

                total_loss += error ** 2

            rmse = np.sqrt(total_loss / len(ratings))
            print(f"Epoch {epoch + 1}: RMSE = {rmse:.4f}")
```

### Alternating Least Squares (ALS)

Fixes one factor matrix, solves for other in closed form:

```python
def als_update_user(ratings_by_user, V, reg, n_factors):
    """Update all user factors given fixed item factors."""
    n_users = len(ratings_by_user)
    U = np.zeros((n_users, n_factors))

    VtV = V.T @ V  # Precompute for efficiency

    for user, items_ratings in ratings_by_user.items():
        items = [i for i, _ in items_ratings]
        ratings = np.array([r for _, r in items_ratings])

        V_u = V[items]  # Items rated by user
        A = V_u.T @ V_u + reg * np.eye(n_factors)
        b = V_u.T @ ratings

        U[user] = np.linalg.solve(A, b)

    return U
```

ALS advantages:
- Parallelizable (users/items independent given other matrix)
- No learning rate tuning
- Handles implicit feedback naturally

## Implicit Feedback

For implicit signals (clicks, views, purchases) rather than explicit ratings:

### Weighted Matrix Factorization

```python
# Implicit feedback formulation
# p_ui = 1 if user interacted with item, 0 otherwise
# c_ui = 1 + alpha * r_ui (confidence, higher for more interactions)

def implicit_als(interactions, n_factors=50, alpha=40, reg=0.01, n_iter=15):
    n_users, n_items = interactions.shape

    # Binary preference matrix
    P = (interactions > 0).astype(float)

    # Confidence matrix
    C = 1 + alpha * interactions

    # Initialize factors
    U = np.random.normal(0, 0.01, (n_users, n_factors))
    V = np.random.normal(0, 0.01, (n_items, n_factors))

    for iteration in range(n_iter):
        # Update users
        VtV = V.T @ V
        for u in range(n_users):
            c_u = np.diag(C[u])
            A = VtV + V.T @ (c_u - np.eye(n_items)) @ V + reg * np.eye(n_factors)
            b = V.T @ c_u @ P[u]
            U[u] = np.linalg.solve(A, b)

        # Update items (symmetric)
        UtU = U.T @ U
        for i in range(n_items):
            c_i = np.diag(C[:, i])
            A = UtU + U.T @ (c_i - np.eye(n_users)) @ U + reg * np.eye(n_factors)
            b = U.T @ c_i @ P[:, i]
            V[i] = np.linalg.solve(A, b)

    return U, V
```

### BPR: Bayesian Personalized Ranking

Optimizes ranking rather than point-wise prediction:

```python
def bpr_update(user, pos_item, neg_item, U, V, lr, reg):
    """Single BPR update step."""
    # Positive item score minus negative item score
    x_uij = np.dot(U[user], V[pos_item] - V[neg_item])

    # Sigmoid of score difference
    sigmoid = 1 / (1 + np.exp(x_uij))

    # Gradient updates
    U[user] += lr * (sigmoid * (V[pos_item] - V[neg_item]) - reg * U[user])
    V[pos_item] += lr * (sigmoid * U[user] - reg * V[pos_item])
    V[neg_item] += lr * (-sigmoid * U[user] - reg * V[neg_item])
```

BPR advantages:
- Directly optimizes ranking metric
- Handles implicit feedback naturally
- No need for confidence weighting

## Practical Implementation

### Using Surprise Library

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)

# SVD with biases
model = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    biased=True
)

# Cross-validate
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train on full data
trainset = data.build_full_trainset()
model.fit(trainset)

# Predict
prediction = model.predict(user_id='user123', item_id='item456')
print(f"Predicted rating: {prediction.est:.2f}")
```

### Using Implicit Library

```python
import implicit
from scipy.sparse import csr_matrix

# Create sparse interaction matrix
user_item_matrix = csr_matrix(interactions)

# Train ALS model for implicit feedback
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=15,
    use_gpu=False
)

model.fit(user_item_matrix)

# Get recommendations for user
user_id = 0
recommendations = model.recommend(
    user_id,
    user_item_matrix[user_id],
    N=10,
    filter_already_liked_items=True
)

for item_id, score in recommendations:
    print(f"Item {item_id}: {score:.4f}")
```

### Spark MLlib

```python
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MF").getOrCreate()

# Load data
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)

# Train ALS model
als = ALS(
    maxIter=10,
    regParam=0.01,
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

model = als.fit(ratings)

# Generate recommendations
user_recs = model.recommendForAllUsers(10)
item_recs = model.recommendForAllItems(10)
```

## Hyperparameter Tuning

### Key Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| n_factors | Model capacity | 20-200 |
| regularization | Prevents overfitting | 0.001-0.1 |
| learning_rate | SGD step size | 0.001-0.01 |
| n_epochs | Training iterations | 10-50 |
| alpha (implicit) | Confidence scaling | 10-100 |

### Tuning Strategy

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_factors': [20, 50, 100],
    'reg': [0.001, 0.01, 0.1],
    'lr': [0.001, 0.005, 0.01]
}

best_rmse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    model = MatrixFactorizationSGD(
        n_users, n_items,
        n_factors=params['n_factors'],
        lr=params['lr'],
        reg=params['reg']
    )
    model.fit(train_ratings, n_epochs=20)

    val_rmse = evaluate(model, val_ratings)
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_params = params
```

## Evaluation Metrics

### Rating Prediction

```python
def rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def mae(predictions, actuals):
    return np.mean(np.abs(predictions - actuals))
```

### Ranking Quality

```python
def precision_at_k(recommendations, relevant_items, k):
    """Precision@K: fraction of top-K that are relevant."""
    top_k = recommendations[:k]
    hits = len(set(top_k) & set(relevant_items))
    return hits / k

def ndcg_at_k(recommendations, relevant_items, k):
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommendations[:k])
              if item in relevant_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    return dcg / idcg if idcg > 0 else 0
```

## Limitations and Extensions

### Cold Start Problem

MF cannot recommend for new users/items without interaction history:

- **User cold start**: Use content features, popular items, or hybrid models
- **Item cold start**: Use content features or side information

### Side Information

Incorporate metadata beyond interactions:

```python
# SVD++ includes implicit feedback
# r_ui = mu + b_u + b_i + (u_i + |N(u)|^-0.5 * sum(y_j for j in N(u))) . v_i

# SVDFeature includes features
# r_ui = mu + sum(b_f * f) + u_i . v_i + sum(feature interactions)
```

### Non-linear Extensions

- **Neural Matrix Factorization**: Replace dot product with neural network
- **Factorization Machines**: Generalize MF to arbitrary feature interactions
- **Deep Factorization Machines**: Combine FM with deep networks

## Comparison with Alternatives

| Method | Pros | Cons |
|--------|------|------|
| Matrix Factorization | Scalable, handles sparsity | Cold start, linear |
| User-based CF | Interpretable | Doesn't scale |
| Item-based CF | Stable | Misses user context |
| Neural CF | Non-linear patterns | More complex, data-hungry |
| Content-based | Handles cold start | Ignores collaborative signal |

## When to Use Matrix Factorization

MF is well-suited for:
- Large-scale recommendation with sparse interactions
- Baseline model for comparison
- When interpretable embeddings are valuable
- Limited computational resources
- Implicit feedback (clicks, views)

Consider alternatives when:
- Rich item/user features available (hybrid)
- Complex interaction patterns (deep learning)
- Sequence matters (sequential models)
- Extreme cold start (content-based)
