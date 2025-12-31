# Alternating Least Squares (ALS)

## Summary

Alternating Least Squares is an optimization algorithm for matrix factorization that alternates between fixing user factors and solving for item factors, then fixing item factors and solving for user factors. Unlike SGD, ALS solves a convex optimization problem at each step, providing closed-form updates that are highly parallelizable. This makes ALS the standard algorithm for large-scale collaborative filtering, particularly for implicit feedback data.

Key points to remember:

- Optimization algorithm for matrix factorization, not a different model
- Alternates: fix U, solve for V; fix V, solve for U
- Each subproblem is convex with closed-form solution
- Highly parallelizable (users/items updated independently)
- No learning rate tuning required
- Naturally handles implicit feedback with confidence weighting
- Industry standard for distributed systems (Spark MLlib)
- Faster convergence than SGD for large datasets
- Compared to SGD: more stable, parallelizable, but higher per-iteration cost
- Compared to gradient descent: no hyperparameter sensitivity

## Algorithm

### Basic ALS for Explicit Ratings

Given rating matrix R (m users x n items), find U (m x k) and V (n x k) minimizing:

```
L = sum_{(u,i) in observed} (r_ui - u_u . v_i)^2 + lambda * (||U||^2 + ||V||^2)
```

The alternating approach:

```
1. Initialize U and V randomly
2. Repeat until convergence:
   a. Fix V, solve for each u_u:
      u_u = (V_u^T V_u + lambda*I)^-1 * V_u^T * r_u
   b. Fix U, solve for each v_i:
      v_i = (U_i^T U_i + lambda*I)^-1 * U_i^T * r_i
```

Where V_u contains item factors for items rated by user u.

### Why It Works

When V is fixed, the loss function is quadratic in U:

```
L(u_u) = ||r_u - V_u * u_u||^2 + lambda * ||u_u||^2
```

This is a ridge regression problem with closed-form solution:

```
u_u = (V_u^T V_u + lambda*I)^-1 * V_u^T * r_u
```

### Implementation

```python
import numpy as np
from scipy.sparse import csr_matrix

class ALS:
    def __init__(self, n_factors=50, reg=0.01, n_iter=15):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter

    def fit(self, ratings):
        """
        ratings: list of (user, item, rating) tuples
        """
        # Build interaction data structures
        n_users = max(u for u, _, _ in ratings) + 1
        n_items = max(i for _, i, _ in ratings) + 1

        # Group ratings by user and item
        user_items = {u: [] for u in range(n_users)}
        item_users = {i: [] for i in range(n_items)}

        for u, i, r in ratings:
            user_items[u].append((i, r))
            item_users[i].append((u, r))

        # Initialize factors
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        for iteration in range(self.n_iter):
            # Update user factors
            for u in range(n_users):
                if not user_items[u]:
                    continue

                items = [i for i, _ in user_items[u]]
                ratings_u = np.array([r for _, r in user_items[u]])
                V_u = self.V[items]

                A = V_u.T @ V_u + self.reg * np.eye(self.n_factors)
                b = V_u.T @ ratings_u
                self.U[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(n_items):
                if not item_users[i]:
                    continue

                users = [u for u, _ in item_users[i]]
                ratings_i = np.array([r for _, r in item_users[i]])
                U_i = self.U[users]

                A = U_i.T @ U_i + self.reg * np.eye(self.n_factors)
                b = U_i.T @ ratings_i
                self.V[i] = np.linalg.solve(A, b)

            # Compute training loss
            loss = self._compute_loss(ratings)
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

    def _compute_loss(self, ratings):
        total_loss = 0
        for u, i, r in ratings:
            pred = np.dot(self.U[u], self.V[i])
            total_loss += (r - pred) ** 2
        total_loss += self.reg * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
        return total_loss

    def predict(self, user, item):
        return np.dot(self.U[user], self.V[item])

    def recommend(self, user, n=10, exclude_seen=None):
        scores = self.U[user] @ self.V.T
        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf
        top_items = np.argsort(scores)[::-1][:n]
        return [(item, scores[item]) for item in top_items]
```

## Implicit Feedback ALS

For implicit data (clicks, views, purchases), use confidence-weighted formulation:

### Formulation

```
Minimize:
L = sum_{u,i} c_ui * (p_ui - u_u . v_i)^2 + lambda * (||U||^2 + ||V||^2)

where:
- p_ui = 1 if user interacted with item, 0 otherwise
- c_ui = 1 + alpha * r_ui (confidence based on interaction strength)
```

### Implementation

```python
class ImplicitALS:
    def __init__(self, n_factors=50, reg=0.01, alpha=40, n_iter=15):
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha  # Confidence scaling
        self.n_iter = n_iter

    def fit(self, interactions):
        """
        interactions: scipy sparse matrix (users x items)
        """
        n_users, n_items = interactions.shape

        # Binary preference
        P = (interactions > 0).astype(np.float32)

        # Confidence = 1 + alpha * interaction count
        C = 1 + self.alpha * interactions

        # Initialize factors
        self.U = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.01, (n_items, self.n_factors))

        for iteration in range(self.n_iter):
            # Update users
            self._update_users(P, C)
            # Update items
            self._update_items(P, C)

            print(f"Iteration {iteration + 1} complete")

    def _update_users(self, P, C):
        """Update all user factors."""
        VtV = self.V.T @ self.V

        for u in range(P.shape[0]):
            # Get confidence-weighted terms for this user
            c_u = C[u].toarray().flatten()
            p_u = P[u].toarray().flatten()

            # C_u - I is sparse (only non-zero for interacted items)
            Cu_I = c_u - 1

            # A = V^T * C_u * V + lambda * I
            # Efficient: V^T * V + V^T * (C_u - I) * V
            A = VtV + self.V.T @ np.diag(Cu_I) @ self.V + self.reg * np.eye(self.n_factors)

            # b = V^T * C_u * p_u
            b = self.V.T @ (c_u * p_u)

            self.U[u] = np.linalg.solve(A, b)

    def _update_items(self, P, C):
        """Update all item factors."""
        UtU = self.U.T @ self.U

        for i in range(P.shape[1]):
            c_i = C[:, i].toarray().flatten()
            p_i = P[:, i].toarray().flatten()

            Ci_I = c_i - 1
            A = UtU + self.U.T @ np.diag(Ci_I) @ self.U + self.reg * np.eye(self.n_factors)
            b = self.U.T @ (c_i * p_i)

            self.V[i] = np.linalg.solve(A, b)
```

### Efficient Sparse Implementation

The naive implementation is O(k^2 * nnz) per iteration. Optimized version:

```python
def update_user_efficient(self, u, P_row, C_row, VtV):
    """Efficient update exploiting sparsity."""
    # Only non-zero interactions matter for (C_u - I)
    interacted_items = P_row.indices
    c_values = C_row.data - 1  # Subtract 1 for C_u - I

    # V_u for interacted items only
    V_u = self.V[interacted_items]

    # A = VtV + V_u^T * diag(c-1) * V_u + reg*I
    A = VtV + V_u.T @ np.diag(c_values) @ V_u + self.reg * np.eye(self.n_factors)

    # b = V^T * C * p = V_u^T * c (where p=1 for interacted)
    c_full = C_row.data
    b = V_u.T @ c_full

    return np.linalg.solve(A, b)
```

## Parallelization

### User-Parallel Updates

Each user update is independent given fixed V:

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_update_users(U, V, user_items, ratings, reg, n_workers=8):
    """Update all users in parallel."""
    n_users = len(U)

    def update_single_user(u):
        if u not in user_items or not user_items[u]:
            return u, U[u]

        items = [i for i, _ in user_items[u]]
        r = np.array([r for _, r in user_items[u]])
        V_u = V[items]

        A = V_u.T @ V_u + reg * np.eye(V.shape[1])
        b = V_u.T @ r
        return u, np.linalg.solve(A, b)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(update_single_user, range(n_users))

    for u, factors in results:
        U[u] = factors

    return U
```

### Distributed ALS (Spark)

```python
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DistributedALS") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load ratings data
ratings = spark.read.parquet("ratings.parquet")

# Configure ALS
als = ALS(
    maxIter=15,
    regParam=0.01,
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",
    coldStartStrategy="drop",
    implicitPrefs=True,  # For implicit feedback
    alpha=40.0,          # Confidence scaling
    nonnegative=True,    # Constrain factors >= 0
    numUserBlocks=100,   # Parallelism
    numItemBlocks=100
)

model = als.fit(ratings)

# Get user/item factors
user_factors = model.userFactors  # DataFrame with (id, features)
item_factors = model.itemFactors

# Generate recommendations
recommendations = model.recommendForAllUsers(10)
recommendations.show()
```

### Block-wise Partitioning

Spark partitions users and items into blocks:

```
Users partitioned:          Items partitioned:
[Block 1: users 0-999]      [Block A: items 0-499]
[Block 2: users 1000-1999]  [Block B: items 500-999]
...                         ...

Rating matrix partitioned into (user block, item block) tiles
Each tile processed on single executor
```

## Hyperparameters

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| n_factors | Latent dimension | 20-200 |
| reg (lambda) | L2 regularization | 0.001-0.1 |
| alpha | Confidence scaling (implicit) | 10-100 |
| n_iter | ALS iterations | 10-20 |

### Tuning Guidelines

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_factors': [20, 50, 100, 200],
    'reg': [0.001, 0.01, 0.1],
    'alpha': [10, 40, 100]  # For implicit
}

best_score = float('inf')
for params in ParameterGrid(param_grid):
    model = ImplicitALS(**params)
    model.fit(train_interactions)

    # Evaluate on held-out interactions
    score = evaluate_ranking(model, test_interactions)
    if score < best_score:
        best_score = score
        best_params = params
```

### Regularization Selection

```python
# Cross-validation for regularization
from sklearn.model_selection import KFold

def cv_score(reg, train_data, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True)
    scores = []

    for train_idx, val_idx in kf.split(train_data):
        train = [train_data[i] for i in train_idx]
        val = [train_data[i] for i in val_idx]

        model = ALS(n_factors=50, reg=reg)
        model.fit(train)

        rmse = evaluate_rmse(model, val)
        scores.append(rmse)

    return np.mean(scores)

# Find optimal regularization
for reg in [0.001, 0.01, 0.1, 1.0]:
    score = cv_score(reg, ratings)
    print(f"reg={reg}: CV RMSE = {score:.4f}")
```

## Practical Considerations

### Initialization

Random initialization works but smart initialization helps:

```python
# Initialize with SVD of mean-centered ratings
from scipy.sparse.linalg import svds

# Mean-center the ratings
mean_rating = ratings.data.mean()
ratings_centered = ratings.copy()
ratings_centered.data -= mean_rating

# Truncated SVD for initialization
u, s, vt = svds(ratings_centered, k=n_factors)
U_init = u @ np.diag(np.sqrt(s))
V_init = np.diag(np.sqrt(s)) @ vt
```

### Handling Scale

For very large datasets:
- Increase number of blocks (Spark)
- Use approximate solvers instead of exact
- Subsample during initial iterations

```python
# Conjugate gradient instead of direct solve
from scipy.sparse.linalg import cg

def solve_cg(A, b, maxiter=10):
    x, info = cg(A, b, maxiter=maxiter)
    return x
```

### Monitoring Convergence

```python
def als_with_monitoring(ratings, n_factors, reg, n_iter):
    # ... initialization ...

    history = {'train_loss': [], 'val_loss': []}

    for iteration in range(n_iter):
        # Update U and V
        update_users(...)
        update_items(...)

        # Compute losses
        train_loss = compute_loss(train_ratings)
        val_loss = compute_loss(val_ratings)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if len(history['val_loss']) > 3:
            if history['val_loss'][-1] > history['val_loss'][-2]:
                print(f"Early stopping at iteration {iteration}")
                break

    return U, V, history
```

## Comparison with SGD

| Aspect | ALS | SGD |
|--------|-----|-----|
| Parallelization | Embarrassingly parallel | Sequential updates |
| Convergence | Faster (closed-form) | Slower |
| Learning rate | Not needed | Requires tuning |
| Per-iteration cost | Higher (matrix solve) | Lower |
| Implicit feedback | Natural fit | Requires modification |
| Online updates | Difficult | Natural |
| Memory | Stores full factors | Can stream |

### When to Use ALS

- Large-scale systems (millions of users/items)
- Implicit feedback data
- Distributed computing environment
- Batch training (not online)
- When tuning sensitivity is a concern

### When to Use SGD

- Smaller datasets
- Need for online updates
- Memory constraints
- Explicit ratings with many observed entries

## Extensions

### Weighted ALS

Different weights for different ratings:

```python
# Weight matrix W where W[u,i] = weight for rating r_ui
# Minimize: sum w_ui * (r_ui - u_u . v_i)^2

def weighted_als_user_update(u, items, ratings, weights, V, reg):
    V_u = V[items]
    W_u = np.diag(weights)

    A = V_u.T @ W_u @ V_u + reg * np.eye(n_factors)
    b = V_u.T @ W_u @ ratings

    return np.linalg.solve(A, b)
```

### Non-negative ALS

Constrain factors to be non-negative (interpretable):

```python
from scipy.optimize import nnls

def nonnegative_als_update(A, b):
    """Non-negative least squares."""
    x, residual = nnls(A, b)
    return x
```

### Temporal ALS

Time-weighted confidence:

```python
# Recent interactions weighted higher
def temporal_confidence(interaction_time, current_time, decay=0.1):
    days_ago = (current_time - interaction_time).days
    return 1 + alpha * np.exp(-decay * days_ago)
```

## When to Use ALS

ALS is well-suited for:
- Large-scale collaborative filtering
- Implicit feedback (clicks, views)
- Distributed training (Spark)
- Production systems needing stable training
- When SGD hyperparameter tuning is problematic

Consider alternatives when:
- Online/streaming updates needed (SGD)
- Very sparse data per user (BPR)
- Complex interaction patterns (neural models)
- Side features important (factorization machines)
