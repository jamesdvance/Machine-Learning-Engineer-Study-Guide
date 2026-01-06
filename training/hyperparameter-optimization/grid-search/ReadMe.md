# Grid Search

## Summary

Grid search exhaustively evaluates all combinations of hyperparameters from a predefined grid. While simple and easy to parallelize, grid search becomes computationally infeasible for large search spaces due to the curse of dimensionality. It's most useful for small, discrete parameter spaces or when fine-tuning around a known good configuration.

Key points to remember:

- Evaluates all combinations in the grid
- Exponential growth: k^n combinations for n params with k values each
- Highly parallelizable (independent trials)
- Wastes compute on unimportant dimensions
- Best for small, discrete search spaces
- Good for final fine-tuning around known good values
- Random search often more efficient
- Easy to implement and understand

## Basic Implementation

### Simple Grid Search

```python
from itertools import product

def grid_search(param_grid, objective_fn):
    """Run grid search over parameter grid."""
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        score = objective_fn(config)
        results.append((config, score))
        print(f"Config: {config}, Score: {score:.4f}")

    # Find best
    best = min(results, key=lambda x: x[1])
    return best

# Define grid
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'hidden_size': [256, 512]
}

# Total: 3 * 3 * 2 = 18 combinations
best_config, best_score = grid_search(param_grid, objective)
```

### Sklearn GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Create grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Parallel Grid Search

### With Joblib

```python
from joblib import Parallel, delayed
from itertools import product

def evaluate_config(config):
    """Evaluate a single configuration."""
    score = train_and_evaluate(config)
    return config, score

def parallel_grid_search(param_grid, n_jobs=-1):
    """Run grid search in parallel."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_config)(config)
        for config in configs
    )

    best = min(results, key=lambda x: x[1])
    return best
```

### With Ray

```python
import ray
from itertools import product

@ray.remote
def evaluate_config_remote(config):
    return config, train_and_evaluate(config)

def ray_grid_search(param_grid):
    """Distributed grid search with Ray."""
    ray.init()

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    # Submit all tasks
    futures = [evaluate_config_remote.remote(c) for c in configs]

    # Gather results
    results = ray.get(futures)

    ray.shutdown()
    return min(results, key=lambda x: x[1])
```

## When to Use Grid Search

### Good Use Cases

```python
# 1. Small, discrete space
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'gelu'],
}  # Only 4 combinations

# 2. Fine-tuning around known good values
param_grid = {
    'learning_rate': [0.9e-4, 1.0e-4, 1.1e-4],
    'weight_decay': [0.009, 0.01, 0.011],
}

# 3. Categorical parameters only
param_grid = {
    'model_type': ['small', 'base', 'large'],
    'tokenizer': ['bpe', 'unigram'],
}
```

### Avoid When

```python
# 1. Large continuous spaces
param_grid = {
    'lr': np.linspace(1e-5, 1e-2, 100),  # 100 values
    'wd': np.linspace(0, 0.1, 50),        # 50 values
}  # 5000 combinations!

# 2. Many parameters
param_grid = {
    'param1': [1, 2, 3],
    'param2': [1, 2, 3],
    'param3': [1, 2, 3],
    'param4': [1, 2, 3],
    'param5': [1, 2, 3],
}  # 3^5 = 243 combinations

# 3. When some params matter more than others
# Grid wastes compute on unimportant combinations
```

## Efficiency Improvements

### Coarse-to-Fine Search

```python
def coarse_to_fine_grid_search(objective_fn):
    """Two-stage grid search."""
    # Stage 1: Coarse search
    coarse_grid = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [16, 64, 256],
    }

    best_coarse = grid_search(coarse_grid, objective_fn)
    print(f"Coarse best: {best_coarse}")

    # Stage 2: Fine search around best
    best_lr = best_coarse[0]['learning_rate']
    best_bs = best_coarse[0]['batch_size']

    fine_grid = {
        'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
        'batch_size': [max(8, best_bs // 2), best_bs, min(512, best_bs * 2)],
    }

    best_fine = grid_search(fine_grid, objective_fn)
    return best_fine
```

### Early Stopping

```python
def grid_search_with_early_stopping(param_grid, objective_fn, max_configs=50):
    """Grid search with budget limit."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    # Limit number of evaluations
    if len(combinations) > max_configs:
        import random
        combinations = random.sample(combinations, max_configs)
        print(f"Sampling {max_configs} of {len(list(product(*values)))} combinations")

    results = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        score = objective_fn(config)
        results.append((config, score))

    return min(results, key=lambda x: x[1])
```

## Comparison with Random Search

### Grid Search Limitations

```
Consider 2 parameters, only one matters:

Grid (9 points):       Random (9 points):
[x] [x] [x]            [x]     [x]  [x]
[x] [x] [x]               [x]      [x]
[x] [x] [x]            [x]    [x] [x]

Grid: Only 3 unique values for important param
Random: Up to 9 unique values for important param
```

### When Grid Beats Random

```python
# Grid better when:
# 1. All parameters are equally important
# 2. Parameters interact strongly
# 3. Discrete choices (not continuous)
# 4. Small search space

# Example where grid makes sense
param_grid = {
    'model': ['bert-base', 'bert-large', 'roberta-base'],
    'freeze_layers': [0, 6, 12],
}
# These interact (model size + frozen layers)
# Discrete choices
# Only 9 combinations
```

## Results Analysis

### Track All Results

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_grid_results(results):
    """Analyze grid search results."""
    df = pd.DataFrame([
        {**config, 'score': score}
        for config, score in results
    ])

    # Best configurations
    print("Top 5 configurations:")
    print(df.nsmallest(5, 'score'))

    # Parameter statistics
    for col in df.columns:
        if col != 'score':
            print(f"\n{col} vs score:")
            print(df.groupby(col)['score'].agg(['mean', 'min', 'std']))

    # Heatmap for 2 params
    if len(df.columns) == 3:  # 2 params + score
        param1, param2 = [c for c in df.columns if c != 'score']
        pivot = df.pivot(index=param1, columns=param2, values='score')
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot.values)
        plt.colorbar(label='Score')
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.savefig('grid_heatmap.png')

    return df
```

## Best Practices

1. **Start with random search**: Find reasonable range first
2. **Use small grids**: < 50 combinations ideal
3. **Coarse-to-fine**: Two-stage search
4. **Parallelize**: All evaluations are independent
5. **Log everything**: Track all trials
6. **Consider parameter importance**: Some params matter more
7. **Use for final tuning**: After narrowing range
8. **Set budget limits**: Prevent runaway experiments
