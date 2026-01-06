# Random Search

## Summary

Random search samples hyperparameter configurations randomly from specified distributions. Despite its simplicity, random search often outperforms grid search, especially when some parameters are more important than others. It provides good coverage of the search space and is highly parallelizable.

Key points to remember:

- Samples randomly from parameter distributions
- More efficient than grid for high-dimensional spaces
- Better coverage when parameters have different importances
- Highly parallelizable (independent samples)
- Easy to add more samples incrementally
- Works well with early stopping
- Good baseline for comparison
- Simple to implement and understand

## Why Random Beats Grid

### The Bergstra & Bengio Insight

```
When only a subset of hyperparameters matter:

Grid Search (9 samples):     Random Search (9 samples):
Only 3 unique values         Up to 9 unique values
for the important param      for the important param

     Important Param              Important Param
     |  |  |                         |      | |
[x] [x] [x]                      [x]     [x]  [x]
[x] [x] [x]                         [x]      [x]
[x] [x] [x]                      [x]    [x] [x]
Unimportant                       Unimportant

Random provides more unique samples along each dimension.
```

## Basic Implementation

### Simple Random Search

```python
import random
import numpy as np

def random_search(search_space, objective_fn, n_trials=50):
    """Simple random search implementation."""
    results = []

    for trial in range(n_trials):
        # Sample configuration
        config = sample_config(search_space)

        # Evaluate
        score = objective_fn(config)
        results.append((config, score))

        print(f"Trial {trial}: score={score:.4f}")

    # Find best
    best = min(results, key=lambda x: x[1])
    return best, results

def sample_config(search_space):
    """Sample a configuration from search space."""
    config = {}
    for param, spec in search_space.items():
        if spec['type'] == 'float':
            if spec.get('log', False):
                log_val = np.random.uniform(np.log(spec['min']), np.log(spec['max']))
                config[param] = np.exp(log_val)
            else:
                config[param] = np.random.uniform(spec['min'], spec['max'])
        elif spec['type'] == 'int':
            config[param] = np.random.randint(spec['min'], spec['max'] + 1)
        elif spec['type'] == 'choice':
            config[param] = random.choice(spec['values'])
    return config

# Define search space
search_space = {
    'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'log': True},
    'batch_size': {'type': 'choice', 'values': [16, 32, 64, 128]},
    'hidden_size': {'type': 'int', 'min': 128, 'max': 1024},
    'dropout': {'type': 'float', 'min': 0.0, 'max': 0.5},
}

best_config, all_results = random_search(search_space, objective, n_trials=50)
```

### Sklearn RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint

# Define distributions
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'min_samples_split': randint(2, 20),
    'subsample': uniform(0.5, 0.5),
}

# Run random search
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

## Search Space Design

### Continuous Parameters

```python
# Learning rate: log scale (spans orders of magnitude)
'learning_rate': {
    'type': 'float',
    'min': 1e-5,
    'max': 1e-2,
    'log': True  # Sample in log space
}

# Dropout: linear scale (bounded 0-1)
'dropout': {
    'type': 'float',
    'min': 0.0,
    'max': 0.5,
    'log': False
}

# Weight decay: log scale
'weight_decay': {
    'type': 'float',
    'min': 1e-6,
    'max': 1e-2,
    'log': True
}
```

### Discrete Parameters

```python
# Categorical choices
'optimizer': {
    'type': 'choice',
    'values': ['adam', 'adamw', 'sgd']
}

# Integer range
'num_layers': {
    'type': 'int',
    'min': 2,
    'max': 12
}

# Powers of 2
'batch_size': {
    'type': 'choice',
    'values': [8, 16, 32, 64, 128, 256]
}
```

### Conditional Parameters

```python
def sample_config_with_conditions(search_space):
    """Sample with conditional parameters."""
    config = {}

    # Sample optimizer
    config['optimizer'] = random.choice(['adam', 'sgd'])

    # Conditional: momentum only for SGD
    if config['optimizer'] == 'sgd':
        config['momentum'] = np.random.uniform(0.8, 0.99)
    else:
        config['betas'] = (0.9, 0.999)  # Adam default

    # Sample learning rate based on optimizer
    if config['optimizer'] == 'sgd':
        config['learning_rate'] = np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1)))
    else:
        config['learning_rate'] = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))

    return config
```

## Parallel Random Search

### With Multiprocessing

```python
from multiprocessing import Pool

def evaluate_config(config):
    """Evaluate a single configuration."""
    try:
        score = train_and_evaluate(config)
        return config, score
    except Exception as e:
        return config, float('inf')

def parallel_random_search(search_space, n_trials, n_workers=4):
    """Parallel random search."""
    # Generate all configurations
    configs = [sample_config(search_space) for _ in range(n_trials)]

    # Evaluate in parallel
    with Pool(n_workers) as pool:
        results = pool.map(evaluate_config, configs)

    return min(results, key=lambda x: x[1]), results
```

### With Ray

```python
import ray

@ray.remote
def evaluate_trial(config):
    return train_and_evaluate(config)

def ray_random_search(search_space, n_trials):
    """Distributed random search with Ray."""
    ray.init()

    configs = [sample_config(search_space) for _ in range(n_trials)]

    # Submit all trials
    futures = [evaluate_trial.remote(c) for c in configs]

    # Gather results
    scores = ray.get(futures)
    results = list(zip(configs, scores))

    ray.shutdown()
    return min(results, key=lambda x: x[1]), results
```

## Early Stopping Integration

### With Successive Halving

```python
def random_search_with_halving(search_space, objective_fn, n_initial=64, reduction_factor=3):
    """Random search with successive halving."""
    # Initial random configurations
    configs = [sample_config(search_space) for _ in range(n_initial)]
    budget = 1  # Initial budget (e.g., epochs)

    while len(configs) > 1:
        print(f"Evaluating {len(configs)} configs with budget {budget}")

        # Evaluate all configs with current budget
        results = [(c, objective_fn(c, budget=budget)) for c in configs]
        results.sort(key=lambda x: x[1])

        # Keep top 1/reduction_factor
        n_keep = max(1, len(configs) // reduction_factor)
        configs = [r[0] for r in results[:n_keep]]

        # Increase budget
        budget *= reduction_factor

    return configs[0], results[0][1]
```

### Trial Pruning

```python
class PruningRandomSearch:
    def __init__(self, search_space, objective_fn, n_trials, prune_fraction=0.5):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.prune_fraction = prune_fraction

    def run(self):
        results = []

        for trial in range(self.n_trials):
            config = sample_config(self.search_space)

            # Evaluate with early stopping
            score = self.objective_fn(config, early_stop_callback=self.should_prune)

            if score is not None:  # Not pruned
                results.append((config, score))

        return min(results, key=lambda x: x[1])

    def should_prune(self, intermediate_score, step):
        """Prune if worse than median of completed trials."""
        if len(self.results) < 5:
            return False
        median_score = np.median([r[1] for r in self.results])
        return intermediate_score > median_score * 1.5
```

## Analysis and Visualization

### Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_random_search(results):
    """Analyze random search results."""
    df = pd.DataFrame([
        {**config, 'score': score}
        for config, score in results
    ])

    # Best configurations
    print("Top 5 configurations:")
    print(df.nsmallest(5, 'score')[['score'] + list(results[0][0].keys())])

    # Parameter correlations with score
    print("\nParameter correlations with score:")
    for col in df.columns:
        if col != 'score' and df[col].dtype in ['int64', 'float64']:
            corr = df[col].corr(df['score'])
            print(f"  {col}: {corr:.3f}")

    # Plot convergence
    plt.figure(figsize=(10, 5))
    best_so_far = [min(df['score'][:i+1]) for i in range(len(df))]
    plt.plot(best_so_far)
    plt.xlabel('Trial')
    plt.ylabel('Best Score')
    plt.title('Random Search Convergence')
    plt.savefig('random_search_convergence.png')

    return df
```

### Parameter Importance Estimate

```python
def estimate_parameter_importance(results):
    """Estimate parameter importance from random search."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    df = pd.DataFrame([
        {**config, 'score': score}
        for config, score in results
    ])

    # Prepare features
    X = pd.get_dummies(df.drop('score', axis=1))
    y = df['score']

    # Fit random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get importances
    importances = dict(zip(X.columns, rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("Parameter importance:")
    for param, imp in sorted_imp:
        print(f"  {param}: {imp:.3f}")

    return sorted_imp
```

## Best Practices

1. **Use appropriate distributions**: Log scale for LR, linear for dropout
2. **Run sufficient trials**: 50-100 for good coverage
3. **Parallelize**: All samples are independent
4. **Combine with early stopping**: Reduce wasted compute
5. **Analyze results**: Look for patterns in top configs
6. **Iterate**: Narrow ranges based on results
7. **Set random seed**: For reproducibility
8. **Log everything**: Track all trials

## Comparison with Other Methods

| Aspect | Random Search | Grid Search | Bayesian |
|--------|---------------|-------------|----------|
| Efficiency | Medium | Low | High |
| Parallelism | High | High | Medium |
| Implementation | Simple | Simple | Complex |
| Coverage | Good | Even | Targeted |
| Scaling | Good | Poor | Good |
| Best for | Exploration | Small spaces | Few trials |
