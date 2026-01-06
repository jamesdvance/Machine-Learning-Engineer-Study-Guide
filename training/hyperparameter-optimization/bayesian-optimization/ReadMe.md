# Bayesian Optimization

## Summary

Bayesian optimization is a sample-efficient approach to hyperparameter optimization that builds a probabilistic model of the objective function. It uses this model to intelligently select the next configuration to evaluate, balancing exploration of unknown regions with exploitation of promising areas. Bayesian optimization excels when evaluations are expensive.

Key points to remember:

- Builds surrogate model of objective function
- Uses acquisition function to select next point
- Sample-efficient: finds good solutions with few trials
- Best for expensive evaluations (hours per trial)
- Handles noisy objectives well
- Can incorporate prior knowledge
- Less parallelizable than random search
- Overhead matters for cheap evaluations

## How It Works

### Bayesian Optimization Loop

```
1. Initialize with random samples
2. Fit surrogate model (typically Gaussian Process)
3. Use acquisition function to find next point
4. Evaluate objective at selected point
5. Update surrogate model
6. Repeat 3-5 until budget exhausted
```

### Key Components

```python
# Surrogate Model: Approximates objective
#   - Gaussian Process (GP): Standard choice
#   - Tree Parzen Estimator (TPE): Optuna default
#   - Random Forest: SMAC

# Acquisition Function: Balances exploration/exploitation
#   - Expected Improvement (EI): Most common
#   - Probability of Improvement (PI)
#   - Upper Confidence Bound (UCB)
#   - Thompson Sampling
```

## Implementation with GPyOpt

### Basic Example

```python
import GPyOpt
import numpy as np

def objective(x):
    """Objective to minimize."""
    lr = 10 ** x[0, 0]  # x is 2D array
    batch_size = int(x[0, 1])

    # Train and return validation loss
    val_loss = train_model(learning_rate=lr, batch_size=batch_size)
    return val_loss

# Define domain
domain = [
    {'name': 'log_lr', 'type': 'continuous', 'domain': (-5, -2)},  # 1e-5 to 1e-2
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)}
]

# Run optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=domain,
    acquisition_type='EI',
    initial_design_numdata=5,
    maximize=False
)

optimizer.run_optimization(max_iter=50)

print(f"Best parameters: {optimizer.x_opt}")
print(f"Best value: {optimizer.fx_opt}")
```

### With scikit-optimize

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def objective(params):
    lr, batch_size, hidden_size, dropout = params
    val_loss = train_model(
        learning_rate=lr,
        batch_size=batch_size,
        hidden_size=hidden_size,
        dropout=dropout
    )
    return val_loss

# Define search space
space = [
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
    Integer(16, 256, name='batch_size'),
    Integer(128, 1024, name='hidden_size'),
    Real(0.0, 0.5, name='dropout')
]

# Run optimization
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    n_random_starts=10,
    random_state=42,
    verbose=True
)

print(f"Best parameters: {result.x}")
print(f"Best value: {result.fun}")
```

## Acquisition Functions

### Expected Improvement (EI)

```python
def expected_improvement(mean, std, best_so_far, xi=0.01):
    """
    Expected Improvement acquisition function.

    Args:
        mean: Predicted mean from surrogate model
        std: Predicted standard deviation
        best_so_far: Best observed value
        xi: Exploration-exploitation tradeoff
    """
    from scipy.stats import norm

    improvement = best_so_far - mean - xi
    Z = improvement / (std + 1e-8)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    return ei
```

### Upper Confidence Bound (UCB)

```python
def upper_confidence_bound(mean, std, kappa=2.0):
    """
    UCB acquisition function.

    Higher kappa = more exploration
    Lower kappa = more exploitation
    """
    # For minimization: LCB = mean - kappa * std
    return mean - kappa * std
```

### Comparison

| Acquisition | Exploration | Behavior |
|-------------|-------------|----------|
| EI | Balanced | Standard choice |
| PI | Lower | Exploitative |
| UCB | Controllable | kappa controls |
| Thompson Sampling | High | Good for parallel |

## Gaussian Process Details

### Kernel Selection

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

# Matern 5/2: Default choice, good for ML hyperparameters
kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel()

# RBF: Very smooth functions
kernel = RBF(length_scale=1.0) + WhiteKernel()

# Construct GP
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True,
    alpha=1e-6
)
```

### Handling Categorical Variables

```python
# Option 1: One-hot encoding
# Transform categorical to binary features before GP

# Option 2: Use TPE (Tree Parzen Estimator) instead
# Handles categorical naturally - this is what Optuna uses

# Option 3: Use specialized kernels
# Mixed kernels for continuous + categorical
```

## Advanced Techniques

### Multi-Fidelity Optimization

```python
from skopt import Optimizer

def multi_fidelity_objective(params, budget):
    """Train with varying budget (epochs)."""
    lr, batch_size = params
    val_loss = train_model(lr=lr, batch_size=batch_size, epochs=budget)
    return val_loss

# Use Hyperband-style scheduling
def hyperband_bayesian():
    optimizer = Optimizer(
        dimensions=[
            Real(1e-5, 1e-2, prior='log-uniform'),
            Integer(16, 256)
        ],
        base_estimator='GP',
        n_initial_points=10
    )

    # Successive halving with BO for suggestions
    configs = [optimizer.ask() for _ in range(27)]
    budget = 1

    while len(configs) > 1:
        # Evaluate
        results = [(c, multi_fidelity_objective(c, budget)) for c in configs]

        # Update optimizer
        for config, result in results:
            optimizer.tell(config, result)

        # Keep top 1/3
        results.sort(key=lambda x: x[1])
        configs = [r[0] for r in results[:len(results)//3]]
        budget *= 3

    return configs[0]
```

### Parallel Bayesian Optimization

```python
from skopt import Optimizer

def parallel_bayesian_opt(n_trials=50, n_parallel=4):
    """Run BO with parallel evaluations."""
    optimizer = Optimizer(
        dimensions=search_space,
        base_estimator='GP',
        acq_func='EI',
        n_initial_points=10
    )

    all_results = []

    while len(all_results) < n_trials:
        # Get batch of suggestions
        suggestions = optimizer.ask(n_points=n_parallel)

        # Evaluate in parallel
        results = parallel_evaluate(suggestions)

        # Update optimizer
        for suggestion, result in zip(suggestions, results):
            optimizer.tell(suggestion, result)
            all_results.append((suggestion, result))

    return min(all_results, key=lambda x: x[1])
```

### Warm Starting

```python
from skopt import Optimizer

# Start with known good configurations
initial_points = [
    [1e-4, 32],   # Previous best
    [1e-3, 64],   # Baseline
]
initial_results = [0.15, 0.18]  # Their scores

optimizer = Optimizer(dimensions=space)

# Warm start
for point, result in zip(initial_points, initial_results):
    optimizer.tell(point, result)

# Continue optimization
for _ in range(50):
    next_point = optimizer.ask()
    result = evaluate(next_point)
    optimizer.tell(next_point, result)
```

## When to Use Bayesian Optimization

### Good Use Cases

```
1. Expensive evaluations
   - Training takes hours/days
   - Limited compute budget

2. Low-dimensional spaces
   - < 20 hyperparameters
   - Continuous parameters

3. Sequential setting
   - Can't run many parallel trials

4. Need sample efficiency
   - < 100 total evaluations
```

### Avoid When

```
1. Cheap evaluations
   - Training takes seconds
   - Random search is fine

2. High-dimensional spaces
   - > 20 hyperparameters
   - GP doesn't scale well

3. Highly parallel setting
   - Many GPUs available
   - Random + early stopping better

4. Categorical-heavy spaces
   - Use TPE instead (Optuna)
```

## Visualization

```python
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

# Convergence plot
plot_convergence(result)

# Parameter importance
plot_objective(result, n_points=10)

# Evaluated points
plot_evaluations(result)
```

## Best Practices

1. **Start with random points**: 10-20 initial samples
2. **Use appropriate kernel**: Matern 5/2 for most cases
3. **Normalize inputs**: Scale to [0, 1]
4. **Handle noise**: Set appropriate noise level
5. **Watch for overfitting**: Surrogate to few points
6. **Use EI for exploration**: Default acquisition
7. **Warm start**: Leverage previous experiments
8. **Visualize**: Check surrogate model fit
