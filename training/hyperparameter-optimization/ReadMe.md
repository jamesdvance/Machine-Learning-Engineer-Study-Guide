# Hyperparameter Optimization

## Summary

Hyperparameter optimization (HPO) systematically searches for the best configuration of training parameters that aren't learned during training. This includes learning rate, batch size, architecture choices, and regularization settings. Effective HPO can significantly improve model performance while automated methods reduce the manual tuning burden.

Key points to remember:

- Hyperparameters are set before training, not learned
- Random search outperforms grid search for most problems
- Bayesian optimization is sample-efficient for expensive evaluations
- Early stopping (Hyperband) reduces wasted compute
- Population-based training adapts during training
- Define search spaces carefully (log scale for LR)
- Use appropriate budgets for exploration
- Track all trials for analysis and reproducibility

## Types of Hyperparameters

### Training Hyperparameters

| Parameter | Typical Range | Scale |
|-----------|--------------|-------|
| Learning rate | 1e-5 to 1e-2 | Log |
| Batch size | 8 to 512 | Powers of 2 |
| Weight decay | 1e-6 to 1e-1 | Log |
| Warmup steps | 100 to 5000 | Linear |
| Dropout | 0.0 to 0.5 | Linear |

### Architecture Hyperparameters

| Parameter | Typical Range | Scale |
|-----------|--------------|-------|
| Hidden size | 128 to 4096 | Powers of 2 |
| Number of layers | 2 to 48 | Linear |
| Attention heads | 4 to 32 | Powers of 2 |
| Intermediate size | 2x to 8x hidden | Linear |

### Regularization Hyperparameters

| Parameter | Typical Range | Scale |
|-----------|--------------|-------|
| Label smoothing | 0.0 to 0.2 | Linear |
| Gradient clipping | 0.1 to 10.0 | Log |
| Layer dropout | 0.0 to 0.3 | Linear |

## Search Methods Comparison

### Quick Reference

| Method | Efficiency | Parallelism | Best For |
|--------|------------|-------------|----------|
| Grid Search | Low | High | Small, discrete spaces |
| Random Search | Medium | High | Exploration, baselines |
| Bayesian Optimization | High | Medium | Expensive evaluations |
| Hyperband | High | High | Many trials, early stopping |
| Population-Based Training | Very High | High | Long training runs |

### Method Selection

```
Budget: < 20 trials
  -> Bayesian optimization

Budget: 20-100 trials
  -> Random search or Hyperband

Budget: > 100 trials
  -> Random search with early stopping

Long training runs:
  -> Population-based training

Quick experiments:
  -> Random search
```

## Search Space Definition

### Good Practices

```python
# Use log scale for learning rate
learning_rate = loguniform(1e-5, 1e-2)

# Use powers of 2 for batch size
batch_size = choice([16, 32, 64, 128, 256])

# Use linear scale for probabilities
dropout = uniform(0.0, 0.5)

# Use categorical for discrete choices
optimizer = choice(['adam', 'adamw', 'sgd'])
```

### Common Mistakes

```python
# Bad: Linear scale for learning rate
learning_rate = uniform(1e-5, 1e-2)  # Most samples near 1e-2

# Good: Log scale
learning_rate = loguniform(1e-5, 1e-2)  # Even coverage

# Bad: Too wide range
hidden_size = randint(1, 10000)

# Good: Constrained, reasonable range
hidden_size = choice([256, 512, 1024, 2048])
```

## Basic HPO Workflow

### Define Objective

```python
def objective(config):
    """Training objective for HPO."""
    model = create_model(
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Train
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

    return val_loss
```

### Run Search

```python
import optuna

study = optuna.create_study(direction='minimize')

def optuna_objective(trial):
    config = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden', [256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'weight_decay': trial.suggest_float('wd', 1e-6, 1e-2, log=True),
        'epochs': 10
    }
    return objective(config)

study.optimize(optuna_objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

## Distributed HPO

### Parallel Trials

```python
import ray
from ray import tune

ray.init()

analysis = tune.run(
    objective,
    config={
        'learning_rate': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([16, 32, 64, 128]),
    },
    num_samples=100,
    resources_per_trial={'cpu': 4, 'gpu': 1},
    max_concurrent_trials=8
)
```

### Multi-Node HPO

```python
# Connect to Ray cluster
ray.init(address='auto')

# Trials distributed across cluster
analysis = tune.run(
    objective,
    config=search_space,
    num_samples=1000
)
```

## Analysis and Visualization

### Parameter Importance

```python
import optuna

# Get parameter importance
importance = optuna.importance.get_param_importances(study)
print("Parameter importance:")
for param, imp in importance.items():
    print(f"  {param}: {imp:.3f}")
```

### Visualization

```python
import optuna.visualization as vis

# Plot optimization history
fig = vis.plot_optimization_history(study)

# Plot parameter importance
fig = vis.plot_param_importances(study)

# Plot parallel coordinates
fig = vis.plot_parallel_coordinate(study)

# Plot contour for 2 params
fig = vis.plot_contour(study, params=['lr', 'batch_size'])
```

### Best Practices for Analysis

```python
# Get top N trials
top_trials = sorted(study.trials, key=lambda t: t.value)[:10]

for trial in top_trials:
    print(f"Value: {trial.value:.4f}")
    print(f"Params: {trial.params}")
    print()

# Check for patterns
import pandas as pd
df = study.trials_dataframe()
print(df.describe())
```

## Integration with Training Frameworks

### PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = LitModel(learning_rate=lr)
    datamodule = DataModule(batch_size=batch_size)

    trainer = Trainer(
        max_epochs=10,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(trial, 'val_loss')
        ]
    )

    trainer.fit(model, datamodule)
    return trainer.callback_metrics['val_loss'].item()
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer
import optuna

def objective(trial):
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=trial.suggest_float('lr', 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical('batch', [8, 16, 32]),
        num_train_epochs=trial.suggest_int('epochs', 2, 5),
        weight_decay=trial.suggest_float('wd', 0.0, 0.3),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics['eval_loss']
```

## Best Practices

1. **Start with random search**: Good baseline, find reasonable ranges
2. **Use appropriate scales**: Log for LR, linear for dropout
3. **Define reasonable ranges**: Based on prior knowledge
4. **Use early stopping**: Prune bad trials quickly
5. **Track all trials**: For reproducibility and analysis
6. **Validate on held-out data**: Avoid overfitting to val set
7. **Consider compute budget**: Balance exploration vs depth
8. **Use warm starting**: Initialize from prior best

## Further Reading

Search methods:
- [Grid Search](grid-search/ReadMe.md): Exhaustive search
- [Random Search](random-search/ReadMe.md): Randomized exploration
- [Bayesian Optimization](bayesian-optimization/ReadMe.md): Sample-efficient search

Frameworks:
- [Optuna](optuna/ReadMe.md): Flexible HPO framework
- [Ray Tune](ray-tune/ReadMe.md): Distributed HPO

Advanced methods:
- [Population-Based Training](population-based-training/ReadMe.md): Adaptive during training
