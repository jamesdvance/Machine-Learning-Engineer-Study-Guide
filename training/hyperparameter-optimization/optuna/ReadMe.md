# Optuna

## Summary

Optuna is a flexible hyperparameter optimization framework that provides efficient sampling algorithms, pruning of unpromising trials, and a clean API for defining search spaces. It uses Tree-structured Parzen Estimator (TPE) by default, which handles both continuous and categorical parameters well. Optuna is widely adopted in the ML community for its ease of use and powerful features.

Key points to remember:

- TPE sampler by default (efficient for mixed spaces)
- Built-in pruning for early stopping
- Clean define-by-run API for search spaces
- Visualization and analysis tools
- Supports distributed optimization
- Framework integrations (PyTorch Lightning, etc.)
- Persistent storage for studies
- Easy to extend with custom samplers/pruners

## Basic Usage

### Simple Example

```python
import optuna

def objective(trial):
    # Define hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 128, 1024)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Train model
    model = create_model(hidden_size=hidden_size, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_loss = train_and_evaluate(model, optimizer, batch_size)
    return val_loss

# Create study
study = optuna.create_study(direction='minimize')

# Optimize
study.optimize(objective, n_trials=100)

# Results
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

### Search Space Definition

```python
def objective(trial):
    # Continuous: linear or log scale
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Integer
    n_layers = trial.suggest_int('n_layers', 1, 10)
    hidden = trial.suggest_int('hidden', 64, 512, step=64)  # Multiples of 64

    # Categorical
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'silu'])

    # Conditional
    if optimizer == 'sgd':
        momentum = trial.suggest_float('momentum', 0.5, 0.99)

    return train(lr, dropout, n_layers, hidden, optimizer, activation)
```

## Pruning

### MedianPruner

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    model = create_model(trial)

    for epoch in range(100):
        train_loss = train_epoch(model)
        val_loss = validate(model)

        # Report intermediate value
        trial.report(val_loss, epoch)

        # Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

# Create study with pruner
study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(
        n_startup_trials=5,     # Min trials before pruning
        n_warmup_steps=10,      # Min steps before pruning
        interval_steps=1         # Check every step
    )
)

study.optimize(objective, n_trials=100)
```

### Hyperband Pruner

```python
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3
    )
)
```

### Successive Halving

```python
from optuna.pruners import SuccessiveHalvingPruner

study = optuna.create_study(
    direction='minimize',
    pruner=SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=4,
        min_early_stopping_rate=0
    )
)
```

## Samplers

### TPE (Default)

```python
from optuna.samplers import TPESampler

sampler = TPESampler(
    n_startup_trials=10,  # Random samples before TPE
    multivariate=True,    # Consider parameter correlations
    seed=42
)

study = optuna.create_study(sampler=sampler)
```

### Other Samplers

```python
from optuna.samplers import (
    RandomSampler,
    GridSampler,
    CmaEsSampler,
    NSGAIISampler
)

# Random search
study = optuna.create_study(sampler=RandomSampler(seed=42))

# Grid search
search_space = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64]
}
study = optuna.create_study(sampler=GridSampler(search_space))

# CMA-ES (for continuous spaces)
study = optuna.create_study(sampler=CmaEsSampler(seed=42))

# Multi-objective
study = optuna.create_study(
    directions=['minimize', 'minimize'],  # Multiple objectives
    sampler=NSGAIISampler(seed=42)
)
```

## Framework Integrations

### PyTorch Lightning

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = LitModel(learning_rate=lr)
    datamodule = DataModule(batch_size=batch_size)

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor='val_loss')
        ],
        enable_progress_bar=False
    )

    trainer.fit(model, datamodule)
    return trainer.callback_metrics['val_loss'].item()

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100)
```

### Hugging Face

```python
from transformers import TrainingArguments, Trainer
import optuna

def objective(trial):
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=trial.suggest_float('lr', 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
        num_train_epochs=trial.suggest_int('epochs', 2, 5),
        weight_decay=trial.suggest_float('weight_decay', 0.0, 0.3),
        warmup_ratio=trial.suggest_float('warmup_ratio', 0.0, 0.2),
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

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## Distributed Optimization

### With SQLite

```python
import optuna

# Create study with storage
study = optuna.create_study(
    study_name='distributed-study',
    storage='sqlite:///study.db',
    load_if_exists=True,
    direction='minimize'
)

# Run on multiple workers (same command on each)
study.optimize(objective, n_trials=25)
```

### With PostgreSQL

```python
study = optuna.create_study(
    study_name='distributed-study',
    storage='postgresql://user:password@host/database',
    load_if_exists=True
)
```

### With Redis

```python
# Using optuna-distributed
from optuna_distributed import DistributedStudy

study = DistributedStudy.create_study(
    storage='redis://localhost:6379',
    study_name='my-study'
)
```

## Visualization

### Built-in Visualizations

```python
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Parallel coordinates
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Contour plot
fig = vis.plot_contour(study, params=['lr', 'batch_size'])
fig.show()

# Slice plot
fig = vis.plot_slice(study)
fig.show()

# Hyperparameter relationships
fig = vis.plot_edf(study)
fig.show()
```

### Analysis

```python
# Get all trials as DataFrame
df = study.trials_dataframe()
print(df.head())

# Best trials
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

# Top N trials
top_trials = sorted(study.trials, key=lambda t: t.value)[:5]
for trial in top_trials:
    print(f"  {trial.value:.4f}: {trial.params}")

# Parameter importance
importance = optuna.importance.get_param_importances(study)
for param, imp in importance.items():
    print(f"  {param}: {imp:.4f}")
```

## Advanced Features

### Multi-Objective Optimization

```python
def objective(trial):
    # Optimize both accuracy and latency
    hidden = trial.suggest_int('hidden', 64, 512)
    layers = trial.suggest_int('layers', 1, 5)

    accuracy = train_and_evaluate(hidden, layers)
    latency = measure_latency(hidden, layers)

    return accuracy, latency

study = optuna.create_study(
    directions=['maximize', 'minimize']
)
study.optimize(objective, n_trials=100)

# Pareto front
pareto_trials = [t for t in study.best_trials]
```

### User Attributes

```python
def objective(trial):
    config = {...}

    # Store additional info
    trial.set_user_attr('config', config)
    trial.set_user_attr('model_path', 'path/to/model')

    return val_loss

# Access later
for trial in study.trials:
    print(trial.user_attrs['config'])
```

### Callbacks

```python
def callback(study, trial):
    """Called after each trial."""
    print(f"Trial {trial.number}: {trial.value}")

    # Save best model
    if study.best_trial.number == trial.number:
        save_model(trial.user_attrs['model_path'])

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### Resume Study

```python
# Create or load existing study
study = optuna.create_study(
    study_name='my-study',
    storage='sqlite:///study.db',
    load_if_exists=True
)

# Continue optimization
study.optimize(objective, n_trials=50)
```

## Best Practices

1. **Use log scale for LR**: `log=True` for learning rates
2. **Enable pruning**: Saves compute on bad trials
3. **Set seed for reproducibility**: `sampler=TPESampler(seed=42)`
4. **Use multivariate TPE**: `multivariate=True`
5. **Persist studies**: Use database storage
6. **Log user attributes**: Store metadata
7. **Visualize results**: Use built-in plots
8. **Start with random trials**: Default 10 is usually good
