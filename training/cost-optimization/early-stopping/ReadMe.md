# Early Stopping

## Summary

Early stopping terminates training runs when they are unlikely to improve, saving compute resources. This applies both to individual training runs (stopping when validation loss plateaus) and to hyperparameter sweeps (stopping poor configurations early). Effective early stopping can reduce training costs by 30-50% without sacrificing model quality.

Key points to remember:

- Stop when validation metric stops improving
- Patience parameter controls stopping sensitivity
- Save best model before stopping
- Use for hyperparameter optimization (Hyperband)
- Monitor multiple metrics for robustness
- Consider training dynamics (early volatility)
- Balance stopping too early vs wasting compute
- Restore best weights after stopping

## Basic Early Stopping

### Manual Implementation

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, value, epoch):
        if self.mode == 'min':
            is_improvement = value < self.best_value - self.min_delta
        else:
            is_improvement = value > self.best_value + self.min_delta

        if is_improvement:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

# Usage
early_stopping = EarlyStopping(patience=5, mode='min')

for epoch in range(100):
    train_loss = train_epoch()
    val_loss = validate()

    if early_stopping(val_loss, epoch):
        print(f"Early stopping at epoch {epoch}")
        print(f"Best epoch was {early_stopping.best_epoch}")
        break
```

### With Best Model Saving

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=5, checkpoint_path='best_model.pt'):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # Restore best model
                model.load_state_dict(torch.load(self.checkpoint_path))
                return True
        return False
```

## Framework Integration

### PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    min_delta=0.001,
    verbose=True
)

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best',
    save_top_k=1,
    mode='min'
)

trainer = Trainer(
    callbacks=[early_stop, checkpoint],
    max_epochs=100
)
```

### Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)
```

### Keras

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=callbacks
)
```

## Hyperparameter Optimization Early Stopping

### Successive Halving

```python
def successive_halving(configs, budget_per_config, reduction_factor=3):
    """Run hyperparameter search with aggressive early stopping."""
    remaining = configs
    budget = budget_per_config

    while len(remaining) > 1:
        # Train each config for current budget
        results = []
        for config in remaining:
            val_loss = train_for_budget(config, budget)
            results.append((config, val_loss))

        # Keep top 1/r configs
        results.sort(key=lambda x: x[1])
        n_keep = max(1, len(remaining) // reduction_factor)
        remaining = [r[0] for r in results[:n_keep]]

        # Increase budget for next round
        budget *= reduction_factor

    return remaining[0]
```

### Hyperband

```python
def hyperband(get_config, max_budget, reduction_factor=3):
    """Hyperband algorithm for hyperparameter optimization."""
    import math

    s_max = int(math.log(max_budget) / math.log(reduction_factor))

    best_config = None
    best_loss = float('inf')

    for s in range(s_max + 1):
        n = int((s_max + 1) * (reduction_factor ** s) / (s + 1))
        r = max_budget / (reduction_factor ** s)

        # Generate n random configs
        configs = [get_config() for _ in range(n)]

        for i in range(s + 1):
            n_i = int(n / (reduction_factor ** i))
            r_i = r * (reduction_factor ** i)

            # Train and evaluate
            results = [(c, train_for_budget(c, r_i)) for c in configs]
            results.sort(key=lambda x: x[1])

            # Update best
            if results[0][1] < best_loss:
                best_loss = results[0][1]
                best_config = results[0][0]

            # Keep top configs for next round
            n_keep = max(1, int(n_i / reduction_factor))
            configs = [r[0] for r in results[:n_keep]]

    return best_config
```

### Optuna Pruning

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)

    model = create_model(lr=lr)

    for epoch in range(100):
        train_loss = train_epoch(model, batch_size)
        val_loss = validate(model)

        # Report intermediate value
        trial.report(val_loss, epoch)

        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

# Create study with pruner
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)

study.optimize(objective, n_trials=100)
```

## Cost Savings Analysis

### Estimate Savings

```python
def estimate_early_stopping_savings(
    epochs_without_stopping,
    epochs_with_stopping,
    cost_per_epoch
):
    """Calculate savings from early stopping."""
    cost_without = epochs_without_stopping * cost_per_epoch
    cost_with = epochs_with_stopping * cost_per_epoch
    savings = cost_without - cost_with
    savings_pct = (savings / cost_without) * 100

    print(f"Without early stopping: ${cost_without:.2f} ({epochs_without_stopping} epochs)")
    print(f"With early stopping: ${cost_with:.2f} ({epochs_with_stopping} epochs)")
    print(f"Savings: ${savings:.2f} ({savings_pct:.1f}%)")

# Example
estimate_early_stopping_savings(
    epochs_without_stopping=100,
    epochs_with_stopping=35,
    cost_per_epoch=10.0
)
```

## Advanced Strategies

### Multi-Metric Early Stopping

```python
class MultiMetricEarlyStopping:
    def __init__(self, metrics_config, patience=5):
        """
        metrics_config: list of (name, mode, weight)
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.counter = 0
        self.best_scores = {}

    def __call__(self, metrics):
        improvements = []

        for name, mode, weight in self.metrics_config:
            value = metrics[name]

            if name not in self.best_scores:
                self.best_scores[name] = value
                improvements.append(True)
            else:
                if mode == 'min':
                    improved = value < self.best_scores[name]
                else:
                    improved = value > self.best_scores[name]

                if improved:
                    self.best_scores[name] = value
                improvements.append(improved)

        if any(improvements):
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

### Warm-Up Period

```python
class EarlyStoppingWithWarmup:
    def __init__(self, patience=5, warmup_epochs=10):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, epoch):
        # Skip warmup period
        if epoch < self.warmup_epochs:
            return False

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
```

## Best Practices

1. **Set appropriate patience**: 5-10 epochs typical
2. **Use min_delta**: Ignore tiny improvements
3. **Save best model**: Don't lose best weights
4. **Monitor validation metric**: Not training metric
5. **Consider warmup**: Skip early volatile epochs
6. **Restore best weights**: After stopping
7. **Log stopping reason**: For debugging
8. **Combine with checkpointing**: Enable resumption

## When NOT to Use Early Stopping

- Learning rate schedules with planned drops
- Known required training duration
- Very small datasets (high variance)
- When all epochs are needed for curriculum
