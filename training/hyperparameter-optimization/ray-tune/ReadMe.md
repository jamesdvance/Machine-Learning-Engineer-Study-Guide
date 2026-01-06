# Ray Tune

## Summary

Ray Tune is a scalable hyperparameter tuning library built on Ray that supports a variety of search algorithms and scheduling strategies. It excels at distributed HPO, handling hundreds of parallel trials across clusters. Ray Tune integrates with popular ML frameworks and provides early stopping through schedulers like ASHA and Population-Based Training.

Key points to remember:

- Built on Ray for distributed execution
- Supports many search algorithms (Optuna, BO, etc.)
- ASHA scheduler for efficient early stopping
- Native support for PBT (Population-Based Training)
- Checkpoint and resume capabilities
- Framework integrations (PyTorch, TensorFlow, etc.)
- Scales to large clusters
- Flexible resource management

## Basic Usage

### Simple Example

```python
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air.config import RunConfig

def objective(config):
    # Training function
    model = create_model(
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )

    for epoch in range(10):
        loss = train_epoch(model, lr=config['lr'])
        tune.report(loss=loss, epoch=epoch)

# Define search space
config = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'hidden_size': tune.choice([128, 256, 512, 1024]),
    'dropout': tune.uniform(0.0, 0.5),
}

# Run tuning
tuner = Tuner(
    objective,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=50,
        metric='loss',
        mode='min',
    ),
    run_config=RunConfig(
        name='my_experiment'
    )
)

results = tuner.fit()
print(f"Best config: {results.get_best_result().config}")
```

### With Resource Specification

```python
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air.config import ScalingConfig

tuner = Tuner(
    tune.with_resources(
        objective,
        resources={'cpu': 4, 'gpu': 1}
    ),
    param_space=config,
    tune_config=TuneConfig(
        num_samples=50,
        max_concurrent_trials=8
    )
)
```

## Search Algorithms

### Optuna Integration

```python
from ray.tune.search.optuna import OptunaSearch

search_alg = OptunaSearch(
    metric='loss',
    mode='min',
    seed=42
)

tuner = Tuner(
    objective,
    param_space=config,
    tune_config=TuneConfig(
        search_alg=search_alg,
        num_samples=100
    )
)
```

### Bayesian Optimization

```python
from ray.tune.search.bayesopt import BayesOptSearch

# For continuous parameters only
config = {
    'lr': tune.uniform(1e-5, 1e-2),
    'dropout': tune.uniform(0.0, 0.5),
}

search_alg = BayesOptSearch(
    metric='loss',
    mode='min',
    random_search_steps=10
)

tuner = Tuner(
    objective,
    param_space=config,
    tune_config=TuneConfig(
        search_alg=search_alg,
        num_samples=50
    )
)
```

### HyperOpt

```python
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# HyperOpt style search space
space = {
    'lr': hp.loguniform('lr', -11.5, -4.6),  # ~1e-5 to 1e-2
    'hidden_size': hp.choice('hidden_size', [128, 256, 512]),
}

search_alg = HyperOptSearch(
    space=space,
    metric='loss',
    mode='min'
)
```

## Schedulers

### ASHA (Async Successive Halving)

```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    metric='loss',
    mode='min',
    max_t=100,          # Max epochs
    grace_period=10,     # Min epochs before pruning
    reduction_factor=3
)

tuner = Tuner(
    objective,
    param_space=config,
    tune_config=TuneConfig(
        scheduler=scheduler,
        num_samples=100
    )
)
```

### Hyperband

```python
from ray.tune.schedulers import HyperBandScheduler

scheduler = HyperBandScheduler(
    metric='loss',
    mode='min',
    max_t=100
)
```

### Population-Based Training

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    perturbation_interval=5,
    hyperparam_mutations={
        'lr': tune.loguniform(1e-5, 1e-2),
        'batch_size': [16, 32, 64, 128],
    },
    quantile_fraction=0.25,  # Bottom 25% replaced
    resample_probability=0.25,
)

tuner = Tuner(
    objective,
    param_space={
        'lr': 1e-3,
        'batch_size': 32
    },
    tune_config=TuneConfig(
        scheduler=pbt,
        num_samples=8  # Population size
    )
)
```

## Framework Integrations

### PyTorch

```python
from ray import tune
from ray.tune import Tuner
import torch

def train_fn(config):
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, optimizer)
        val_loss = validate(model)

        # Checkpoint for fault tolerance
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save(model.state_dict(), f'{checkpoint_dir}/model.pt')

        tune.report(loss=val_loss, epoch=epoch)

tuner = Tuner(
    train_fn,
    param_space={
        'lr': tune.loguniform(1e-5, 1e-2),
        'epochs': 100
    }
)
```

### PyTorch Lightning

```python
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import pytorch_lightning as pl

def train_fn(config):
    model = LitModel(lr=config['lr'])
    datamodule = DataModule(batch_size=config['batch_size'])

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={'loss': 'val_loss'},
                on='validation_end'
            )
        ]
    )

    trainer.fit(model, datamodule)

tuner = Tuner(
    train_fn,
    param_space={
        'lr': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([16, 32, 64])
    },
    tune_config=TuneConfig(
        num_samples=50,
        scheduler=ASHAScheduler(metric='loss', mode='min')
    )
)
```

### Hugging Face

```python
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from transformers import Trainer, TrainingArguments

def train_fn(config):
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=config['lr'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Report metrics after each epoch
    class TuneReportCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            tune.report(**metrics)

    trainer.add_callback(TuneReportCallback())
    trainer.train()

tuner = Tuner(
    train_fn,
    param_space={...}
)
```

## Distributed Tuning

### Multi-GPU Trials

```python
from ray import tune
from ray.tune import Tuner

tuner = Tuner(
    tune.with_resources(
        train_fn,
        resources={'cpu': 8, 'gpu': 4}  # 4 GPUs per trial
    ),
    tune_config=TuneConfig(
        num_samples=20,
        max_concurrent_trials=2  # 2 trials x 4 GPUs = 8 GPUs
    )
)
```

### Cluster Execution

```python
import ray

# Connect to existing cluster
ray.init(address='auto')

# Or start cluster
ray.init(
    num_cpus=64,
    num_gpus=8
)

tuner = Tuner(
    train_fn,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=100
    )
)

results = tuner.fit()
```

## Checkpointing and Resume

### Save Checkpoints

```python
from ray import tune
import os

def train_fn(config):
    # Load checkpoint if resuming
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            state = torch.load(os.path.join(checkpoint_dir, 'state.pt'))
            model.load_state_dict(state['model'])
            start_epoch = state['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 100):
        train_epoch(model)
        val_loss = validate(model)

        # Save checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch
            }, os.path.join(checkpoint_dir, 'state.pt'))

        tune.report(loss=val_loss)
```

### Resume Experiment

```python
from ray.tune import Tuner

# Resume from previous run
tuner = Tuner.restore(
    path='~/ray_results/my_experiment',
    trainable=train_fn,
    resume_unfinished=True,
    resume_errored=True
)

results = tuner.fit()
```

## Analysis

### Access Results

```python
from ray.tune import ResultGrid

results = tuner.fit()

# Best result
best_result = results.get_best_result(metric='loss', mode='min')
print(f"Best config: {best_result.config}")
print(f"Best loss: {best_result.metrics['loss']}")

# All results as DataFrame
df = results.get_dataframe()
print(df[['config/lr', 'loss']].head())

# Iterate over results
for result in results:
    print(f"{result.config}: {result.metrics['loss']}")
```

### Visualization

```python
from ray.tune import Analysis

analysis = Analysis('~/ray_results/my_experiment')

# Get best config
best_config = analysis.get_best_config(metric='loss', mode='min')

# Plot
df = analysis.dataframe()
df.plot(x='training_iteration', y='loss')
```

## Best Practices

1. **Use ASHA scheduler**: Efficient early stopping
2. **Set appropriate resources**: Match to your hardware
3. **Enable checkpointing**: For fault tolerance
4. **Use search algorithms**: OptunaSearch is good default
5. **Start with small population**: Then scale up
6. **Monitor with Dashboard**: `ray dashboard`
7. **Log to cloud storage**: For long experiments
8. **Set max_concurrent_trials**: Control parallelism
