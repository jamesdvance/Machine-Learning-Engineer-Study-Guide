# Population-Based Training

## Summary

Population-Based Training (PBT) combines hyperparameter optimization with training by maintaining a population of models that share information during training. Unlike traditional HPO that trains models independently, PBT periodically evaluates models, copies weights from better-performing models, and perturbs hyperparameters. This enables adaptation of hyperparameters throughout training.

Key points to remember:

- Trains population of models in parallel
- Periodically exploits best models and explores new hyperparameters
- Adapts hyperparameters during training (not fixed)
- Can find learning rate schedules automatically
- Requires more compute but finds better configurations
- Best for long training runs
- Implemented in Ray Tune
- Combines training and HPO into single run

## How PBT Works

### The PBT Loop

```
1. Initialize population with random hyperparameters
2. Train all models for interval steps
3. Evaluate all models
4. For each model in bottom 20%:
   a. Replace weights with copy from top 20% (exploit)
   b. Randomly perturb hyperparameters (explore)
5. Return to step 2 until training complete
```

### Key Concepts

```
Population: Set of models trained in parallel
Exploit: Copy weights from better model
Explore: Perturb hyperparameters randomly
Truncation selection: Replace bottom X% with top X%
```

## Basic Implementation

### Manual PBT

```python
import torch
import random
import copy

class PBTTrainer:
    def __init__(self, create_model_fn, population_size=8):
        self.population_size = population_size
        self.population = []

        # Initialize population
        for _ in range(population_size):
            config = self.sample_config()
            model = create_model_fn(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            self.population.append({
                'model': model,
                'optimizer': optimizer,
                'config': config,
                'score': float('inf')
            })

    def sample_config(self):
        return {
            'lr': 10 ** random.uniform(-5, -2),
            'batch_size': random.choice([16, 32, 64, 128])
        }

    def perturb_config(self, config):
        """Perturb hyperparameters."""
        new_config = config.copy()

        # Perturb learning rate
        if random.random() < 0.5:
            new_config['lr'] *= random.choice([0.8, 1.2])
            new_config['lr'] = max(1e-6, min(1e-1, new_config['lr']))

        # Perturb batch size
        if random.random() < 0.5:
            idx = [16, 32, 64, 128].index(config['batch_size'])
            idx = max(0, min(3, idx + random.choice([-1, 1])))
            new_config['batch_size'] = [16, 32, 64, 128][idx]

        return new_config

    def exploit_and_explore(self):
        """Replace bottom performers with top performers."""
        # Sort by score
        sorted_pop = sorted(self.population, key=lambda x: x['score'])

        # Top and bottom 20%
        n_replace = max(1, self.population_size // 5)
        top = sorted_pop[:n_replace]
        bottom = sorted_pop[-n_replace:]

        for bad in bottom:
            # Select random top performer
            good = random.choice(top)

            # Copy weights (exploit)
            bad['model'].load_state_dict(
                copy.deepcopy(good['model'].state_dict())
            )

            # Copy optimizer state
            bad['optimizer'].load_state_dict(
                copy.deepcopy(good['optimizer'].state_dict())
            )

            # Perturb hyperparameters (explore)
            bad['config'] = self.perturb_config(good['config'])

            # Update learning rate in optimizer
            for param_group in bad['optimizer'].param_groups:
                param_group['lr'] = bad['config']['lr']

    def train(self, train_fn, eval_fn, total_steps, eval_interval):
        """Run PBT training."""
        for step in range(0, total_steps, eval_interval):
            # Train each member for interval
            for member in self.population:
                train_fn(member['model'], member['optimizer'],
                        member['config'], eval_interval)

            # Evaluate all
            for member in self.population:
                member['score'] = eval_fn(member['model'])

            # Exploit and explore
            if step > 0:  # Skip first interval
                self.exploit_and_explore()

            # Log progress
            best = min(self.population, key=lambda x: x['score'])
            print(f"Step {step}: Best score = {best['score']:.4f}")

        return min(self.population, key=lambda x: x['score'])
```

## Ray Tune PBT

### Basic Usage

```python
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

# Define perturbation space
pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    perturbation_interval=5,  # Perturb every 5 iterations
    hyperparam_mutations={
        'lr': tune.loguniform(1e-5, 1e-2),
        'batch_size': [16, 32, 64, 128],
    },
    quantile_fraction=0.25,  # Bottom 25% replaced
    resample_probability=0.25,  # 25% chance to resample
)

def train_fn(config):
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(100):
        train_epoch(model, optimizer, config['batch_size'])
        loss = validate(model)
        tune.report(loss=loss)

tuner = tune.Tuner(
    train_fn,
    param_space={
        'lr': 1e-3,  # Initial value
        'batch_size': 32,
    },
    tune_config=tune.TuneConfig(
        scheduler=pbt,
        num_samples=8,  # Population size
    )
)

results = tuner.fit()
```

### With Checkpointing

```python
import os
import torch
from ray import tune

def train_fn(config):
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    start_epoch = 0

    # Load checkpoint if available
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            state = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']

    for epoch in range(start_epoch, 100):
        train_epoch(model, optimizer, config['batch_size'])
        loss = validate(model)

        # Save checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        tune.report(loss=loss, training_iteration=epoch)
```

## Hyperparameter Schedule Discovery

### Adaptive Learning Rate

```python
pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    perturbation_interval=10,
    hyperparam_mutations={
        # LR can vary widely
        'lr': tune.loguniform(1e-6, 1e-2),
    }
)

# PBT naturally discovers schedules:
# - High LR early (fast learning)
# - Low LR late (fine-tuning)
```

### Extract Discovered Schedule

```python
from ray.tune import Analysis

analysis = Analysis('~/ray_results/pbt_experiment')

# Get best trial's hyperparameter history
best_trial = analysis.get_best_trial('loss', 'min')
config_history = best_trial.config_history

# Plot LR over training
lrs = [c['lr'] for c in config_history]
plt.plot(lrs)
plt.xlabel('Training iteration')
plt.ylabel('Learning rate')
plt.title('Discovered LR schedule')
```

## Advanced Configurations

### Custom Perturbation

```python
def custom_explore(config):
    """Custom perturbation function."""
    new_config = config.copy()

    # Multiplicative perturbation for LR
    new_config['lr'] *= random.choice([0.5, 0.8, 1.0, 1.25, 2.0])

    # Clip to valid range
    new_config['lr'] = max(1e-6, min(1e-1, new_config['lr']))

    # Additive perturbation for dropout
    new_config['dropout'] += random.choice([-0.1, 0, 0.1])
    new_config['dropout'] = max(0.0, min(0.5, new_config['dropout']))

    return new_config

pbt = PopulationBasedTraining(
    metric='loss',
    mode='min',
    perturbation_interval=5,
    custom_explore_fn=custom_explore
)
```

### Multiple Hyperparameters

```python
pbt = PopulationBasedTraining(
    metric='loss',
    mode='min',
    perturbation_interval=10,
    hyperparam_mutations={
        'lr': tune.loguniform(1e-5, 1e-2),
        'weight_decay': tune.loguniform(1e-6, 1e-2),
        'dropout': tune.uniform(0.0, 0.5),
        'batch_size': [8, 16, 32, 64, 128],
        'warmup_steps': tune.randint(0, 1000),
    }
)
```

## When to Use PBT

### Good Use Cases

```
1. Long training runs (hours to days)
   - Amortizes overhead of exploitation/exploration

2. Learning rate scheduling
   - Automatically discovers schedules

3. RL training
   - Hyperparameters interact with policy

4. When training is the bottleneck
   - No need for separate HPO phase
```

### Avoid When

```
1. Short training runs
   - Overhead not worth it

2. Small populations needed
   - PBT needs 8+ members

3. Limited compute
   - Requires parallel training

4. Need exact reproducibility
   - PBT is inherently stochastic
```

## Comparison with Other Methods

| Aspect | PBT | Random/Bayesian | ASHA |
|--------|-----|-----------------|------|
| Parallelism | High | Medium-High | High |
| Adaptation | Yes | No | No |
| Schedules | Discovers | Fixed | Fixed |
| Compute | High | Variable | Medium |
| Best for | Long runs | Any | Early stopping |

## Best Practices

1. **Population size**: 8-16 typically sufficient
2. **Perturbation interval**: 5-10% of total steps
3. **Quantile fraction**: 0.2-0.25 (20-25%)
4. **Checkpointing**: Required for exploit step
5. **Warm start**: Train briefly before first exploit
6. **Log hyperparameters**: Track schedule discovery
7. **Monitor diversity**: Ensure population doesn't collapse
8. **Use for LR**: Most effective for learning rate

## Implementation Tips

```python
# Ensure optimizer state is properly copied
for target_param, source_param in zip(
    target_optimizer.param_groups[0]['params'],
    source_optimizer.param_groups[0]['params']
):
    if source_param in source_optimizer.state:
        target_optimizer.state[target_param] = copy.deepcopy(
            source_optimizer.state[source_param]
        )

# Update hyperparameters in optimizer
for param_group in optimizer.param_groups:
    param_group['lr'] = new_config['lr']
    param_group['weight_decay'] = new_config['weight_decay']
```
