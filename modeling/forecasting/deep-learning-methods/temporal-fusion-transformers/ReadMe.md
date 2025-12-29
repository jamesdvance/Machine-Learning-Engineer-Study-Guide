# Temporal Fusion Transformers (TFT)

## Summary

The Temporal Fusion Transformer is a deep learning architecture designed for multi-horizon forecasting with interpretability. Developed by Google, TFT combines high-performance forecasting with the ability to explain which inputs matter and when. It handles multiple input types (static metadata, known future inputs, observed historical inputs), provides attention-based interpretability, and outputs probabilistic forecasts via quantile regression.

Key points to remember:

- TFT provides interpretable attention patterns showing which time steps and features matter
- Handles three input types: static metadata, known future inputs (e.g., holidays), and observed past values
- Uses variable selection networks to learn feature importance automatically
- Produces multi-horizon probabilistic forecasts via quantile regression
- Combines LSTMs for local temporal patterns with self-attention for long-range dependencies
- Computationally expensive compared to simpler models but provides rich insights
- Available in PyTorch Forecasting and GluonTS

## When to Use TFT

TFT is appropriate when:

- Interpretability is important alongside accuracy
- You have mixed input types (static, known future, observed past)
- Multiple related time series can be modeled together
- Probabilistic forecasts are valuable for decision-making
- You need to understand feature importance over time
- Computational resources are available (TFT is heavier than simpler models)

Consider alternatives when:

- Speed is critical (use N-BEATS or classical methods)
- You have limited data (classical methods may be more robust)
- Only univariate forecasting is needed (N-BEATS may be simpler)
- Maximum accuracy matters more than interpretability (ensemble methods)

## Architecture Overview

### Component Overview

```
                    Static Covariates
                           |
                           v
               +-------------------+
               | Variable Selection|
               +-------------------+
                     |       |
              +------+       +------+
              |                     |
    Historical Inputs       Known Future Inputs
              |                     |
              v                     v
    +-------------------+  +-------------------+
    | Variable Selection|  | Variable Selection|
    +-------------------+  +-------------------+
              |                     |
              +----------+----------+
                         |
                         v
               +-------------------+
               |   LSTM Encoder    |
               +-------------------+
                         |
                         v
               +-------------------+
               |   Gated Residual  |
               |     Networks      |
               +-------------------+
                         |
                         v
               +-------------------+
               | Interpretable     |
               | Multi-Head        |
               | Attention         |
               +-------------------+
                         |
                         v
               +-------------------+
               | Quantile Outputs  |
               +-------------------+
```

### Key Components

| Component | Purpose |
|-----------|---------|
| Variable Selection Networks | Learn which inputs matter |
| Gated Residual Networks (GRN) | Flexible non-linear processing with skip connections |
| LSTM Encoder-Decoder | Capture local temporal dependencies |
| Interpretable Multi-Head Attention | Long-range dependencies with explainable weights |
| Quantile Outputs | Probabilistic forecasts |

### Variable Selection

TFT uses gated variable selection to learn feature importance:

```
Raw Inputs (per time step)
        |
        v
[Embedding + Linear Transform]
        |
        v
[GRN for each variable]
        |
        v
[Softmax-weighted combination]
        |
        v
Selected representation
```

This provides interpretable feature importance weights that can be extracted after training.

### Gated Residual Networks

GRNs are TFT's building block, providing:
- Non-linear transformations
- Skip connections for gradient flow
- Gating mechanisms for selective activation

```python
# Conceptual GRN
def GRN(x, context=None):
    hidden = Dense(x, hidden_size)
    if context is not None:
        hidden = hidden + Dense(context, hidden_size)
    hidden = ELU(hidden)
    hidden = Dense(hidden, hidden_size)
    gate = Sigmoid(Dense(hidden, hidden_size))
    output = gate * Dense(hidden, input_size)
    return LayerNorm(x + output)
```

### Interpretable Attention

TFT modifies standard multi-head attention to be interpretable:

1. Attention weights are averaged across heads
2. The resulting weights show which historical time steps influence each forecast step
3. No positional encoding is needed (temporal order preserved by LSTM)

## Implementation with PyTorch Forecasting

### Data Preparation

```python
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

# Prepare DataFrame with required structure
df = pd.DataFrame({
    "time_idx": time_indices,           # Integer time index
    "series_id": series_ids,            # Group identifier
    "target": values,                   # Target variable
    # Static features
    "category": category_values,        # Static categorical
    "store_size": store_sizes,          # Static real
    # Known future inputs
    "month": months,                    # Known categorical
    "holiday": holiday_flags,           # Known real
    # Observed inputs
    "temperature": temperatures,        # Observed real
    "price": prices,                    # Observed real
})

# Create dataset
training = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["series_id"],
    max_encoder_length=60,              # Historical context
    max_prediction_length=7,            # Forecast horizon

    # Feature specification
    static_categoricals=["category"],
    static_reals=["store_size"],
    time_varying_known_categoricals=["month"],
    time_varying_known_reals=["holiday"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["target", "temperature", "price"],

    # Normalization
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set from training set specification
validation = TimeSeriesDataSet.from_dataset(
    training,
    df[df["time_idx"] > training_cutoff],
    min_prediction_idx=training_cutoff + 1
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=64)
val_dataloader = validation.to_dataloader(train=False, batch_size=64)
```

### Model Configuration

```python
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=32,                     # Size of hidden layers
    attention_head_size=4,              # Attention heads
    dropout=0.1,
    hidden_continuous_size=16,          # Size for continuous variable processing
    output_size=7,                      # Number of quantiles (default: 7)
    loss=QuantileLoss(),                # Quantile regression
    reduce_on_plateau_patience=4,
)

# Print number of parameters
print(f"Parameters: {model.size() / 1e6:.1f}M")
```

### Training

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10),
        ModelCheckpoint(monitor="val_loss"),
    ],
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

### Prediction

```python
# Best model from training
best_model = TemporalFusionTransformer.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)

# Predict on test data
predictions = best_model.predict(test_dataloader, mode="prediction")

# Get prediction with all quantiles
predictions_raw = best_model.predict(
    test_dataloader,
    mode="raw",
    return_x=True
)

# Access specific quantiles
median = predictions_raw.output[:, :, 3]  # Central quantile
lower = predictions_raw.output[:, :, 0]   # Lowest quantile
upper = predictions_raw.output[:, :, -1]  # Highest quantile
```

## Interpretability

### Variable Importance

```python
# Get variable importance
interpretation = best_model.interpret_output(predictions_raw, reduction="sum")

# Encoder variable importance (historical features)
encoder_importance = interpretation["encoder_variables"]
# Static variable importance
static_importance = interpretation["static_variables"]
# Decoder variable importance (known future features)
decoder_importance = interpretation["decoder_variables"]

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Encoder variables
axes[0].barh(list(encoder_importance.keys()), list(encoder_importance.values()))
axes[0].set_title("Encoder Variable Importance")

# Static variables
axes[1].barh(list(static_importance.keys()), list(static_importance.values()))
axes[1].set_title("Static Variable Importance")

# Decoder variables
axes[2].barh(list(decoder_importance.keys()), list(decoder_importance.values()))
axes[2].set_title("Decoder Variable Importance")

plt.tight_layout()
plt.savefig("variable_importance.png")
```

### Attention Patterns

```python
# Get attention weights
attention_weights = interpretation["attention"]

# Shape: (batch, prediction_length, encoder_length)
# Shows which historical time steps influence each prediction step

# Plot attention for a single sample
import seaborn as sns

sample_idx = 0
plt.figure(figsize=(12, 6))
sns.heatmap(
    attention_weights[sample_idx].numpy(),
    cmap="Blues",
    xticklabels=10,
    yticklabels=True
)
plt.xlabel("Encoder Time Step")
plt.ylabel("Prediction Step")
plt.title("Attention Weights")
plt.savefig("attention_weights.png")
```

### Partial Dependence

```python
# Analyze relationship between feature and prediction
partial_dependence = best_model.plot_prediction(
    predictions_raw.x,
    predictions_raw.output,
    idx=0,
    add_loss_to_title=True,
    show_future_observed=True,
)
```

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| hidden_size | 16-256 | Model capacity |
| attention_head_size | 1-8 | Attention expressiveness |
| dropout | 0.0-0.3 | Regularization |
| hidden_continuous_size | 8-64 | Continuous variable processing |
| learning_rate | 1e-4 to 1e-2 | Training speed/stability |
| max_encoder_length | Domain-specific | Historical context window |

### Automated Tuning

```python
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning.callbacks import EarlyStopping

# Create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=100,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(1e-4, 1e-2),
    dropout_range=(0.0, 0.3),
    trainer_kwargs=dict(
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        accelerator="gpu",
    ),
    reduce_on_plateau_patience=4,
)

# Best parameters
print(study.best_trial.params)
```

### Architecture Size Guidelines

| Dataset Size | hidden_size | attention_head_size | Recommendation |
|--------------|-------------|---------------------|----------------|
| Small (<1000) | 16-32 | 1-2 | Heavy regularization |
| Medium (1K-100K) | 32-64 | 2-4 | Default settings |
| Large (>100K) | 64-256 | 4-8 | Can increase capacity |

## Loss Functions

### Quantile Loss (Default)

```python
from pytorch_forecasting import QuantileLoss

# Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

model = TemporalFusionTransformer.from_dataset(
    training,
    loss=loss,
    output_size=3,  # Number of quantiles
)
```

### Alternative Losses

```python
from pytorch_forecasting import RMSE, MAE, SMAPE

# Point forecast (no prediction intervals)
model = TemporalFusionTransformer.from_dataset(
    training,
    loss=MAE(),
    output_size=1,
)

# Multiple loss combination
from pytorch_forecasting import MultiLoss
loss = MultiLoss([QuantileLoss(), MAE()])
```

## Handling Special Cases

### Multiple Target Variables

```python
from pytorch_forecasting import MultiNormalizer

# Multi-target setup
training = TimeSeriesDataSet(
    df,
    target=["sales", "revenue"],  # Multiple targets
    target_normalizer=MultiNormalizer([
        GroupNormalizer(groups=["series_id"]),
        GroupNormalizer(groups=["series_id"]),
    ]),
    ...
)
```

### Irregular Time Series

```python
# TFT requires regular intervals
# Pre-process to fill gaps
df = df.set_index("timestamp").resample("D").asfreq()
df = df.interpolate(method="linear")
df = df.reset_index()

# Or use time_idx that reflects actual intervals
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.days
```

### Cold Start (New Series)

```python
# TFT handles cold start via static features
# Ensure meaningful static features for new series

# Example: Use category embeddings
training = TimeSeriesDataSet(
    df,
    static_categoricals=["category", "region"],  # Help with cold start
    ...
)
```

## Common Pitfalls

### Pitfall 1: Incorrect Feature Classification

**Symptom**: Poor forecasts, feature importance doesn't make sense.

**Solution**: Carefully classify features:
- `time_varying_known_reals`: Known for future (holidays, day of week)
- `time_varying_unknown_reals`: Only observed in past (target, external observations)

### Pitfall 2: Data Leakage

**Symptom**: Unrealistically good validation performance.

**Solution**: Ensure unknown features aren't available at forecast time.

### Pitfall 3: Too Short Encoder Length

**Symptom**: Misses long-term patterns.

**Solution**: Encoder length should capture at least one full seasonal cycle.

### Pitfall 4: Not Normalizing Data

**Symptom**: Training instability, NaN losses.

**Solution**: Use GroupNormalizer or similar for the target and continuous inputs.

### Pitfall 5: Ignoring Attention Patterns

**Symptom**: Model is a black box.

**Solution**: Regularly inspect attention weights; they reveal model reasoning.

## TFT vs Alternatives

| Aspect | TFT | DeepAR | N-BEATS | Classical |
|--------|-----|--------|---------|-----------|
| Interpretability | High (attention + variables) | Low | Medium | High |
| Covariates | Excellent | Good | Limited | Limited |
| Probabilistic | Quantile regression | Full distribution | Via ensemble | Limited |
| Multi-series | Native | Native | Per series | Per series |
| Training speed | Slow | Moderate | Fast | Very fast |
| Inference speed | Moderate | Slow | Fast | Very fast |

**Guidance**:
- TFT when interpretability and covariates both matter
- DeepAR when you need full distributions and have many series
- N-BEATS for pure accuracy on univariate series
- Classical methods for speed and simplicity

## Production Considerations

### Model Export

```python
# Save best model
torch.save(best_model.state_dict(), "tft_model.pt")

# Or use PyTorch Lightning checkpointing
trainer.save_checkpoint("tft_checkpoint.ckpt")

# Load for inference
model = TemporalFusionTransformer.load_from_checkpoint("tft_checkpoint.ckpt")
```

### Inference Optimization

```python
# Batch predictions
model.eval()
with torch.no_grad():
    predictions = model.predict(
        test_dataloader,
        mode="prediction",
        n_samples=None,  # Deterministic for speed
    )

# Use GPU
model = model.to("cuda")
```

### Monitoring

```python
# Track prediction intervals
coverage = (actual >= lower) & (actual <= upper)
coverage_rate = coverage.mean()

if coverage_rate < 0.90:  # For 90% PI
    # Model may need retraining
    pass
```

## Key Takeaways

1. **Interpretability + accuracy**: TFT uniquely combines strong forecasting with explainable attention and feature importance.

2. **Feature types matter**: Correctly classifying features as static, known future, or observed past is critical.

3. **Rich input handling**: TFT excels when you have diverse inputs (metadata, calendar features, historical observations).

4. **Quantile outputs**: Built-in probabilistic forecasts via quantile regression.

5. **Computationally expensive**: TFT is heavier than alternatives; justified when interpretability matters.

6. **Inspect attention**: The attention weights are a powerful debugging and explanation tool.

7. **Group modeling**: Designed for multiple related series; knowledge transfers across groups.
