# Deep Learning Methods for Time Series Forecasting

## Summary

Deep learning has transformed time series forecasting, particularly for complex patterns, long sequences, and multi-series learning. Modern architectures like Temporal Fusion Transformers (TFT), N-BEATS, DeepAR, and TimesNet offer capabilities beyond classical methods: learning from many related series, handling complex covariates, producing rich probabilistic outputs, and discovering patterns automatically. However, they require more data and computation, and may not always outperform well-tuned classical methods.

Key points to remember:

- Deep learning excels with long series, complex patterns, and many related series
- N-BEATS: Pure accuracy for univariate forecasting, fast and simple
- DeepAR: Probabilistic forecasts across many related series
- TFT: Interpretable attention with rich covariate handling
- TimesNet: Multi-task architecture exploiting periodicities
- Always compare against classical baselines (ARIMA, ETS, Prophet)
- More data generally needed (hundreds to thousands of observations per series)
- Ensemble approaches often improve robustness

## When to Use Deep Learning

### Good Candidates

| Scenario | Why Deep Learning Helps |
|----------|------------------------|
| Many related series | Cross-series learning transfers patterns |
| Complex patterns | Non-linear relationships captured automatically |
| Rich covariates | Native handling of diverse input types |
| Long sequences | Attention/convolution captures long-range dependencies |
| Multi-task needs | Same architecture for forecasting, imputation, classification |
| Probabilistic requirements | Built-in uncertainty quantification |

### Poor Candidates

| Scenario | Why Classical May Be Better |
|----------|----------------------------|
| Single short series | Insufficient data for neural networks |
| Simple patterns | Classical methods are faster and sufficient |
| Maximum interpretability | Statistical models are more transparent |
| Minimal compute budget | Deep learning is resource-intensive |
| Need for stability | Classical methods are more predictable |

### Decision Framework

```
Start
  |
  v
How many series?
  |
  +-- One series --> Length > 500? --> Yes --> N-BEATS/Classical Ensemble
  |                       |
  |                       No --> Classical methods (ARIMA, ETS)
  |
  +-- Many series --> Continue
  |
  v
Need interpretability?
  |
  +-- Yes --> TFT (attention + variable importance)
  |
  +-- No --> Continue
  |
  v
Need probabilistic forecasts?
  |
  +-- Yes --> DeepAR (full distributions)
  |
  +-- No --> Continue
  |
  v
Clear periodicity in data?
  |
  +-- Yes --> TimesNet (exploits periods)
  |
  +-- No --> N-BEATS or TFT
```

## Architecture Comparison

### Feature Matrix

| Feature | N-BEATS | DeepAR | TFT | TimesNet |
|---------|---------|--------|-----|----------|
| Core mechanism | FC blocks | Autoregressive RNN | Transformer attention | 2D Conv on periods |
| Multi-series | Per series | Native | Native | Per series |
| Covariates | Limited | Yes | Excellent | Limited |
| Probabilistic | Via ensemble | Native (distribution) | Quantile regression | Via ensemble |
| Interpretability | Trend/season decomposition | Low | High (attention, variables) | Medium (periods) |
| Speed (train) | Fast | Moderate | Slow | Moderate |
| Speed (inference) | Fast | Slow (autoregressive) | Moderate | Fast |

### Architecture Details

**N-BEATS**: Stacks of fully connected blocks with backward (residual) and forward (forecast) outputs. Pure deep learning without time-series-specific components. Optional interpretable variant provides trend/seasonality decomposition.

**DeepAR**: Autoregressive LSTM that learns to output distribution parameters (e.g., mean and variance for Gaussian). Trains across many series, learning shared patterns while handling individual series characteristics.

**TFT**: Transformer with variable selection networks for automatic feature importance, gated residual networks for flexible processing, and interpretable attention showing which historical time steps matter for each forecast.

**TimesNet**: Discovers dominant periods via FFT, reshapes 1D series into 2D based on periods, and applies inception-style 2D convolutions to capture both intra-period and inter-period patterns.

## Implementation Considerations

### Data Requirements

| Model | Minimum per Series | Recommended | Notes |
|-------|-------------------|-------------|-------|
| N-BEATS | ~100 | 500+ | More for long horizons |
| DeepAR | ~20 (with many series) | 100+ | Leverages cross-series |
| TFT | ~100 | 500+ | Rich covariates help |
| TimesNet | 2x longest period | 5x longest period | Needs periodic structure |

### Computational Resources

| Model | GPU Recommended | Training Time | Memory |
|-------|-----------------|---------------|--------|
| N-BEATS | Optional | Minutes-hours | Low |
| DeepAR | Yes | Hours | Medium |
| TFT | Yes | Hours-days | High |
| TimesNet | Yes | Hours | Medium |

### Framework Options

| Framework | Models Available | Notes |
|-----------|-----------------|-------|
| PyTorch Forecasting | TFT, DeepAR, N-BEATS | Rich features, PyTorch Lightning |
| GluonTS | DeepAR, N-BEATS, Transformer | Amazon, MXNet/PyTorch |
| NeuralForecast | N-BEATS, TimesNet, TFT, many more | Unified API, fast |
| Darts | N-BEATS, TFT, many more | User-friendly, scikit-learn style |
| Time-Series-Library | TimesNet, Informer, etc. | Research implementations |

## Practical Workflow

### Step 1: Baseline with Classical Methods

Always start with classical baselines:

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

sf = StatsForecast(
    models=[AutoARIMA(season_length=12), AutoETS(season_length=12)],
    freq='M'
)
sf.fit(df)
baseline = sf.predict(h=12)
```

### Step 2: Select Deep Learning Approach

Based on data characteristics and requirements:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, DeepAR, TFT

# Choose models based on needs
models = []

# For pure accuracy (univariate)
models.append(NBEATS(h=12, input_size=36))

# For probabilistic forecasts (many series)
if num_series > 50:
    models.append(DeepAR(h=12, input_size=36))

# For interpretability with covariates
if has_covariates:
    models.append(TFT(h=12, input_size=36))

nf = NeuralForecast(models=models, freq='M')
nf.fit(df)
forecasts = nf.predict()
```

### Step 3: Ensemble for Robustness

```python
# Simple average ensemble
forecasts = nf.predict()
baseline = sf.predict(h=12)

# Combine
ensemble = (
    forecasts['NBEATS'] * 0.3 +
    forecasts['DeepAR'] * 0.3 +
    baseline['AutoARIMA'] * 0.2 +
    baseline['AutoETS'] * 0.2
)
```

### Step 4: Evaluate and Iterate

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Compare all models
results = {}
for model_name in ['NBEATS', 'DeepAR', 'AutoARIMA', 'ensemble']:
    results[model_name] = evaluate_forecast(actual, forecasts[model_name])

best_model = min(results, key=lambda x: results[x]['MAPE'])
```

## Common Challenges

### Challenge 1: Overfitting

**Symptoms**: Great training loss, poor test performance.

**Solutions**:
- Increase regularization (dropout, weight decay)
- Reduce model size (fewer layers, smaller hidden size)
- Add more training data
- Use early stopping

### Challenge 2: Training Instability

**Symptoms**: Loss spikes, NaN values, poor convergence.

**Solutions**:
- Normalize data (per-series normalization)
- Reduce learning rate
- Add gradient clipping
- Check for outliers in data

### Challenge 3: Slow Inference

**Symptoms**: Production latency too high.

**Solutions**:
- Use N-BEATS (direct multi-step)
- Reduce model size
- Batch predictions
- Use GPU for inference
- Consider distillation to smaller model

### Challenge 4: Cold Start

**Symptoms**: Poor forecasts for new series with little history.

**Solutions**:
- Use static features (DeepAR, TFT)
- Train hierarchical models
- Fall back to classical methods for new series

## Hyperparameter Recommendations

### General Guidelines

| Parameter | Conservative | Moderate | Aggressive |
|-----------|--------------|----------|------------|
| Hidden size | 16-32 | 32-64 | 64-256 |
| Layers | 1-2 | 2-3 | 3-5 |
| Dropout | 0.2-0.3 | 0.1-0.2 | 0.0-0.1 |
| Learning rate | 1e-4 | 1e-3 | 1e-2 |
| Batch size | 16-32 | 32-64 | 64-256 |

Start conservative and increase capacity if underfitting.

### Model-Specific Tips

**N-BEATS**:
- Use 30 stacks for competition settings
- Ensemble 5-10 models with different seeds
- Generic architecture often beats interpretable

**DeepAR**:
- Match distribution to data (Gaussian for continuous, NegBin for counts)
- Context length should cover seasonal cycles
- More series is better than longer series

**TFT**:
- Correctly classify features (known vs unknown)
- Inspect attention weights to verify model reasoning
- Consider variable selection for feature engineering insights

**TimesNet**:
- Verify data has clear periodicity
- Set top_k based on expected periods
- Input length should cover multiple cycles

## Production Deployment

### Model Serving

```python
# Save trained model
model.save("model_checkpoint")

# Load for inference
model = NBEATS.load("model_checkpoint")

# Batch prediction
predictions = model.predict(new_data)
```

### Monitoring

Track these metrics in production:

| Metric | Purpose | Alert Threshold |
|--------|---------|-----------------|
| MAPE | Accuracy | >2x training MAPE |
| Prediction interval coverage | Calibration | <80% for 90% PI |
| Inference latency | Performance | >SLA |
| Data drift | Input distribution | Significant shift |

### Retraining Strategy

| Strategy | When to Use |
|----------|------------|
| Scheduled (weekly/monthly) | Stable patterns |
| Triggered (performance drop) | Changing patterns |
| Online/incremental | High-frequency updates |
| Full retrain | Major distribution shifts |

## Further Reading

For detailed information on each deep learning method, see:

- [DeepAR](deepar/ReadMe.md) - Probabilistic forecasting with autoregressive RNNs
- [N-BEATS](n-beats/ReadMe.md) - Neural basis expansion for univariate forecasting
- [Temporal Fusion Transformers](temporal-fusion-transformers/ReadMe.md) - Interpretable multi-horizon forecasting
- [TimesNet](timesnet/ReadMe.md) - Multi-task learning with period-based 2D convolutions

For classical alternatives, see the Classical Methods section.
