# N-BEATS (Neural Basis Expansion Analysis for Time Series)

## Summary

N-BEATS is a deep learning architecture for univariate time series forecasting that achieved state-of-the-art results on the M3 and M4 forecasting competitions. It uses a stack of fully connected networks with a unique backward and forward residual connection structure. N-BEATS is notable for being "pure" deep learning with no time-series-specific components, yet it matches or beats specialized statistical methods. The interpretable variant decomposes forecasts into trend and seasonality components.

Key points to remember:

- Architecture uses stacks of blocks with backward (to input) and forward (to output) residual connections
- Two variants: Generic (maximum flexibility) and Interpretable (trend/seasonality decomposition)
- No recurrence, convolutions, or attention; just fully connected layers
- Direct multi-step forecasting (outputs all horizons at once)
- Designed for univariate series; does not natively handle covariates
- Fast training and inference compared to RNN/Transformer models
- Works well on single series without requiring many related series
- Strong benchmark performance, especially in ensemble configuration

## When to Use N-BEATS

N-BEATS excels when:

- You need high accuracy on univariate forecasting
- Speed is important (faster than RNN/Transformer methods)
- You want a robust baseline for forecasting competitions
- Interpretable decomposition would be valuable
- The series is long enough for deep learning (hundreds+ of observations)
- Multi-step ahead forecasting is required

Consider alternatives when:

- You have many related series to leverage (use DeepAR or TFT)
- Covariates are critical (N-BEATS has limited covariate support)
- Probabilistic forecasts are essential (use DeepAR or TFT)
- Series is very short (use classical methods)
- You need attention to specific time steps (use Transformers)

## Architecture

### The Block Structure

Each N-BEATS block contains:

1. **Fully connected stack**: 4 FC layers with ReLU activation
2. **Two linear projections**:
   - Backcast: Reconstructs the input (for residual learning)
   - Forecast: Produces the output prediction

```
Input (lookback window)
        |
        v
   [FC -> ReLU] x 4
        |
   +---------+
   |         |
   v         v
Backcast   Forecast
   |         |
   v         |
Input - Backcast --> Next Block
             |
             v
        Accumulate Forecasts
```

### The Stack Structure

Multiple blocks are organized into stacks:

```
Input
  |
  v
[Block 1] --> residual --> [Block 2] --> residual --> [Block 3]
    |                          |                          |
    v                          v                          v
forecast_1               forecast_2                 forecast_3
    |                          |                          |
    +----------+---------------+
               |
               v
         Sum of Forecasts
```

The backcast from each block is subtracted from the input before passing to the next block. This creates a hierarchical decomposition where each block focuses on what previous blocks couldn't explain.

### Generic vs Interpretable

**Generic N-BEATS**:
- Backcast and forecast are unconstrained linear projections
- Maximum flexibility for complex patterns
- Less interpretable but often more accurate

**Interpretable N-BEATS**:
- Constrained basis functions for backcast and forecast
- Trend stack: polynomial basis (linear, quadratic, etc.)
- Seasonality stack: Fourier basis (sine/cosine terms)
- Outputs decomposition: forecast = trend + seasonality

```python
# Interpretable: Polynomial trend basis
theta_trend = FC(hidden_state)  # [batch, polynomial_degree]
trend = theta_trend @ polynomial_basis  # [batch, horizon]

# Interpretable: Fourier seasonality basis
theta_season = FC(hidden_state)  # [batch, 2*num_harmonics]
seasonality = theta_season @ fourier_basis  # [batch, horizon]
```

## Implementation with Darts

```python
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import pandas as pd

# Prepare data
series = TimeSeries.from_dataframe(df, 'date', 'value')

# Scale data
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# Split
train, test = series_scaled.split_before(0.8)

# Configure model
model = NBEATSModel(
    input_chunk_length=30,      # Lookback window
    output_chunk_length=7,      # Forecast horizon
    generic_architecture=True,  # True for generic, False for interpretable
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    n_epochs=100,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu"}
)

# Train
model.fit(train, verbose=True)

# Predict
pred = model.predict(n=len(test))

# Inverse transform
pred = scaler.inverse_transform(pred)
```

### Interpretable Configuration

```python
model = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    generic_architecture=False,  # Interpretable
    num_stacks=2,                # Typically 2: trend + seasonality
    num_blocks=3,                # Blocks per stack
    num_layers=4,
    layer_widths=256,
    trend_polynomial_degree=3,   # Polynomial degree for trend
    n_epochs=100,
)
```

## Implementation with GluonTS

```python
from gluonts.mx.model.n_beats import NBEATSEstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset

# Prepare data
training_data = ListDataset(
    [{"start": start_date, "target": values}],
    freq="D"
)

# Configure estimator
estimator = NBEATSEstimator(
    freq="D",
    prediction_length=7,
    context_length=30,
    num_stacks=30,
    num_blocks=[1],
    widths=[256],
    sharing=[False],
    expansion_coefficient_lengths=[32],
    trainer=Trainer(epochs=100, learning_rate=1e-3)
)

# Train
predictor = estimator.train(training_data)
```

## Implementation with NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
import pandas as pd

# Prepare data (long format with unique_id, ds, y columns)
df = pd.DataFrame({
    'unique_id': ['series_1'] * len(values),
    'ds': dates,
    'y': values
})

# Configure model
model = NBEATS(
    h=7,                        # Forecast horizon
    input_size=30,              # Lookback
    stack_types=['generic'] * 3,
    n_blocks=[1, 1, 1],
    mlp_units=[[256, 256], [256, 256], [256, 256]],
    max_steps=1000,
    learning_rate=1e-3,
)

# Train
nf = NeuralForecast(models=[model], freq='D')
nf.fit(df=df)

# Predict
forecasts = nf.predict()
```

## Hyperparameters

### Key Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| num_stacks | 2-30 | More stacks = more capacity |
| num_blocks | 1-3 per stack | More blocks = deeper hierarchy |
| num_layers | 2-4 | FC layers per block |
| layer_widths | 128-512 | Network width |
| input_chunk_length | 2-7x output length | Longer = more context |
| output_chunk_length | Application-specific | Direct forecast horizon |

### Architecture Recommendations

**For M4-like competition settings**:
```python
# Generic configuration (30 stacks, 1 block each)
num_stacks=30
num_blocks=1
num_layers=4
layer_widths=512
```

**For interpretable decomposition**:
```python
# Two stacks: trend and seasonality
num_stacks=2
num_blocks=3
generic_architecture=False
trend_polynomial_degree=3
```

**For fast training**:
```python
# Smaller architecture
num_stacks=4
num_blocks=1
layer_widths=128
```

### Input/Output Length

```python
# Rule of thumb
input_chunk_length = 3 * output_chunk_length  # At minimum
input_chunk_length = seasonality_period + output_chunk_length  # Capture seasonality

# Examples
# Weekly forecast with daily data
input_chunk_length = 28  # 4 weeks
output_chunk_length = 7

# Monthly forecast with daily data
input_chunk_length = 90  # 3 months
output_chunk_length = 30
```

## Ensemble Methods

N-BEATS performs best in ensemble configurations:

### Bagging Ensemble

```python
from darts.models import NBEATSModel
import numpy as np

# Train multiple models with different seeds
models = []
for i in range(5):
    model = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=7,
        random_state=i,
        n_epochs=100,
    )
    model.fit(train)
    models.append(model)

# Ensemble prediction
predictions = [m.predict(n=7) for m in models]
ensemble_mean = sum(predictions) / len(predictions)

# Uncertainty from ensemble spread
ensemble_std = np.std([p.values() for p in predictions], axis=0)
```

### Generic + Interpretable Ensemble

```python
# Combine both architectures
generic_model = NBEATSModel(generic_architecture=True, ...)
interp_model = NBEATSModel(generic_architecture=False, ...)

generic_model.fit(train)
interp_model.fit(train)

pred_generic = generic_model.predict(n=7)
pred_interp = interp_model.predict(n=7)

ensemble = (pred_generic + pred_interp) / 2
```

## Decomposition Analysis

For interpretable N-BEATS, extract trend and seasonality:

```python
# With NeuralForecast
from neuralforecast.models import NBEATS

model = NBEATS(
    h=7,
    input_size=30,
    stack_types=['trend', 'seasonality'],
    n_blocks=[3, 3],
)

# Access decomposed outputs
# (Implementation depends on library)
```

```python
# Manual decomposition with separate models
trend_model = NBEATSModel(
    generic_architecture=False,
    trend_polynomial_degree=2,
    num_stacks=1,
)

seasonality_model = NBEATSModel(
    generic_architecture=False,
    trend_polynomial_degree=0,  # No trend in this stack
    num_stacks=1,
)

# Train on same data, get components
```

## Common Pitfalls

### Pitfall 1: Insufficient Input Length

**Symptom**: Misses seasonal patterns.

**Solution**: Input length should be at least 2x the longest seasonal period.

### Pitfall 2: Overfitting on Short Series

**Symptom**: Great training loss, poor test performance.

**Solution**: Reduce model size (fewer stacks, smaller widths) or use regularization.

### Pitfall 3: Not Using Ensembles

**Symptom**: High variance in forecasts.

**Solution**: N-BEATS benefits significantly from ensembling (5-10 models).

### Pitfall 4: Expecting Covariates

**Symptom**: Cannot incorporate external features.

**Solution**: N-BEATS is designed for univariate. Use TFT or DeepAR for covariates.

### Pitfall 5: Ignoring Data Scaling

**Symptom**: Training instability, poor convergence.

**Solution**: Always scale data before training (standardization or min-max).

## Performance Comparison

### M4 Competition Results

N-BEATS achieved strong results on the M4 competition:

| Horizon | N-BEATS | ETS | ARIMA |
|---------|---------|-----|-------|
| Yearly | Competitive | Strong | Moderate |
| Quarterly | Strong | Strong | Moderate |
| Monthly | Strong | Moderate | Moderate |
| Weekly | Very Strong | Moderate | Weak |
| Daily | Very Strong | Weak | Weak |

N-BEATS particularly excels on higher-frequency data (daily, weekly).

### Speed Comparison

| Model | Training (1000 obs) | Inference |
|-------|---------------------|-----------|
| N-BEATS | ~30s | ~10ms |
| DeepAR | ~60s | ~100ms |
| TFT | ~120s | ~50ms |
| ARIMA | ~1s | ~1ms |

N-BEATS offers a good balance of accuracy and speed.

## N-BEATS vs Alternatives

| Aspect | N-BEATS | DeepAR | TFT | Classical |
|--------|---------|--------|-----|-----------|
| Target | Univariate | Many series | Many series | Univariate |
| Covariates | Limited | Yes | Yes | Limited |
| Probabilistic | Via ensemble | Native | Native | Via bootstrap |
| Interpretable | Optional | No | Yes | Yes |
| Speed | Fast | Moderate | Slow | Very fast |
| Architecture | FC only | RNN | Transformer | Statistical |

**Guidance**:
- N-BEATS for maximum accuracy on single/few series without covariates
- DeepAR when you have many related series
- TFT when covariates and interpretability both matter
- Classical for baseline and maximum speed

## Key Takeaways

1. **Simple but effective**: Pure fully connected networks with clever residual design outperform complex architectures.

2. **Direct forecasting**: Outputs all horizons at once, avoiding error accumulation of autoregressive methods.

3. **Interpretable option**: The interpretable variant provides trend/seasonality decomposition.

4. **Ensemble is essential**: Single N-BEATS models have high variance; ensembles are much more robust.

5. **Univariate focus**: Designed for single series; use alternatives for covariates.

6. **Fast training and inference**: Significantly faster than RNN/Transformer alternatives.

7. **Strong baseline**: N-BEATS is a reliable benchmark for any forecasting project.
