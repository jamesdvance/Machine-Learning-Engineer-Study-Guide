# Time Series Forecasting

## Summary

Time series forecasting predicts future values based on historical temporal patterns. It is fundamental to ML engineering across domains: demand forecasting in retail, capacity planning in infrastructure, financial modeling, and energy optimization. This section covers both classical statistical methods (ARIMA, Exponential Smoothing, Prophet) and modern deep learning approaches (DeepAR, N-BEATS, TFT, TimesNet), along with essential concepts for understanding and validating time series models.

Key points to remember:

- Start with classical methods as baselines; they often perform surprisingly well
- Understand stationarity, seasonality, and trend before modeling
- Use proper time series cross-validation (never random splits)
- Deep learning shines with many series, complex patterns, or rich covariates
- Model selection depends on: data volume, interpretability needs, and computational budget
- Ensemble methods often outperform individual models
- Production forecasting requires monitoring for data drift and model degradation

## The Forecasting Landscape

### Method Categories

| Category | Methods | Best For |
|----------|---------|----------|
| Classical Statistical | ARIMA, ETS, Prophet | Single series, interpretability, speed |
| Deep Learning | DeepAR, N-BEATS, TFT, TimesNet | Many series, complex patterns, covariates |
| Machine Learning | XGBoost, LightGBM | Tabular features, fast iteration |
| Ensemble | Combinations | Production robustness |

### Choosing an Approach

```
How many related series?
|
+-- One/Few --> Classical methods (ARIMA, ETS, Prophet)
|                     |
|                     +-- Need interpretability? --> ETS/Prophet
|                     +-- Complex autocorrelation? --> ARIMA
|                     +-- Multiple seasonalities? --> Prophet
|
+-- Many (100+) --> Consider deep learning
                    |
                    +-- Need probabilistic forecasts? --> DeepAR
                    +-- Need interpretability? --> TFT
                    +-- Pure accuracy focus? --> N-BEATS
                    +-- Multi-task (forecast + other)? --> TimesNet
```

## Forecasting Workflow

### Step 1: Understand the Data

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

# Load and visualize
series = pd.read_csv('data.csv', index_col='date', parse_dates=True)['value']
series.plot(figsize=(12, 4), title='Time Series')
plt.show()

# Check stationarity
adf_result = adfuller(series.dropna())
print(f"ADF p-value: {adf_result[1]:.4f}")

# Decompose
stl = STL(series, period=12, robust=True)
result = stl.fit()
result.plot()
plt.tight_layout()
```

**Key questions**:
- Is there trend? Seasonality? Both?
- Is the series stationary?
- Are there outliers or missing values?
- What is the natural forecast horizon?

### Step 2: Establish Baselines

Always compare against simple baselines:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

def naive_forecast(train, horizon):
    """Last value repeated."""
    return np.full(horizon, train.iloc[-1])

def seasonal_naive(train, horizon, period):
    """Same period from last cycle."""
    return train.iloc[-period:].values[:horizon]

# Evaluate baselines
train, test = series[:-12], series[-12:]
naive_mae = mean_absolute_error(test, naive_forecast(train, 12))
seasonal_mae = mean_absolute_error(test, seasonal_naive(train, 12, 12))

print(f"Naive MAE: {naive_mae:.2f}")
print(f"Seasonal Naive MAE: {seasonal_mae:.2f}")
```

### Step 3: Fit Classical Models

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# Exponential Smoothing
ets_model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=True
)
ets_fitted = ets_model.fit()
ets_forecast = ets_fitted.forecast(12)

# ARIMA (automatic selection)
arima_model = auto_arima(
    train,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True
)
arima_forecast = arima_model.predict(n_periods=12)

# Evaluate
print(f"ETS MAE: {mean_absolute_error(test, ets_forecast):.2f}")
print(f"ARIMA MAE: {mean_absolute_error(test, arima_forecast):.2f}")
```

### Step 4: Consider Deep Learning (If Appropriate)

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

# Prepare data format
df = pd.DataFrame({
    'unique_id': ['series_1'] * len(series),
    'ds': series.index,
    'y': series.values
})

# Fit N-BEATS
model = NBEATS(h=12, input_size=36, max_steps=500)
nf = NeuralForecast(models=[model], freq='M')
nf.fit(df=df[:-12])
nbeats_forecast = nf.predict()

print(f"N-BEATS MAE: {mean_absolute_error(test, nbeats_forecast['NBEATS'].values):.2f}")
```

### Step 5: Validate with Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def evaluate_model(model_fn, series, n_splits=5, horizon=12):
    """Walk-forward cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
    scores = []

    for train_idx, test_idx in tscv.split(series):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]

        model = model_fn(train)
        forecast = model.forecast(len(test))

        scores.append(mean_absolute_error(test, forecast))

    return np.mean(scores), np.std(scores)

# Compare models with CV
ets_cv = evaluate_model(
    lambda x: ExponentialSmoothing(x, trend='add', seasonal='add',
                                    seasonal_periods=12, damped_trend=True).fit(),
    series
)
print(f"ETS CV MAE: {ets_cv[0]:.2f} (+/- {ets_cv[1]:.2f})")
```

### Step 6: Create Ensemble (Production)

```python
def ensemble_forecast(series, horizon):
    """Simple ensemble of classical methods."""
    # Fit multiple models
    ets = ExponentialSmoothing(
        series, trend='add', seasonal='add',
        seasonal_periods=12, damped_trend=True
    ).fit()

    arima = auto_arima(series, seasonal=True, m=12,
                       stepwise=True, suppress_warnings=True)

    # Combine forecasts
    ets_fc = ets.forecast(horizon)
    arima_fc = arima.predict(n_periods=horizon)

    # Simple average
    ensemble = (ets_fc + arima_fc) / 2

    return ensemble, {'ets': ets_fc, 'arima': arima_fc}

ensemble, components = ensemble_forecast(train, 12)
print(f"Ensemble MAE: {mean_absolute_error(test, ensemble):.2f}")
```

## Method Comparison

### Feature Matrix

| Feature | ARIMA | ETS | Prophet | N-BEATS | DeepAR | TFT |
|---------|-------|-----|---------|---------|--------|-----|
| Stationarity required | Yes | No | No | No | No | No |
| Multiple seasonality | Limited | No | Yes | Learned | Learned | Learned |
| Holidays | Manual | Manual | Built-in | Learned | Features | Features |
| Covariates | ARIMAX | Limited | Yes | No | Yes | Yes |
| Probabilistic | CI | PI | CI | Ensemble | Native | Quantiles |
| Multi-series | No | No | No | Per-series | Yes | Yes |
| Interpretability | Coefficients | Components | Components | Decompose | Low | Attention |
| Speed | Fast | Fast | Slow | Fast | Moderate | Slow |

### When to Use Each

| Scenario | Recommended Methods |
|----------|---------------------|
| Quick baseline | ETS, Seasonal Naive |
| Single series, interpretability | ARIMA, Prophet |
| Multiple seasonalities, holidays | Prophet |
| Many related series | DeepAR, TFT |
| Maximum accuracy (single series) | N-BEATS ensemble |
| Rich covariates + interpretability | TFT |
| Multi-task (forecast + imputation) | TimesNet |
| Production with robustness | Ensemble of classical + DL |

## Evaluation Metrics

### Common Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean(\|y - y_hat\|) | Interpretable units |
| RMSE | sqrt(mean((y - y_hat)^2)) | Penalizes large errors |
| MAPE | mean(\|y - y_hat\| / y) * 100 | Percentage, problematic near zero |
| sMAPE | mean(2\|y - y_hat\| / (\|y\| + \|y_hat\|)) * 100 | Symmetric version |
| MASE | MAE / naive_MAE | Scale-free, compares to naive |

### Choosing Metrics

```python
def comprehensive_metrics(actual, forecast, train_series):
    """Calculate multiple forecasting metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    # MASE: compared to naive forecast
    naive_errors = np.abs(np.diff(train_series))
    mase = mae / np.mean(naive_errors)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }
```

## Production Considerations

### Monitoring

Track these in production:

| Metric | Purpose | Alert When |
|--------|---------|------------|
| Forecast accuracy | Model performance | >2x historical error |
| Prediction interval coverage | Calibration | <80% for 90% PI |
| Data drift | Input changes | Significant distribution shift |
| Inference latency | System health | >SLA threshold |

### Retraining Strategies

| Strategy | When |
|----------|------|
| Scheduled | Weekly/monthly for stable patterns |
| Triggered | When monitoring detects degradation |
| Online | High-frequency streaming data |
| Seasonal | Before known pattern changes (holiday season) |

### Common Production Patterns

```python
class ForecastingPipeline:
    """Production forecasting pipeline."""

    def __init__(self, models, horizon):
        self.models = models
        self.horizon = horizon
        self.history = []

    def fit(self, series):
        """Fit all models."""
        for model in self.models:
            model.fit(series)

    def predict(self):
        """Generate ensemble forecast."""
        forecasts = [m.forecast(self.horizon) for m in self.models]
        ensemble = np.mean(forecasts, axis=0)
        return ensemble

    def update(self, new_observation):
        """Update with new data."""
        self.history.append(new_observation)
        # Refit periodically or on trigger

    def evaluate(self, actual):
        """Track performance."""
        predicted = self.predict()
        error = mean_absolute_error(actual, predicted)
        return error
```

## Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Random train/test splits | Data leakage, optimistic estimates | Use time-ordered splits |
| Ignoring stationarity | Invalid ARIMA, poor forecasts | Test and transform |
| Wrong seasonality period | Misses patterns | Verify with ACF, domain knowledge |
| Over-engineering simple problems | Wasted effort, complexity | Start simple, add complexity if needed |
| Single train/test split | Unreliable evaluation | Use multiple CV folds |
| Ignoring baselines | Can't assess improvement | Always compare to naive |

## Further Reading

For detailed information on forecasting methods and concepts, see:

### Methods
- [Classical Methods](classical-methods/ReadMe.md) - ARIMA, Exponential Smoothing, Prophet
- [Deep Learning Methods](deep-learning-methods/ReadMe.md) - DeepAR, N-BEATS, TFT, TimesNet

### Concepts
- [Concepts](concepts/ReadMe.md) - Stationarity, seasonality, trend decomposition, cross-validation
