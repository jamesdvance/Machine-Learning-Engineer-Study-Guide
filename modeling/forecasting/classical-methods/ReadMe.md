# Classical Forecasting Methods

## Summary

Classical forecasting methods form the foundation of time series prediction. These statistical approaches have been refined over decades and remain highly competitive with modern machine learning techniques, especially for univariate series and short to medium forecast horizons. Understanding ARIMA, Exponential Smoothing, and Prophet provides both practical forecasting tools and conceptual foundations for more advanced methods.

Key points to remember:

- Classical methods are fast, interpretable, and often surprisingly accurate
- ARIMA models autocorrelation structure; Exponential Smoothing decomposes into level/trend/seasonality; Prophet handles complex business patterns
- Start with classical methods as baselines before trying deep learning
- For most business forecasting, one of these three methods will be sufficient
- Automatic selection tools (auto_arima, ETS, Prophet defaults) make these methods accessible
- Classical methods assume patterns continue; they cannot capture regime changes or external causal relationships
- Computational efficiency allows running thousands of forecasts in production

## When to Use Classical Methods

Classical methods are the right choice when:

- You have a single univariate time series
- The series length is moderate (dozens to low thousands of observations)
- Forecast horizons are short to medium (days to a few seasonal cycles)
- Interpretability matters more than marginal accuracy gains
- Computational resources are limited or latency is critical
- You need reliable prediction intervals
- Historical patterns are expected to continue

Consider alternatives when:

- Multiple related series could inform predictions (use hierarchical methods or ML)
- Complex non-linear relationships exist (use gradient boosting, neural networks)
- Very long series with complex patterns exist (consider deep learning)
- Real-time streaming predictions are needed (use online learning methods)
- Causal relationships need to be modeled (use causal inference methods)

## Method Comparison

### Overview

| Aspect | ARIMA | Exponential Smoothing | Prophet |
|--------|-------|----------------------|---------|
| Core idea | Model autocorrelation | Weighted averages of past | Additive regression |
| Components | AR, I, MA, (SARIMA: seasonal) | Level, trend, seasonality | Trend, seasonality, holidays |
| Strengths | Autocorrelation, flexibility | Simplicity, speed | Multiple seasonalities, holidays |
| Weaknesses | Complex selection | Single seasonality | Slow, installation |
| Best for | General purpose, autocorrelated series | Clear patterns, speed | Business data with holidays |

### When to Choose Each

| Scenario | Recommended Method |
|----------|-------------------|
| Quick baseline forecast | Exponential Smoothing (ETS) |
| Strong autocorrelation in residuals | ARIMA/SARIMA |
| Multiple seasonal patterns | Prophet |
| Holiday effects important | Prophet |
| Sub-daily data without clear patterns | ARIMA |
| Daily/weekly business data | Prophet or ETS |
| Maximum speed needed | Exponential Smoothing |
| Missing data, outliers present | Prophet |
| Minimal configuration desired | Prophet |
| Maximum control over model | ARIMA |

## The Forecasting Workflow

### Step 1: Exploratory Analysis

Before selecting a method, understand your data:

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Visualize the series
series.plot(figsize=(12, 4), title='Time Series')
plt.show()

# Check for seasonality and trend
decomposition = seasonal_decompose(series, period=12)
decomposition.plot()
plt.show()

# Examine autocorrelation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, ax=axes[0], lags=40)
plot_pacf(series, ax=axes[1], lags=40)
plt.show()
```

**Key questions**:
- Is there a trend? (Rising/falling over time)
- Is there seasonality? (Repeating patterns)
- Are there level shifts or structural breaks?
- Are there obvious outliers?
- Is the variance constant or changing?

### Step 2: Method Selection

Based on exploratory analysis:

| Observation | Suggested Approach |
|-------------|-------------------|
| Strong trend | Holt's method, ARIMA with d>0, Prophet |
| Clear seasonality | SARIMA, Holt-Winters, Prophet |
| Multiple seasonalities | Prophet |
| Holidays matter | Prophet |
| Strong ACF/PACF patterns | ARIMA |
| Simple, stable patterns | Exponential Smoothing |
| Missing values | Prophet |
| Need maximum speed | Exponential Smoothing |

### Step 3: Model Fitting

#### Exponential Smoothing (Fast Default)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    series,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=True
)
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```

#### ARIMA (Automatic Selection)

```python
from pmdarima import auto_arima

model = auto_arima(
    series,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True
)
forecast = model.predict(n_periods=12)
```

#### Prophet (Business Forecasting)

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.add_country_holidays(country_name='US')
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

### Step 4: Validation

Always validate with proper time series cross-validation:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def time_series_cv(series, model_fn, n_splits=5, horizon=1):
    """Walk-forward validation"""
    n = len(series)
    fold_size = n // (n_splits + 1)

    errors = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = min(train_end + horizon, n)

        train = series[:train_end]
        test = series[train_end:test_end]

        # Fit and forecast
        model = model_fn(train)
        forecast = model.forecast(len(test))

        errors.append(mean_absolute_error(test, forecast))

    return np.mean(errors), np.std(errors)
```

### Step 5: Model Comparison

Compare multiple methods systematically:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

results = {}

# Exponential Smoothing
try:
    es_model = ExponentialSmoothing(train, trend='add', seasonal='add',
                                     seasonal_periods=12, damped_trend=True)
    es_fitted = es_model.fit()
    es_forecast = es_fitted.forecast(len(test))
    results['ETS'] = mean_absolute_error(test, es_forecast)
except:
    pass

# ARIMA
try:
    arima_model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
    arima_forecast = arima_model.predict(n_periods=len(test))
    results['ARIMA'] = mean_absolute_error(test, arima_forecast)
except:
    pass

# Prophet
try:
    train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_df)
    future = pd.DataFrame({'ds': test.index})
    prophet_forecast = prophet_model.predict(future)['yhat']
    results['Prophet'] = mean_absolute_error(test, prophet_forecast.values)
except:
    pass

best_model = min(results, key=results.get)
print(f'Best model: {best_model} with MAE: {results[best_model]:.2f}')
```

## Forecast Combination

Combining forecasts often outperforms individual methods:

### Simple Average

```python
final_forecast = (arima_forecast + ets_forecast + prophet_forecast) / 3
```

### Weighted Average (by CV performance)

```python
# Weights inversely proportional to error
weights = {k: 1/v for k, v in cv_errors.items()}
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

final_forecast = (
    weights['ARIMA'] * arima_forecast +
    weights['ETS'] * ets_forecast +
    weights['Prophet'] * prophet_forecast
)
```

### Stacking with ML

```python
from sklearn.linear_model import Ridge

# Stack forecasts as features
X_stack = np.column_stack([arima_forecast, ets_forecast, prophet_forecast])
y_stack = test.values

# Learn optimal combination
stacker = Ridge(alpha=1.0)
stacker.fit(X_stack, y_stack)

# Apply to future forecasts
future_stack = np.column_stack([arima_future, ets_future, prophet_future])
final_forecast = stacker.predict(future_stack)
```

## Practical Considerations

### Handling Data Issues

| Issue | ARIMA | ETS | Prophet |
|-------|-------|-----|---------|
| Missing values | Impute first | Impute first | Handles natively |
| Outliers | Sensitive | Sensitive | Robust |
| Level shifts | Difficult | Difficult | Changepoints |
| Varying variance | GARCH extension | Multiplicative | Multiplicative |

### Computational Performance

| Method | Fit Time (1000 obs) | Forecast Time | Memory |
|--------|---------------------|---------------|--------|
| SES | ~1ms | <1ms | Low |
| Holt-Winters | ~10ms | <1ms | Low |
| ARIMA(1,1,1) | ~50ms | <1ms | Low |
| auto_arima | ~1-5s | <1ms | Low |
| Prophet | ~2-10s | ~100ms | Medium |

### Production Considerations

```python
# Fast production forecasting with ETS
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

# Fit multiple series in parallel
models = [AutoETS(season_length=12), AutoARIMA(season_length=12)]
sf = StatsForecast(models=models, freq='M', n_jobs=-1)
sf.fit(df)
forecasts = sf.predict(h=12)
```

### Updating Models

For production systems, decide on retraining strategy:

| Strategy | When to Use |
|----------|------------|
| Full retrain | Daily/weekly, when new patterns emerge |
| Recursive update | Real-time, stable patterns |
| Sliding window | Balance between adaptation and stability |

```python
# Recursive forecast updating
def update_forecast(model, new_observation):
    # For ETS: use last fitted state
    # For ARIMA: extend and refit
    # For Prophet: append and refit (slow)
    pass
```

## Common Pitfalls

### Pitfall 1: Wrong Seasonality Period

**Symptom**: Poor fit despite obvious seasonality.

**Solution**: Verify the period matches your data frequency (12 for monthly, 52 for weekly, 7 for daily).

### Pitfall 2: Ignoring Stationarity

**Symptom**: ARIMA fails or produces poor forecasts.

**Solution**: Check and apply differencing. Use ADF test.

### Pitfall 3: Overfitting with Complex Models

**Symptom**: Great in-sample fit, poor out-of-sample.

**Solution**: Use information criteria (AIC/BIC), cross-validation.

### Pitfall 4: Extrapolating Too Far

**Symptom**: Unrealistic long-horizon forecasts.

**Solution**: Classical methods are best for short to medium horizons. Use damped trends.

### Pitfall 5: Random Train/Test Splits

**Symptom**: Overly optimistic performance estimates.

**Solution**: Always use time-based splits that respect temporal order.

## Choosing the Right Method

### Decision Flowchart

```
START
  |
  v
Data has strong autocorrelation?
  |
  +-- Yes --> ARIMA/SARIMA
  |
  +-- No --> Continue
  |
  v
Multiple seasonalities or holidays?
  |
  +-- Yes --> Prophet
  |
  +-- No --> Continue
  |
  v
Need maximum speed?
  |
  +-- Yes --> Exponential Smoothing
  |
  +-- No --> Continue
  |
  v
Missing data or outliers?
  |
  +-- Yes --> Prophet
  |
  +-- No --> Continue
  |
  v
Clear trend and single seasonality?
  |
  +-- Yes --> Exponential Smoothing (Holt-Winters)
  |
  +-- No --> Try all, use CV to select
```

### Rule of Thumb

1. **Start with ETS**: Fast, often works well
2. **Add ARIMA**: Captures what ETS might miss
3. **Use Prophet for business data**: When holidays and multiple seasonalities matter
4. **Combine if close**: Simple average often beats individual models

## Further Reading

For detailed information on each classical method, see:

- [ARIMA](arima/ReadMe.md) - Autoregressive Integrated Moving Average
- [Exponential Smoothing](exponential-smoothing/ReadMe.md) - Level, trend, seasonality decomposition
- [Prophet](prophet/ReadMe.md) - Meta's business forecasting tool

For modern deep learning alternatives, see the Deep Learning Methods section.
