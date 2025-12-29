# Trend Decomposition

## Summary

Trend decomposition separates a time series into its constituent components: trend, seasonality, and residuals. This technique is fundamental for understanding time series behavior, improving forecasts, and explaining patterns to stakeholders. Modern decomposition methods like STL handle varying trends and multiple seasonalities, while classical methods provide simpler baselines.

Key points to remember:

- Decomposition separates series into trend, seasonal, and residual components
- Classical decomposition uses moving averages; STL uses local regression
- Additive: y = trend + seasonal + residual; Multiplicative: y = trend * seasonal * residual
- STL is more robust to outliers and allows varying seasonality
- MSTL handles multiple seasonal patterns
- Decomposition aids visualization, anomaly detection, and feature engineering
- Not all series decompose cleanly; residuals should be checked

## Why Decompose?

### Understanding Data

Decomposition answers key questions:
- Is there an underlying trend? (Growing, declining, stable)
- What is the seasonal pattern? (When are highs and lows)
- What remains unexplained? (Residuals, anomalies)

### Practical Applications

| Application | How Decomposition Helps |
|-------------|------------------------|
| Forecasting | Model components separately, combine forecasts |
| Anomaly detection | Unusual residuals indicate anomalies |
| Deseasonalization | Reveal underlying trend for business decisions |
| Feature engineering | Components become model inputs |
| Communication | Explain patterns to non-technical stakeholders |

## The Decomposition Model

### Additive Model

```
y_t = T_t + S_t + R_t
```

Where:
- y_t: Observed value at time t
- T_t: Trend component
- S_t: Seasonal component
- R_t: Residual (remainder)

**Use when**: Seasonal variation is constant regardless of level.

### Multiplicative Model

```
y_t = T_t * S_t * R_t
```

**Use when**: Seasonal variation scales with the level.

### Pseudo-Additive (Mixed) Model

```
y_t = T_t * S_t + R_t
```

**Use when**: Seasonality is multiplicative but residuals are additive.

### Choosing the Model

```python
import numpy as np

def recommend_model(series):
    """Recommend additive or multiplicative decomposition."""
    # Compute seasonal amplitude at different levels
    yearly_std = series.groupby(series.index.year).std()
    yearly_mean = series.groupby(series.index.year).mean()

    # Coefficient of variation (CV)
    cv = yearly_std / yearly_mean

    # If CV is roughly constant, multiplicative
    # If std is roughly constant, additive
    cv_variation = cv.std() / cv.mean()
    std_variation = yearly_std.std() / yearly_std.mean()

    if std_variation < cv_variation:
        return 'additive'
    else:
        return 'multiplicative'
```

## Classical Decomposition

### Moving Average Method

Classical decomposition uses centered moving averages:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
result_add = seasonal_decompose(series, model='additive', period=12)

# Multiplicative decomposition
result_mul = seasonal_decompose(series, model='multiplicative', period=12)

# Plot components
result_add.plot()
plt.tight_layout()
plt.show()
```

### How It Works

1. **Estimate trend** using centered moving average:
   ```
   T_t = MA(y_t, period)
   ```

2. **Calculate detrended series**:
   ```
   Additive: D_t = y_t - T_t
   Multiplicative: D_t = y_t / T_t
   ```

3. **Estimate seasonal component** by averaging detrended values for each period position:
   ```
   S_i = mean(D_t for all t where position = i)
   ```

4. **Calculate residuals**:
   ```
   Additive: R_t = y_t - T_t - S_t
   Multiplicative: R_t = y_t / (T_t * S_t)
   ```

### Limitations

| Limitation | Impact |
|------------|--------|
| Missing values at ends | Loses period/2 observations at each end |
| Fixed seasonality | Cannot capture changing seasonal patterns |
| Outlier sensitive | Outliers affect trend and seasonal estimates |
| Single seasonality | Cannot handle multiple seasonal patterns |

## STL Decomposition

STL (Seasonal and Trend decomposition using Loess) is more flexible and robust.

### Basic Usage

```python
from statsmodels.tsa.seasonal import STL

stl = STL(
    series,
    period=12,
    seasonal=7,    # Seasonal smoother span (odd number)
    trend=None,    # Auto-select based on period
    robust=True    # Robust to outliers
)
result = stl.fit()

# Access components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Plot
result.plot()
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| period | Seasonal period | Must match data (12 for monthly) |
| seasonal | Seasonal smoother span | Higher = smoother seasonal |
| trend | Trend smoother span | Higher = smoother trend |
| robust | Use robust fitting | True if outliers present |

### How STL Works

STL uses an iterative algorithm:

1. **Outer loop** (optional, for robustness):
   - Calculate robustness weights based on residuals

2. **Inner loop**:
   - Detrend by subtracting current trend estimate
   - Smooth cycle-subseries to get seasonal
   - Compute deseasonalized series
   - Smooth deseasonalized series to get trend

The Loess (locally weighted scatterplot smoothing) regression allows both trend and seasonality to vary smoothly over time.

### STL Advantages

- Handles any seasonal period (even non-integer)
- Trend and seasonality can vary
- Robust to outliers (with robust=True)
- Fewer missing values at ends

## MSTL for Multiple Seasonalities

### Basic Usage

```python
from statsmodels.tsa.seasonal import MSTL

# Hourly data with daily (24) and weekly (168) patterns
mstl = MSTL(
    series,
    periods=[24, 168],
    stl_kwargs={'robust': True}
)
result = mstl.fit()

# Access components
trend = result.trend
seasonals = result.seasonal  # Shape: (n, num_periods)
residual = result.resid

# Individual seasonalities
daily_seasonal = result.seasonal[:, 0]
weekly_seasonal = result.seasonal[:, 1]
```

### How MSTL Works

MSTL applies STL iteratively:

1. Start with the longest period
2. Extract that seasonal component
3. Remove from series
4. Apply STL to residual with next period
5. Repeat for all periods
6. Refine iteratively

## Decomposition for Forecasting

### Component-Based Forecasting

Forecast components separately and recombine:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Decompose
stl = STL(series, period=12, robust=True)
result = stl.fit()

# Forecast trend (Holt's linear trend)
trend_model = ExponentialSmoothing(
    result.trend.dropna(),
    trend='add',
    damped_trend=True
)
trend_forecast = trend_model.fit().forecast(12)

# Seasonal is periodic (last year's pattern)
seasonal_forecast = result.seasonal[-12:]

# Combine
forecast = trend_forecast + seasonal_forecast.values
```

### Feature Engineering

Use components as features:

```python
# Extract components
stl = STL(series, period=12, robust=True).fit()

# Create feature DataFrame
features = pd.DataFrame({
    'original': series,
    'trend': stl.trend,
    'seasonal': stl.seasonal,
    'residual': stl.resid,
    'deseasonalized': series - stl.seasonal,
    'detrended': series - stl.trend
})

# Use in ML model
from sklearn.ensemble import RandomForestRegressor
# Features: lagged deseasonalized values
# Target: original series
```

## Anomaly Detection

### Residual-Based Detection

Anomalies appear as unusual residuals:

```python
stl = STL(series, period=12, robust=True).fit()

# Calculate residual statistics
residual_mean = stl.resid.mean()
residual_std = stl.resid.std()

# Flag anomalies (> 3 standard deviations)
lower_bound = residual_mean - 3 * residual_std
upper_bound = residual_mean + 3 * residual_std

anomalies = (stl.resid < lower_bound) | (stl.resid > upper_bound)
print(f"Detected {anomalies.sum()} anomalies")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
series.plot(ax=axes[0], label='Original')
series[anomalies].plot(ax=axes[0], style='ro', label='Anomalies')
axes[0].legend()

stl.resid.plot(ax=axes[1], label='Residuals')
axes[1].axhline(upper_bound, color='r', linestyle='--')
axes[1].axhline(lower_bound, color='r', linestyle='--')
axes[1].legend()
```

### Trend Break Detection

Sudden changes in trend component:

```python
# First difference of trend
trend_diff = stl.trend.diff()

# Flag large changes
threshold = 2 * trend_diff.std()
trend_breaks = trend_diff.abs() > threshold
```

## Advanced Techniques

### X-13ARIMA-SEATS

Industry-standard decomposition (used by Census Bureau):

```python
# Requires x13 binary installation
from statsmodels.tsa.x13 import x13_arima_analysis

# Run X-13
result = x13_arima_analysis(
    series,
    x12path='/path/to/x13as',
    prefer_x13=True
)

# Components
trend = result.trend
seasonal = result.seasadj - result.trend
irregular = result.irregular
```

### Prophet Decomposition

Prophet's components are interpretable:

```python
from prophet import Prophet

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Access components
trend = forecast['trend']
yearly = forecast['yearly']
weekly = forecast['weekly']
holidays = forecast.get('holidays', 0)

# Reconstructed: trend + yearly + weekly + holidays
```

### Wavelet Decomposition

For multi-scale analysis:

```python
import pywt

# Wavelet decomposition
coeffs = pywt.wavedec(series.values, 'db4', level=4)

# Reconstruct components at different scales
approx = pywt.waverec([coeffs[0]] + [None]*4, 'db4')[:len(series)]
detail_1 = pywt.waverec([None, coeffs[1]] + [None]*3, 'db4')[:len(series)]
# ... additional detail levels
```

## Best Practices

### Before Decomposition

1. **Check for missing values**: Interpolate or handle appropriately
2. **Identify the period**: Use ACF, domain knowledge
3. **Choose additive vs multiplicative**: Examine variance patterns
4. **Consider multiple seasonalities**: Use MSTL if present

### After Decomposition

1. **Inspect residuals**: Should be roughly white noise
2. **Check seasonal pattern**: Should look reasonable for domain
3. **Validate trend**: Should be smooth, not capturing noise
4. **Test stationarity of residuals**: ADF test on residuals

### Validation

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

stl = STL(series, period=12, robust=True).fit()

# Residuals should be stationary
adf_result = adfuller(stl.resid.dropna())
print(f"Residual ADF p-value: {adf_result[1]:.4f}")

# Residuals should have no autocorrelation
lb_result = acorr_ljungbox(stl.resid.dropna(), lags=20)
if lb_result['lb_pvalue'].min() > 0.05:
    print("No significant autocorrelation in residuals")
else:
    print("Residuals still contain autocorrelation")
```

## Common Pitfalls

### Pitfall 1: Wrong Period

**Symptom**: Seasonal component looks irregular or trend captures seasonal pattern.

**Solution**: Verify period with ACF or domain knowledge.

### Pitfall 2: Over-Smoothing Trend

**Symptom**: Trend is too smooth, legitimate changes absorbed into residuals.

**Solution**: Reduce trend smoother span in STL.

### Pitfall 3: Under-Smoothing Seasonal

**Symptom**: Seasonal component looks noisy, varies too much year-to-year.

**Solution**: Increase seasonal smoother span in STL.

### Pitfall 4: Ignoring End Effects

**Symptom**: Poor decomposition at series ends.

**Solution**: Classical decomposition loses observations; STL is better but still less reliable at ends.

## Key Takeaways

1. **Decomposition reveals structure**: Separate trend, seasonal, and residual for understanding.

2. **Choose the right method**: Classical for simple cases, STL for robustness, MSTL for multiple seasonalities.

3. **Match additive/multiplicative**: Based on whether seasonal amplitude is constant or proportional.

4. **Check residuals**: Good decomposition leaves white noise residuals.

5. **Use for feature engineering**: Components become useful model inputs.

6. **Enable anomaly detection**: Unusual residuals signal anomalies.

7. **STL is usually better**: More flexible, robust, and fewer missing values.
