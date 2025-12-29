# Seasonality

## Summary

Seasonality refers to regular, predictable patterns that repeat over fixed time intervals in time series data. Recognizing, measuring, and properly modeling seasonality is crucial for accurate forecasting. Seasonal patterns arise from calendar effects (days of week, months), natural cycles (weather, daylight), and human behavior (holidays, business cycles). Different forecasting methods handle seasonality differently, and choosing the wrong approach can lead to poor predictions.

Key points to remember:

- Seasonality is a repeating pattern with a fixed, known period
- Common periods: 7 (weekly), 12 (monthly), 52 (weekly in annual), 365 (daily in annual)
- Additive seasonality has constant amplitude; multiplicative scales with level
- Multiple seasonalities can coexist (daily and weekly patterns in hourly data)
- ACF at seasonal lags and seasonal decomposition are key diagnostic tools
- SARIMA, Holt-Winters, and Prophet directly model seasonality
- Seasonal differencing and dummy variables are alternative approaches
- Fourier terms allow flexible seasonal patterns in regression models

## Identifying Seasonality

### Visual Inspection

Always start by plotting the data:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Time series plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Full series
series.plot(ax=axes[0], title='Full Time Series')

# Seasonal subseries plot (by month for monthly data)
for month in range(1, 13):
    monthly_data = series[series.index.month == month]
    axes[1].plot(monthly_data.values, alpha=0.7, label=f'Month {month}')
axes[1].set_title('Seasonal Subseries')

# Box plot by season
series.groupby(series.index.month).boxplot(ax=axes[2])
axes[2].set_title('Box Plot by Month')

plt.tight_layout()
```

### Autocorrelation Analysis

Seasonal patterns appear as spikes at seasonal lags in the ACF:

```python
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(series, lags=48, ax=ax)
ax.axvline(x=12, color='r', linestyle='--', label='Seasonal lag')
ax.legend()
plt.title('ACF with Seasonal Lag Highlighted')
```

**Interpretation**:
- Spike at lag 12: Monthly seasonality in monthly data
- Spike at lag 7: Weekly seasonality in daily data
- Multiple spikes at 12, 24, 36: Strong yearly seasonality

### Statistical Tests

```python
from scipy import stats

# Test for seasonal differences using ANOVA
# Group by season and test if means differ
groups = [series[series.index.month == m] for m in range(1, 13)]
f_stat, p_value = stats.f_oneway(*groups)

if p_value < 0.05:
    print(f"Significant seasonal differences (F={f_stat:.2f}, p={p_value:.4f})")
```

### Periodogram Analysis

Identify dominant frequencies:

```python
from scipy import signal
import numpy as np

# Compute periodogram
frequencies, power = signal.periodogram(series.values)

# Convert to periods
periods = 1 / frequencies[1:]  # Skip zero frequency
power = power[1:]

# Find dominant periods
top_indices = np.argsort(power)[-5:][::-1]
print("Dominant periods:")
for idx in top_indices:
    print(f"  Period: {periods[idx]:.1f}, Power: {power[idx]:.2f}")
```

## Types of Seasonality

### Additive Seasonality

Seasonal effect is constant regardless of series level:

```
y_t = Trend_t + Seasonal_t + Error_t
```

**Characteristics**:
- Seasonal swings have roughly constant amplitude
- January always adds 100 units, regardless of base level
- Appropriate when level changes don't affect seasonal magnitude

### Multiplicative Seasonality

Seasonal effect scales with series level:

```
y_t = Trend_t * Seasonal_t * Error_t
```

**Characteristics**:
- Seasonal swings proportional to level
- January is always 20% higher, regardless of base level
- Appropriate when seasonal effect is a percentage

### Determining Type

```python
# Visual approach: Look for "fanning" in seasonal peaks
# If peaks get larger as trend increases --> multiplicative

# Quantitative approach: Correlation between seasonal amplitude and level
seasonal_decomp = seasonal_decompose(series, model='additive', period=12)
seasonal_range = series.groupby(series.index.year).apply(lambda x: x.max() - x.min())
yearly_mean = series.groupby(series.index.year).mean()

correlation = seasonal_range.corr(yearly_mean)

if correlation > 0.5:
    print("Likely multiplicative seasonality")
else:
    print("Likely additive seasonality")
```

### Multiple Seasonalities

Many series have overlapping seasonal patterns:

| Data Frequency | Common Seasonalities |
|----------------|---------------------|
| Hourly | Daily (24), Weekly (168) |
| Daily | Weekly (7), Yearly (365) |
| Weekly | Yearly (52) |
| Monthly | Yearly (12) |

```python
# Hourly data with daily and weekly patterns
# Period 1: 24 (daily)
# Period 2: 168 (weekly = 24 * 7)

# Prophet handles multiple seasonalities automatically
from prophet import Prophet

model = Prophet(
    daily_seasonality=True,   # Period 24
    weekly_seasonality=True,  # Period 168
    yearly_seasonality=False
)
```

## Modeling Seasonality

### Seasonal Differencing

Remove seasonality by differencing at the seasonal lag:

```python
# Monthly data with yearly seasonality
seasonal_diff = series.diff(12).dropna()

# Combined with regular differencing
combined = series.diff(1).diff(12).dropna()
```

### SARIMA

Explicitly models seasonal autoregression and moving average:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p, d, q)(P, D, Q, s)
model = SARIMAX(
    series,
    order=(1, 1, 1),          # Non-seasonal
    seasonal_order=(1, 1, 1, 12)  # Seasonal with period 12
)
fitted = model.fit()
```

### Holt-Winters (ETS)

Decomposition-based approach:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Additive seasonality
model_add = ExponentialSmoothing(
    series,
    seasonal='add',
    seasonal_periods=12,
    trend='add',
    damped_trend=True
)

# Multiplicative seasonality
model_mul = ExponentialSmoothing(
    series,
    seasonal='mul',
    seasonal_periods=12,
    trend='add',
    damped_trend=True
)
```

### Prophet

Additive model with Fourier terms:

```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=10,   # Fourier order
    weekly_seasonality=3,
    seasonality_mode='additive'  # or 'multiplicative'
)

# Add custom seasonality
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)
```

### Fourier Terms in Regression

Flexible seasonality via sine/cosine pairs:

```python
import numpy as np

def fourier_features(t, period, n_terms):
    """Generate Fourier terms for seasonality."""
    features = {}
    for k in range(1, n_terms + 1):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

# Example: yearly seasonality with 5 harmonics
t = np.arange(len(series))
fourier_df = fourier_features(t, period=12, n_terms=5)

# Use in regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(fourier_df, series)
```

### Dummy Variables

Simple but effective:

```python
# Create dummy variables for each season
dummies = pd.get_dummies(series.index.month, prefix='month', drop_first=True)

# Use in regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(dummies, series)
```

## Handling Complex Seasonality

### Non-Integer Periods

Some seasonalities don't have integer periods:

```python
# Business days in a year: ~252 (not 365)
# Weeks in a year: 52.18 (not 52)

# Solution: Fourier terms work with non-integer periods
fourier_features(t, period=52.18, n_terms=5)
```

### Varying Seasonal Patterns

When seasonality changes over time:

```python
# Prophet with seasonality prior
model = Prophet(
    yearly_seasonality=True,
    seasonality_prior_scale=0.1  # Regularization
)

# Or: Local seasonality with time-varying models
# Or: Multiple regime models
```

### Calendar Adjustments

Account for varying month lengths:

```python
# Sales per day instead of per month
adjusted = series / series.index.days_in_month
```

## Seasonal Decomposition

### Classical Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
decomp_add = seasonal_decompose(series, model='additive', period=12)

# Multiplicative decomposition
decomp_mul = seasonal_decompose(series, model='multiplicative', period=12)

# Access components
trend = decomp_add.trend
seasonal = decomp_add.seasonal
residual = decomp_add.resid

# Plot
decomp_add.plot()
plt.tight_layout()
```

### STL Decomposition

More robust to outliers and allows trend/seasonality to vary:

```python
from statsmodels.tsa.seasonal import STL

stl = STL(series, period=12, robust=True)
result = stl.fit()

# Components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

result.plot()
```

### MSTL for Multiple Seasonalities

```python
from statsmodels.tsa.seasonal import MSTL

# Hourly data with daily (24) and weekly (168) seasonality
mstl = MSTL(series, periods=[24, 168])
result = mstl.fit()

# Access each seasonal component
daily_seasonal = result.seasonal[:, 0]
weekly_seasonal = result.seasonal[:, 1]
```

## Common Pitfalls

### Pitfall 1: Wrong Period

**Symptom**: Seasonal pattern not captured, residuals show seasonal ACF.

**Solution**: Verify the correct period:
- Daily data, weekly pattern: period = 7
- Daily data, yearly pattern: period = 365 (or 365.25)
- Monthly data, yearly pattern: period = 12

### Pitfall 2: Confusing Trend with Long-Cycle Seasonality

**Symptom**: What looks like a trend might be a very long seasonal cycle.

**Solution**: Use longer history or domain knowledge.

### Pitfall 3: Ignoring Holiday Effects

**Symptom**: Large residuals around holidays.

**Solution**: Model holidays explicitly:

```python
from prophet import Prophet

model = Prophet()
model.add_country_holidays(country_name='US')
```

### Pitfall 4: Assuming Constant Seasonality

**Symptom**: Recent predictions worse than historical.

**Solution**: Use methods that allow time-varying seasonality (STL, Prophet with regularization).

## Seasonality in Deep Learning

### Explicit Encoding

```python
# Add seasonal features as inputs
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

### Letting Models Learn

Modern deep learning models often learn seasonality automatically:

```python
# N-BEATS, TFT, TimesNet can capture seasonal patterns
# if given sufficient lookback covering the seasonal period

model = NBEATS(
    input_size=365,  # Full year for yearly seasonality
    ...
)
```

## Key Takeaways

1. **Always check for seasonality**: Use ACF, periodograms, and visual inspection.

2. **Identify the type**: Additive (constant amplitude) vs multiplicative (proportional).

3. **Know your periods**: Match period to your data frequency (12 for monthly, 7 for daily).

4. **Handle multiple seasonalities**: Prophet and MSTL decomposition are built for this.

5. **Consider Fourier terms**: Flexible way to add seasonality to any model.

6. **Account for holidays**: They're special seasonal effects that need explicit handling.

7. **Seasonality can change**: Use robust methods (STL) or regularization if patterns evolve.
