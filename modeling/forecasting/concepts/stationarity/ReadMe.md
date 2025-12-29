# Stationarity

## Summary

Stationarity is a fundamental property of time series that determines which forecasting methods are applicable. A stationary series has constant statistical properties over time: the mean, variance, and autocorrelation structure do not change. Most classical forecasting methods (ARIMA, exponential smoothing) assume or require stationarity. Understanding and testing for stationarity, and knowing how to transform non-stationary data, is essential for effective time series modeling.

Key points to remember:

- A stationary series has constant mean, variance, and autocorrelation over time
- Most forecasting methods require or assume stationarity
- Two types: strict stationarity (all moments constant) and weak stationarity (first two moments constant)
- Unit root tests (ADF, KPSS) formally test for stationarity
- Differencing is the primary tool to achieve stationarity (removes trend)
- Seasonal differencing handles seasonal non-stationarity
- Log transformation stabilizes variance that changes with level
- Some modern deep learning methods are more robust to non-stationarity

## Why Stationarity Matters

### The Core Problem

Forecasting assumes that patterns learned from the past will continue into the future. If the statistical properties of a series change over time (non-stationary), past patterns may not predict future behavior:

```
Stationary:     Past patterns --> Similar future patterns
Non-stationary: Past patterns --> Different future patterns (unreliable forecasts)
```

### Practical Impact

| Consequence of Non-Stationarity | Example |
|---------------------------------|---------|
| Unreliable forecasts | Model predicts constant, but series trends upward |
| Invalid confidence intervals | Intervals too narrow or wide |
| Spurious relationships | Two trending series appear correlated by chance |
| Model instability | Parameter estimates change with sample period |

## Types of Stationarity

### Strict Stationarity

The joint probability distribution of any collection of observations is invariant to time shifts:

```
P(y_t1, y_t2, ..., y_tk) = P(y_{t1+h}, y_{t2+h}, ..., y_{tk+h})
```

This is a strong condition rarely tested in practice.

### Weak (Wide-Sense) Stationarity

A more practical definition requiring only:

1. **Constant mean**: E[y_t] = mu for all t
2. **Constant variance**: Var(y_t) = sigma^2 for all t
3. **Autocovariance depends only on lag**: Cov(y_t, y_{t+h}) = gamma(h) for all t

Weak stationarity is what most forecasting methods assume and what statistical tests check.

### Trend Stationarity vs Difference Stationarity

| Type | Description | Example | Treatment |
|------|-------------|---------|-----------|
| Trend stationary | Deterministic trend, stationary residuals | y_t = a + bt + e_t | Remove trend by regression |
| Difference stationary | Stochastic trend (random walk) | y_t = y_{t-1} + e_t | Differencing |

Differencing works for both, but trend-stationary series can also be handled by regressing on time.

## Common Patterns of Non-Stationarity

### Trending Mean

```
Time: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y:    [10, 12, 14, 15, 17, 19, 21, 22, 24, 26]

The mean is increasing over time.
```

**Solution**: First differencing (d=1) or detrending.

### Changing Variance

```
Time: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y:    [100, 102, 98, 200, 210, 190, 400, 420, 380, 800]

Variance increases with level.
```

**Solution**: Log transformation or Box-Cox transformation.

### Seasonal Patterns

```
Monthly data with yearly seasonality:
Jan: High, Feb: Medium, ..., Dec: High (repeating)
```

**Solution**: Seasonal differencing or explicit seasonal modeling.

### Structural Breaks

```
Before COVID: Mean = 100
After COVID: Mean = 50
```

**Solution**: Model separately, include intervention variables, or use robust methods.

## Testing for Stationarity

### Visual Inspection

Always start with visual analysis:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series plot
axes[0, 0].plot(series)
axes[0, 0].set_title('Time Series')

# Rolling statistics
rolling_mean = series.rolling(window=12).mean()
rolling_std = series.rolling(window=12).std()
axes[0, 1].plot(series, label='Original')
axes[0, 1].plot(rolling_mean, label='Rolling Mean')
axes[0, 1].plot(rolling_std, label='Rolling Std')
axes[0, 1].legend()
axes[0, 1].set_title('Rolling Statistics')

# Histogram (check for normality)
axes[1, 0].hist(series, bins=30)
axes[1, 0].set_title('Distribution')

# ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, ax=axes[1, 1], lags=40)

plt.tight_layout()
```

**Signs of non-stationarity**:
- Trend in time series plot
- Rolling mean/std not constant
- ACF decays very slowly (characteristic of unit root)

### Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis that the series has a unit root (non-stationary):

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series, autolag='AIC')

print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'  {key}: {value:.4f}')

# Interpretation
if result[1] < 0.05:
    print("Reject null hypothesis: Series is stationary")
else:
    print("Cannot reject null hypothesis: Series may be non-stationary")
```

**Important**: ADF tests for unit root, not stationarity directly. Low p-value suggests no unit root (likely stationary).

### KPSS Test

Tests the null hypothesis that the series is stationary (opposite of ADF):

```python
from statsmodels.tsa.stattools import kpss

result = kpss(series, regression='c', nlags='auto')

print(f'KPSS Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
print('Critical Values:')
for key, value in result[3].items():
    print(f'  {key}: {value:.4f}')

# Interpretation
if result[1] < 0.05:
    print("Reject null hypothesis: Series is non-stationary")
else:
    print("Cannot reject null hypothesis: Series may be stationary")
```

### Combined Testing Strategy

Use both ADF and KPSS together:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject H0 | Don't reject H0 | Stationary |
| Don't reject H0 | Reject H0 | Non-stationary (unit root) |
| Reject H0 | Reject H0 | Trend stationary |
| Don't reject H0 | Don't reject H0 | Inconclusive |

```python
def check_stationarity(series):
    # ADF test
    adf_result = adfuller(series, autolag='AIC')
    adf_stationary = adf_result[1] < 0.05

    # KPSS test
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stationary = kpss_result[1] >= 0.05

    if adf_stationary and kpss_stationary:
        return "Stationary"
    elif not adf_stationary and not kpss_stationary:
        return "Non-stationary (unit root)"
    elif adf_stationary and not kpss_stationary:
        return "Trend stationary"
    else:
        return "Inconclusive"
```

### Phillips-Perron Test

Alternative to ADF that handles serial correlation differently:

```python
from statsmodels.tsa.stattools import phillips_perron

result = phillips_perron(series)
print(f'PP Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
```

## Achieving Stationarity

### Differencing

The most common transformation for trend removal:

```python
import pandas as pd

# First difference (removes linear trend)
diff1 = series.diff().dropna()

# Second difference (removes quadratic trend, rarely needed)
diff2 = series.diff().diff().dropna()

# Check if stationary after differencing
adf_result = adfuller(diff1)
print(f'After differencing - p-value: {adf_result[1]:.4f}')
```

**How differencing works**:
```
Original:    [100, 102, 105, 109, 114, 120]  (trending up)
First diff:  [2, 3, 4, 5, 6]                  (still trending)
Second diff: [1, 1, 1, 1]                     (constant, stationary)
```

### Seasonal Differencing

For seasonal non-stationarity:

```python
# Seasonal difference (period = 12 for monthly data with yearly seasonality)
seasonal_diff = series.diff(12).dropna()

# Combined: first difference then seasonal difference
combined_diff = series.diff().diff(12).dropna()
```

### Log Transformation

Stabilizes variance that increases with level:

```python
import numpy as np

# Log transformation
log_series = np.log(series)

# Log transformation then differencing (common combo)
log_diff = np.log(series).diff().dropna()
```

**When to use**:
- Variance increases proportionally with level
- Data is strictly positive
- Multiplicative relationships

### Box-Cox Transformation

Generalizes log transformation:

```python
from scipy import stats

# Find optimal lambda
transformed, lambda_opt = stats.boxcox(series)
print(f'Optimal lambda: {lambda_opt:.4f}')

# lambda = 0: log transform
# lambda = 0.5: square root
# lambda = 1: no transform
```

### Detrending

Remove deterministic trend:

```python
from scipy import signal

# Linear detrending
detrended = signal.detrend(series)

# Or fit trend explicitly
import numpy as np
t = np.arange(len(series))
coeffs = np.polyfit(t, series, deg=1)
trend = np.polyval(coeffs, t)
detrended = series - trend
```

## Determining Differencing Order

### Automatic Detection

```python
from pmdarima.arima import ndiffs, nsdiffs

# Number of non-seasonal differences
d = ndiffs(series, test='adf')
print(f'Recommended d: {d}')

# Number of seasonal differences (for monthly data)
D = nsdiffs(series, m=12, test='ocsb')
print(f'Recommended D: {D}')
```

### Manual Approach

```python
def find_d(series, max_d=2):
    for d in range(max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()

        adf_pvalue = adfuller(test_series)[1]

        if adf_pvalue < 0.05:
            return d

    return max_d

d = find_d(series)
```

## Stationarity in Different Contexts

### ARIMA

ARIMA explicitly handles non-stationarity through the I (Integrated) component:

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p, d, q) - d is the differencing order
model = ARIMA(series, order=(1, 1, 1))  # d=1 means one difference
fitted = model.fit()
```

### Exponential Smoothing

ETS models can handle trends and seasonality directly without differencing:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    series,
    trend='add',      # Handles trend internally
    seasonal='add',   # Handles seasonality internally
    seasonal_periods=12
)
```

### Deep Learning

Many deep learning methods are more robust to non-stationarity:

```python
# N-BEATS, TFT, etc. often work without explicit differencing
# But normalization is still important

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized = scaler.fit_transform(series.values.reshape(-1, 1))
```

### Cointegration

When multiple non-stationary series share a common trend:

```python
from statsmodels.tsa.stattools import coint

# Test if two series are cointegrated
score, pvalue, _ = coint(series1, series2)

if pvalue < 0.05:
    print("Series are cointegrated")
    # Can model the spread between them
```

## Common Pitfalls

### Pitfall 1: Over-Differencing

**Symptom**: Series becomes too noisy, ACF shows negative spike at lag 1.

**Solution**: Use unit root tests to determine correct order. Usually d <= 2.

### Pitfall 2: Ignoring Seasonal Non-Stationarity

**Symptom**: ADF suggests stationarity but ACF shows spikes at seasonal lags.

**Solution**: Check for and apply seasonal differencing.

### Pitfall 3: Forgetting to Reverse Transformations

**Symptom**: Forecasts are in wrong units.

**Solution**: Always reverse differencing and log transforms for final forecasts:

```python
# If series was differenced
forecast_original = series.iloc[-1] + forecast_diff.cumsum()

# If series was log-transformed
forecast_original = np.exp(forecast_log)
```

### Pitfall 4: Testing on Entire Series

**Symptom**: Stationarity tests give different results on train vs test.

**Solution**: Test on training data only. Be aware that stationarity can change over time.

## Practical Example

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Load data
series = pd.read_csv('data.csv', index_col='date', parse_dates=True)['value']

# Step 1: Visual inspection
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
series.plot(ax=axes[0, 0], title='Original Series')
series.rolling(12).mean().plot(ax=axes[0, 0], label='12-period MA')
axes[0, 0].legend()

# Step 2: Statistical tests
def run_tests(s, name):
    adf_result = adfuller(s.dropna(), autolag='AIC')
    kpss_result = kpss(s.dropna(), regression='c', nlags='auto')
    print(f'{name}:')
    print(f'  ADF p-value: {adf_result[1]:.4f}')
    print(f'  KPSS p-value: {kpss_result[1]:.4f}')

run_tests(series, 'Original')

# Step 3: Transform if needed
# Check if variance increases with level
if series.rolling(12).std().corr(series.rolling(12).mean()) > 0.5:
    series_transformed = np.log(series)
    print('Applied log transformation')
else:
    series_transformed = series

# Step 4: Difference if needed
diff1 = series_transformed.diff()
run_tests(diff1, 'First difference')

# Step 5: Seasonal difference if needed (for monthly data)
seasonal_diff = diff1.diff(12)
run_tests(seasonal_diff, 'Seasonal difference')

# Step 6: Use appropriate transformation
# For ARIMA: specify d and D based on tests
# For deep learning: normalize

print(f'\nRecommended: d=1, D=1 for monthly data with trend and seasonality')
```

## Key Takeaways

1. **Test before modeling**: Always check stationarity before applying classical methods.

2. **Use multiple tests**: Combine ADF and KPSS for reliable conclusions.

3. **Difference appropriately**: First differencing for trend, seasonal differencing for seasonal patterns.

4. **Stabilize variance**: Log or Box-Cox transformation if variance changes with level.

5. **Don't over-difference**: Usually d <= 2 and D <= 1 are sufficient.

6. **Remember to reverse**: Undo transformations when generating final forecasts.

7. **Modern methods are more robust**: Deep learning often handles non-stationarity internally, but normalization still helps.
