# ARIMA (AutoRegressive Integrated Moving Average)

## Summary

ARIMA is the foundational statistical model for time series forecasting. It combines autoregression (using past values to predict future values), differencing (making the series stationary), and moving average (using past forecast errors). Understanding ARIMA provides the conceptual basis for most time series work, even when using more modern approaches.

Key points to remember:

- ARIMA(p, d, q) has three parameters: p (autoregressive order), d (differencing order), q (moving average order)
- The series must be stationary for ARIMA to work; differencing (the "I" in ARIMA) handles non-stationarity
- Use ACF and PACF plots to identify appropriate p and q values, or rely on auto_arima for automatic selection
- ARIMA assumes linear relationships and struggles with complex seasonality, regime changes, and multivariate inputs
- SARIMA extends ARIMA with seasonal components for data with recurring patterns
- For quick prototyping, statsmodels or pmdarima in Python are the standard tools
- ARIMA serves as a strong baseline; always compare more complex models against it

## When to Use ARIMA

ARIMA is appropriate when:

- You have a single univariate time series
- The data shows autocorrelation (past values predict future values)
- Trends are linear or can be removed via differencing
- You need an interpretable, well-understood baseline
- Computational resources are limited
- You have relatively short series (hundreds to low thousands of observations)

Consider alternatives when:

- Multiple related series could inform predictions (use VAR, machine learning)
- Strong seasonality exists (use SARIMA, Prophet, or Exponential Smoothing)
- Non-linear patterns dominate (use tree-based models, neural networks)
- You have exogenous variables to incorporate (use ARIMAX or machine learning)
- The series is very long with complex patterns (consider deep learning)

## The ARIMA Components

### Autoregressive (AR) Component

The AR component models the current value as a linear combination of past values:

```
y_t = c + phi_1 * y_{t-1} + phi_2 * y_{t-2} + ... + phi_p * y_{t-p} + epsilon_t
```

Where:
- p is the order (number of lagged terms)
- phi values are the learned coefficients
- epsilon_t is white noise error

An AR(1) model uses only the immediately preceding value. AR(2) uses the two most recent values, and so on.

**Intuition**: If today's stock price depends heavily on yesterday's price, an AR(1) model captures this. If it depends on both yesterday and the day before, AR(2) is more appropriate.

### Integrated (I) Component

The integrated component represents differencing applied to make the series stationary. A stationary series has constant mean and variance over time.

First-order differencing (d=1):
```
y'_t = y_t - y_{t-1}
```

Second-order differencing (d=2):
```
y''_t = y'_t - y'_{t-1} = y_t - 2*y_{t-1} + y_{t-2}
```

Most series require d=0, 1, or 2. Higher orders are rare and often indicate issues with the data.

**Intuition**: A stock price might trend upward (non-stationary), but the daily changes in price might fluctuate around zero (stationary). Differencing converts prices to returns.

### Moving Average (MA) Component

The MA component models the current value as a linear combination of past forecast errors:

```
y_t = c + epsilon_t + theta_1 * epsilon_{t-1} + theta_2 * epsilon_{t-2} + ... + theta_q * epsilon_{t-q}
```

Where:
- q is the order (number of lagged error terms)
- theta values are the learned coefficients

**Intuition**: If an unexpected shock (like a one-time event) affects the series, the MA component captures how that shock's influence decays over subsequent periods.

### Combined ARIMA(p, d, q)

The full ARIMA model applies AR and MA to the differenced series:

```
y'_t = c + phi_1 * y'_{t-1} + ... + phi_p * y'_{t-p} + epsilon_t + theta_1 * epsilon_{t-1} + ... + theta_q * epsilon_{t-q}
```

Where y'_t is the differenced series (after d rounds of differencing).

## Identifying ARIMA Orders

### The Stationarity Check

Before fitting ARIMA, verify the series is stationary (or determine d):

1. **Visual inspection**: Plot the series. Look for trends, changing variance, or seasonality.

2. **Augmented Dickey-Fuller (ADF) test**: Tests the null hypothesis that the series has a unit root (non-stationary).
   - p-value < 0.05: Reject null, series is stationary
   - p-value >= 0.05: Cannot reject null, series may be non-stationary

3. **KPSS test**: Tests the null hypothesis that the series is stationary.
   - Use in conjunction with ADF for confirmation

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF test
result = adfuller(series)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

# If non-stationary, difference and test again
if result[1] > 0.05:
    diff_series = series.diff().dropna()
    result_diff = adfuller(diff_series)
    print(f'After differencing - p-value: {result_diff[1]:.4f}')
```

### ACF and PACF Analysis

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots help identify p and q:

| Pattern in ACF | Pattern in PACF | Suggested Model |
|----------------|-----------------|-----------------|
| Tails off (gradual decay) | Cuts off after lag p | AR(p) |
| Cuts off after lag q | Tails off | MA(q) |
| Tails off | Tails off | ARMA(p, q) |

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, ax=axes[0], lags=20)
plot_pacf(series, ax=axes[1], lags=20)
plt.show()
```

**Practical guidance**:
- PACF "cuts off" means it drops to near-zero after lag p and stays there
- ACF "cuts off" means it drops to near-zero after lag q and stays there
- "Tails off" means gradual decay toward zero
- Significant spikes at seasonal lags (12 for monthly data) suggest SARIMA

### Automatic Order Selection

In practice, use pmdarima's auto_arima to search for optimal orders:

```python
from pmdarima import auto_arima

model = auto_arima(
    series,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # Auto-detect differencing
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic'
)

print(model.summary())
print(f'Order: {model.order}')
```

**Information criteria** (lower is better):
- AIC (Akaike Information Criterion): Balances fit and complexity
- BIC (Bayesian Information Criterion): Penalizes complexity more than AIC

## Fitting and Forecasting

### Basic ARIMA with statsmodels

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Fit the model
model = ARIMA(series, order=(p, d, q))
fitted = model.fit()

# Model diagnostics
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=10)

# Forecast with confidence intervals
forecast_df = fitted.get_forecast(steps=10)
mean = forecast_df.predicted_mean
conf_int = forecast_df.conf_int()
```

### Model Diagnostics

Always validate the model before trusting forecasts:

```python
# Residual analysis
residuals = fitted.resid

# 1. Residuals should be white noise (no autocorrelation)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)  # p-value > 0.05 means no significant autocorrelation

# 2. Residuals should be normally distributed
from scipy import stats
_, p_value = stats.shapiro(residuals[:5000])  # Shapiro-Wilk test

# 3. Visual diagnostics
fitted.plot_diagnostics(figsize=(12, 8))
plt.show()
```

**Diagnostic checklist**:
- Standardized residuals: Should look like white noise, centered around zero
- Histogram: Should approximate a normal distribution
- Q-Q plot: Points should follow the diagonal line
- Correlogram: No significant autocorrelation in residuals

### Out-of-Sample Validation

Never evaluate solely on in-sample fit. Use time series cross-validation:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Simple train/test split (respecting time order)
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

model = ARIMA(train, order=(p, d, q))
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%')
```

### Rolling Forecast

For multi-step forecasting, rolling (walk-forward) validation provides more realistic error estimates:

```python
def rolling_forecast(series, train_size, order, horizon=1):
    predictions = []
    actuals = []

    for i in range(train_size, len(series) - horizon + 1):
        train = series[:i]
        model = ARIMA(train, order=order)
        fitted = model.fit()
        pred = fitted.forecast(steps=horizon)

        predictions.append(pred.iloc[-1])
        actuals.append(series.iloc[i + horizon - 1])

    return np.array(predictions), np.array(actuals)

preds, acts = rolling_forecast(series, train_size, order=(1, 1, 1))
```

## SARIMA for Seasonal Data

SARIMA adds seasonal components to handle recurring patterns:

SARIMA(p, d, q)(P, D, Q, s)

Where:
- (p, d, q): Non-seasonal orders (same as ARIMA)
- (P, D, Q): Seasonal orders
- s: Seasonal period (12 for monthly, 4 for quarterly, 7 for daily with weekly pattern)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Monthly data with yearly seasonality
model = SARIMAX(
    series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```

### Identifying Seasonal Orders

1. Look at ACF for significant spikes at seasonal lags (lag 12, 24, 36 for monthly data)
2. Use seasonal differencing if there's a seasonal unit root
3. auto_arima with `seasonal=True` handles this automatically:

```python
model = auto_arima(
    series,
    seasonal=True,
    m=12,  # Seasonal period
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=None,  # Auto-detect seasonal differencing
)
```

## ARIMAX: Including Exogenous Variables

ARIMAX (or ARIMA with exogenous regressors) incorporates external predictors:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# exog should be aligned with the series
model = SARIMAX(
    series,
    exog=exogenous_features,
    order=(1, 1, 1)
)
fitted = model.fit()

# For forecasting, you need future values of exogenous variables
future_exog = get_future_features(steps=10)
forecast = fitted.forecast(steps=10, exog=future_exog)
```

**Caution**: Exogenous variables must be known at forecast time. You cannot use variables that need to be predicted themselves unless you forecast them separately.

## Common Pitfalls and Solutions

### Pitfall 1: Non-Stationary Residuals

**Symptom**: Significant autocorrelation in residuals, Ljung-Box test rejects null.

**Solutions**:
- Increase p or q
- Check if additional differencing is needed
- Consider if the series has structural breaks

### Pitfall 2: Overfitting

**Symptom**: Excellent in-sample fit, poor out-of-sample performance.

**Solutions**:
- Use BIC instead of AIC (heavier complexity penalty)
- Set maximum order constraints
- Validate with time series cross-validation, not random splits

### Pitfall 3: Seasonality Not Captured

**Symptom**: Residuals show patterns at seasonal lags.

**Solutions**:
- Use SARIMA instead of ARIMA
- Ensure the seasonal period m is correct
- Consider seasonal dummy variables or Prophet for complex seasonality

### Pitfall 4: Model Doesn't Converge

**Symptom**: Convergence warnings during fitting.

**Solutions**:
- Try different starting values
- Reduce model order
- Increase maximum iterations: `model.fit(maxiter=500)`
- Check for outliers or data issues

### Pitfall 5: Confidence Intervals Too Wide

**Symptom**: Forecasts quickly become uninformative.

**Reality**: This is often correct behavior. ARIMA uncertainty grows with forecast horizon.

**Solutions**:
- Accept that long-horizon forecasts are uncertain
- Use shorter forecast horizons with regular updates
- Consider if ARIMA is appropriate for your forecast horizon

## Practical Code Example

Complete workflow for a monthly sales forecast:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('monthly_sales.csv', parse_dates=['date'], index_col='date')
series = df['sales']

# Check stationarity
adf_result = adfuller(series)
print(f'ADF p-value: {adf_result[1]:.4f}')

# Automatic order selection
auto_model = auto_arima(
    series,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print(f'Best order: {auto_model.order}')
print(f'Seasonal order: {auto_model.seasonal_order}')

# Fit final model
model = SARIMAX(
    series,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
)
fitted = model.fit(disp=False)

# Diagnostics
fitted.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig('diagnostics.png')

# Forecast next 12 months
forecast_result = fitted.get_forecast(steps=12)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

# Plot results
fig, ax = plt.subplots(figsize=(12, 6))
series.plot(ax=ax, label='Observed')
forecast_mean.plot(ax=ax, label='Forecast')
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.3
)
ax.legend()
plt.savefig('forecast.png')
```

## ARIMA vs Other Methods

| Criterion | ARIMA | Exponential Smoothing | Prophet |
|-----------|-------|----------------------|---------|
| Interpretability | High | High | Medium |
| Handles trend | Via differencing | Via trend component | Automatic |
| Handles seasonality | SARIMA needed | Built-in (ETS) | Automatic |
| Multiple seasonalities | Limited | Limited | Excellent |
| Holiday effects | Manual | Manual | Built-in |
| Exogenous variables | ARIMAX | Limited | Built-in regressors |
| Automatic fitting | pmdarima | Some libraries | Fully automatic |
| Uncertainty quantification | Confidence intervals | Prediction intervals | Credible intervals |
| Computational cost | Low | Very low | Medium |

**Guidance**:
- Start with ARIMA as a baseline for any forecasting problem
- Move to Exponential Smoothing for simpler problems with clear trend/seasonality
- Use Prophet when you have holidays, multiple seasonalities, or need ease of use
- Consider deep learning methods only when you have long series and complex patterns

## Key Takeaways

1. **ARIMA is the foundation**: Understanding ARIMA helps you understand most classical time series methods.

2. **Stationarity matters**: Always check and achieve stationarity before fitting.

3. **Use automatic selection**: pmdarima's auto_arima saves time and often finds good models.

4. **Validate properly**: Time series cross-validation respects temporal order; random splits do not.

5. **Check diagnostics**: Residuals should be white noise. If not, the model is missing something.

6. **Know the limitations**: ARIMA assumes linearity and struggles with complex patterns, regime changes, and multiple seasonalities.

7. **It's a baseline**: Always compare fancier methods against ARIMA. Sometimes the simple model wins.
