# Time Series Forecasting Concepts

## Summary

Successful time series forecasting requires understanding fundamental concepts that govern temporal data behavior. Stationarity determines which methods are applicable, seasonality and trend decomposition reveal the underlying structure, and proper cross-validation ensures reliable performance estimates. Mastering these concepts is essential before applying any forecasting method, whether classical statistical models or modern deep learning.

Key points to remember:

- Stationarity: Statistical properties constant over time; required by most methods
- Seasonality: Repeating patterns with fixed periods (daily, weekly, yearly)
- Trend: Long-term increase or decrease in the series level
- Decomposition: Separating series into trend, seasonal, and residual components
- Cross-validation: Time-ordered splits essential; never use random shuffling
- These concepts apply to all forecasting methods, classical and deep learning
- Understanding fundamentals prevents common pitfalls and improves model selection

## The Time Series Analysis Workflow

```
Raw Time Series
       |
       v
+-------------------+
| Visual Inspection |
| - Plot series     |
| - Check for trend |
| - Look for season |
+-------------------+
       |
       v
+-------------------+
| Stationarity Test |
| - ADF test        |
| - KPSS test       |
| - Visual checks   |
+-------------------+
       |
       v
+-------------------+
| Decomposition     |
| - STL/Classical   |
| - Identify parts  |
| - Check residuals |
+-------------------+
       |
       v
+-------------------+
| Model Selection   |
| - Choose method   |
| - Set up CV       |
| - Define metrics  |
+-------------------+
       |
       v
+-------------------+
| Evaluation        |
| - Walk-forward CV |
| - Compare models  |
| - Final selection |
+-------------------+
```

## Concept Relationships

### How Concepts Connect

| Concept | Affects | Why It Matters |
|---------|---------|----------------|
| Stationarity | Model choice | ARIMA requires stationarity; differencing achieves it |
| Seasonality | Period selection | Must identify correct period for SARIMA, Holt-Winters |
| Trend | Differencing order | Trend removal via differencing or explicit modeling |
| Decomposition | Feature engineering | Components become useful predictors |
| Cross-validation | Model selection | Proper CV prevents overfitting and leakage |

### Stationarity and Other Concepts

Stationarity interacts with seasonality and trend:

- **Trend causes non-stationarity**: Differencing removes it
- **Seasonality can cause non-stationarity**: Seasonal differencing helps
- **After differencing**: Series should be stationary, residuals white noise

### Decomposition and Modeling

Decomposition guides model selection:

| Decomposition Finding | Modeling Implication |
|----------------------|---------------------|
| Strong trend | Use differencing or trend component |
| Clear seasonality | Use SARIMA, Holt-Winters, or Prophet |
| Changing seasonality | Use STL-based or Prophet |
| Large residuals | More complex models may help |
| Outliers in residuals | Use robust methods or anomaly handling |

## Diagnostic Workflow

### Step 1: Visual Analysis

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series plot
series.plot(ax=axes[0, 0], title='Time Series')

# Rolling statistics
rolling_mean = series.rolling(window=12).mean()
rolling_std = series.rolling(window=12).std()
axes[0, 1].plot(series, label='Original', alpha=0.5)
axes[0, 1].plot(rolling_mean, label='Rolling Mean')
axes[0, 1].plot(rolling_std, label='Rolling Std')
axes[0, 1].legend()
axes[0, 1].set_title('Rolling Statistics')

# ACF
plot_acf(series.dropna(), ax=axes[1, 0], lags=40)

# PACF
plot_pacf(series.dropna(), ax=axes[1, 1], lags=40)

plt.tight_layout()
```

### Step 2: Statistical Tests

```python
from statsmodels.tsa.stattools import adfuller, kpss

def diagnose_series(series):
    """Run diagnostic tests on time series."""
    results = {}

    # Stationarity tests
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf_pvalue'] = adf_result[1]
    results['adf_stationary'] = adf_result[1] < 0.05

    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    results['kpss_pvalue'] = kpss_result[1]
    results['kpss_stationary'] = kpss_result[1] >= 0.05

    # Seasonality detection (via ACF peaks)
    from statsmodels.tsa.stattools import acf
    acf_values = acf(series.dropna(), nlags=40)
    potential_periods = [i for i in [7, 12, 24, 52, 365] if i < len(acf_values)]
    seasonal_acf = {p: acf_values[p] for p in potential_periods}
    results['seasonal_acf'] = seasonal_acf

    return results

diagnosis = diagnose_series(series)
print(f"ADF suggests stationarity: {diagnosis['adf_stationary']}")
print(f"KPSS suggests stationarity: {diagnosis['kpss_stationary']}")
print(f"Seasonal ACF values: {diagnosis['seasonal_acf']}")
```

### Step 3: Decomposition

```python
from statsmodels.tsa.seasonal import STL

def decompose_and_analyze(series, period):
    """Decompose and analyze components."""
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Check residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(result.resid.dropna(), lags=20)
    residuals_white_noise = lb_result['lb_pvalue'].min() > 0.05

    # Variance explained
    total_var = series.var()
    trend_var = result.trend.var()
    seasonal_var = result.seasonal.var()
    residual_var = result.resid.var()

    return {
        'components': result,
        'residuals_white_noise': residuals_white_noise,
        'trend_variance_pct': trend_var / total_var * 100,
        'seasonal_variance_pct': seasonal_var / total_var * 100,
        'residual_variance_pct': residual_var / total_var * 100
    }

decomp = decompose_and_analyze(series, period=12)
print(f"Trend explains {decomp['trend_variance_pct']:.1f}% of variance")
print(f"Seasonality explains {decomp['seasonal_variance_pct']:.1f}% of variance")
print(f"Residuals are white noise: {decomp['residuals_white_noise']}")
```

## Common Patterns and Treatments

### Pattern Recognition Guide

| What You See | Likely Issue | Treatment |
|--------------|--------------|-----------|
| Upward/downward trend | Non-stationary (trend) | Differencing (d=1) |
| Repeating pattern | Seasonality | Seasonal model or differencing |
| Slow ACF decay | Unit root | Differencing |
| ACF spikes at lag 12, 24, ... | Yearly seasonality | Seasonal differencing |
| Variance increases with level | Non-constant variance | Log transform |
| Sudden level shifts | Structural break | Intervention analysis or separate models |

### Treatment Summary

```python
def recommend_treatment(diagnosis, decomposition):
    """Recommend transformations based on diagnostics."""
    treatments = []

    # Stationarity
    if not diagnosis['adf_stationary']:
        treatments.append("Apply first differencing (d=1)")

    # Seasonality
    max_seasonal = max(diagnosis['seasonal_acf'].items(), key=lambda x: x[1])
    if max_seasonal[1] > 0.3:
        treatments.append(f"Model seasonality with period {max_seasonal[0]}")

    # Variance
    if decomposition['trend_variance_pct'] > 50:
        treatments.append("Consider variance-stabilizing transform (log)")

    # Residuals
    if not decomposition['residuals_white_noise']:
        treatments.append("Residuals have structure; consider more complex model")

    return treatments

treatments = recommend_treatment(diagnosis, decomp)
for t in treatments:
    print(f"- {t}")
```

## Concept Application by Method

### How Methods Use These Concepts

| Method | Stationarity | Seasonality | Trend | CV Approach |
|--------|--------------|-------------|-------|-------------|
| ARIMA | Required (via d) | SARIMA extension | Via differencing | Walk-forward |
| ETS | Handled internally | Explicit component | Explicit component | Walk-forward |
| Prophet | Robust | Fourier terms | Piecewise linear | Built-in CV |
| N-BEATS | Implicit | Can learn | Can learn | Walk-forward |
| DeepAR | Implicit | Via features | Via features | Walk-forward |
| TFT | Implicit | Via features | Via features | Walk-forward |

### Pre-processing Checklist

Before fitting any model:

1. Check for and handle missing values
2. Test for stationarity (ADF + KPSS)
3. Identify seasonality period(s)
4. Examine trend presence
5. Choose additive vs multiplicative decomposition
6. Apply necessary transformations
7. Set up proper time series cross-validation

## Integration with Modeling

### From Concepts to Models

```python
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def select_model(series, diagnosis, decomposition, period):
    """Select model based on diagnostics."""

    # If simple patterns, try classical
    if decomposition['residuals_white_noise']:
        if diagnosis['seasonal_acf'].get(period, 0) > 0.3:
            # Clear seasonality
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=period,
                damped_trend=True
            )
        else:
            # No seasonality
            model = ExponentialSmoothing(
                series,
                trend='add',
                damped_trend=True
            )
    else:
        # Complex patterns, use auto selection
        model = auto_arima(
            series,
            seasonal=True,
            m=period,
            stepwise=True,
            suppress_warnings=True
        )

    return model
```

## Further Reading

For detailed information on each concept, see:

- [Stationarity](stationarity/ReadMe.md) - Testing and achieving stationarity
- [Seasonality](seasonality/ReadMe.md) - Identifying and modeling seasonal patterns
- [Trend Decomposition](trend-decomposition/ReadMe.md) - Separating series components
- [Cross-Validation for Time Series](cross-validation-for-time-series/ReadMe.md) - Proper evaluation methods
