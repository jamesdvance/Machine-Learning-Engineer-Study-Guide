# Prophet

## Summary

Prophet is a forecasting library developed by Meta (Facebook) designed for business time series with strong seasonal patterns and holiday effects. It uses an additive regression model that decomposes time series into trend, seasonality, and holidays. Prophet is particularly valuable for analysts who need quick, reasonable forecasts without deep time series expertise, and for series with missing data, outliers, or irregular patterns.

Key points to remember:

- Prophet models time series as: y(t) = g(t) + s(t) + h(t) + e(t) (trend + seasonality + holidays + error)
- Handles multiple seasonalities automatically (daily, weekly, yearly)
- Built-in support for holidays and special events
- Robust to missing data, outliers, and trend changes
- Provides intuitive decomposition plots for interpretability
- Works well "out of the box" with minimal tuning for business data
- Installation can be complex due to Stan dependency; consider NeuralProphet as an alternative
- Best for daily/weekly data with strong seasonality; less suitable for high-frequency or purely autoregressive series

## When to Use Prophet

Prophet excels when:

- Data has strong seasonal patterns (weekly, yearly)
- Holidays and special events affect the series
- The series has missing values or irregular observations
- Trend changes have occurred (regime shifts)
- You need quick baseline forecasts with minimal configuration
- Interpretable component decomposition is valuable
- The forecast horizon is days to months (not seconds or years)

Consider alternatives when:

- The series is high-frequency (sub-hourly) without clear seasonality
- Strong autoregressive patterns dominate (use ARIMA)
- You have multivariate dependencies (use machine learning)
- The series is very short (under one year of daily data)
- Installation constraints exist (consider statsforecast or NeuralProphet)

## The Prophet Model

### Decomposition

Prophet decomposes the time series into components:

```
y(t) = g(t) + s(t) + h(t) + e(t)
```

| Component | Description |
|-----------|-------------|
| g(t) | Trend: Long-term increase or decrease |
| s(t) | Seasonality: Periodic patterns (daily, weekly, yearly) |
| h(t) | Holidays: Effects of holidays and events |
| e(t) | Error: Residual not explained by the model |

### Trend Component

Prophet offers two trend models:

**Linear Trend with Changepoints**:
```
g(t) = (k + a(t)' * delta) * t + (m + a(t)' * gamma)
```

Where:
- k is the base growth rate
- delta is a vector of rate changes at changepoints
- m is the offset
- a(t) indicates which changepoints have occurred by time t

**Logistic Growth** (saturating trend):
```
g(t) = C(t) / (1 + exp(-(k + a(t)' * delta) * (t - (m + a(t)' * gamma))))
```

Where C(t) is the carrying capacity, allowing for growth that saturates at a maximum.

### Seasonality Component

Prophet models seasonality using Fourier series:

```
s(t) = sum_{n=1}^{N} (a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))
```

Where:
- P is the period (365.25 for yearly, 7 for weekly)
- N controls the flexibility (higher N = more complex patterns)

Default Fourier orders:
- Yearly: N=10
- Weekly: N=3
- Daily: N=4

### Holiday Component

Holidays are modeled as indicator functions:

```
h(t) = Z(t) * kappa
```

Where Z(t) is a matrix of holiday indicators and kappa is the vector of holiday effects.

## Basic Usage

### Installation

```bash
# Prophet requires cmdstan or pystan
pip install prophet

# Alternative: lighter installation with cmdstanpy
pip install prophet
```

### Minimal Example

```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns named 'ds' (date) and 'y' (target)
df = pd.read_csv('data.csv')
df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# Fit model
model = Prophet()
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=30)

# Predict
forecast = model.predict(future)

# Key output columns
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

### Visualization

```python
# Full forecast plot
fig1 = model.plot(forecast)

# Component decomposition
fig2 = model.plot_components(forecast)
```

## Trend Configuration

### Changepoint Detection

Prophet automatically detects trend changes. Configure with:

```python
model = Prophet(
    changepoint_prior_scale=0.05,  # Flexibility of trend changes (default: 0.05)
    changepoint_range=0.8,          # Proportion of history to consider (default: 0.8)
    n_changepoints=25               # Number of potential changepoints (default: 25)
)
```

| Parameter | Effect of Increasing |
|-----------|---------------------|
| changepoint_prior_scale | More flexible trend, fits data more closely |
| n_changepoints | More potential locations for trend changes |

**Guidance**:
- If trend is underfitting (too smooth): increase changepoint_prior_scale (e.g., 0.5)
- If trend is overfitting (too wiggly): decrease changepoint_prior_scale (e.g., 0.01)

### Manual Changepoints

When you know trend change dates:

```python
model = Prophet(
    changepoints=['2020-03-15', '2021-01-01']  # COVID, new year regime
)
```

### Saturating Growth

For metrics with natural upper bounds:

```python
# Logistic growth requires cap (and optionally floor)
df['cap'] = 1000000  # Market size
df['floor'] = 0

model = Prophet(growth='logistic')
model.fit(df)

# Future dataframe also needs cap/floor
future = model.make_future_dataframe(periods=30)
future['cap'] = 1000000
future['floor'] = 0
```

### Flat Trend

For series without trend:

```python
model = Prophet(growth='flat')
```

## Seasonality Configuration

### Default Seasonalities

Prophet includes yearly, weekly, and daily seasonality by default (when applicable):

```python
# Yearly: auto-enabled if >2 years of data
# Weekly: auto-enabled if >2 weeks of data
# Daily: only for sub-daily data
```

### Disabling Seasonalities

```python
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
```

### Custom Seasonality

Add domain-specific patterns:

```python
model = Prophet()

# Monthly seasonality
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Quarterly business cycles
model.add_seasonality(
    name='quarterly',
    period=91.25,
    fourier_order=3
)

model.fit(df)
```

### Conditional Seasonality

Different patterns for different conditions:

```python
# Add boolean column for condition
df['is_weekend'] = df['ds'].dt.dayofweek >= 5

model = Prophet(weekly_seasonality=False)

# Different weekly patterns for weekday vs weekend
model.add_seasonality(
    name='weekly_weekday',
    period=7,
    fourier_order=3,
    condition_name='is_weekday'
)
model.add_seasonality(
    name='weekly_weekend',
    period=7,
    fourier_order=3,
    condition_name='is_weekend'
)

# Must include condition in future dataframe
future['is_weekday'] = ~(future['ds'].dt.dayofweek >= 5)
future['is_weekend'] = future['ds'].dt.dayofweek >= 5
```

### Multiplicative Seasonality

When seasonal effects scale with trend:

```python
model = Prophet(seasonality_mode='multiplicative')
```

Or per-seasonality:

```python
model.add_seasonality(
    name='yearly',
    period=365.25,
    fourier_order=10,
    mode='multiplicative'
)
```

## Holidays and Events

### Country Holidays

```python
model = Prophet()
model.add_country_holidays(country_name='US')
model.fit(df)
```

Available countries include US, UK, DE, FR, and many others.

### Custom Holidays

```python
# Create holiday dataframe
holidays = pd.DataFrame({
    'holiday': 'black_friday',
    'ds': pd.to_datetime(['2022-11-25', '2023-11-24', '2024-11-29']),
    'lower_window': -1,  # Effect starts 1 day before
    'upper_window': 1,   # Effect ends 1 day after
})

# Add another holiday
cyber_monday = pd.DataFrame({
    'holiday': 'cyber_monday',
    'ds': pd.to_datetime(['2022-11-28', '2023-11-27', '2024-12-02']),
    'lower_window': 0,
    'upper_window': 0,
})

holidays = pd.concat([holidays, cyber_monday])

model = Prophet(holidays=holidays)
model.fit(df)
```

### Holiday Windows

The window parameters capture effects before and after the actual date:

| Parameter | Meaning |
|-----------|---------|
| lower_window | Days before the holiday date (negative = before) |
| upper_window | Days after the holiday date |

Example: Black Friday effect from Wednesday to Sunday:
```python
lower_window=-2  # Wednesday, Thursday
upper_window=2   # Saturday, Sunday
```

### Holiday Prior Scale

Control holiday effect regularization:

```python
model = Prophet(holidays_prior_scale=10.0)  # Default: 10.0
```

Higher values allow larger holiday effects.

## Additional Regressors

### Adding External Features

```python
# Add weather as regressor
df['temperature'] = weather_data['temp']

model = Prophet()
model.add_regressor('temperature')
model.fit(df)

# Must include regressor in future dataframe
future = model.make_future_dataframe(periods=30)
future['temperature'] = get_weather_forecast(periods=30)  # Need future values
```

### Regressor Configuration

```python
model.add_regressor(
    'temperature',
    prior_scale=0.5,           # Regularization (default: inferred)
    standardize='auto',        # 'auto', True, or False
    mode='additive'            # 'additive' or 'multiplicative'
)
```

**Important**: You must know or forecast regressor values for the forecast horizon.

## Uncertainty Intervals

### Prediction Intervals

```python
model = Prophet(interval_width=0.95)  # 95% interval (default: 0.8)
model.fit(df)
forecast = model.predict(future)

# yhat_lower and yhat_upper contain interval bounds
```

### Sources of Uncertainty

Prophet models three sources:

1. **Trend uncertainty**: Future trend may differ from history
2. **Seasonality uncertainty**: Estimated from data
3. **Observation noise**: Residual variation

```python
model = Prophet(
    interval_width=0.95,
    uncertainty_samples=1000  # MCMC samples for uncertainty
)
```

### Disabling Uncertainty

For faster predictions when intervals aren't needed:

```python
model = Prophet(uncertainty_samples=0)
```

## Model Evaluation

### Cross-Validation

Prophet provides built-in time series cross-validation:

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Evaluate on rolling windows
df_cv = cross_validation(
    model,
    initial='730 days',  # Initial training period
    period='180 days',   # Spacing between cutoff dates
    horizon='30 days'    # Forecast horizon
)

# Calculate metrics
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'mape', 'rmse', 'mae']])
```

### Visualization of CV Results

```python
from prophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')
```

### Manual Train/Test Split

```python
# Train on history
train = df[df['ds'] < '2023-01-01']
test = df[df['ds'] >= '2023-01-01']

model = Prophet()
model.fit(train)

# Predict test period
forecast = model.predict(test[['ds']])

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test['y'], forecast['yhat'])
rmse = mean_squared_error(test['y'], forecast['yhat'], squared=False)
```

## Tuning Prophet

### Hyperparameter Tuning

Key parameters to tune:

| Parameter | Range | Effect |
|-----------|-------|--------|
| changepoint_prior_scale | 0.001 - 0.5 | Trend flexibility |
| seasonality_prior_scale | 0.01 - 10 | Seasonality flexibility |
| holidays_prior_scale | 0.01 - 10 | Holiday effect size |
| seasonality_mode | additive, multiplicative | Seasonality type |

```python
import itertools
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Grid search
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes = []

for params in all_params:
    model = Prophet(**params)
    model.fit(df)
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    mapes.append(df_p['mape'].values[0])

best_params = all_params[np.argmin(mapes)]
print(f'Best params: {best_params}')
```

### Fourier Order Tuning

Increase for more complex patterns, decrease for simpler:

```python
model = Prophet(
    yearly_seasonality=20,  # Increase from default 10
    weekly_seasonality=5    # Increase from default 3
)
```

## Handling Outliers

### Removing Outliers

Prophet is robust to outliers but extreme values can still affect the model:

```python
# Remove outliers by setting y to NaN
df.loc[df['y'] > threshold, 'y'] = None
# Prophet handles NaN values automatically
```

### Capping Values

```python
# Cap extreme values
df['y'] = df['y'].clip(lower=0, upper=df['y'].quantile(0.99))
```

## Performance and Scaling

### Fitting Speed

Prophet can be slow for large datasets. Options:

```python
# Reduce MCMC iterations for faster fitting
model = Prophet(
    mcmc_samples=0  # Use MAP estimation instead of MCMC
)
```

### Parallel Forecasting

For multiple time series:

```python
from concurrent.futures import ProcessPoolExecutor

def fit_and_forecast(series_df):
    model = Prophet()
    model.fit(series_df)
    future = model.make_future_dataframe(periods=30)
    return model.predict(future)

with ProcessPoolExecutor(max_workers=4) as executor:
    forecasts = list(executor.map(fit_and_forecast, list_of_dfs))
```

## Common Pitfalls

### Pitfall 1: Forgetting Future Regressors

**Symptom**: Error when predicting with regressors.

**Solution**: Always include regressor values in the future dataframe.

### Pitfall 2: Overfitting Trend

**Symptom**: Trend captures noise instead of signal.

**Solution**: Reduce changepoint_prior_scale (e.g., 0.01).

### Pitfall 3: Wrong Seasonality Mode

**Symptom**: Poor fit, especially when trend changes.

**Solution**: Try multiplicative seasonality if effects scale with level.

### Pitfall 4: Installation Issues

**Symptom**: Stan/cmdstan errors.

**Solution**:
- Use conda: `conda install -c conda-forge prophet`
- Or try NeuralProphet: `pip install neuralprophet`

### Pitfall 5: Short History

**Symptom**: Unreliable seasonal estimates.

**Solution**: Need at least 2 full cycles of each seasonality (2 years for yearly).

## Prophet vs Alternatives

| Aspect | Prophet | ARIMA | Exponential Smoothing |
|--------|---------|-------|----------------------|
| Multiple seasonalities | Native | Limited | Single |
| Holiday effects | Built-in | Manual | Manual |
| Missing data | Handles automatically | Requires imputation | Requires imputation |
| Interpretability | Excellent (components) | Moderate | Good |
| Automatic configuration | Mostly | Requires selection | Partially |
| Exogenous variables | Regressors | ARIMAX | Limited |
| Speed | Slow (Stan) | Fast | Very fast |
| Installation | Complex | Simple | Simple |

**Guidance**:
- Use Prophet for business forecasting with holidays and multiple seasonalities
- Use ARIMA when autocorrelation matters and installation simplicity is needed
- Use Exponential Smoothing for simpler patterns and maximum speed

## Complete Example

```python
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('daily_sales.csv')
df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# Create holidays dataframe
holidays = pd.DataFrame({
    'holiday': 'black_friday',
    'ds': pd.to_datetime(['2021-11-26', '2022-11-25', '2023-11-24']),
    'lower_window': -1,
    'upper_window': 1,
})

# Configure model
model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)

# Add country holidays
model.add_country_holidays(country_name='US')

# Add custom regressor (if available)
# df['marketing_spend'] = marketing_data
# model.add_regressor('marketing_spend')

# Fit
model.fit(df)

# Cross-validation
df_cv = cross_validation(
    model,
    initial='365 days',
    period='90 days',
    horizon='30 days'
)
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'mape', 'mae', 'rmse']])

# Final forecast
future = model.make_future_dataframe(periods=90)
# future['marketing_spend'] = future_marketing  # If using regressor
forecast = model.predict(future)

# Visualization
fig1 = model.plot(forecast)
plt.title('Sales Forecast')
plt.savefig('prophet_forecast.png')

fig2 = model.plot_components(forecast)
plt.savefig('prophet_components.png')

# Extract components for analysis
components = forecast[['ds', 'trend', 'weekly', 'yearly', 'holidays', 'yhat']]
print(components.tail())
```

## Key Takeaways

1. **Designed for business time series**: Prophet excels at the daily/weekly data common in business with holidays and seasonality.

2. **Decomposition is powerful**: The ability to inspect trend, seasonality, and holiday components makes Prophet highly interpretable.

3. **Robust defaults**: Prophet works well out of the box; tuning is often unnecessary for initial forecasts.

4. **Know the limitations**: Prophet struggles with high-frequency data, purely autoregressive patterns, and very short histories.

5. **Cross-validation is built in**: Use Prophet's CV tools to reliably evaluate forecasts.

6. **Consider alternatives**: For simpler needs, statsforecast or NeuralProphet may be easier to install and faster.

7. **Trend flexibility is key**: changepoint_prior_scale is the most important parameter to tune for underfitting/overfitting.
