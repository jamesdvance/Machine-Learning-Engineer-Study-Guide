# Cross-Validation for Time Series

## Summary

Cross-validation for time series requires special handling because observations are temporally ordered and often autocorrelated. Standard k-fold cross-validation, which randomly shuffles data, violates the temporal structure and leads to overly optimistic performance estimates. Time series cross-validation methods respect temporal order, ensuring that models are always trained on past data and tested on future data, mimicking real-world forecasting scenarios.

Key points to remember:

- Never use random train/test splits for time series (causes data leakage)
- Walk-forward validation trains on expanding or sliding windows
- Test sets should always be chronologically after training sets
- Gap between train and test prevents leakage from lagged features
- Multiple test folds provide robust performance estimates
- Forecast horizon affects evaluation (multi-step vs single-step)
- Nested CV needed for hyperparameter tuning

## Why Standard CV Fails

### The Temporal Ordering Problem

Standard k-fold CV randomly assigns observations to folds:

```
Standard k-fold (WRONG for time series):
Fold 1: [Aug, Dec, Feb] train, [May, Oct] test
Fold 2: [May, Oct, Feb] train, [Aug, Dec] test
...

Problem: Training on December to predict August
         Future data used to predict past!
```

### Data Leakage Issues

| Leakage Source | How It Happens |
|----------------|----------------|
| Temporal leakage | Future values in training set |
| Lagged features | Future information in engineered features |
| Global statistics | Mean/std computed on full data including test |
| Shuffled sequences | Sequence order ignored |

### Consequences

- Overestimated performance
- Models fail in production
- False confidence in forecasts
- Unreliable model selection

## Walk-Forward Validation

### Expanding Window

Training window grows with each fold:

```
Fold 1: [----Train----][Test]..............
Fold 2: [------Train------][Test]..........
Fold 3: [--------Train--------][Test]......
Fold 4: [----------Train----------][Test]..
Fold 5: [------------Train------------][Test]
```

```python
def expanding_window_cv(series, n_splits=5, test_size=1):
    """Generate expanding window train/test splits."""
    n = len(series)
    min_train_size = n - (n_splits * test_size)

    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        train_idx = range(0, train_end)
        test_idx = range(train_end, test_end)

        yield train_idx, test_idx

# Usage
for train_idx, test_idx in expanding_window_cv(series, n_splits=5, test_size=12):
    train = series.iloc[list(train_idx)]
    test = series.iloc[list(test_idx)]
    # Fit and evaluate
```

### Sliding Window

Fixed-size training window moves forward:

```
Fold 1: [----Train----][Test]..............
Fold 2: ..[----Train----][Test]............
Fold 3: ....[----Train----][Test]..........
Fold 4: ......[----Train----][Test]........
Fold 5: ........[----Train----][Test]......
```

```python
def sliding_window_cv(series, train_size, test_size, n_splits=5):
    """Generate sliding window train/test splits."""
    n = len(series)
    step = (n - train_size - test_size) // (n_splits - 1)

    for i in range(n_splits):
        train_start = i * step
        train_end = train_start + train_size
        test_end = train_end + test_size

        if test_end > n:
            break

        train_idx = range(train_start, train_end)
        test_idx = range(train_end, test_end)

        yield train_idx, test_idx
```

### Choosing Between Them

| Criterion | Expanding Window | Sliding Window |
|-----------|-----------------|----------------|
| More training data | Yes (later folds) | Fixed amount |
| Concept drift sensitivity | Less sensitive | More sensitive |
| Computational cost | Higher (later folds) | Constant |
| Realistic for production | Depends | Better for drifting data |

## Using scikit-learn's TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

# Basic usage
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(series):
    train = series.iloc[train_idx]
    test = series.iloc[test_idx]
    print(f"Train: {len(train)}, Test: {len(test)}")
```

### With Gap (Prevent Leakage from Lags)

```python
tscv = TimeSeriesSplit(n_splits=5, gap=7)  # 7-period gap

# Gap ensures lagged features don't leak
# If model uses lag-7 feature, gap=7 prevents leakage
```

### With Fixed Test Size

```python
tscv = TimeSeriesSplit(n_splits=5, test_size=30)

# Each test fold has exactly 30 observations
```

## Multi-Step Forecast Evaluation

### Horizon Matters

Different forecast horizons require different evaluation:

```python
def evaluate_multistep(model, train, test, horizon):
    """Evaluate multi-step ahead forecasts."""
    model.fit(train)
    forecast = model.forecast(steps=horizon)

    # Evaluate at each horizon
    errors_by_horizon = {}
    for h in range(1, horizon + 1):
        actual = test.iloc[:h]
        predicted = forecast.iloc[:h]
        errors_by_horizon[h] = mean_absolute_error(actual, predicted)

    return errors_by_horizon
```

### Rolling Origin Evaluation

Most rigorous approach for multi-step:

```python
def rolling_origin_cv(series, train_size, horizon, step=1):
    """Rolling origin (walk-forward) validation."""
    n = len(series)
    origins = range(train_size, n - horizon + 1, step)

    results = []
    for origin in origins:
        train = series.iloc[:origin]
        test = series.iloc[origin:origin + horizon]

        # Fit model and forecast
        model = fit_model(train)
        forecast = model.forecast(horizon)

        # Evaluate
        for h in range(horizon):
            results.append({
                'origin': origin,
                'horizon': h + 1,
                'actual': test.iloc[h],
                'forecast': forecast.iloc[h]
            })

    return pd.DataFrame(results)
```

## Grouped/Panel Time Series CV

For multiple time series with the same time index:

```python
from sklearn.model_selection import TimeSeriesSplit

def grouped_time_series_cv(df, group_col, time_col, n_splits=5):
    """Cross-validation respecting both groups and time."""
    # Get unique time points
    times = df[time_col].unique()
    times = np.sort(times)

    # Split time points
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_times_idx, test_times_idx in tscv.split(times):
        train_times = times[train_times_idx]
        test_times = times[test_times_idx]

        train_mask = df[time_col].isin(train_times)
        test_mask = df[time_col].isin(test_times)

        yield train_mask, test_mask

# Usage
for train_mask, test_mask in grouped_time_series_cv(df, 'store_id', 'date'):
    train = df[train_mask]
    test = df[test_mask]
```

## Nested Cross-Validation

For hyperparameter tuning with unbiased performance estimation:

```python
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

def nested_time_series_cv(series, param_grid, n_outer=5, n_inner=3):
    """Nested CV for time series."""
    outer_cv = TimeSeriesSplit(n_splits=n_outer)
    results = []

    for outer_train_idx, outer_test_idx in outer_cv.split(series):
        outer_train = series.iloc[outer_train_idx]
        outer_test = series.iloc[outer_test_idx]

        # Inner CV for hyperparameter tuning
        inner_cv = TimeSeriesSplit(n_splits=n_inner)
        best_score = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            inner_scores = []

            for inner_train_idx, inner_test_idx in inner_cv.split(outer_train):
                inner_train = outer_train.iloc[inner_train_idx]
                inner_test = outer_train.iloc[inner_test_idx]

                # Fit and score
                model = fit_model(inner_train, **params)
                forecast = model.forecast(len(inner_test))
                score = mean_absolute_error(inner_test, forecast)
                inner_scores.append(score)

            avg_score = np.mean(inner_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params

        # Evaluate best model on outer test set
        final_model = fit_model(outer_train, **best_params)
        forecast = final_model.forecast(len(outer_test))
        outer_score = mean_absolute_error(outer_test, forecast)

        results.append({
            'best_params': best_params,
            'outer_score': outer_score
        })

    return results
```

## Library Implementations

### statsmodels

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(series):
    train = series.iloc[train_idx]
    test = series.iloc[test_idx]

    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(len(test))

    scores.append(mean_absolute_error(test, forecast))

print(f"Mean MAE: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

### Prophet

```python
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

model = Prophet()
model.fit(df)

# Built-in cross-validation
df_cv = cross_validation(
    model,
    initial='730 days',  # Initial training period
    period='180 days',   # Spacing between cutoffs
    horizon='30 days'    # Forecast horizon
)

# Calculate metrics
metrics = performance_metrics(df_cv)
print(metrics[['horizon', 'mape', 'rmse', 'mae']])
```

### GluonTS

```python
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.evaluation.backtest import backtest_metrics

# Backtest with evaluation
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,
    predictor=predictor,
    num_samples=100
)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(
    iter(ts_it),
    iter(forecast_it)
)
```

### PyTorch Forecasting

```python
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.model_selection import TimeSeriesSplit

# Create folds based on time_idx
unique_times = df['time_idx'].unique()
tscv = TimeSeriesSplit(n_splits=5)

for train_times, test_times in tscv.split(unique_times):
    train_df = df[df['time_idx'].isin(unique_times[train_times])]
    test_df = df[df['time_idx'].isin(unique_times[test_times])]

    # Create datasets and train
    training = TimeSeriesDataSet(train_df, ...)
    validation = TimeSeriesDataSet.from_dataset(training, test_df)
```

## Metrics for Time Series CV

### Aggregate Metrics

```python
def calculate_cv_metrics(actuals, forecasts):
    """Calculate comprehensive CV metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(actuals, forecasts)
    rmse = np.sqrt(mean_squared_error(actuals, forecasts))
    mape = np.mean(np.abs((actuals - forecasts) / actuals)) * 100

    # MASE (Mean Absolute Scaled Error)
    naive_errors = np.abs(np.diff(actuals))
    mase = mae / np.mean(naive_errors)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }
```

### Per-Horizon Metrics

```python
def metrics_by_horizon(results_df):
    """Calculate metrics for each forecast horizon."""
    return results_df.groupby('horizon').apply(
        lambda g: pd.Series({
            'MAE': mean_absolute_error(g['actual'], g['forecast']),
            'RMSE': np.sqrt(mean_squared_error(g['actual'], g['forecast']))
        })
    )
```

## Common Pitfalls

### Pitfall 1: Random Shuffling

**Problem**: Using standard k-fold or random splits.

**Solution**: Always use time-ordered splits.

### Pitfall 2: No Gap with Lagged Features

**Problem**: Lagged features leak future information.

**Solution**: Set gap >= max_lag in TimeSeriesSplit.

### Pitfall 3: Global Normalization

**Problem**: Normalizing with full dataset statistics.

**Solution**: Fit scaler on training fold only:

```python
from sklearn.preprocessing import StandardScaler

for train_idx, test_idx in tscv.split(series):
    train = series.iloc[train_idx]
    test = series.iloc[test_idx]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
```

### Pitfall 4: Single Train/Test Split

**Problem**: One split is not robust to when the split occurs.

**Solution**: Use multiple folds and report mean/std.

### Pitfall 5: Ignoring Forecast Horizon

**Problem**: Evaluating only 1-step ahead when production needs multi-step.

**Solution**: Evaluate at actual production horizon:

```python
# If production needs 30-day forecast, evaluate 30-day forecast
forecast = model.forecast(30)
mae_30 = mean_absolute_error(test[:30], forecast)
```

## Best Practices

### Recommended Workflow

1. **Define evaluation strategy** before modeling
2. **Use multiple folds** (5-10 for robust estimates)
3. **Match evaluation to production** (same horizon, metrics)
4. **Include gap** if using lagged features
5. **Report uncertainty** (std across folds)
6. **Use nested CV** for hyperparameter tuning
7. **Compare to baseline** (naive forecast)

### Baseline Comparisons

```python
def naive_forecast(train, horizon):
    """Last observed value repeated."""
    return np.full(horizon, train.iloc[-1])

def seasonal_naive(train, horizon, period):
    """Same period last cycle."""
    return train.iloc[-period:-period+horizon].values

# Always compare models to baselines
naive_mae = evaluate(naive_forecast(train, len(test)), test)
model_mae = evaluate(model.forecast(len(test)), test)
improvement = (naive_mae - model_mae) / naive_mae * 100
print(f"Improvement over naive: {improvement:.1f}%")
```

## Key Takeaways

1. **Never shuffle time series**: Use time-ordered splits only.

2. **Use walk-forward validation**: Expanding or sliding windows.

3. **Match production conditions**: Same horizon, gap, and metrics.

4. **Multiple folds for robustness**: Report mean and standard deviation.

5. **Gap prevents leakage**: Set gap >= max lag for lagged features.

6. **Nested CV for tuning**: Separate inner and outer loops.

7. **Compare to baselines**: Naive forecasts establish minimum bar.
