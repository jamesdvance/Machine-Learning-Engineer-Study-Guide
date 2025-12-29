# DeepAR

## Summary

DeepAR is a probabilistic forecasting method based on autoregressive recurrent neural networks, developed by Amazon. It learns a global model across many related time series, sharing patterns and transferring knowledge between series. DeepAR outputs full probability distributions rather than point forecasts, making it valuable when uncertainty quantification matters. It is particularly effective when you have many related time series with varying histories and the ability to leverage cross-series learning.

Key points to remember:

- DeepAR trains one model on many related time series, enabling knowledge transfer between series
- Produces probabilistic forecasts (full distributions, not just point estimates)
- Uses autoregressive RNN architecture with learned likelihood parameters
- Handles varying length histories and missing values naturally
- Particularly strong when you have hundreds to thousands of related series
- Available in GluonTS, PyTorch Forecasting, and Amazon SageMaker
- Requires more data than classical methods; not suitable for single short series
- Inference is sequential (autoregressive), making it slower than direct methods

## When to Use DeepAR

DeepAR is appropriate when:

- You have many related time series (tens to thousands)
- Series share common patterns that can be learned jointly
- Probabilistic forecasts are needed (prediction intervals, quantiles)
- Series have varying lengths and start times
- You have sufficient data (at least a few hundred observations per series)
- Cross-series features exist (categories, hierarchies, static metadata)
- Uncertainty quantification is important for decision-making

Consider alternatives when:

- You have a single time series (use ARIMA, ETS, or Prophet)
- Series are too short (under ~100 observations each)
- You need maximum interpretability (use classical methods)
- Computational resources are very limited
- Real-time inference is critical (consider N-BEATS or simpler models)

## DeepAR Architecture

### Core Concept

DeepAR uses an autoregressive recurrent neural network that generates forecasts by:

1. Encoding the historical context with an LSTM/GRU
2. Sampling from a learned probability distribution at each time step
3. Feeding sampled values back as inputs for subsequent steps

```
Historical Values      Forecast Horizon
[y_1, y_2, ..., y_T] -> [y_T+1, y_T+2, ..., y_T+H]
        |                       |
        v                       v
   [LSTM Encoder]  ->  [Autoregressive Decoder with Sampling]
```

### Training Objective

DeepAR maximizes the log-likelihood of the training data:

```
L = sum_{i,t} log p(y_{i,t} | y_{i,1:t-1}, x_{i,1:t}, theta)
```

Where:
- i indexes the time series
- t indexes time steps
- x represents covariates
- theta are the model parameters

The model learns parameters of a probability distribution (e.g., mean and variance for Gaussian, shape and rate for Negative Binomial).

### Probability Distributions

DeepAR supports different distributions based on data characteristics:

| Distribution | Use Case | Learned Parameters |
|--------------|----------|-------------------|
| Gaussian | Continuous, unbounded | mu (mean), sigma (std) |
| Negative Binomial | Count data, overdispersed | mu, alpha |
| Student's t | Heavy-tailed, outliers | mu, sigma, df |
| Beta | Bounded [0, 1] data | alpha, beta |

```python
# GluonTS configuration
from gluonts.mx.distribution import GaussianOutput, NegativeBinomialOutput

# For continuous data
estimator = DeepAREstimator(
    distr_output=GaussianOutput(),
    ...
)

# For count data
estimator = DeepAREstimator(
    distr_output=NegativeBinomialOutput(),
    ...
)
```

### Covariates

DeepAR incorporates three types of features:

| Feature Type | Description | Example |
|--------------|-------------|---------|
| Static (real/categorical) | Constant per series | Store ID, product category |
| Known time-varying | Known for past and future | Day of week, holiday indicator |
| Unknown time-varying | Only known for past | Actual temperature (past) |

## Implementation with GluonTS

### Basic Usage

```python
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import pandas as pd

# Prepare data as ListDataset
training_data = ListDataset(
    [
        {
            "start": pd.Timestamp("2020-01-01", freq="D"),
            "target": series_1_values,  # numpy array
            "feat_static_cat": [0],     # Category index
        },
        {
            "start": pd.Timestamp("2020-01-01", freq="D"),
            "target": series_2_values,
            "feat_static_cat": [1],
        },
        # ... more series
    ],
    freq="D"
)

# Configure estimator
estimator = DeepAREstimator(
    freq="D",
    prediction_length=30,
    context_length=60,
    num_layers=2,
    num_cells=40,
    cell_type="lstm",
    dropout_rate=0.1,
    use_feat_static_cat=True,
    cardinality=[10],  # Number of categories
    trainer=Trainer(
        epochs=100,
        learning_rate=1e-3,
        batch_size=32
    )
)

# Train
predictor = estimator.train(training_data)

# Generate forecasts
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)
```

### Accessing Forecast Distribution

```python
# Each forecast contains samples from the predictive distribution
forecast = forecasts[0]

# Point forecast (median)
point_forecast = forecast.median

# Mean forecast
mean_forecast = forecast.mean

# Quantiles
lower_90 = forecast.quantile(0.05)
upper_90 = forecast.quantile(0.95)

# Prediction interval
pi_90 = forecast.quantile_ts(0.05), forecast.quantile_ts(0.95)

# All samples (for custom analysis)
samples = forecast.samples  # Shape: (num_samples, prediction_length)
```

### Time-Varying Features

```python
# Include known covariates (available for both past and future)
training_data = ListDataset(
    [
        {
            "start": pd.Timestamp("2020-01-01", freq="D"),
            "target": series_values,
            "feat_dynamic_real": [
                day_of_week_values,    # Numeric features
                is_holiday_values,
            ],
        }
    ],
    freq="D"
)

estimator = DeepAREstimator(
    freq="D",
    prediction_length=30,
    use_feat_dynamic_real=True,
    num_feat_dynamic_real=2,
    ...
)
```

## Implementation with PyTorch Forecasting

```python
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

# Prepare DataFrame
df = pd.DataFrame({
    "time_idx": time_indices,
    "series_id": series_identifiers,
    "target": values,
    "known_covariate": covariate_values
})

# Create dataset
training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="target",
    group_ids=["series_id"],
    max_encoder_length=60,
    max_prediction_length=30,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["series_id"]),
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=64)

# Configure model
model = DeepAR.from_dataset(
    training,
    hidden_size=40,
    rnn_layers=2,
    dropout=0.1,
    learning_rate=1e-3,
)

# Train
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, train_dataloaders=train_dataloader)

# Predict
predictions = model.predict(test_dataloader, mode="prediction", n_samples=100)
```

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| context_length | 1-5x prediction_length | More context = better patterns but more compute |
| num_layers | 1-3 | Deeper = more capacity, risk of overfitting |
| num_cells | 20-100 | Wider = more capacity |
| dropout_rate | 0.0-0.3 | Higher = more regularization |
| cell_type | lstm, gru | GRU faster, LSTM more expressive |
| num_samples | 100-1000 | More samples = better quantile estimates |

### Context Length Selection

```python
# Rule of thumb: 1-3x the prediction length
# But must capture at least one full seasonal cycle

prediction_length = 30  # 1 month
context_length = 90     # 3 months (captures monthly patterns)

# For yearly seasonality with daily data
prediction_length = 30
context_length = 365 + 30  # Full year + buffer
```

### Cross-Validation

```python
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

# Generate forecasts
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=100
)

# Evaluate
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(
    iter(ts_it),
    iter(forecast_it),
    num_series=len(test_data)
)

print(f"MASE: {agg_metrics['MASE']:.3f}")
print(f"MAPE: {agg_metrics['MAPE']:.3f}")
print(f"QuantileLoss[0.5]: {agg_metrics['QuantileLoss[0.5]']:.3f}")
print(f"Coverage[0.9]: {agg_metrics['Coverage[0.9]']:.3f}")
```

## Scaling and Performance

### Data Normalization

DeepAR uses per-series normalization to handle series with different scales:

```python
# GluonTS handles normalization internally by default
# For custom normalization:
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
)
```

### Training on Many Series

```python
# For thousands of series, increase batch size
trainer = Trainer(
    epochs=100,
    batch_size=128,  # Larger batch for more series
    num_batches_per_epoch=50,  # Limit batches per epoch
)

# Use sampling to avoid memory issues
estimator = DeepAREstimator(
    ...
    train_sampler=ExpectedNumInstanceSampler(num_instances=1.0)
)
```

### GPU Training

```python
# GluonTS with MXNet
import mxnet as mx
trainer = Trainer(
    ctx=mx.gpu(),
    epochs=100,
    hybridize=True
)

# PyTorch Forecasting
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    precision=16  # Mixed precision for speed
)
```

## Cold Start and Missing Data

### Handling New Series

DeepAR handles cold start through:

1. Static features transfer knowledge from similar series
2. Learned prior distribution when history is short

```python
# Encode series similarity via static features
training_data = ListDataset(
    [
        {
            "target": new_series_values[:10],  # Short history
            "feat_static_cat": [category_id],   # Category helps!
            "feat_static_real": [avg_sales],    # Scale hint
        }
    ],
    freq="D"
)
```

### Missing Values

DeepAR naturally handles missing values during training:

```python
# NaN values in target are handled
training_data = ListDataset(
    [
        {
            "target": series_with_nans,  # np.nan allowed
        }
    ],
    freq="D"
)
```

## Common Pitfalls

### Pitfall 1: Not Enough Series

**Symptom**: Poor generalization, overfitting.

**Solution**: DeepAR needs many series (ideally 100+) to learn shared patterns. For few series, use classical methods.

### Pitfall 2: Too Short Context

**Symptom**: Misses seasonal patterns.

**Solution**: Context length should capture at least one full seasonal cycle.

### Pitfall 3: Ignoring Static Features

**Symptom**: Poor cold start performance.

**Solution**: Include category information as static features to enable cross-series learning.

### Pitfall 4: Wrong Distribution

**Symptom**: Poor uncertainty estimates, nonsensical values.

**Solution**: Match distribution to data type (Gaussian for continuous, Negative Binomial for counts).

### Pitfall 5: Over-reliance on Samples

**Symptom**: High variance in quantile estimates.

**Solution**: Use more samples (100-1000) for stable estimates.

## DeepAR vs Alternatives

| Aspect | DeepAR | N-BEATS | TFT | Classical |
|--------|--------|---------|-----|-----------|
| Probabilistic | Native | Via ensemble | Native | Via bootstrap |
| Cross-series learning | Native | No | Native | No |
| Interpretability | Low | Medium | High | High |
| Covariates | Yes | Limited | Yes | Limited |
| Cold start | Good | Poor | Good | Poor |
| Training speed | Moderate | Fast | Slow | Fast |
| Inference speed | Slow (autoregressive) | Fast | Moderate | Fast |

**Guidance**:
- Use DeepAR for many related series with probabilistic needs
- Use N-BEATS for pure accuracy on individual series
- Use TFT when interpretability and covariates both matter
- Use classical methods for single series or maximum speed

## Production Deployment

### Amazon SageMaker

```python
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import image_uris

role = get_execution_role()
image = image_uris.retrieve("forecasting-deepar", region, version="latest")

estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/output'
)

estimator.set_hyperparameters(
    time_freq='D',
    context_length=60,
    prediction_length=30,
    epochs=100,
)

estimator.fit({'train': 's3://bucket/train'})
```

### Inference Optimization

```python
# Pre-compute static features
# Cache LSTM states for known history
# Batch predictions for multiple series

# Reduce samples for faster inference
forecast_it, _ = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=20  # Fewer samples for production
)
```

## Key Takeaways

1. **Many series, one model**: DeepAR shines when you have many related series to learn from collectively.

2. **Probabilistic by design**: Full predictive distributions enable risk-aware decision making.

3. **Covariates matter**: Static features enable knowledge transfer; time-varying features capture external effects.

4. **Distribution choice is critical**: Match the output distribution to your data type.

5. **Context captures patterns**: Ensure context length spans at least one full seasonal cycle.

6. **Not for single series**: For individual series, classical methods or N-BEATS are better choices.

7. **Inference is sequential**: Autoregressive sampling makes inference slower than direct methods.
