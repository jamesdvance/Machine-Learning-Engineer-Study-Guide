# AutoGluon

## Summary

AutoGluon is an open-source AutoML framework developed by Amazon that produces high-accuracy models on tabular, image, text, and time series data with minimal code. Its distinguishing approach is that it does not rely primarily on hyperparameter optimization. Instead, it trains a diverse set of models, applies k-fold bagging, and combines them via multi-layer stack ensembling to produce a final prediction that consistently outperforms individual models.

AutoGluon-Tabular is the most widely used module and the focus of this chapter. In benchmarks across 50+ classification and regression tasks from Kaggle and the OpenML AutoML Benchmark, AutoGluon-Tabular was shown to be faster, more robust, and more accurate than most competing AutoML frameworks.

Key points to remember:

- Ensemble-first design: AutoGluon achieves accuracy through model diversity and stacking rather than extensive hyperparameter search
- Multi-layer stacking: Base models feed predictions into stacker models across multiple layers, with a final weighted ensemble on top
- Bagging via cross-validation: Each model type is trained K times (typically 8 folds), producing out-of-fold predictions that prevent data leakage during stacking
- Presets system: Built-in presets like `best_quality`, `high_quality`, `medium_quality`, and `light` let you trade off accuracy against compute time with a single parameter
- Minimal code: A full training pipeline can be launched in three lines of Python
- Automatic preprocessing: Handles missing values, data type detection, categorical encoding, and text features without manual intervention
- Broad model portfolio: Trains LightGBM, CatBoost, XGBoost, random forests, extra trees, k-nearest neighbors, and neural networks by default
- Distillation support: Can compress the full ensemble into a single lightweight model for faster inference in production

## What AutoGluon Is

AutoGluon is a framework that automates the end-to-end machine learning pipeline for tabular data: data preprocessing, model selection, hyperparameter configuration, model training, ensembling, and prediction. It was developed by Amazon Web Services and first released in 2020, accompanied by the research paper "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data."

The key insight behind AutoGluon is that most AutoML frameworks spend the majority of their compute budget searching for optimal hyperparameters for a single model type. AutoGluon instead allocates that budget to training many different model types with known-good default configurations and then combining them. This approach consistently yields better results within a fixed time budget.

AutoGluon supports:

- Binary classification
- Multiclass classification
- Regression
- Ranking problems
- Quantile regression

While AutoGluon also has modules for image classification, object detection, text classification, and time series forecasting, this chapter focuses on the tabular prediction module, which is its most mature and widely adopted component.

## How It Works

### The Three Pillars

AutoGluon's architecture rests on three principles that work together:

1. **Model diversity**: Train many different algorithm types (gradient boosting, neural networks, tree ensembles, nearest neighbors) rather than tuning one algorithm extensively.

2. **Bagging**: Train each model type across K cross-validation folds, then average their predictions at inference time. This reduces variance and produces out-of-fold (OOF) predictions needed for stacking.

3. **Multi-layer stack ensembling**: Use the OOF predictions from one layer of models as additional input features for the next layer of models, building progressively more powerful meta-learners.

### Multi-Layer Stacking Architecture

The stacking architecture is AutoGluon's core innovation and works as follows:

```
Layer 1 (Base Layer):
  - LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, KNN, Neural Network
  - Each model is trained on the original features only
  - Each model is bagged across K folds, producing OOF predictions

Layer 2 (Stacker Layer):
  - Same model types as Layer 1
  - Trained on original features PLUS the OOF predictions from Layer 1
  - Again bagged across K folds

Layer N (Additional Stacker Layers):
  - Same pattern, adding predictions from the previous layer

Final Layer (Weighted Ensemble):
  - A greedy weighted ensemble that learns optimal combination weights
  - Selects which stacker models to include and how much weight each receives
```

The total number of models trained is: M x N x K + 1, where M is the number of ensemble layers (excluding the final layer), N is the number of model types per layer, K is the number of folds, and 1 is the final weighted ensemble meta-model.

### Preventing Data Leakage

A critical detail in the stacking process is how AutoGluon avoids overfitting through data leakage. During training, each layer uses only out-of-fold predictions from the previous layer. Since each model in a K-fold bag only generates predictions for the data it did not train on, the stacker models never see predictions that were made on data the base model already trained on. During inference, predictions from all K fold models are simply averaged.

### Model Portfolio

AutoGluon evaluated over 1,300 model configurations across 200 datasets to determine which combinations of models and hyperparameters work best as defaults. The result is a curated portfolio of models with pre-tuned hyperparameters that are encoded into the preset system. This means that even without any hyperparameter tuning, AutoGluon's defaults are strong.

The default model types include:

- **LightGBM**: Multiple configurations with different hyperparameters
- **CatBoost**: Gradient boosting with native categorical support
- **XGBoost**: Gradient boosting
- **Random Forest**: Bagged decision trees
- **Extra Trees**: Extremely randomized trees
- **K-Nearest Neighbors**: Distance-based model
- **Neural Networks**: Custom tabular neural network (TabularNeuralNetTorch)
- **FastAI Neural Network**: Alternative neural network implementation

## Basic Usage

### Installation

```bash
pip install autogluon
```

For a lighter installation with only the tabular module:

```bash
pip install autogluon.tabular
```

### Three-Line Training

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target_column').fit(train_data)
predictions = predictor.predict(test_data)
```

This single call handles data preprocessing, trains multiple models with bagging, builds a multi-layer stacking ensemble, and selects the best combination.

### Classification

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# AutoGluon automatically detects classification vs regression
predictor = TabularPredictor(
    label='target',
    eval_metric='roc_auc',       # Metric to optimize
    path='autogluon_models/'     # Where to save models
).fit(
    train_data,
    time_limit=3600,             # Train for up to 1 hour
    presets='best_quality'       # Use the highest quality preset
)

# Predict class labels
predictions = predictor.predict(test_data)

# Predict probabilities
probabilities = predictor.predict_proba(test_data)

# View leaderboard of all trained models
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
```

### Regression

```python
predictor = TabularPredictor(
    label='price',
    eval_metric='root_mean_squared_error',
    problem_type='regression'
).fit(
    train_data,
    time_limit=1800,
    presets='high_quality'
)

predictions = predictor.predict(test_data)
```

### Evaluating Models

```python
# Performance summary on test data
performance = predictor.evaluate(test_data)
print(performance)

# Detailed leaderboard
leaderboard = predictor.leaderboard(test_data, extra_info=True)

# Feature importance (via permutation)
importance = predictor.feature_importance(test_data)
print(importance)
```

## Presets

Presets are the primary way to control AutoGluon's behavior. Each preset configures the model portfolio, whether bagging and stacking are used, and how aggressively resources are allocated.

| Preset | Stacking | Bagging | Models | Use Case |
|--------|----------|---------|--------|----------|
| best_quality | Yes (multi-layer) | Yes (8 folds) | Full portfolio (~100 configs) | Competitions, maximum accuracy |
| high_quality | Yes (multi-layer) | Yes (8 folds) | Full portfolio | Good accuracy, reasonable time |
| medium_quality | No | No | Default set | Balanced speed and accuracy |
| light | No | No | Reduced set | Fast prototyping |
| very_light | No | No | Minimal set | Quick experiments |

The `best_quality` preset uses a model portfolio called "zeroshot" that was learned from ensemble simulation on 200 datasets from TabRepo. It contains roughly 100 model configurations and represents AutoGluon's most powerful mode.

```python
# Quick prototyping
predictor = TabularPredictor(label='target').fit(train_data, presets='light')

# Production quality
predictor = TabularPredictor(label='target').fit(
    train_data,
    presets='best_quality',
    time_limit=7200
)
```

## Time Budget Management

AutoGluon is designed to work within a time budget. The `time_limit` parameter (in seconds) controls how long training runs. AutoGluon will attempt to train as many models as possible within the budget, prioritizing models that are likely to contribute the most to the final ensemble.

```python
# 10-minute quick run
predictor = TabularPredictor(label='target').fit(train_data, time_limit=600)

# 4-hour thorough run
predictor = TabularPredictor(label='target').fit(train_data, time_limit=14400)
```

If no time limit is set, AutoGluon will train all models to completion, which can take a very long time on large datasets.

## Advanced Configuration

### Custom Hyperparameters

You can override the default model types and hyperparameters:

```python
from autogluon.tabular import TabularPredictor

hyperparameters = {
    'GBM': [
        {'num_boost_round': 1000, 'learning_rate': 0.01},
        {'num_boost_round': 500, 'learning_rate': 0.05},
    ],
    'CAT': {},
    'XGB': {},
    'RF': {},
    'NN_TORCH': {},
}

predictor = TabularPredictor(label='target').fit(
    train_data,
    hyperparameters=hyperparameters,
    time_limit=3600
)
```

### Controlling Stacking and Bagging

```python
# Enable bagging with custom fold count
predictor = TabularPredictor(label='target').fit(
    train_data,
    num_bag_folds=5,         # Number of bagging folds (0 disables bagging)
    num_bag_sets=1,           # Number of times to repeat bagging
    num_stack_levels=1,       # Number of stacking layers (0 disables stacking)
    time_limit=3600
)
```

### Excluding Specific Models

```python
predictor = TabularPredictor(label='target').fit(
    train_data,
    excluded_model_types=['KNN', 'NN_TORCH'],  # Skip slow models
    time_limit=1800
)
```

## Deployment and Production

### Reducing Model Size

AutoGluon ensembles can be large. For production deployment, there are several strategies to reduce size and improve inference speed.

```python
# Clone with minimal artifacts for prediction only
predictor.clone_for_deployment('production_model/')

# Or use distillation to compress the ensemble into a single model
distilled = predictor.distill(
    time_limit=600,
    augment_method='spunge'
)
```

### Model Distillation

Distillation trains a single smaller model to mimic the predictions of the full ensemble. The distilled model is often more accurate than the same model trained directly on the original data, because it learns from the ensemble's soft predictions.

```python
# Distill into a single LightGBM model
distilled_predictor = predictor.distill(
    time_limit=600,
    hyperparameters={'GBM': {}},
    augment_method='spunge'      # Generates synthetic data for distillation
)
```

### Saving and Loading

```python
# Save (happens automatically during fit)
predictor.save()

# Load
predictor = TabularPredictor.load('autogluon_models/')

# Predict
predictions = predictor.predict(new_data)
```

### Version Compatibility

When loading a saved predictor, you must use the same Python version and AutoGluon version that were used during training. Mismatched versions can cause instability or errors.

## Memory and Compute Considerations

AutoGluon can be resource-intensive, especially with `best_quality` presets. Key considerations:

- **Memory**: Most issues with AutoGluon arise from insufficient memory. The multi-layer stacking with bagging trains M x N x K models, each of which stays in memory or on disk. Use machines with ample RAM. For AWS, m5.24xlarge instances are recommended for large datasets.

- **Disk space**: All trained models are saved to disk. A `best_quality` run can produce several gigabytes of model artifacts.

- **GPU usage**: AutoGluon supports GPU training for its neural network models but does not support distributed training across multiple machines. Gradient boosting models (LightGBM, XGBoost, CatBoost) can also use GPU if available.

- **Inference latency**: The full ensemble can be too slow for real-time applications. Consider distillation or using a single model from the leaderboard if inference speed is critical.

```python
# Select a single fast model for real-time inference
fast_model = predictor.leaderboard()
# Pick the best LightGBM model
predictor.predict(test_data, model='LightGBM_BAG_L1')
```

## AutoGluon vs H2O vs FLAML

These three frameworks represent different philosophies in the AutoML space. Understanding the trade-offs helps in selecting the right tool.

### AutoGluon

- **Philosophy**: Ensemble-first. Train many diverse models with strong defaults, combine via stacking.
- **Strengths**: Highest out-of-the-box accuracy in most benchmarks. Minimal configuration needed. Excellent for competitions and offline batch prediction.
- **Weaknesses**: High memory and disk usage. Large ensembles are not ideal for real-time serving. Python-only. No native distributed training.
- **Best for**: Getting the highest possible accuracy when compute resources and inference latency are not primary constraints.

### H2O AutoML

- **Philosophy**: Enterprise-grade distributed AutoML with broad language and platform support.
- **Strengths**: Scales to very large datasets via distributed computing across clusters. Supports Python, R, Java, and Scala. Mature enterprise features (MOJO export, governance, explainability). Reliable stacking ensemble.
- **Weaknesses**: Can be resource-intensive and slow during optimization. Requires running an H2O cluster (JVM-based). Fewer model types in the default ensemble compared to AutoGluon.
- **Best for**: Large-scale enterprise deployments, teams using R or Java, organizations needing distributed training across clusters.

### FLAML

- **Philosophy**: Lightweight and cost-efficient. Uses cost-effective hyperparameter optimization to find good models quickly with minimal resources.
- **Strengths**: Very fast training. Low memory footprint. Works well in resource-constrained environments (cloud cost optimization, edge computing). Excellent cost-performance ratio.
- **Weaknesses**: Lower peak accuracy compared to AutoGluon's stacking ensembles. Moderate reliability on larger or more complex datasets. Relies more on HPO than ensembling.
- **Best for**: Resource-constrained environments, rapid prototyping, scenarios where compute cost matters more than squeezing out the last percentage point of accuracy.

### Comparison Table

| Dimension | AutoGluon | H2O AutoML | FLAML |
|-----------|-----------|------------|-------|
| Peak accuracy | Highest | High | Good |
| Training speed | Moderate | Slow | Fastest |
| Memory usage | High | High | Low |
| Inference speed | Slow (ensemble) | Moderate | Fast |
| Language support | Python | Python, R, Java, Scala | Python |
| Distributed training | No | Yes | No |
| Ease of use | Very easy | Moderate (JVM setup) | Very easy |
| Model compression | Distillation | MOJO export | N/A |
| Stacking ensemble | Multi-layer | Yes | Limited |

### Decision Framework

Choose **AutoGluon** when:
- You want the best possible accuracy and have sufficient compute resources
- You are running Kaggle competitions or offline batch prediction jobs
- You want a simple Python API with minimal configuration
- You plan to deploy on AWS (good SageMaker integration)

Choose **H2O** when:
- You need distributed training across a cluster for very large datasets
- Your team uses R, Java, or Scala in addition to Python
- You need enterprise features like model governance and MOJO deployment
- You are already in the H2O ecosystem

Choose **FLAML** when:
- You have limited compute budget or are optimizing cloud costs
- You need fast iteration during prototyping
- Inference speed matters more than maximum accuracy
- You are working in a resource-constrained environment

## Practical Tips

1. **Start with `medium_quality` and a short time limit** to verify your data pipeline and get a baseline. Then switch to `best_quality` for the final run.

2. **Always set a `time_limit`**. Without one, AutoGluon will train all models to completion, which can take hours or days on large datasets.

3. **Give AutoGluon more time rather than tuning hyperparameters**. AutoGluon is designed so that additional time translates directly into better models. Spending time on manual hyperparameter tuning is usually less productive than simply increasing the time budget.

4. **Check the leaderboard** after training. It shows every model that was trained, its validation score, inference time, and training time. This tells you which models contributed most and helps you decide if distillation to a single model is viable.

5. **Use `feature_importance()`** to understand which features matter. This uses permutation importance and gives reliable estimates, but requires a held-out test set.

6. **For production deployment, consider distillation**. The full stacking ensemble can be too large and slow for serving. Distilling into a single LightGBM or CatBoost model often retains most of the accuracy with much faster inference.

7. **Monitor memory usage**. If you run out of memory, try reducing `num_bag_folds`, reducing `num_stack_levels`, or excluding memory-hungry model types like KNN.

8. **Use `excluded_model_types` strategically**. If you know certain model types are too slow for your use case (such as KNN on large datasets), exclude them to give more time budget to other models.

9. **Version-pin your AutoGluon installation**. Saved predictors must be loaded with the same AutoGluon version. Pin the version in your requirements.txt to avoid breaking model loading in production.

10. **For AWS deployments**, AutoGluon has first-class integration with SageMaker, including built-in AutoGluon-Tabular algorithms in SageMaker Autopilot and SageMaker training jobs.

## See Also

- [H2O AutoML](../h2o/ReadMe.md) - Enterprise distributed AutoML
- [FLAML](../flaml/ReadMe.md) - Lightweight cost-efficient AutoML
- [AutoML Overview](../ReadMe.md) - Parent concept
- [XGBoost](../../gradient-boosting/xgboost/ReadMe.md) - One of AutoGluon's base models
- [LightGBM](../../gradient-boosting/lightgbm/ReadMe.md) - AutoGluon's primary gradient boosting model
