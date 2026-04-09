# FLAML

## Summary

FLAML (Fast and Lightweight AutoML) is a Python library from Microsoft Research that automates model selection and hyperparameter tuning with a focus on computational efficiency. Unlike heavier AutoML frameworks that throw large amounts of compute at the problem, FLAML is designed to find good models quickly and cheaply. It is scikit-learn compatible, works out of the box with a simple `fit()` call, and is particularly well suited for scenarios where you have a limited time or compute budget.

Key points to remember:

- Cost-aware search: FLAML starts with cheap configurations and moves toward expensive ones only when justified, unlike frameworks that explore the space uniformly
- Two core search algorithms: CFO (Cost-Frugal Optimization) for sequential search and BlendSearch for parallel or complex search spaces
- Time-budget driven: You specify a wall-clock time budget and FLAML manages everything within it, including model selection, hyperparameter tuning, and cross-validation
- Lightweight: Minimal dependencies, fast installation, low memory footprint compared to AutoGluon or H2O
- Default learner pool: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, and linear models for classification and regression
- Scikit-learn compatible: The AutoML object works as a drop-in estimator with fit/predict/predict_proba
- Zero-shot AutoML: Can apply data-dependent default hyperparameters without any tuning at all, useful for quick baselines
- Best for: Constrained environments, rapid prototyping, hyperparameter tuning of specific models, and situations where compute cost matters more than squeezing out the last fraction of a percent of accuracy

## What FLAML Is

FLAML is an open-source AutoML library developed by Microsoft Research, first published in 2020. It addresses a practical gap in the AutoML landscape: most AutoML tools optimize for accuracy at the expense of compute, but many real-world ML projects face strict time or resource constraints. FLAML was designed to produce competitive models with significantly less computation than alternatives.

At its core, FLAML does two things:

1. Task-oriented AutoML: Given a dataset and a time budget, it jointly selects the best model type and hyperparameters.
2. Generic hyperparameter tuning: Through `flaml.tune`, it provides a general-purpose tuning API that can optimize any user-defined function, not just ML models.

FLAML is built on research into economical hyperparameter optimization. The key insight is that different hyperparameter configurations have vastly different evaluation costs. A random forest with 10 trees trains orders of magnitude faster than one with 10,000 trees. FLAML exploits this by searching cheap regions of the space first and only moving to expensive configurations when the cheap ones stop improving.

## Installation and Basic Usage

### Installation

```bash
pip install flaml
```

For additional features:

```bash
pip install "flaml[automl]"    # Full AutoML dependencies
pip install "flaml[notebook]"  # Notebook visualization support
```

FLAML requires Python 3.10 or later.

### Classification

```python
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task="classification",
    time_budget=120,          # 2 minutes
    metric="roc_auc",
    eval_method="cv",
    n_splits=5,
    seed=42
)

# Results
print(f"Best model: {automl.best_estimator}")
print(f"Best config: {automl.best_config}")
print(f"Best CV score: {1 - automl.best_loss:.4f}")

# Predict
predictions = automl.predict(X_test)
probabilities = automl.predict_proba(X_test)
```

### Regression

```python
automl = AutoML()
automl.fit(
    X_train, y_train,
    task="regression",
    time_budget=60,
    metric="rmse",
    estimator_list=["lgbm", "xgboost", "rf"],
)

y_pred = automl.predict(X_test)
```

### Restricting the Estimator List

If you already know which model family you want and just need FLAML for hyperparameter tuning, restrict the search:

```python
automl.fit(
    X_train, y_train,
    task="classification",
    time_budget=60,
    estimator_list=["lgbm"],   # Only tune LightGBM
    metric="f1"
)
```

This is a common pattern: use FLAML as a fast hyperparameter tuner for a single model type rather than as a full model selection tool.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| task | required | "classification", "regression", "ts_forecast", "rank", etc. |
| time_budget | 60 | Total wall-clock seconds for the search |
| metric | "auto" | Optimization metric (accuracy, roc_auc, rmse, mae, f1, log_loss, etc.) |
| estimator_list | "auto" | List of learners to consider, or "auto" for task defaults |
| eval_method | "auto" | "cv" for cross-validation, "holdout" for train/val split |
| n_splits | 5 | Number of CV folds when eval_method="cv" |
| split_ratio | 0.1 | Validation fraction when eval_method="holdout" |
| train_time_limit | None | Max seconds per individual trial |
| mem_thres | None | Memory threshold in bytes to abort a trial |
| retrain_full | True | Whether to retrain the best model on full data after search |
| hpo_method | "auto" | "cfo", "bs" (BlendSearch), "optuna", "random", or "auto" |
| log_file_name | "default" | Path for the search log |
| seed | None | Random seed for reproducibility |
| n_jobs | 1 | Number of parallel jobs for model training |

## Architecture and Search Strategy

### The Problem with Naive HPO

Standard hyperparameter optimization methods like random search or Bayesian optimization (e.g., Optuna, Hyperopt) treat all configurations equally in terms of cost. They might evaluate a LightGBM model with 2,000 boosting rounds right after one with 10 rounds, wasting budget on slow evaluations that may not yield improvements. FLAML addresses this with cost-aware search.

### CFO: Cost-Frugal Optimization

CFO is FLAML's default search algorithm for sequential (single-threaded) tuning. It is built on a method called FLOW2 (Frugal Optimization for Locally-optimal Worth).

How CFO works:

1. It starts from a low-cost initial configuration. For example, a small number of boosting rounds, shallow tree depth, and small subsample ratios.
2. It performs local search around the current best configuration, proposing small perturbations.
3. It uses an adaptive step size: steps shrink as the search converges and expand after random restarts.
4. Crucially, it moves through the search space along a cost-increasing trajectory. It only explores more expensive configurations when cheaper ones stop yielding improvements.
5. When local search converges, it performs random restarts to escape local optima, but again starts from low-cost points.

The result is that CFO spends most of its budget evaluating cheap configurations and only invests in expensive ones when there is evidence they will pay off. This is why FLAML can often match the accuracy of other tools in a fraction of the wall-clock time.

### BlendSearch: Blended Local and Global Search

BlendSearch extends CFO by combining local search with global search (typically Bayesian optimization). It was published at ICLR 2021.

How BlendSearch works:

1. It maintains one global search thread (e.g., Bayesian optimization using Optuna or a random forest surrogate).
2. It gradually creates local search threads (CFO instances) seeded at promising configurations found by the global search.
3. At each step, it prioritizes among the global thread and all active local threads based on their real-time performance and cost.
4. New local threads are only created at points that are sufficiently distant from existing threads in cost-related dimensions, avoiding redundant search.

When to use BlendSearch vs CFO:

- CFO: Use when running sequentially (single thread), when the search space is relatively simple and continuous, or when you want maximum cost frugality.
- BlendSearch: Use when running in parallel, when the search space is complex (multiple disjoint or discontinuous subspaces), or when you want better exploration of the global space.

FLAML defaults to CFO for sequential execution and BlendSearch for parallel execution, which is a sensible default for most users.

### Per-Estimator Search States

FLAML maintains independent search states for each estimator type. This means the search for LightGBM hyperparameters is tracked separately from the search for XGBoost hyperparameters. The system allocates trials across estimators based on their relative performance, naturally spending more time tuning promising model types.

### Time Budget Management

FLAML's budget management is more sophisticated than a simple timer:

- It tracks per-trial training time and uses this to estimate whether another trial can fit within the remaining budget.
- It will not start a trial if it estimates the trial will exceed the remaining time.
- If `retrain_full="budget"` is set, remaining time after search is used to retrain the best model on the full training set.
- Early stopping is applied automatically when budget exhaustion is imminent.

## Data Preprocessing

FLAML includes lightweight automatic preprocessing:

- Numerical features: Missing values are imputed with the median.
- Categorical features: Missing values are filled with a placeholder, then encoded.
- No heavy feature engineering: FLAML does not perform feature generation, target encoding, or stacking-based preprocessing like AutoGluon does.

You can skip the built-in preprocessing entirely with `skip_transform=True` if you prefer to handle it yourself, which is common in production pipelines.

## Advanced Features

### Zero-Shot AutoML

FLAML offers a zero-shot mode that applies learned default hyperparameters based on dataset characteristics, with no tuning at all:

```python
from flaml.default import LGBMClassifier

model = LGBMClassifier()
model.fit(X_train, y_train)
```

This replaces the standard LightGBM import and automatically selects data-dependent hyperparameters. It is useful for quick baselines or when you have no time budget for tuning.

### Custom Learners

You can register custom estimators with FLAML:

```python
from flaml.automl.model import SKLearnEstimator

class MyEstimator(SKLearnEstimator):
    @classmethod
    def search_space(cls, data_size, task):
        return {
            "n_estimators": {
                "domain": tune.randint(10, 500),
                "init_value": 100,
                "low_cost_init_value": 10,
            },
            "max_depth": {
                "domain": tune.randint(3, 15),
                "init_value": 6,
                "low_cost_init_value": 3,
            },
        }

automl = AutoML()
automl.add_learner("my_model", MyEstimator)
automl.fit(X_train, y_train, estimator_list=["my_model"], time_budget=60)
```

The `low_cost_init_value` is important here: it tells CFO where to start its search in the cheap region of the space.

### Generic Hyperparameter Tuning

FLAML's tune API works beyond ML models:

```python
from flaml import tune

def my_objective(config):
    # Any function that returns a metric
    score = train_and_evaluate(config)
    return {"loss": score}

analysis = tune.run(
    my_objective,
    config={
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.randint(1, 5),
    },
    metric="loss",
    mode="min",
    time_budget_s=300,
    num_samples=-1,   # Unlimited trials within time budget
)

print(analysis.best_config)
```

This is useful for tuning deep learning models, pipeline configurations, or any black-box function.

### Logging and Reproducibility

```python
automl.fit(
    X_train, y_train,
    task="classification",
    time_budget=120,
    log_file_name="flaml_experiment.log",
    seed=42
)

# Access search history
print(automl.best_config_per_estimator)
```

The log file records every trial with its configuration, metric, and training time, which is useful for post-hoc analysis.

## When to Choose FLAML vs Alternatives

### FLAML vs AutoGluon

AutoGluon (from AWS) takes a fundamentally different approach. It focuses on multi-layer model ensembling and stacking, training many models and combining them. AutoGluon tends to produce higher raw accuracy on benchmarks, especially with generous time budgets, because its ensembling strategy is very effective.

Choose FLAML when:
- You have a tight time or compute budget
- You want a single best model rather than a large ensemble
- You need low inference latency (ensembles are slower at prediction time)
- You want a lightweight dependency footprint
- You are tuning a specific model type rather than doing full model selection

Choose AutoGluon when:
- Maximum accuracy is the priority and you have the compute
- You do not mind deploying an ensemble
- You want more aggressive automated feature engineering
- You need strong out-of-the-box defaults without configuration

### FLAML vs H2O AutoML

H2O AutoML is a mature, JVM-based framework focused on enterprise scalability. It trains many models using random search and creates stacked ensembles.

Choose FLAML when:
- You want a pure Python solution without JVM dependencies
- Your data fits in memory on a single machine
- You need fast iteration and lightweight integration
- You prefer cost-aware search over brute-force random search

Choose H2O when:
- You need distributed training across a cluster
- You are working with very large datasets that require H2O's out-of-core processing
- You are in an enterprise environment that already uses the H2O ecosystem
- You want a built-in web UI (H2O Flow)

### Summary Comparison Table

| Aspect | FLAML | AutoGluon | H2O AutoML |
|--------|-------|-----------|------------|
| Search strategy | CFO / BlendSearch (cost-aware) | Multi-layer stacking + ensembling | Random search + stacking |
| Compute efficiency | High (cost-frugal by design) | Moderate (trains many models) | Moderate to low |
| Typical output | Single best model | Weighted ensemble | Stacked ensemble |
| Inference speed | Fast (single model) | Slower (ensemble) | Slower (ensemble) |
| Dependencies | Lightweight (Python) | Moderate (Python) | Heavy (JVM required) |
| Scalability | Single machine | Single machine (GPU support) | Distributed cluster |
| Best accuracy (unlimited budget) | Good | Best | Good |
| Best accuracy (tight budget) | Good to best | Good | Moderate |
| Feature preprocessing | Minimal | Aggressive | Moderate |
| Custom model support | Yes (add_learner) | Yes (custom models) | Limited |

## Practical Tips

1. Start with a short time budget (60-120 seconds) to get an initial read on which model families work well for your data, then increase the budget for a final run.

2. Use the log file. FLAML's trial logs tell you exactly how time was spent and what configurations were tried. This is invaluable for understanding whether to extend the budget or restrict the estimator list.

3. Restrict estimators when you know what you want. If your production system requires a single LightGBM model, do not waste budget exploring random forests. Set `estimator_list=["lgbm"]`.

4. Set `retrain_full=True` (the default) for final models. After the search completes, FLAML retrains the best configuration on the full training set (train + validation), which typically improves performance.

5. Use `eval_method="cv"` for small datasets. Cross-validation gives more reliable estimates than a single holdout split. For large datasets, holdout is faster and usually sufficient.

6. Pay attention to the `best_config` caveat. FLAML internally transforms some hyperparameters (e.g., using `log_max_bin` instead of `max_bin`). If you need to extract the config for use outside FLAML, use the model object directly rather than the raw config dictionary.

7. For production deployment, extract the trained model with `automl.model` and serialize it independently. You do not need FLAML as a runtime dependency for inference.

8. Combine FLAML with your own preprocessing. Since FLAML's built-in preprocessing is minimal, pair it with a scikit-learn Pipeline or your own feature engineering for best results.

9. For parallel execution, set `n_jobs` greater than 1 and FLAML will automatically switch to BlendSearch, which is better suited for parallel exploration.

10. Use zero-shot mode for quick baselines. Before investing in a full AutoML run, `from flaml.default import LGBMClassifier` gives you tuned defaults instantly.

## Supported Tasks

- Binary and multiclass classification
- Regression
- Time series forecasting (Prophet, ARIMA, SARIMAX, Holt-Winters)
- Ranking
- NLP tasks (via HuggingFace Transformers integration)
- Custom tasks via the tune API

## See Also

- [AutoGluon](../autogluon/ReadMe.md) - Ensemble-focused AutoML from AWS
- [H2O AutoML](../h2o/ReadMe.md) - Enterprise-grade distributed AutoML
