# LightGBM

## Summary

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It's designed for distributed and efficient training, making it particularly suited for large datasets. LightGBM's key innovations include leaf-wise tree growth and histogram-based algorithms that significantly speed up training while maintaining high accuracy.

Key points to remember:

- Leaf-wise growth: Grows trees by splitting the leaf with maximum gain (vs level-wise)
- Histogram-based: Bins continuous features for faster split finding
- Gradient-based One-Side Sampling (GOSS): Focuses on samples with large gradients
- Exclusive Feature Bundling (EFB): Bundles mutually exclusive features
- Native categorical support: Handles categorical features without encoding
- Fastest training: Generally faster than XGBoost and CatBoost
- Low memory: Efficient memory usage with histogram algorithm
- Overfitting risk: Leaf-wise growth can overfit on small datasets

## Core Concepts

### Leaf-wise vs Level-wise Growth

```
Level-wise (XGBoost):
       [Root]
      /      \
   [L1]      [L1]      # Split all nodes at level 1
   /  \      /  \
[L2] [L2] [L2] [L2]    # Split all nodes at level 2

Leaf-wise (LightGBM):
       [Root]
      /      \
   [L1]     [Best]     # Split only the best leaf
           /    \
        [New]  [New]   # Continue splitting best leaves
```

Leaf-wise grows deeper trees faster and typically achieves lower loss, but can overfit on small datasets.

### Histogram Algorithm

```
Traditional: Sort-based (O(n × features × log n))
- Sort feature values
- Scan for best split

Histogram: Bin-based (O(n × features))
- Bin continuous values into discrete bins (default 255)
- Build histograms per feature
- Find best split from histogram

Benefits:
- 8x fewer comparisons
- Cache-efficient
- Supports parallel histogram building
```

## Basic Usage

### Classification

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Dataset (LightGBM's optimized data structure)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Predict
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(classification_report(y_test, y_pred))
```

### Regression

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Using sklearn API
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50)]
)

y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Multiclass Classification

```python
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50)]
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
```

## Key Hyperparameters

### Tree Structure

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| num_leaves | 31 | 10-256 | Max number of leaves per tree |
| max_depth | -1 | 3-15 | Max tree depth (-1 = no limit) |
| min_data_in_leaf | 20 | 10-1000 | Minimum samples in a leaf |
| min_sum_hessian_in_leaf | 1e-3 | 0-100 | Minimum sum of hessian in leaf |

### Learning Control

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| learning_rate | 0.1 | 0.01-0.3 | Shrinkage factor |
| n_estimators | 100 | 100-10000 | Number of boosting rounds |
| max_bin | 255 | 63-511 | Max number of bins for histograms |

### Sampling

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| feature_fraction | 1.0 | 0.5-1.0 | Fraction of features per tree |
| bagging_fraction | 1.0 | 0.5-1.0 | Fraction of data per tree |
| bagging_freq | 0 | 1-10 | Frequency of bagging (0=disabled) |

### Regularization

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| lambda_l1 | 0 | 0-10 | L1 regularization |
| lambda_l2 | 0 | 0-10 | L2 regularization |
| min_gain_to_split | 0 | 0-1 | Minimum gain for split |
| path_smooth | 0 | 0-100 | Smoothing for leaf predictions |

## Native Categorical Feature Support

LightGBM can handle categorical features directly without one-hot encoding.

```python
# Method 1: Specify categorical features by name
categorical_features = ['category_a', 'category_b', 'category_c']

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_features
)

# Method 2: Specify by column index
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=[0, 3, 7]  # Column indices
)

# Method 3: Using sklearn API with pandas DataFrame
# LightGBM auto-detects columns with 'category' dtype
df[categorical_features] = df[categorical_features].astype('category')

model = lgb.LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=categorical_features)
```

### How LightGBM Handles Categoricals

```
Traditional: One-hot encoding creates sparse features

LightGBM: Optimal split finding
1. Sort categories by sum of gradients
2. Find best split point on sorted categories
3. Results in many-vs-many splits (not one-vs-all)

Benefits:
- Faster than one-hot for high-cardinality
- Can find complex groupings
- Lower memory usage
```

## Hyperparameter Tuning

### Optuna Optimization

```python
import optuna

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')
        ]
    )

    return model.best_score['valid_0']['binary_logloss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
```

### Tuning Strategy

```python
# Step 1: Set num_leaves based on max_depth heuristic
# num_leaves should be less than 2^max_depth
# For max_depth=7: num_leaves < 128

# Step 2: Tune learning_rate and n_estimators together
# Lower learning_rate needs more n_estimators

# Step 3: Tune tree parameters
tree_params = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 7, 10],
    'min_data_in_leaf': [20, 50, 100]
}

# Step 4: Tune sampling parameters
sampling_params = {
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'bagging_freq': [0, 5, 10]
}

# Step 5: Tune regularization
reg_params = {
    'lambda_l1': [0, 0.1, 1.0],
    'lambda_l2': [0, 0.1, 1.0],
    'min_gain_to_split': [0, 0.1, 0.5]
}
```

## Feature Importance

```python
import matplotlib.pyplot as plt

# Train model
model.fit(X_train, y_train)

# Method 1: Split importance (number of times feature is used)
importance_split = model.feature_importances_

# Method 2: Gain importance (total gain from splits using feature)
importance_gain = model.booster_.feature_importance(importance_type='gain')

# Plot
lgb.plot_importance(model, importance_type='gain', max_num_features=20)
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()

# Get feature names and importance as DataFrame
import pandas as pd
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance_gain
}).sort_values('importance', ascending=False)
```

### SHAP Integration

```python
import shap

# TreeExplainer for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list of 2 arrays
# shap_values[1] for positive class

# Summary plot
shap.summary_plot(shap_values[1], X_test)

# Dependence plot
shap.dependence_plot('feature_name', shap_values[1], X_test)
```

## Handling Imbalanced Data

### Class Weights

```python
# Method 1: is_unbalance parameter
params = {
    'objective': 'binary',
    'is_unbalance': True  # Automatically adjusts weights
}

# Method 2: scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
params = {
    'objective': 'binary',
    'scale_pos_weight': scale_pos_weight
}

# Method 3: Sample weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
```

### Focal Loss

```python
import numpy as np

def focal_loss(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))

    grad = alpha * (y_true * (1 - p) ** gamma * (gamma * p * np.log(p + 1e-8) + p - 1) +
                   (1 - y_true) * p ** gamma * (p - gamma * (1 - p) * np.log(1 - p + 1e-8)))

    hess = alpha * (y_true * (1 - p) ** gamma * (
        gamma * (gamma + 1) * p ** 2 * np.log(p + 1e-8) +
        2 * gamma * p ** 2 - 2 * gamma * p + p) +
        (1 - y_true) * p ** gamma * (
        gamma * (gamma + 1) * (1 - p) ** 2 * np.log(1 - p + 1e-8) +
        2 * gamma * (1 - p) ** 2 - 2 * gamma * (1 - p) + 1 - p))

    return grad, hess

model = lgb.train(params, train_data, fobj=focal_loss)
```

## GPU Training

```python
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'objective': 'binary',
    'metric': 'binary_logloss'
}

# For CUDA devices
params = {
    'device': 'cuda',
    'objective': 'binary'
}

model = lgb.train(params, train_data, num_boost_round=1000)
```

## Distributed Training

### Dask Integration

```python
import dask.dataframe as dd
from dask.distributed import Client
import lightgbm as lgb

# Start Dask client
client = Client()

# Load distributed data
ddf = dd.read_parquet('large_dataset.parquet')

# Train distributed model
model = lgb.DaskLGBMClassifier(
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.1
)

model.fit(
    ddf[feature_columns],
    ddf[target_column],
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50)]
)
```

## Cross-Validation

```python
# Built-in cross-validation
train_data = lgb.Dataset(X, label=y)

cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=1000,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics=['auc', 'binary_logloss'],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ],
    seed=42,
    return_cvbooster=True
)

print(f"Best iteration: {len(cv_results['valid auc-mean'])}")
print(f"Best AUC: {cv_results['valid auc-mean'][-1]:.4f} ± {cv_results['valid auc-stdv'][-1]:.4f}")
```

## Model Persistence

```python
# Method 1: Native LightGBM format (recommended)
model.save_model('model.txt')  # Text format
model.save_model('model.txt', num_iteration=model.best_iteration)

loaded_model = lgb.Booster(model_file='model.txt')

# Method 2: Pickle for sklearn wrapper
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 3: Joblib
import joblib
joblib.dump(model, 'model.joblib')
```

## Monotonic Constraints

```python
# Enforce monotonic relationships
# 1 = increasing, -1 = decreasing, 0 = no constraint
params = {
    'objective': 'binary',
    'monotone_constraints': [1, -1, 0, 0, 1]  # One per feature
}

# Or by feature name
params = {
    'monotone_constraints_method': 'advanced',
    'monotone_constraints': 'feature_a:1,feature_b:-1'
}
```

## Custom Objectives and Metrics

```python
def custom_asymmetric_loss(y_pred, dtrain):
    """Asymmetric loss penalizing false negatives more."""
    y_true = dtrain.get_label()

    residual = y_true - y_pred
    grad = np.where(residual < 0, -2 * residual, -10 * residual)
    hess = np.where(residual < 0, 2, 10)

    return grad, hess

def custom_metric(y_pred, dtrain):
    """Custom evaluation metric."""
    y_true = dtrain.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_true == y_pred_binary)
    return 'custom_accuracy', accuracy, True  # True = higher is better

model = lgb.train(
    params,
    train_data,
    fobj=custom_asymmetric_loss,
    feval=custom_metric
)
```

## Best Practices

### Preventing Overfitting

1. **Limit num_leaves**: Keep `num_leaves < 2^max_depth`
2. **Increase min_data_in_leaf**: 20-100 for small datasets
3. **Use subsampling**: Both feature_fraction and bagging_fraction ~0.8
4. **Add regularization**: lambda_l1 and lambda_l2
5. **Early stopping**: Essential for leaf-wise growth

### Parameter Starting Points

```python
# Small dataset (<10K samples)
small_params = {
    'num_leaves': 15,
    'max_depth': 5,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'learning_rate': 0.05
}

# Medium dataset (10K-100K samples)
medium_params = {
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'learning_rate': 0.1
}

# Large dataset (>100K samples)
large_params = {
    'num_leaves': 63,
    'max_depth': -1,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'learning_rate': 0.1
}
```

## Comparison with XGBoost and CatBoost

| Feature | LightGBM | XGBoost | CatBoost |
|---------|----------|---------|----------|
| Tree growth | Leaf-wise | Level-wise | Symmetric |
| Speed | Fastest | Fast | Moderate |
| Memory | Lowest | Moderate | Higher |
| Categorical support | Good | Encoding required | Best |
| Default performance | Good | Needs tuning | Best |
| Overfitting risk | Higher (small data) | Lower | Lower |
| GPU support | Good | Good | Best |

## See Also

- [XGBoost](../xgboost/ReadMe.md) - Alternative gradient boosting
- [CatBoost](../catboost/ReadMe.md) - Best for categorical features
- [Gradient Boosting](../ReadMe.md) - Parent concept
