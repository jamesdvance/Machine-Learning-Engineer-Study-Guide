# XGBoost

## Summary

XGBoost (eXtreme Gradient Boosting) is a highly optimized gradient boosting library that has become the go-to algorithm for structured/tabular data. It builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous trees. XGBoost is known for its speed, scalability, and strong performance in machine learning competitions.

Key points to remember:

- Gradient boosting: Trees are added sequentially to minimize a loss function
- Regularization: L1 and L2 regularization prevent overfitting
- Handling missing values: Built-in support for sparse data and missing values
- Tree pruning: Max depth-based pruning with gamma for additional regularization
- Parallelization: Column-block structure enables parallel tree construction
- Distributed training: Native support for multi-GPU and distributed computing
- Feature importance: Multiple methods for understanding feature contributions
- Hyperparameters: Learning rate, max depth, and subsample are most critical

## Core Concepts

### Gradient Boosting Framework

```
Iteration 1: y = F(x)     # First tree
Iteration 2: y = F(x) + F‚(x)     # Add second tree
Iteration 3: y = F(x) + F‚(x) + Fƒ(x)     # Add third tree
...
Final:       y = £ F–(x)   # Sum of all trees

Each tree F– is trained to predict the negative gradient (residuals)
of the loss function with respect to the current prediction.
```

### XGBoost Objective Function

```
Objective = £ L(yb, wb) + £ ©(f–)

Where:
- L(yb, wb) is the loss function (e.g., MSE, log loss)
- ©(f–) = ³T + ½»||w||² is the regularization term
  - T = number of leaves in tree
  - w = leaf weights
  - ³ = complexity penalty for number of leaves
  - » = L2 regularization on leaf weights
```

## Basic Usage

### Classification

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'binary:logistic',  # or 'multi:softmax' for multiclass
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train with early stopping
evallist = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predict
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

print(classification_report(y_test, y_pred))
```

### Regression

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Using sklearn API for convenience
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Fit with evaluation set
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

# Predict
y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Multiclass Classification

```python
model = xgb.XGBClassifier(
    objective='multi:softmax',  # Returns class labels
    # or 'multi:softprob' for probabilities
    num_class=3,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
```

## Key Hyperparameters

### Tree Structure

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| max_depth | 6 | 3-10 | Maximum tree depth |
| min_child_weight | 1 | 1-10 | Minimum sum of instance weight in child |
| gamma | 0 | 0-5 | Minimum loss reduction for split |
| max_leaves | 0 | 0-unlimited | Maximum number of leaves (0=unlimited) |

### Regularization

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| lambda (reg_lambda) | 1 | 0-10 | L2 regularization |
| alpha (reg_alpha) | 0 | 0-10 | L1 regularization |
| gamma | 0 | 0-5 | Minimum loss reduction to split |

### Sampling

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| subsample | 1 | 0.5-1 | Row sampling ratio |
| colsample_bytree | 1 | 0.5-1 | Column sampling per tree |
| colsample_bylevel | 1 | 0.5-1 | Column sampling per level |
| colsample_bynode | 1 | 0.5-1 | Column sampling per split |

### Learning

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| learning_rate (eta) | 0.3 | 0.01-0.3 | Shrinkage factor |
| n_estimators | 100 | 100-10000 | Number of trees |

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### Optuna Optimization

```python
import optuna

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_params}")
```

## Handling Missing Values

XGBoost handles missing values natively by learning the optimal direction for missing values at each split.

```python
import numpy as np

# XGBoost handles NaN values automatically
X_train_with_missing = X_train.copy()
X_train_with_missing[np.random.random(X_train.shape) < 0.1] = np.nan

model = xgb.XGBClassifier()
model.fit(X_train_with_missing, y_train)  # Works directly

# Can also specify missing value indicator
dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999)
```

## Feature Importance

### Built-in Methods

```python
import matplotlib.pyplot as plt

model.fit(X_train, y_train)

# Method 1: Weight (number of times feature is used in splits)
importance_weight = model.get_booster().get_score(importance_type='weight')

# Method 2: Gain (average gain of splits using feature)
importance_gain = model.get_booster().get_score(importance_type='gain')

# Method 3: Cover (average coverage of splits using feature)
importance_cover = model.get_booster().get_score(importance_type='cover')

# Plot
xgb.plot_importance(model, importance_type='gain', max_num_features=20)
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()
```

### SHAP Values

```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for single prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=feature_names
)

# Dependence plot
shap.dependence_plot('feature_name', shap_values, X_test)
```

## Handling Imbalanced Data

### Scale Position Weight

```python
# Calculate scale_pos_weight for binary classification
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic'
)
```

### Sample Weights

```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y_train)

model.fit(X_train, y_train, sample_weight=sample_weights)
```

## GPU Training

```python
# GPU training parameters
params = {
    'tree_method': 'gpu_hist',  # GPU-accelerated
    'device': 'cuda',            # or 'cuda:0' for specific GPU
    'predictor': 'gpu_predictor',
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1
}

# For sklearn API
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    device='cuda',
    **other_params
)
```

## Distributed Training

### Dask Integration

```python
import dask.dataframe as dd
from xgboost import dask as dxgb
from dask.distributed import Client

# Start Dask client
client = Client()

# Load data as Dask DataFrame
ddf = dd.read_parquet('large_dataset.parquet')
X = ddf[feature_columns]
y = ddf[target_column]

# Create DaskDMatrix
dtrain = dxgb.DaskDMatrix(client, X, y)

# Train
output = dxgb.train(
    client,
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train')]
)

# Predict
predictions = dxgb.predict(client, output['booster'], X)
```

## Model Persistence

### Save and Load

```python
# Method 1: Native XGBoost format (recommended)
model.save_model('model.json')  # JSON format
model.save_model('model.ubj')   # Binary format

loaded_model = xgb.XGBClassifier()
loaded_model.load_model('model.json')

# Method 2: Pickle (includes sklearn wrapper state)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Method 3: Joblib (better for large models)
import joblib
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
```

## Early Stopping

```python
# Using sklearn API
model = xgb.XGBClassifier(
    n_estimators=10000,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")

# Using native API
evallist = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=evallist,
    early_stopping_rounds=50
)
```

## Cross-Validation

```python
# Built-in cross-validation
dtrain = xgb.DMatrix(X, label=y)

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    metrics=['auc', 'logloss'],
    early_stopping_rounds=50,
    seed=42,
    as_pandas=True
)

print(f"Best iteration: {cv_results['test-auc-mean'].idxmax()}")
print(f"Best AUC: {cv_results['test-auc-mean'].max():.4f}")
```

## Monotonic Constraints

```python
# Ensure feature 0 has positive relationship with target
# Feature 1 has negative relationship
# Feature 2 has no constraint
model = xgb.XGBClassifier(
    monotone_constraints=(1, -1, 0)
)
```

## Custom Objective Functions

```python
def custom_objective(y_pred, dtrain):
    """Custom loss function."""
    y_true = dtrain.get_label()

    # Gradient of loss
    grad = y_pred - y_true

    # Hessian of loss
    hess = np.ones_like(y_true)

    return grad, hess

def custom_metric(y_pred, dtrain):
    """Custom evaluation metric."""
    y_true = dtrain.get_label()
    error = np.mean((y_pred - y_true) ** 2)
    return 'custom_mse', error

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=custom_objective,
    custom_metric=custom_metric
)
```

## Best Practices

### Typical Tuning Order

1. **n_estimators + learning_rate**: Start with high n_estimators, tune learning_rate
2. **max_depth + min_child_weight**: Control tree complexity
3. **subsample + colsample_bytree**: Add randomness
4. **gamma**: Fine-tune split threshold
5. **reg_alpha + reg_lambda**: Final regularization tuning

### Starting Parameters

```python
# Good starting point for most problems
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42
}
```

### Preventing Overfitting

1. **Reduce max_depth**: Simpler trees generalize better
2. **Increase min_child_weight**: Require more samples per leaf
3. **Increase gamma**: Require larger gain for splits
4. **Decrease learning_rate** + increase n_estimators
5. **Add subsampling**: Both rows and columns
6. **Increase regularization**: Both L1 and L2

## Comparison with Other Boosting Libraries

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree growth | Level-wise | Leaf-wise | Symmetric |
| Speed | Fast | Fastest | Moderate |
| GPU support | Good | Good | Best |
| Categorical handling | Encoding required | Native | Best native |
| Missing values | Native | Native | Native |
| Default parameters | Needs tuning | Good defaults | Best defaults |
| Memory | Moderate | Low | Higher |

## See Also

- [LightGBM](../lightgbm/ReadMe.md) - Faster gradient boosting
- [CatBoost](../catboost/ReadMe.md) - Best for categorical features
- [Gradient Boosting](../ReadMe.md) - Parent concept
