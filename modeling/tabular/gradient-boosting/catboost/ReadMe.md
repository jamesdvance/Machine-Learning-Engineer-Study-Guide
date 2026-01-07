# CatBoost

## Summary

CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex that excels at handling categorical features and provides robust out-of-the-box performance. It uses ordered boosting to combat prediction shift and symmetric trees for faster inference. CatBoost typically requires less hyperparameter tuning than other boosting libraries while delivering competitive results.

Key points to remember:

- Categorical handling: Best-in-class native support for categorical features
- Ordered boosting: Uses permutation-driven approach to prevent target leakage
- Symmetric trees: All leaves at same depth, faster inference
- Robust defaults: Works well with minimal tuning
- GPU optimization: Excellent multi-GPU training support
- Overfitting resistance: Built-in mechanisms reduce overfitting
- Feature combinations: Automatically creates feature interactions
- Higher memory: Uses more memory than XGBoost/LightGBM

## Core Concepts

### Ordered Boosting

Traditional gradient boosting suffers from target leakage when calculating target statistics for categorical features. CatBoost uses ordered boosting to address this:

```
Traditional: Calculate target mean using ALL training data
Problem: Target statistics leak future information

Ordered Boosting:
1. Random permutation of training data
2. For sample i, calculate statistics using only samples 0...i-1
3. Different permutation for each tree

Result: No target leakage, better generalization
```

### Symmetric Trees

```
XGBoost/LightGBM: Asymmetric trees
- Different splits at each node
- More flexible
- Slower inference

CatBoost: Symmetric (oblivious) trees
- Same split condition at all nodes of same depth
- Less flexible
- Much faster inference
- Better regularization

Example symmetric tree (depth=2):
         [feature_a < 5]
        /               \
 [feature_b < 3]   [feature_b < 3]
   /    \            /    \
 leaf  leaf        leaf  leaf
```

### Categorical Feature Handling

```
Traditional encoding methods:
- One-hot: High dimensionality, ignores ordering
- Label encoding: Arbitrary ordering
- Target encoding: Target leakage risk

CatBoost approach:
1. Ordered target statistics (no leakage)
2. Random permutations for robustness
3. Prior for regularization:
   target_stat = (count * mean_in_category + prior) / (count + 1)
```

## Basic Usage

### Classification

```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify categorical columns
categorical_features = ['category_a', 'category_b', 'category_c']

# Create Pool (CatBoost's data structure)
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize and train
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=50
)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
```

### Regression

```python
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Multiclass Classification

```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    classes_count=3,
    random_seed=42,
    verbose=False
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
```

## Key Hyperparameters

### Tree Structure

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| depth | 6 | 4-10 | Tree depth (symmetric trees) |
| min_data_in_leaf | 1 | 1-100 | Minimum samples in leaf |
| max_leaves | 31 | - | Maximum leaves (only if grow_policy='Lossguide') |
| grow_policy | 'SymmetricTree' | - | Tree growth policy |

### Learning Control

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| iterations | 1000 | 100-10000 | Number of boosting iterations |
| learning_rate | Auto | 0.01-0.3 | Shrinkage factor |
| l2_leaf_reg | 3.0 | 0-10 | L2 regularization coefficient |
| random_strength | 1.0 | 0-10 | Random noise for scoring splits |

### Categorical Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| cat_features | None | List of categorical feature indices/names |
| one_hot_max_size | 2 | Max categories for one-hot encoding |
| max_ctr_complexity | 4 | Max feature combinations for target statistics |
| ctr_target_border_count | 1 | Number of borders for target quantization |

### Sampling

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| subsample | 1.0 | 0.5-1.0 | Row sampling ratio |
| colsample_bylevel | 1.0 | 0.5-1.0 | Column sampling per tree level |
| bagging_temperature | 1.0 | 0-10 | Bayesian bootstrap intensity |

## Categorical Feature Handling

### Automatic Detection

```python
import pandas as pd

# Method 1: Specify by index
cat_features = [0, 3, 5]

# Method 2: Specify by column name
cat_features = ['city', 'category', 'brand']

# Method 3: Auto-detect from pandas dtype
df['city'] = df['city'].astype('category')
df['category'] = df['category'].astype('category')

# CatBoost auto-detects 'category' dtype
model.fit(df[features], df[target])
```

### Feature Combinations

CatBoost automatically creates combinations of categorical features:

```python
# Control combination complexity
model = CatBoostClassifier(
    max_ctr_complexity=4,  # Max number of features to combine
    simple_ctr=['Borders:TargetBorderCount=10'],  # For numerical target
    combinations_ctr=['Borders:TargetBorderCount=10']  # For combinations
)

# Example: If you have features [city, category, brand]
# CatBoost may create: city_category, city_brand, city_category_brand
```

### Text Features

CatBoost can handle text features directly:

```python
# Specify text features
model = CatBoostClassifier(
    text_features=['description', 'title'],
    tokenizers=[
        {'tokenizer_id': 'Space', 'delimiter': ' '},
        {'tokenizer_id': 'Ngram', 'token_types': ['Word'], 'gramOrder': [2]}
    ],
    dictionaries=[
        {'dictionary_id': 'Word', 'token_level_type': 'Word'}
    ],
    feature_calcers=['BoW:top_tokens_count=10000']
)
```

## Hyperparameter Tuning

### Grid Search with Built-in CV

```python
from catboost import CatBoostClassifier, cv, Pool

params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42
}

train_pool = Pool(X_train, y_train, cat_features=categorical_features)

# Built-in cross-validation
cv_results = cv(
    params,
    train_pool,
    fold_count=5,
    shuffle=True,
    stratified=True,
    verbose=False,
    early_stopping_rounds=50
)

print(f"Best iteration: {cv_results['test-AUC-mean'].idxmax()}")
print(f"Best AUC: {cv_results['test-AUC-mean'].max():.4f}")
```

### Optuna Optimization

```python
import optuna

def objective(trial):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    return model.get_best_score()['validation']['AUC']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best AUC: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_params}")
```

### Randomized Search

```python
from catboost import CatBoostClassifier

# CatBoost's built-in randomized search
model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42
)

grid = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 10],
    'iterations': [500, 1000]
}

randomized_search_result = model.randomized_search(
    grid,
    X=train_pool,
    cv=5,
    n_iter=50,
    search_by_train_test_split=True,
    calc_cv_statistics=True,
    verbose=False
)
```

## Feature Importance

```python
# Multiple importance types
feature_importance = model.get_feature_importance(prettified=True)

# Types: PredictionValuesChange, LossFunctionChange, FeatureImportance, etc.
importance_types = [
    'PredictionValuesChange',
    'LossFunctionChange',
    'ShapValues',
    'Interaction'
]

for imp_type in importance_types:
    importance = model.get_feature_importance(
        type=imp_type,
        data=test_pool
    )
    print(f"\n{imp_type}:")
    print(importance[:10])  # Top 10 features
```

### SHAP Values

```python
# Built-in SHAP support
shap_values = model.get_feature_importance(
    type='ShapValues',
    data=test_pool
)

# Plot with shap library
import shap
shap.summary_plot(shap_values[:, :-1], X_test)  # Exclude expected value column

# For single prediction explanation
shap.force_plot(
    shap_values[0, -1],  # Expected value
    shap_values[0, :-1],  # SHAP values
    X_test.iloc[0]
)
```

### Feature Interactions

```python
# Get feature interactions
interactions = model.get_feature_importance(type='Interaction')

# Returns pairs of features and their interaction strengths
for i, (f1, f2, strength) in enumerate(interactions[:10]):
    print(f"{f1} × {f2}: {strength:.4f}")
```

## Handling Imbalanced Data

### Class Weights

```python
# Automatic balancing
model = CatBoostClassifier(
    auto_class_weights='Balanced',  # or 'SqrtBalanced'
    loss_function='Logloss'
)

# Manual class weights
class_weights = {0: 1, 1: 10}  # Weight positive class 10x
model = CatBoostClassifier(
    class_weights=class_weights
)

# Scale position weight for binary
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = CatBoostClassifier(
    scale_pos_weight=scale_pos_weight
)
```

### Focal Loss

```python
from catboost import CatBoostClassifier

# CatBoost has built-in focal loss support
model = CatBoostClassifier(
    loss_function='Logloss:focal_alpha=0.25;focal_gamma=2',
    random_seed=42
)

# Alternative: Custom loss function
model = CatBoostClassifier(
    loss_function='FocalLoss',  # Requires recent CatBoost version
    random_seed=42
)
```

## GPU Training

```python
# Single GPU
model = CatBoostClassifier(
    task_type='GPU',
    devices='0',  # GPU device ID
    iterations=1000,
    learning_rate=0.1,
    depth=6
)

# Multi-GPU
model = CatBoostClassifier(
    task_type='GPU',
    devices='0:1:2:3',  # Multiple GPU IDs
    gpu_ram_part=0.95,  # Use 95% of GPU memory
)

# Mixed precision training
model = CatBoostClassifier(
    task_type='GPU',
    gpu_cat_features_storage='GpuRam',  # Store categorical features on GPU
)
```

## Model Persistence

```python
# Method 1: Native CatBoost format (recommended)
model.save_model('model.cbm')  # Binary format
model.save_model('model.json', format='json')  # JSON format
model.save_model('model.onnx', format='onnx')  # ONNX for deployment

loaded_model = CatBoostClassifier()
loaded_model.load_model('model.cbm')

# Method 2: Pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 3: CoreML export (for iOS)
model.save_model('model.mlmodel', format='coreml')
```

## Monotonic Constraints

```python
# Define monotonic constraints
# 1 = increasing, -1 = decreasing, 0 = no constraint
model = CatBoostClassifier(
    monotone_constraints={
        'age': 1,           # Older = higher probability
        'income': 1,        # Higher income = higher probability
        'risk_score': -1    # Higher risk = lower probability
    }
)

# Or by index
model = CatBoostClassifier(
    monotone_constraints=[1, 0, -1, 0, 1]
)
```

## Feature Selection

```python
# Built-in feature selection
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(iterations=1000, random_seed=42)

# Select optimal feature set
summary = model.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select=list(range(X_train.shape[1])),
    num_features_to_select=10,
    algorithm='RecursiveByLossFunctionChange',  # or 'RecursiveByPredictionValuesChange'
    steps=3,
    train_final_model=True
)

print(f"Selected features: {summary['selected_features']}")
print(f"Eliminated features: {summary['eliminated_features_names']}")
```

## Ranking

```python
from catboost import CatBoostRanker, Pool

# Prepare ranking data with group information
train_pool = Pool(
    X_train,
    y_train,
    group_id=group_ids_train,  # Query/group identifiers
    cat_features=categorical_features
)

model = CatBoostRanker(
    loss_function='YetiRank',  # or 'PairLogit', 'QueryRMSE'
    iterations=1000,
    learning_rate=0.1,
    depth=6
)

model.fit(train_pool, eval_set=test_pool)

# Predict relevance scores
scores = model.predict(X_test)
```

## Custom Metrics

```python
from catboost import CatBoostClassifier

class CustomMetric:
    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return True  # Higher is better

    def evaluate(self, approxes, target, weight):
        # approxes: model predictions
        # target: true labels
        # weight: sample weights

        predictions = np.array(approxes[0])
        targets = np.array(target)

        # Custom metric calculation
        metric_value = np.mean((predictions > 0.5) == targets)

        return metric_value, 1.0  # Return (metric, weight)

model = CatBoostClassifier(
    iterations=1000,
    eval_metric=CustomMetric()
)
```

## Best Practices

### Default Parameters (Usually Good Enough)

```python
# CatBoost's defaults are well-tuned
model = CatBoostClassifier(
    random_seed=42,
    verbose=100
)

# These parameters are auto-tuned:
# - learning_rate: Adjusted based on iterations
# - border_count: 254 for CPU, 128 for GPU
# - l2_leaf_reg: 3.0
```

### When to Tune

```python
# Tune these first if needed:
priority_params = {
    'depth': [4, 6, 8, 10],            # Tree complexity
    'learning_rate': [0.03, 0.1, 0.3],  # Learning speed
    'l2_leaf_reg': [1, 3, 5, 10],       # Regularization
}

# Tune these second:
secondary_params = {
    'random_strength': [0, 1, 5],       # Randomization
    'bagging_temperature': [0, 0.5, 1], # Bootstrap intensity
    'border_count': [32, 128, 254],     # Split candidates
}
```

### Performance Tips

```python
# Speed optimization
model = CatBoostClassifier(
    task_type='GPU',                  # Use GPU
    bootstrap_type='Bernoulli',       # Faster than MVS
    grow_policy='SymmetricTree',      # Fastest inference
    max_ctr_complexity=1,             # Reduce categorical combinations
    sparse_features_conflict_fraction=0.5  # For sparse data
)

# Memory optimization
model = CatBoostClassifier(
    max_bin=128,                      # Reduce bin count
    one_hot_max_size=10,              # Limit one-hot encoding
    feature_border_type='GreedyLogSum'  # Memory-efficient borders
)
```

## Comparison with XGBoost and LightGBM

| Feature | CatBoost | XGBoost | LightGBM |
|---------|----------|---------|----------|
| Categorical handling | Best | Encoding required | Good |
| Default performance | Best | Needs tuning | Good |
| Training speed | Moderate | Fast | Fastest |
| Inference speed | Fastest | Moderate | Fast |
| Memory usage | Highest | Moderate | Lowest |
| GPU support | Best | Good | Good |
| Overfitting resistance | Best | Good | Moderate |
| Text feature support | Yes | No | No |

## See Also

- [XGBoost](../xgboost/ReadMe.md) - Flexible gradient boosting
- [LightGBM](../lightgbm/ReadMe.md) - Fastest gradient boosting
- [Gradient Boosting](../ReadMe.md) - Parent concept
