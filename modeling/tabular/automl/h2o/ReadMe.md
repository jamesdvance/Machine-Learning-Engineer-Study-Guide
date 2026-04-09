# H2O

## Summary

H2O-3 is an open-source, in-memory, distributed machine learning platform built on the JVM. It provides a broad library of supervised and unsupervised algorithms and, most notably for this chapter, an AutoML module that automates model selection, hyperparameter tuning, and ensembling with minimal user configuration.

H2O AutoML trains models across six algorithm families -- XGBoost, GBM, GLM, Distributed Random Forest (including Extremely Randomized Trees), Deep Learning (feedforward neural networks), and Stacked Ensembles -- then ranks them on a leaderboard. The entire pipeline can be launched with two required arguments: a response column and a training frame.

Key points to remember:

- H2O-3 is the underlying platform; H2O AutoML is a module built on top of it. You interact with both through the same Python, R, or Flow interfaces.
- Architecture is JVM-based and distributed. Data is stored in columnar, compressed, in-memory format across cluster nodes. There is no master node; all nodes are peers.
- AutoML builds two Stacked Ensembles automatically: one combining all base models and one combining only the best model from each algorithm family.
- Models export as MOJO (Model ObJect, Optimized) or POJO (Plain Old Java Object) for production scoring without needing a running H2O cluster. MOJO is strongly preferred -- it is smaller and faster.
- Compared to AutoGluon, H2O is better suited for distributed big data workloads but typically scores lower on raw accuracy benchmarks. Compared to FLAML, H2O is heavier and more feature-rich, while FLAML prioritizes lightweight, cost-efficient hyperparameter search.
- Reproducibility requires setting `max_models` (not a time budget), providing a `seed`, and excluding Deep Learning.
- XGBoost within H2O AutoML is unavailable on Windows and Apple Silicon.

---

## What Is H2O-3?

H2O-3 is the third major version of the H2O platform, developed by H2O.ai. It is an in-memory machine learning platform that runs on the JVM and supports distributed computing across clusters of machines. It integrates with Hadoop and Spark ecosystems and provides client APIs in Python, R, Java, Scala, and a web-based GUI called Flow.

The platform implements a wide range of algorithms:

- Generalized Linear Models (GLM with Elastic Net regularization)
- Gradient Boosting Machines (GBM)
- XGBoost
- Distributed Random Forest and Extremely Randomized Trees (XRT)
- Deep Learning (multi-layer feedforward neural networks)
- K-Means clustering
- PCA
- Generalized Additive Models (GAM)
- Cox Proportional Hazards
- RuleFit
- Support Vector Machines
- Stacked Ensembles
- AutoML (the focus of this chapter)

H2O-3 is distinct from H2O.ai's commercial products like Driverless AI and H2O AI Cloud. This chapter covers the open-source H2O-3 platform and its AutoML module.

## Architecture

Understanding H2O's architecture helps explain both its strengths and its quirks.

### JVM and Distributed Computing

Each H2O node is a single JVM process. A cluster is formed by multiple nodes that communicate as peers -- there is no designated master. This peer-to-peer design simplifies fault tolerance reasoning but means all nodes must be started together; you cannot add nodes to a running cluster.

The JVM architecture has three layers:

1. **Language layer** -- REST API, client bindings for Python/R, Flow UI
2. **Algorithm layer** -- ML algorithm implementations
3. **Core infrastructure** -- distributed data storage, map-reduce framework, memory management

### Data Storage Model

Data is stored in a three-level hierarchy:

- **Frame**: the user-visible table (equivalent to a DataFrame)
- **Vec**: a single distributed column
- **Chunk**: a contiguous block of rows within a Vec, stored on one node

All Vecs in a Frame share a VectorGroup, which guarantees chunk alignment. Chunk `i` of column A covers the same rows as chunk `i` of column B. This alignment makes row-wise operations across columns efficient without shuffling.

Data is stored in compressed columnar format in JVM memory. A distributed key-value store provides access to data, models, and objects across all nodes.

### Computation Model

H2O uses its own in-memory map-reduce framework called MRTask, which is separate from Hadoop MapReduce. Each chunk is processed on the node where it resides, and partial results are reduced up a binary tree back to the calling node. Within each node, the Java fork/join framework handles multi-threading.

### Client Communication

All client interactions go through a versioned REST API. When you use the Python `h2o` package, your Python process sends HTTP requests to an H2O cluster (which could be running locally or remotely). Data is uploaded to the cluster; all computation happens there.

This means H2O has a client-server split that other AutoML tools (AutoGluon, FLAML) do not. Your Python process is a thin client. This is important to understand because:

- You need to start or connect to an H2O cluster before doing anything (`h2o.init()`)
- Data must be uploaded to the cluster as H2OFrames
- You cannot pass arbitrary Python objects or custom loss functions into H2O algorithms the way you can with scikit-learn-style APIs

## H2O AutoML

### Overview

H2O AutoML automates the process of training a large selection of candidate models within user-specified time or model-count constraints. It handles algorithm selection, hyperparameter search, and ensembling.

### Minimal Example

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Load data into H2O cluster
train = h2o.import_file("train.csv")
test = h2o.import_file("test.csv")

# Identify response and predictors
y = "target"
x = train.columns
x.remove(y)

# Run AutoML
aml = H2OAutoML(max_models=20, seed=42)
aml.train(x=x, y=y, training_frame=train)

# View leaderboard
print(aml.leaderboard)

# Predict with the best model
preds = aml.predict(test)

# Get the best model of a specific type
best_gbm = aml.get_best_model(algorithm="gbm")
```

### Required Parameters

Only two things are truly required:

- **y**: the response column
- **training_frame**: the training data

You must also specify at least one stopping criterion, or AutoML defaults to a one-hour runtime budget:

- **max_runtime_secs**: total wall-clock time budget
- **max_models**: maximum number of base models to train (excludes Stacked Ensembles)

### Algorithm Pipeline

AutoML trains models in a specific order. It begins with fixed-hyperparameter defaults for each algorithm family (to get baseline models quickly), then moves to random grid search over hyperparameter spaces for XGBoost, GBM, and Deep Learning. GLM uses its own internal lambda search. Finally, it trains Stacked Ensembles on top of the base models.

The hyperparameter search spaces are predefined. Key examples:

- **GBM**: max_depth 3-17, sample_rate 0.5-1.0, col_sample_rate 0.4-1.0, min_rows from 1 to 100, fixed learn_rate of 0.1, up to 10,000 trees with early stopping
- **XGBoost**: similar ranges plus regularization parameters (reg_alpha, reg_lambda), booster types (gbtree, dart)
- **Deep Learning**: 1-3 hidden layers with 20/50/100 units, dropout, Rectifier activation, ADADELTA optimizer variations
- **GLM**: alpha grid from 0.0 to 1.0 in steps of 0.2

### Stacked Ensembles

AutoML automatically builds two Stacked Ensemble models:

1. **All Models Ensemble**: combines every base model
2. **Best of Family Ensemble**: takes only the single best model from each algorithm family

The metalearner is a non-negative GLM with regularization (Lasso or Elastic Net, chosen by cross-validation). For classification, predictions are logit-transformed before being fed to the metalearner.

Stacked Ensembles require cross-validation predictions from base models, so they are disabled when `nfolds=0`.

### Cross-Validation and Early Stopping

By default, AutoML uses 5-fold cross-validation. You can customize this:

- `nfolds`: number of folds (default uses AutoML's selection, typically 5)
- `fold_column`: a column in the data specifying custom fold assignments
- Setting `nfolds=0` disables cross-validation and Stacked Ensembles

Early stopping applies to individual models:

- `stopping_metric`: AUTO (logloss for classification, deviance for regression), or explicitly AUC, RMSE, MAE, AUCPR, etc.
- `stopping_rounds`: number of scoring rounds without improvement before stopping (default 3)
- `stopping_tolerance`: minimum relative improvement required

### Leaderboard

The leaderboard ranks all trained models by a default metric:

- Binary classification: AUC
- Multiclass classification: mean per-class error
- Regression: RMSE (reported as deviance internally)

You can override this with `sort_metric`. The leaderboard can show extended information including training time and prediction latency per row by accessing `aml.leaderboard.head()` or requesting extra columns.

### Key Configuration Options

**Controlling which algorithms run:**

```python
# Only train tree-based models
aml = H2OAutoML(include_algos=["GBM", "XGBoost", "DRF"],
                max_models=20, seed=42)

# Exclude Deep Learning for reproducibility
aml = H2OAutoML(exclude_algos=["DeepLearning"],
                max_models=20, seed=42)
```

`include_algos` and `exclude_algos` are mutually exclusive.

**Per-model time budget:**

```python
aml = H2OAutoML(max_runtime_secs=3600,
                max_runtime_secs_per_model=300,
                seed=42)
```

**Class imbalance handling:**

```python
aml = H2OAutoML(max_models=20, seed=42,
                balance_classes=True,
                max_after_balance_size=5.0)
```

**Monotone constraints (for interpretability):**

```python
aml = H2OAutoML(max_models=20, seed=42,
                monotone_constraints={"age": 1, "income": -1})
```

**Preprocessing (experimental):**

```python
# Enable target encoding for high-cardinality categoricals
aml = H2OAutoML(max_models=20, seed=42,
                preprocessing=["target_encoding"])
```

**Exploitation ratio (experimental):**

```python
# Spend 10% of budget fine-tuning top GBM/XGBoost models
aml = H2OAutoML(max_models=20, seed=42,
                exploitation_ratio=0.1)
```

### Model Inspection and Explainability

```python
# Get the overall best model
best = aml.leader

# Get best model by a specific metric
best_logloss = aml.get_best_model(criterion="logloss")

# Get best model of a specific algorithm
best_xgb = aml.get_best_model(algorithm="xgboost")

# View the event log (what AutoML did and when)
print(aml.event_log)

# Explainability -- generates variable importance,
# partial dependence, SHAP, and other plots
aml.explain(test)

# Or explain the leader model specifically
aml.leader.explain(test)
```

The `explain()` integration is one of H2O's stronger features. It produces a multi-panel report including variable importance, partial dependence plots, SHAP contributions, and learning curves with a single call.

## Production Deployment

### MOJO and POJO Export

H2O models can be exported as standalone scoring artifacts:

- **MOJO (Model ObJect, Optimized)**: the recommended format. Uses a generic tree-walker approach that produces small, fast artifacts. No size restrictions.
- **POJO (Plain Old Java Object)**: legacy format. Generates a Java source file that must be compiled. Fails for source files larger than 1 GB.

MOJOs are 20-25x smaller on disk, 2-3x faster in "hot" scoring (after JVM optimization), and 10-40x faster in "cold" scoring compared to POJOs.

```python
# Export the best model as MOJO
best = aml.leader
best.download_mojo(path="./models/", get_genmodel_jar=True)
```

The only runtime dependency is the `h2o-genmodel.jar` file. This means you can score in any Java environment without a running H2O cluster.

### Deployment Considerations

- Input columns must contain only categorical levels seen during training. Unseen levels are treated as NA.
- Any transformations applied before training must also be applied before scoring.
- Additional input columns not used for training are ignored.
- All AutoML models support MOJO export.

### Scoring in Java

```java
import hex.genmodel.easy.*;

EasyPredictModelWrapper model = new EasyPredictModelWrapper(
    MojoModel.load("GBM_model.zip"));

RowData row = new RowData();
row.put("feature1", "value1");
row.put("feature2", 42.0);

BinomialModelPrediction pred = model.predictBinomial(row);
System.out.println("Prediction: " + pred.label);
System.out.println("Probability: " + pred.classProbabilities[1]);
```

## When to Choose H2O vs Alternatives

### H2O vs AutoGluon

AutoGluon tends to outperform H2O AutoML on accuracy benchmarks, particularly in multiclass classification. AutoGluon uses more aggressive multi-layer stacking and includes more model families by default. On a suite of 50 classification and regression tasks from the OpenML AutoML Benchmark, AutoGluon was faster, more robust, and more accurate overall.

Choose H2O over AutoGluon when:

- You need distributed computing across a cluster for datasets that do not fit in single-machine memory
- You need MOJO export for Java-based production scoring pipelines
- Your organization uses Hadoop or Spark and you want native integration
- You need the Flow UI for non-programmers on your team
- You have existing H2O infrastructure or expertise

Choose AutoGluon over H2O when:

- Maximum predictive accuracy is the priority
- You are working on a single machine (AutoGluon has no distributed mode but is highly optimized for single-node)
- You need multi-modal support (text, images, tabular combined)
- You want a simpler, more Pythonic API without client-server overhead

### H2O vs FLAML

FLAML is a lightweight AutoML library from Microsoft that focuses on cost-efficient hyperparameter optimization. It uses novel search strategies (like cost-frugal optimization) to find good configurations quickly with minimal compute.

Choose H2O over FLAML when:

- You need distributed training on large datasets
- You want automatic ensembling (FLAML does not build stacked ensembles by default)
- You need built-in explainability and a model leaderboard
- Production deployment in Java environments is a requirement

Choose FLAML over H2O when:

- You are resource-constrained (limited CPU, memory, or cloud budget)
- You want a library that works naturally with scikit-learn pipelines and custom estimators
- You need fast turnaround on small-to-medium datasets
- You want more control over the search process with less overhead

### Decision Summary

| Factor | H2O | AutoGluon | FLAML |
|--------|-----|-----------|-------|
| Best accuracy (benchmarks) | Good | Best | Good |
| Distributed computing | Yes | No | No |
| Resource efficiency | Heavy | Moderate | Light |
| Production scoring (Java) | MOJO/POJO | No native | No native |
| Multi-modal data | No | Yes | No |
| Automatic ensembling | Yes | Yes (stronger) | No |
| API complexity | Higher (client-server) | Simple | Simple |
| GUI | Flow | No | No |

## Practical Tips

1. **Start with `max_models`, not `max_runtime_secs`**, if you need reproducibility. Time-based budgets produce different results depending on hardware speed.

2. **Exclude Deep Learning for reproducibility.** H2O's Deep Learning algorithm is not reproducible across runs even with the same seed, due to non-deterministic floating point operations.

3. **Set `max_runtime_secs_per_model`** in addition to a total budget. Without it, a single slow model (often Deep Learning) can consume most of your time budget.

4. **Memory management matters.** H2O recommends giving the JVM no more than two-thirds of available RAM. Use `h2o.init(max_mem_size="16G")` to set this explicitly. Running out of memory mid-training produces cryptic JVM errors.

5. **Use `include_algos` to focus.** If you know XGBoost works well for your problem, restrict the search to XGBoost and GBM. This lets AutoML explore more hyperparameter configurations for those algorithms within the same time budget.

6. **Check XGBoost availability.** XGBoost is not available in H2O on Windows or Apple Silicon. If your development machine runs Windows but production is Linux, be aware that your local AutoML runs will skip XGBoost models.

7. **Prefer MOJO over POJO for production.** There is no reason to use POJO for new projects. MOJO is smaller, faster, and has no size limitations.

8. **Use the leaderboard for model selection, not just the leader.** The best model on cross-validation may not be the best on your holdout set. Inspect the top several models and their characteristics.

9. **Provide a `leaderboard_frame`.** If you have a dedicated test set, pass it as the leaderboard frame so models are ranked on truly held-out data rather than cross-validation estimates.

10. **Watch for data leakage in the H2OFrame.** Since H2O handles its own train/validation splits, ensure your training frame does not contain the response or derived features that would leak information. H2O will not catch this for you.

11. **Use `h2o.explain()` after training.** The built-in explainability suite is comprehensive and saves significant time compared to manually generating SHAP plots and partial dependence plots.

12. **Shut down the cluster when done.** H2O clusters hold memory until explicitly stopped. Call `h2o.cluster().shutdown()` or `h2o.shutdown()` to release resources, especially in shared environments.

## Common Pitfalls

- **Forgetting `h2o.init()`**: Unlike AutoGluon or FLAML, H2O requires an active cluster. Every script must start with `h2o.init()`.
- **Data type mismatches**: H2O infers types on import. A numeric column with a single non-numeric value becomes categorical. Use `train[col] = train[col].asnumeric()` to fix this.
- **Large categorical cardinality**: Columns with thousands of categories slow down training and may cause memory issues. Use `preprocessing=["target_encoding"]` or manually reduce cardinality before training.
- **Silent fallback behavior**: If an algorithm fails (e.g., XGBoost not available), AutoML silently skips it. Check the event log (`aml.event_log`) to verify which models actually trained.
- **Cross-validation overhead**: With large datasets, 5-fold CV means training 5x the models. Consider setting `nfolds=3` or using a validation frame (`nfolds=0`) to reduce compute.
