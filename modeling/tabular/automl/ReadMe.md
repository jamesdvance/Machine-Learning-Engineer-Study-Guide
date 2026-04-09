# AutoML

## Summary

AutoML (Automated Machine Learning) refers to the process of automating the end-to-end pipeline of applying machine learning to real-world problems. This includes data preprocessing, feature engineering, model selection, hyperparameter tuning, and ensembling. For tabular data, AutoML frameworks have matured to the point where they routinely match or exceed hand-tuned models, particularly when given sufficient compute budget. The three most practical open-source AutoML frameworks for tabular data today are AutoGluon, H2O, and FLAML, each representing a distinct philosophy in the trade-off between accuracy, compute cost, and deployment flexibility.

Key points to remember:

- AutoML automates model selection, hyperparameter optimization, and ensembling -- it does not replace understanding the problem, cleaning data, or engineering domain-specific features
- The three dominant open-source tabular AutoML frameworks are AutoGluon (ensemble-first, highest accuracy), H2O (enterprise-grade, distributed), and FLAML (lightweight, cost-efficient)
- Hyperparameter optimization (HPO) is a core AutoML component, with approaches ranging from grid search and random search to Bayesian optimization and cost-aware methods
- Model selection and ensembling are equally important -- the best AutoML tools combine multiple model families rather than perfecting a single one
- AutoML is most valuable for establishing strong baselines, rapid prototyping, and scenarios where ML expertise is limited
- Production deployment of AutoML outputs requires the same rigor as any ML system: monitoring, versioning, reproducibility, and latency constraints
- AutoML does not eliminate the need for feature engineering -- domain-specific features still provide significant lift, and AutoML tools vary widely in how much preprocessing they automate
- The choice between AutoML frameworks depends on your constraints: accuracy requirements, compute budget, deployment target, team skills, and dataset scale

## What AutoML Automates

The machine learning pipeline for tabular data involves many steps that are tedious, repetitive, and require expertise to do well. AutoML tools aim to automate some or all of them.

### The Full Pipeline

```
Raw Data
    |
 1. Data Preprocessing
    - Type detection, missing values, encoding
    |
 2. Feature Engineering
    - Transformations, interactions, selection
    |
 3. Model Selection
    - Which algorithm families to try
    |
 4. Hyperparameter Optimization
    - Finding the best configuration per model
    |
 5. Ensembling
    - Combining models for better predictions
    |
 6. Model Evaluation
    - Cross-validation, holdout scoring
    |
Deployed Model
```

Different AutoML tools automate different subsets of this pipeline and to different degrees. AutoGluon aggressively automates steps 1 through 6. FLAML focuses primarily on steps 3 and 4. H2O covers steps 1 through 6 with an emphasis on scalability. No current AutoML tool fully replaces step 2 (feature engineering) when domain knowledge is available -- see the [Feature Engineering](../feature-engineering/ReadMe.md) chapter for more on this.

### What AutoML Does Not Do

It is important to be clear about what falls outside the scope of AutoML:

- **Problem formulation**: Deciding what to predict, how to frame the problem, and what data to collect
- **Data collection and labeling**: Gathering training data and ensuring label quality
- **Domain feature engineering**: Creating features that require business knowledge (e.g., "days since last purchase" in a churn model)
- **Deployment infrastructure**: Building serving systems, monitoring, CI/CD
- **Fairness and bias auditing**: Ensuring the model behaves appropriately across subgroups
- **Ongoing maintenance**: Retraining schedules, drift detection, performance monitoring

AutoML handles the middle of the pipeline -- the part between having clean, labeled data and having a trained model ready for evaluation.

## Core AutoML Techniques

### Hyperparameter Optimization (HPO)

Hyperparameter optimization is the most fundamental AutoML technique. Every model has hyperparameters (learning rate, tree depth, regularization strength, number of layers) that must be set before training. The goal of HPO is to find the configuration that maximizes performance on held-out data.

**Grid Search**: Evaluates every combination in a predefined grid. Exhaustive but scales exponentially with the number of hyperparameters. Only practical for small search spaces (2-3 parameters with a few values each).

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000]
}

grid = GridSearchCV(estimator, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
```

**Random Search**: Samples configurations randomly from the search space. Surprisingly effective because it explores more distinct values per hyperparameter than grid search for the same budget. Bergstra and Bengio (2012) showed that random search finds good configurations faster than grid search in most cases because not all hyperparameters are equally important.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.001, 0.3),
    'n_estimators': randint(50, 2000)
}

random = RandomizedSearchCV(estimator, param_distributions,
                            n_iter=100, cv=5, scoring='roc_auc')
random.fit(X_train, y_train)
```

**Bayesian Optimization**: Builds a probabilistic surrogate model of the objective function and uses an acquisition function to decide which configuration to try next. This balances exploration (trying diverse configurations) with exploitation (refining promising regions). Common implementations include Optuna, Hyperopt, and SMAC.

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Cost-Aware Optimization**: The approach used by FLAML. It prioritizes cheap configurations first (e.g., few boosting rounds, shallow trees) and only moves to expensive ones when cheap ones stop improving. This is more efficient than methods that treat all configurations equally in terms of cost.

**Multi-Fidelity Methods**: Techniques like Successive Halving and Hyperband train many configurations on a small subset of data, then progressively allocate more resources to the most promising ones. This avoids wasting full training runs on poor configurations.

### Model Selection

Model selection is the process of choosing which algorithm families to evaluate. The key insight behind modern AutoML is that no single algorithm dominates across all datasets. Gradient boosting (LightGBM, XGBoost, CatBoost) wins most often on tabular data, but random forests, linear models, and neural networks each have niches where they excel.

AutoML frameworks handle model selection in different ways:

- **AutoGluon**: Trains all algorithm families simultaneously with known-good defaults, relying on ensembling to combine their strengths
- **H2O**: Trains models from six algorithm families sequentially, starting with defaults then exploring hyperparameter spaces via random search
- **FLAML**: Maintains independent search states per algorithm and allocates trials based on relative performance, naturally spending more time on promising model types

### Ensembling

Ensembling combines multiple models to produce predictions that are more accurate and robust than any single model. It is one of the most reliable ways to improve predictive performance on tabular data.

**Averaging / Voting**: The simplest approach. Average the predictions of multiple models (regression) or take a majority vote (classification). Even this basic method often outperforms any single model.

**Weighted Averaging**: Learn optimal weights for each model's contribution. Models that perform better on validation data receive higher weights. AutoGluon uses a greedy weighted ensemble as its final layer.

**Stacking (Stacked Generalization)**: Train a meta-model on the out-of-fold predictions of base models. The meta-model learns how to best combine base model predictions. This is more powerful than simple averaging because the meta-model can learn complex relationships between base model outputs.

```
Base Models (Layer 1):
  LightGBM  ->  OOF predictions
  XGBoost   ->  OOF predictions
  CatBoost  ->  OOF predictions
  RF        ->  OOF predictions

Meta-Model (Layer 2):
  GLM or another learner trained on the OOF predictions from Layer 1
```

**Multi-Layer Stacking**: AutoGluon extends stacking to multiple layers, where each layer receives the original features plus predictions from all previous layers. This is its core architectural innovation.

**Bagging**: Train the same model on different bootstrap samples or cross-validation folds, then average predictions. Reduces variance. AutoGluon bags every model across K folds, producing out-of-fold predictions needed for stacking without data leakage.

### Neural Architecture Search (NAS)

Neural Architecture Search automates the design of neural network architectures. While most relevant for deep learning on images and text, it has some application in tabular AutoML. Tools like Auto-sklearn and TPOT explore pipeline configurations that can include neural network components. AutoGluon includes custom tabular neural networks (TabularNeuralNetTorch) with pre-tuned architectures but does not perform NAS at runtime.

For most tabular problems, NAS is less important than model selection and ensembling because gradient boosting methods dominate tabular benchmarks.

### Automated Feature Engineering

Some AutoML tools include automated feature engineering:

- **AutoGluon**: Performs automatic data type detection, categorical encoding, missing value handling, and text feature extraction. Does not generate interaction features or domain-specific features.
- **H2O**: Handles categorical encoding natively (including target encoding as an experimental option). Provides automatic handling of missing values and data type inference.
- **FLAML**: Minimal built-in preprocessing -- median imputation for numericals, placeholder encoding for categoricals. Designed to work with user-provided feature pipelines.

For deeper coverage of feature engineering techniques, see the [Feature Engineering](../feature-engineering/ReadMe.md) chapter.

## Comparing AutoGluon, H2O, and FLAML

These three frameworks represent three distinct philosophies for solving the same problem. Understanding their approaches helps you choose the right tool.

### Philosophy and Approach

**AutoGluon** (Amazon, 2020) follows an ensemble-first philosophy. Its core insight is that spending compute on hyperparameter tuning for a single model is less effective than training many diverse models with strong defaults and combining them via multi-layer stacking. AutoGluon evaluated over 1,300 model configurations across 200 datasets to determine optimal default portfolios. The result is that even without any tuning, its defaults are competitive.

**H2O AutoML** (H2O.ai) takes an enterprise-distributed approach. Built on the JVM with a client-server architecture, it prioritizes scalability across clusters and broad language support (Python, R, Java, Scala). It combines random hyperparameter search with stacked ensembles and provides mature deployment artifacts (MOJO export for Java scoring without a running cluster).

**FLAML** (Microsoft Research, 2020) prioritizes compute efficiency. Its CFO (Cost-Frugal Optimization) algorithm starts from cheap configurations and only explores expensive ones when justified. This makes FLAML particularly effective under tight time or resource budgets. It produces a single best model rather than a large ensemble, which simplifies deployment.

### Comparison Table

| Dimension | AutoGluon | H2O AutoML | FLAML |
|-----------|-----------|------------|-------|
| Developer | Amazon (AWS) | H2O.ai | Microsoft Research |
| Core strategy | Multi-layer stacking + ensembling | Random search + stacking | Cost-frugal HPO |
| Peak accuracy (benchmarks) | Highest | High | Good |
| Accuracy under tight budget | Good | Moderate | Good to best |
| Training speed | Moderate | Slow | Fastest |
| Memory footprint | High | High (JVM) | Low |
| Inference speed | Slow (ensemble) | Moderate | Fast (single model) |
| Language support | Python | Python, R, Java, Scala | Python |
| Distributed training | No | Yes (multi-node clusters) | No |
| Ease of use | Very easy (3 lines) | Moderate (requires cluster init) | Very easy |
| Production export | Distillation, clone_for_deployment | MOJO (standalone Java artifact) | Extract sklearn-compatible model |
| Automatic ensembling | Multi-layer stacking | Two stacked ensembles | No (single best model) |
| Feature preprocessing | Automatic, aggressive | Automatic, moderate | Minimal |
| Custom model support | Yes | Limited | Yes (add_learner) |
| GPU support | Yes (neural nets, boosting) | Yes (XGBoost, Deep Learning) | Depends on estimator |
| GUI | No | Flow (web-based) | No |
| Scikit-learn compatible | No (own API) | No (own API, H2OFrames) | Yes (fit/predict) |

### Minimal Code Comparison

**AutoGluon**:
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target').fit(train_data, time_limit=3600)
predictions = predictor.predict(test_data)
```

**H2O AutoML**:
```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.import_file("train.csv")
aml = H2OAutoML(max_models=20, seed=42)
aml.train(x=features, y="target", training_frame=train)
predictions = aml.predict(h2o.import_file("test.csv"))
```

**FLAML**:
```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=120)
predictions = automl.predict(X_test)
```

### When to Choose Each

**Choose AutoGluon when:**
- Maximum predictive accuracy is the top priority
- You have sufficient compute resources (RAM, disk, time)
- You are doing offline batch prediction or competitions
- You want minimal configuration -- strong defaults out of the box
- You plan to deploy on AWS (native SageMaker integration)
- You can accept ensemble-level inference latency or will use distillation

**Choose H2O when:**
- Your dataset is too large for single-machine memory and you need distributed training
- Your team works across multiple languages (R, Java, Scala, not just Python)
- You need standalone Java scoring artifacts (MOJO) for production
- You want a built-in web UI (Flow) for non-programmers
- You are in an enterprise environment with existing Hadoop/Spark infrastructure
- You need built-in explainability reports (`explain()`)

**Choose FLAML when:**
- You are working under a tight compute or cloud cost budget
- You want fast iteration during prototyping (results in 60-120 seconds)
- You need a single lightweight model for low-latency serving
- You want scikit-learn compatibility and integration with existing pipelines
- You are tuning a specific model type rather than doing full model selection
- You need a minimal dependency footprint

### Benchmark Performance

On the OpenML AutoML Benchmark (a standardized suite of classification and regression tasks), the general ranking is:

1. AutoGluon consistently ranks first or second in raw accuracy, particularly with the `best_quality` preset
2. H2O AutoML places in the middle -- competitive but rarely the top performer on accuracy
3. FLAML achieves good accuracy relative to its compute budget, often matching H2O with far less compute

However, accuracy alone does not determine the best tool. A model that is 0.2% more accurate but takes 10x longer to train and requires 5x more memory for inference may not be the right choice for a production system with latency constraints.

## Other Notable AutoML Tools

While AutoGluon, H2O, and FLAML are the focus of this guide, several other tools are worth knowing about for context.

### Auto-sklearn

Auto-sklearn is built on scikit-learn and uses Bayesian optimization (SMAC) to search over scikit-learn pipelines. It was one of the first successful AutoML frameworks and won early AutoML challenges. It includes meta-learning (using performance on prior datasets to warm-start search) and automatic ensemble construction. However, it is limited to scikit-learn estimators and does not include modern gradient boosting implementations (LightGBM, CatBoost) by default. Development has slowed relative to newer alternatives. Auto-sklearn 2.0 simplified the approach by dropping meta-learning in favor of portfolio-based configuration selection, which influenced AutoGluon's design.

### TPOT

TPOT (Tree-based Pipeline Optimization Tool) uses genetic programming to evolve complete scikit-learn pipelines, including preprocessing steps, feature selection, and model selection. It explores a broader space than most AutoML tools because it can discover novel pipeline structures. The downside is that genetic search is slow and resource-intensive. TPOT is best for exploratory analysis where you want to discover unexpected pipeline configurations.

### Google Cloud AutoML (Vertex AI)

Google's managed AutoML service handles tabular, image, text, and video data. It abstracts away all infrastructure concerns and provides a web UI for non-technical users. The trade-offs are cost (pay-per-use cloud pricing), vendor lock-in, limited customization, and opacity (you cannot inspect the exact models or pipelines used). Best for organizations that want AutoML as a managed service rather than a library.

### Amazon SageMaker Autopilot

AWS's managed AutoML offering. It generates candidate pipelines, trains them, and provides notebooks showing what it did. It uses AutoGluon under the hood for tabular tasks. Useful for AWS-native teams who want managed infrastructure with more transparency than Google's offering.

### Azure Automated ML

Microsoft's cloud AutoML service within Azure Machine Learning. Supports tabular, image, and text tasks. Provides explainability dashboards and integrates with Azure's MLOps tooling. Like Google's offering, it is a managed service with the associated trade-offs of cost and vendor lock-in.

### Ray Tune

Not an AutoML tool per se, but a distributed hyperparameter tuning framework that can serve as the backend for AutoML workflows. FLAML integrates with Ray Tune for distributed execution. Ray Tune provides schedulers (Hyperband, ASHA, PBT) and search algorithms (Bayesian, Optuna) that can be composed into custom AutoML pipelines.

## When to Use AutoML vs Manual Tuning

### AutoML Is the Right Choice When:

- You need a strong baseline quickly. AutoML in 10 minutes often beats a hand-tuned model built over days.
- ML expertise is limited on the team. AutoML encodes best practices in code.
- You are exploring many datasets or problem formulations. AutoML provides consistent, comparable results across experiments.
- The problem is well-defined tabular classification or regression. This is where AutoML tools are most mature.
- You want to validate whether ML can solve the problem before investing in a custom pipeline.

### Manual Tuning Is the Right Choice When:

- You have deep domain expertise that translates into feature engineering. AutoML cannot discover features that require business knowledge.
- Inference latency is a hard constraint. Hand-crafted single models with known latency profiles are easier to optimize than AutoML ensembles.
- The problem has unusual requirements (custom loss functions, fairness constraints, specific model architectures). Most AutoML tools support standard objectives but not arbitrary constraints.
- You need full reproducibility and explainability for regulatory purposes. Some AutoML tools have non-deterministic components.
- The data pipeline is complex (streaming, multi-modal with custom fusion, graph-structured). AutoML tools are designed for static tabular datasets.

### The Hybrid Approach

In practice, the best results often come from combining AutoML with manual work:

1. **Start with AutoML** to get a strong baseline and understand which model families work well for your data
2. **Invest in feature engineering** using domain knowledge -- this is where human expertise adds the most value
3. **Re-run AutoML** on the enriched feature set
4. **Select the deployment model** based on production constraints (latency, memory, interpretability), possibly using a single model from the leaderboard or a distilled version of the ensemble

```python
# Hybrid approach example with AutoGluon
import pandas as pd
from autogluon.tabular import TabularPredictor

# Step 1: Manual feature engineering
train_data = pd.read_csv('train.csv')
train_data['revenue_per_employee'] = train_data['revenue'] / (train_data['employees'] + 1)
train_data['years_since_founded'] = 2026 - train_data['founded_year']

# Step 2: AutoML on enriched features
predictor = TabularPredictor(label='target', eval_metric='roc_auc').fit(
    train_data,
    presets='best_quality',
    time_limit=3600
)

# Step 3: Evaluate and select deployment model
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Step 4: Optionally distill for production
distilled = predictor.distill(time_limit=600, hyperparameters={'GBM': {}})
```

## Production Considerations

### Model Serialization and Serving

Each framework has different deployment characteristics:

| Framework | Export Format | Runtime Dependencies | Serving Model |
|-----------|-------------|---------------------|---------------|
| AutoGluon | Pickled ensemble on disk | Full AutoGluon installation | Python process, SageMaker |
| H2O | MOJO (standalone JAR) | h2o-genmodel.jar only | Any JVM environment |
| FLAML | Pickle or joblib | scikit-learn + estimator library | Standard Python serving |

H2O's MOJO export is a significant advantage for Java-based production environments. AutoGluon's `clone_for_deployment()` and distillation features help reduce the footprint. FLAML's scikit-learn compatibility makes it the easiest to integrate into existing Python serving infrastructure.

### Inference Latency

Ensemble models (AutoGluon, H2O stacked ensembles) are inherently slower at inference than single models. For real-time serving with sub-millisecond latency requirements:

- Use FLAML, which outputs a single model
- Use AutoGluon with distillation to compress the ensemble into a single model
- Use a single model from the AutoGluon or H2O leaderboard instead of the full ensemble
- Export H2O models as MOJOs for optimized Java-based scoring

### Reproducibility

Reproducibility is a common challenge with AutoML:

- **AutoGluon**: Generally reproducible with the same data, version, and hardware. Results may differ across machines due to timing-dependent model selection.
- **H2O**: Reproducible when using `max_models` (not time-based stopping), a fixed `seed`, and excluding Deep Learning (which is non-deterministic). Time-based budgets produce different results on different hardware.
- **FLAML**: Reproducible with a fixed `seed` and `time_budget` on the same hardware. Cost-aware search is deterministic given the same sequence of trial timings.

For regulatory or audit requirements, prefer `max_models` over `max_runtime_secs` and pin all library versions.

### Monitoring and Retraining

AutoML does not eliminate the need for production ML monitoring:

- Track prediction distributions for data drift
- Monitor model performance metrics against ground truth when available
- Set up automated retraining pipelines that re-run AutoML on fresh data
- Version your AutoML configurations alongside your data and code
- Be aware that retraining with AutoML may produce a different model architecture each time, which can complicate A/B testing and rollback

## Common Pitfalls

1. **Treating AutoML as a black box.** Always inspect the leaderboard, feature importance, and model types selected. Understanding what AutoML chose and why is essential for debugging and improving results.

2. **Skipping data quality work.** AutoML cannot fix fundamentally flawed data. Duplicate records, label errors, data leakage, and selection bias will produce misleading results regardless of the framework.

3. **Ignoring time and resource limits.** Running AutoML without a time budget on a large dataset can consume hours of compute and produce models too large to deploy. Always set explicit constraints.

4. **Overfitting to the validation set.** AutoML frameworks optimize against their internal validation metric. If you also use the same validation set to make framework or feature decisions, you are effectively double-dipping. Hold out a true test set that is never used during AutoML runs.

5. **Deploying the full ensemble without evaluating alternatives.** The best model on the leaderboard may be within 0.1% of the full ensemble while being 10x faster at inference. Check single-model performance before committing to ensemble deployment.

6. **Assuming AutoML handles feature engineering.** The preprocessing built into AutoML tools is basic -- type detection, encoding, missing values. Domain features (ratios, time-based aggregations, business rules) still need to be engineered manually and consistently provide the largest accuracy gains.

7. **Version mismatch in production.** AutoGluon and H2O both require matching library versions between training and inference. Pin your versions and test model loading in your deployment environment before going live.

8. **Not accounting for the training-serving skew.** If your AutoML tool applies preprocessing (encoding, imputation) during training, the same preprocessing must be applied identically during serving. This is automatic if you use the framework's predict function, but can break if you extract the model and serve it independently.

## Practical Tips

1. **Start with a short run.** Use 5-10 minutes to validate your data pipeline and get an initial baseline. Then increase the time budget for a production run.

2. **Give AutoML more data, not more time.** If you have the choice between running AutoML longer on the same data or collecting more/better data, choose the data. Feature engineering and data quality improvements almost always outperform longer search times.

3. **Use the leaderboard strategically.** The leaderboard tells you which model families work well for your data. If LightGBM consistently ranks highest, consider a focused FLAML run with `estimator_list=["lgbm"]` for a production-optimized single model.

4. **Combine frameworks.** There is no rule against using AutoGluon for exploration and FLAML for the production model. Use AutoGluon's `best_quality` preset to understand what is possible, then use FLAML to produce a lightweight, deployable model.

5. **Profile your data first.** Check for class imbalance, high-cardinality categoricals, missing value patterns, and feature correlations before running AutoML. Some issues (severe imbalance, data leakage) will produce misleading AutoML results.

6. **Set memory limits.** AutoGluon's multi-layer stacking can exhaust memory on large datasets. H2O requires explicit JVM memory allocation. Always configure memory limits appropriate to your hardware.

7. **Log everything.** FLAML produces detailed trial logs. AutoGluon provides leaderboards. H2O has event logs. Use these to understand how your compute budget was spent and whether extending it would help.

8. **Test inference latency early.** Do not wait until deployment to discover that your AutoML ensemble is too slow. Measure prediction time per row immediately after training and factor this into your framework choice.

9. **Consider the full cost.** Compare frameworks not just on accuracy but on total cost: training compute, inference compute, engineering time for deployment, and ongoing maintenance. A model that is 0.5% less accurate but deploys in an hour instead of a week may be the better business decision.

10. **Keep it simple when possible.** If a single well-tuned LightGBM model meets your accuracy requirements, there is no need for a multi-layer stacking ensemble. AutoML is a tool, not an obligation.

## Chapter Index

- [AutoGluon](./autogluon/ReadMe.md) - Ensemble-first AutoML from AWS
- [H2O AutoML](./h2o/ReadMe.md) - Enterprise-grade distributed AutoML
- [FLAML](./flaml/ReadMe.md) - Lightweight cost-efficient AutoML from Microsoft

## See Also

- [Feature Engineering](../feature-engineering/ReadMe.md) - Manual and automated feature engineering techniques
- [Gradient Boosting](../gradient-boosting/ReadMe.md) - The dominant algorithm family in tabular ML
- [Tabular Data Modeling](../ReadMe.md) - Parent topic
