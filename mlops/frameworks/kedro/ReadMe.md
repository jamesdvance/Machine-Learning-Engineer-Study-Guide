# Kedro

## Summary

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. Developed by McKinsey's QuantumBlack, it applies software engineering best practices to data science projects through a structured project template, data catalog abstraction, and pipeline definition system.

Key points to remember:

- Kedro enforces project structure and coding standards through templates
- The Data Catalog abstracts data loading and saving, separating logic from I/O
- Pipelines are defined as pure Python functions connected by data dependencies
- Strong emphasis on software engineering practices: testing, documentation, modularity
- Best suited for data-heavy ML projects where data engineering is significant

## Core Concepts

### Project Structure

Kedro enforces a standardized project layout:

```
my_project/
    conf/
        base/
            catalog.yml        # Data catalog
            parameters.yml     # Configuration
        local/
            credentials.yml    # Secrets (not committed)
    data/
        01_raw/
        02_intermediate/
        03_primary/
        04_feature/
        05_model_input/
        06_models/
        07_model_output/
        08_reporting/
    src/
        my_project/
            pipelines/
                data_engineering/
                    nodes.py
                    pipeline.py
                data_science/
                    nodes.py
                    pipeline.py
            pipeline_registry.py
    tests/
```

This structure promotes:
- Separation of code, configuration, and data
- Environment-specific configuration
- Logical data organization

### Data Catalog

The Data Catalog abstracts data I/O:

```yaml
# conf/base/catalog.yml
raw_customers:
  type: pandas.CSVDataSet
  filepath: data/01_raw/customers.csv

processed_customers:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/customers.parquet

model:
  type: pickle.PickleDataSet
  filepath: data/06_models/model.pkl

features:
  type: pandas.ParquetDataSet
  filepath: s3://bucket/features.parquet
  credentials: aws_credentials
```

Nodes reference datasets by name, not path:

```python
def preprocess_customers(raw_customers: pd.DataFrame) -> pd.DataFrame:
    # raw_customers is loaded automatically
    return processed_df

# The function doesn't know or care about file paths
```

Catalog benefits:
- Decouples logic from storage details
- Easy to switch between local files and cloud storage
- Centralizes data configuration
- Enables versioning and caching

### Nodes

Nodes are pure Python functions that transform data:

```python
# pipelines/data_science/nodes.py

def split_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets."""
    X = data.drop(columns=parameters["target_column"])
    y = data[parameters["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> sklearn.base.BaseEstimator:
    """Train a model."""
    model = RandomForestClassifier(**parameters["model_params"])
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: sklearn.base.BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }
```

Nodes are:
- Pure functions with no side effects
- Framework-agnostic Python code
- Easy to test in isolation

### Pipelines

Pipelines connect nodes through data dependencies:

```python
# pipelines/data_science/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_data,
            inputs=["processed_features", "parameters"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train", "parameters"],
            outputs="model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["model", "X_test", "y_test"],
            outputs="metrics",
            name="evaluate_model_node"
        )
    ])
```

Kedro infers execution order from data dependencies.

### Parameters

Configuration through YAML:

```yaml
# conf/base/parameters.yml
data_science:
  target_column: "is_fraud"
  test_size: 0.2
  random_state: 42
  model_params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
```

Access in nodes:

```python
def train_model(X_train, y_train, parameters):
    model_params = parameters["data_science"]["model_params"]
    model = RandomForestClassifier(**model_params)
```

## Running Pipelines

### CLI Execution

```bash
# Run full pipeline
kedro run

# Run specific pipeline
kedro run --pipeline data_science

# Run specific node
kedro run --node train_model_node

# Run from specific node to end
kedro run --from-nodes split_data_node

# Run with different configuration
kedro run --env production
```

### Programmatic Execution

```python
from kedro.runner import SequentialRunner
from kedro.framework.session import KedroSession

with KedroSession.create() as session:
    context = session.load_context()
    runner = SequentialRunner()
    runner.run(context.pipeline)
```

### Parallel Execution

```python
from kedro.runner import ParallelRunner

with KedroSession.create() as session:
    context = session.load_context()
    runner = ParallelRunner()
    runner.run(context.pipeline)
```

## Data Versioning

Kedro supports dataset versioning:

```yaml
# conf/base/catalog.yml
trained_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/model.pkl
  versioned: true
```

Versioned datasets:
- Automatically save with timestamps
- Load latest version by default
- Can load specific versions

```python
# Load specific version
from kedro.io import DataCatalog

catalog.load("trained_model", version="2024-01-15T10.30.00.000Z")
```

## Testing

### Unit Testing Nodes

```python
# tests/pipelines/data_science/test_nodes.py

import pytest
import pandas as pd
from my_project.pipelines.data_science.nodes import train_model

def test_train_model():
    # Arrange
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    parameters = {"model_params": {"n_estimators": 10}}

    # Act
    model = train_model(X_train, y_train, parameters)

    # Assert
    assert hasattr(model, "predict")
    assert model.n_estimators == 10
```

### Integration Testing

```python
def test_data_science_pipeline():
    with KedroSession.create() as session:
        context = session.load_context()
        runner = SequentialRunner()
        result = runner.run(
            context.pipelines["data_science"],
            context.catalog
        )
        assert "metrics" in result
        assert result["metrics"]["accuracy"] > 0.7
```

## Visualization

Kedro-Viz provides interactive pipeline visualization:

```bash
pip install kedro-viz
kedro viz
```

Features:
- Interactive DAG visualization
- Data preview
- Node execution status
- Experiment tracking integration

## Integration with Orchestrators

Kedro pipelines can run on external orchestrators:

### Airflow

```bash
pip install kedro-airflow
kedro airflow create
```

Generates Airflow DAG from Kedro pipeline.

### Kubeflow

```bash
pip install kedro-kubeflow
kedro kubeflow compile
```

### Argo Workflows

```bash
pip install kedro-argo
kedro argo generate
```

## Best Practices

### Node Design

Write focused, testable nodes:

```python
# Good: single responsibility
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    ...

def train_model(features: pd.DataFrame, params: Dict) -> Model:
    ...

# Bad: doing too much
def do_everything(data: pd.DataFrame) -> Model:
    features = calculate_features(data)
    model = train_model(features)
    metrics = evaluate(model)
    save_report(metrics)
    return model
```

### Catalog Organization

Use the data layer convention:

```
01_raw/       # Untouched source data
02_intermediate/  # Cleaned data
03_primary/   # Domain-specific tables
04_feature/   # ML features
05_model_input/   # Final training data
06_models/    # Trained models
07_model_output/  # Predictions
08_reporting/ # Reports and dashboards
```

### Configuration Management

Use environment-specific configuration:

```
conf/
    base/          # Shared configuration
        catalog.yml
        parameters.yml
    local/         # Local development
        catalog.yml  # Override with local paths
    production/    # Production
        catalog.yml  # S3 paths, production settings
```

### Modular Pipelines

Organize complex projects into modular pipelines:

```python
# pipeline_registry.py
def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering = de_pipeline.create_pipeline()
    data_science = ds_pipeline.create_pipeline()
    deployment = deploy_pipeline.create_pipeline()

    return {
        "de": data_engineering,
        "ds": data_science,
        "deploy": deployment,
        "__default__": data_engineering + data_science
    }
```

## Comparison with Alternatives

### vs ZenML

Kedro advantages:
- Stronger data catalog abstraction
- Better for data engineering
- More opinionated structure

ZenML advantages:
- Stack-based infrastructure abstraction
- Cloud-native orchestration
- Model deployment focus

### vs Metaflow

Kedro advantages:
- Data catalog concept
- Stricter project structure

Metaflow advantages:
- Simpler scaling to cloud
- Better data versioning
- More Pythonic feel

### vs Plain Python

Kedro provides:
- Enforced project structure
- Data I/O abstraction
- Pipeline visualization
- Configuration management
- Testing patterns

Worth the overhead for larger projects; overkill for quick experiments.

## Common Pitfalls

### Over-Engineering Nodes

Not every function needs to be a node. Use nodes for significant transformations, not every small operation.

### Ignoring the Catalog

Hardcoding file paths defeats the purpose:

```python
# Bad
def load_data():
    return pd.read_csv("data/raw/data.csv")

# Good: let catalog handle it
def process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    return transform(raw_data)
```

### Tight Coupling

Nodes should not depend on specific file formats or storage locations. The catalog handles that.

### Not Testing

Pure function nodes are easy to test. Take advantage of this design for comprehensive testing.
