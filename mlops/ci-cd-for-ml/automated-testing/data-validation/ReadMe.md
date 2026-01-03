# Data Validation

## Summary

Data validation in ML pipelines ensures that incoming data meets expected quality standards before being used for training or inference. Unlike traditional software testing where inputs are often well-defined, ML systems must handle statistical distributions, detect subtle data drift, and validate schema compliance across potentially billions of records.

Key points to remember:

- Data validation should occur at pipeline boundaries: ingestion, feature engineering, and model input
- Schema validation catches structural issues; statistical validation catches distribution shifts
- Great Expectations and Pandera are the leading Python libraries for declarative data validation
- Validation should fail fast in development but often needs graceful degradation in production
- Testing data quality is distinct from testing data pipelines; both are necessary

## Core Validation Types

### Schema Validation

Schema validation ensures data conforms to expected types, column names, and structural constraints. This is the most basic form of data validation and should always be implemented first.

Common schema checks include:
- Column presence and naming
- Data types (integer, float, string, datetime)
- Nullability constraints
- Uniqueness constraints
- Value ranges and enumerations

Schema validation catches obvious errors like missing columns, type mismatches, or malformed records. These issues typically indicate upstream data source changes or pipeline bugs rather than natural data evolution.

### Statistical Validation

Statistical validation compares incoming data distributions against reference distributions or historical baselines. This type of validation detects data drift, which can degrade model performance even when schema validation passes.

Statistical checks include:
- Distribution comparisons using KS tests, chi-squared tests, or PSI (Population Stability Index)
- Summary statistics: mean, median, standard deviation, percentiles
- Cardinality checks for categorical variables
- Correlation stability between features
- Missing value rates

Statistical validation requires maintaining reference datasets or summary statistics from training data. The challenge lies in setting appropriate thresholds: too strict causes false alarms, too lenient misses real drift.

### Semantic Validation

Semantic validation checks business logic and domain-specific constraints that go beyond schema and statistics. Examples include:
- Transaction amounts should be positive
- End dates must be after start dates
- User ages should fall within reasonable bounds
- Product categories must match known taxonomy

These rules encode domain knowledge and often catch data quality issues that statistical methods miss because they violate business invariants rather than statistical properties.

## Validation Frameworks

### Great Expectations

Great Expectations is the most widely adopted data validation framework in the Python ecosystem. It provides a declarative approach where you define expectations about your data, and the framework validates data against those expectations.

Core concepts:
- Expectations: individual assertions about data (e.g., column values should be unique)
- Expectation Suites: collections of expectations that define a data contract
- Checkpoints: runnable validation jobs that apply suites to data batches
- Data Docs: auto-generated documentation of expectations and validation results

Great Expectations integrates with pandas, Spark, and SQL databases. It supports both interactive expectation authoring and programmatic definition. The framework can automatically profile data to suggest initial expectations, though these suggestions require human review.

A typical workflow involves:
1. Profile sample data to generate candidate expectations
2. Refine expectations based on domain knowledge
3. Store expectation suites in version control
4. Run checkpoints in CI/CD and production pipelines
5. Monitor validation results and adjust thresholds as needed

### Pandera

Pandera focuses on pandas and integrates more directly with the Python type system. It uses a schema definition approach that feels natural to developers familiar with dataclasses or Pydantic.

Pandera advantages:
- Lighter weight than Great Expectations
- Better integration with type hints and static analysis
- Schemas can be used for both validation and documentation
- Supports hypothesis-based testing via integration with Hypothesis library

Pandera is well-suited for projects already using pandas where you want validation without the operational overhead of Great Expectations. For Spark workloads or complex multi-datasource scenarios, Great Expectations typically scales better.

### TensorFlow Data Validation (TFDV)

TFDV is part of TFX (TensorFlow Extended) and provides schema inference, statistics generation, and anomaly detection specifically designed for ML pipelines. It excels at:
- Automatic schema inference from training data
- Comparing training and serving data distributions
- Detecting schema skew and data drift
- Integration with other TFX components

TFDV uses Protocol Buffers for schema definition, which may feel less Pythonic but enables strong cross-language support. If you are using TFX for your ML pipeline, TFDV is the natural choice for data validation.

## Implementation Patterns

### Validation in Feature Pipelines

Feature pipelines should validate data at three points:
1. Raw data ingestion: validate source data before any transformation
2. Post-transformation: validate feature values after engineering
3. Feature store writes: validate before persisting to feature store

The ingestion validation focuses on schema and completeness. Post-transformation validation checks that feature engineering logic produced valid outputs. Feature store validation ensures the final features meet serving requirements.

For batch pipelines, validation failures should halt processing and alert data engineers. For streaming pipelines, you typically need a strategy for handling invalid records: dead-letter queues, imputation, or graceful degradation.

### Validation in Training Pipelines

Training pipelines should validate:
- Training data before model fitting
- Validation/test splits for data leakage
- Feature distributions match expectations
- Label distributions are not unexpectedly skewed

Training validation failures should block model training. An invalid training run wastes compute and produces unreliable models. Catching data issues before a multi-hour training job saves significant resources.

### Validation in Serving Pipelines

Serving-time validation balances thoroughness with latency. You cannot run expensive statistical tests on every inference request. Typical approaches:

- Schema validation on every request (fast, essential)
- Sampling-based statistical validation (background process)
- Aggregated drift detection over time windows

When serving validation detects issues, the system needs a fallback strategy. Options include:
- Reject the request with an error
- Return a default prediction with low confidence
- Route to a simpler fallback model
- Log and continue, alerting for later investigation

The right choice depends on the application. A fraud detection system might prefer false positives from strict validation, while a recommendation system might prefer graceful degradation.

## Testing Data Validation

Data validation code itself needs testing. Common testing patterns:

Unit tests for individual expectations:
- Test that valid data passes
- Test that specific invalid data fails
- Test edge cases and boundary conditions

Integration tests for validation pipelines:
- Test full validation suites against known-good data
- Test that expected failures are caught
- Test performance under realistic data volumes

Synthetic data generation for testing:
- Generate data that should pass all validations
- Generate data with specific defects to test detection
- Use property-based testing (Hypothesis) for thorough coverage

## Monitoring and Alerting

Production data validation requires monitoring infrastructure:

Metrics to track:
- Validation pass/fail rates over time
- Which specific expectations fail most often
- Latency of validation checks
- Volume of data validated

Alerting considerations:
- Alert on validation failure rate exceeding threshold
- Alert on new expectation failures (not seen before)
- Distinguish data issues from validation bugs
- Avoid alert fatigue with appropriate thresholds

Data validation metrics should feed into broader ML monitoring dashboards alongside model performance metrics. Correlation between validation failures and model degradation helps justify investment in data quality.

## Common Pitfalls

### Overfitting Expectations to Training Data

If you generate expectations purely from training data statistics, your validation may be too strict. Production data often has legitimate variation that training data did not capture. Start with loose bounds and tighten based on observed issues rather than starting strict and loosening.

### Ignoring Temporal Patterns

Data often has time-based patterns: daily cycles, weekly seasonality, holiday effects. Validating Friday data against Monday baselines may produce false drift alerts. Build time-aware validation that compares like periods.

### Validating Too Late

Validation at the end of a pipeline means you have already wasted compute on invalid data. Validate as early as possible, ideally at ingestion, to fail fast.

### Missing the Forest for the Trees

Individual column validations might all pass while the overall dataset is problematic. Include holistic checks like row counts, cross-column relationships, and join cardinalities. A table with the right schema but wrong number of rows indicates a serious issue.
