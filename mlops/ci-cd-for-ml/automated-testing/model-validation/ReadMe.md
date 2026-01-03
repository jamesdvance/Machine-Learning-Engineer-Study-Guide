# Model Validation

## Summary

Model validation in CI/CD pipelines ensures that trained models meet quality standards before deployment. This goes beyond offline evaluation metrics to include behavioral testing, bias detection, performance regression checks, and validation against production requirements. Model validation serves as a quality gate between training and serving.

Key points to remember:

- Model validation should test behavior, not just aggregate metrics
- Compare new models against production baselines to detect regressions
- Validate on data slices to catch hidden bias and performance disparities
- Include latency and resource usage validation, not just accuracy
- Automate validation in CI/CD but maintain human review for significant changes

## Validation vs Evaluation

Model evaluation measures how well a model performs on held-out test data, producing metrics like accuracy, F1, or AUC. Model validation determines whether those metrics and the model's behavior meet deployment criteria.

A model with 95% accuracy might still fail validation because:
- The previous production model had 96% accuracy
- Performance on a critical customer segment dropped
- Inference latency exceeds SLA requirements
- The model exhibits new bias patterns
- Predictions on known critical examples are wrong

Validation asks: "Is this model good enough to deploy?" Evaluation just asks: "How good is this model?"

## Types of Model Validation

### Performance Regression Testing

Compare the candidate model against the current production model on standardized test sets:

- Overall metric comparison (accuracy, F1, precision, recall)
- Per-class or per-label comparisons
- Metric confidence intervals to assess statistical significance
- Win/loss analysis on individual examples

Set thresholds for acceptable regression. A new model might be acceptable if it regresses by less than 0.5% on overall accuracy while improving on a target segment. Define these tradeoffs explicitly in validation criteria.

### Slice-Based Validation

Aggregate metrics hide disparate performance across subgroups. Slice-based validation computes metrics on data segments:

- Demographic groups (age, gender, geography)
- Business segments (customer tier, product category)
- Data characteristics (input length, image resolution)
- Temporal segments (weekday vs weekend, seasonal)

Define minimum performance thresholds per slice. A model that performs well overall but poorly on a protected group should fail validation. Tools like SliceFinder can automatically discover problematic slices.

### Behavioral Testing

Behavioral tests check specific model behaviors beyond aggregate metrics:

Invariance tests: Model predictions should not change when inputs are perturbed in semantically meaningless ways. A sentiment classifier should give the same prediction for "Great product!" and "Great product."

Directional tests: Model predictions should change appropriately when inputs are modified in semantically meaningful ways. Adding negative words to a review should decrease sentiment scores.

Minimum functionality tests: Model should handle known critical cases correctly. A content moderation model must correctly classify specific harmful content examples.

Capability tests: Model should demonstrate required capabilities. A language model should handle multiple languages if that is a requirement.

The CheckList framework provides a systematic approach to behavioral testing for NLP models.

### Bias and Fairness Validation

Validate models against fairness criteria before deployment:

- Demographic parity: Prediction rates equal across groups
- Equalized odds: True positive and false positive rates equal across groups
- Calibration: Predicted probabilities accurate across groups
- Individual fairness: Similar individuals receive similar predictions

Choose fairness metrics appropriate to your application and legal requirements. Different metrics can conflict; explicitly document which you prioritize and why.

Fairness validation should use curated test sets that include adequate representation of protected groups. Production data may not have sufficient representation for reliable bias measurement.

### Robustness Validation

Test model behavior on perturbed or adversarial inputs:

- Typos and misspellings for text models
- Noise and compression artifacts for image models
- Feature corruption and missing values for tabular models
- Out-of-distribution inputs

Robustness testing reveals whether the model will fail gracefully or catastrophically when facing imperfect real-world data. Set thresholds for acceptable performance degradation under perturbation.

### Consistency Validation

Validate that model behavior is consistent with business rules and common sense:

- Monotonicity: Increasing credit score should not decrease loan approval probability
- Stability: Small input changes should produce small output changes
- Transitivity: If A > B and B > C, then A > C for ranking models
- Range constraints: Predictions should fall within valid ranges

Consistency validation catches models that achieve good metrics through unexpected or undesirable learned patterns.

## Performance Validation

### Latency Validation

Measure inference latency under realistic conditions:

- P50, P95, P99 latency
- Batch inference throughput
- Cold start time
- Latency under concurrent load

Set latency budgets based on SLA requirements. A model that meets accuracy requirements but exceeds latency limits should fail validation.

### Resource Usage Validation

Validate computational requirements:

- Memory footprint
- GPU memory usage
- CPU utilization
- Model file size

Resource validation ensures the model can run on target infrastructure. A model that requires more GPU memory than available will fail in production even if it passes all accuracy tests.

### Throughput Validation

Validate that the model can handle expected request volumes:

- Maximum requests per second
- Sustained throughput over extended periods
- Throughput under realistic traffic patterns

Throughput validation helps size serving infrastructure and identify scaling requirements.

## Validation Implementation

### Validation Pipelines

Structure validation as a pipeline of checks:

1. Load candidate model and baseline model
2. Load validation datasets (overall and per-slice)
3. Run inference on validation sets
4. Compute metrics for all validation types
5. Compare against thresholds and baseline
6. Generate validation report
7. Pass or fail based on criteria

Each step should be independently runnable for debugging. Failed validations should produce detailed reports explaining which checks failed and why.

### Validation Datasets

Maintain curated validation datasets separate from training and evaluation data:

- Golden test sets with known correct labels
- Slice-specific test sets for subgroup validation
- Behavioral test suites with expected outcomes
- Adversarial examples for robustness testing

Version validation datasets and update them as business requirements evolve. Document why specific examples are included.

### Threshold Management

Validation thresholds require ongoing calibration:

- Start with thresholds based on current production performance
- Tighten thresholds as the system matures
- Adjust thresholds when business requirements change
- Allow temporary threshold exceptions with explicit justification

Store thresholds in configuration files under version control. Changing thresholds should require the same review as code changes.

### Human Review Integration

Automated validation catches many issues but cannot replace human judgment for significant model changes. Integrate human review:

- Automatic approval for minor improvements that pass all checks
- Human review required when metrics change significantly
- Human review required for new model architectures or features
- Escalation path for borderline cases

Define clear criteria for what constitutes a significant change requiring review.

## Validation in CI/CD

### Pull Request Validation

Run validation on every model-related pull request:

- Fast validation checks on PR (subset of full validation)
- Full validation on merge to main branch
- Block merges when validation fails

Fast PR validation might include behavioral tests and basic metric checks. Full validation includes comprehensive slice analysis and performance testing.

### Continuous Validation

Run validation periodically on production models:

- Detect performance degradation over time
- Catch data drift effects
- Validate against updated test sets

Continuous validation catches issues that emerge gradually and might not appear in one-time deployment validation.

### Staging Validation

Before production deployment:

- Run full validation suite in staging environment
- Test with production-like traffic patterns
- Validate integration with production dependencies

Staging validation catches environment-specific issues that may not appear in CI/CD pipeline tests.

## Validation Reports

Generate detailed validation reports:

- Summary pass/fail status with overall recommendation
- Metric comparisons with confidence intervals
- Per-slice breakdowns
- Failed behavioral tests with examples
- Performance metrics with comparisons to baseline
- Identified risks and mitigation suggestions

Store validation reports for audit purposes. Reports should be machine-readable for automation and human-readable for review.

## Common Pitfalls

### Validating Only on Aggregate Metrics

High overall accuracy can mask poor performance on critical segments. Always include slice-based validation.

### Ignoring Metric Variance

A model might appear to regress when the difference is within noise. Use statistical tests or confidence intervals to assess whether differences are significant.

### Static Validation Thresholds

Thresholds appropriate at launch may be too lenient as the system matures or too strict as requirements change. Review and update thresholds regularly.

### Insufficient Baseline Comparisons

Validating only against absolute thresholds misses regressions. Always compare against the current production model.

### Skipping Performance Validation

A model that is accurate but slow or resource-hungry will fail in production. Include latency and resource usage in validation criteria.

### Outdated Validation Data

Validation datasets can become stale as production data evolves. Periodically refresh validation sets with recent examples.
