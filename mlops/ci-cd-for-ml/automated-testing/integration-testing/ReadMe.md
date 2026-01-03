# Integration Testing for ML Systems

## Summary

Integration testing for ML systems validates that individual components work correctly together as a complete pipeline. Unlike unit tests that isolate components, integration tests verify data flows, model loading, preprocessing-inference alignment, and end-to-end behavior under realistic conditions.

Key points to remember:

- Integration tests catch issues that unit tests miss: serialization bugs, environment mismatches, and component interface drift
- Test the full inference path from raw input to final output, not just the model forward pass
- Use representative data samples, not synthetic data, for integration tests
- Test both happy paths and failure modes: malformed inputs, missing features, model load failures
- Integration tests should run in environments that closely mirror production

## Why Integration Testing Matters for ML

Traditional software integration testing focuses on API contracts and data flow between services. ML systems have additional integration concerns:

Model-preprocessing coupling: If preprocessing during training differs from preprocessing during serving, predictions will be wrong even if both components work correctly in isolation. This train-serve skew is a common production failure mode.

Feature engineering consistency: Features computed during training must be computed identically during inference. Integration tests verify that feature pipelines produce consistent outputs.

Serialization and deserialization: Models serialized with one library version may fail to load with another. Integration tests catch version incompatibilities before deployment.

Environment dependencies: ML models often depend on specific CUDA versions, library versions, and system configurations. Integration tests in production-like environments catch these issues.

## Integration Test Categories

### Preprocessing-Model Integration

These tests verify that the preprocessing pipeline produces outputs the model can consume correctly. Common checks:

- Output shapes match model input expectations
- Data types are correct (float32 vs float64 matters for some frameworks)
- Normalization parameters match those used during training
- Categorical encoding produces expected vocabulary indices
- Missing value handling aligns with training expectations

A minimal test loads the preprocessing pipeline and model, runs a known input through both, and verifies the output matches expected values. This catches interface drift when either component changes.

### Feature Store Integration

If using a feature store, integration tests should verify:

- Feature retrieval returns expected features for known entities
- Feature freshness meets SLA requirements
- Online and offline feature values are consistent
- Feature schemas match model expectations
- Fallback behavior works when features are missing

Feature store integration tests often require test fixtures: known entities with predetermined feature values that persist across test runs.

### Model Loading and Initialization

Test that models load correctly in the target serving environment:

- Model artifacts load without errors
- Model weights are correct (compare predictions against known baseline)
- Warm-up inference runs without errors
- Memory usage is within expected bounds
- GPU memory is allocated correctly if applicable

Model loading tests should run in Docker containers or environments matching production configuration. A model that loads fine locally but fails in production due to library version differences is a common issue.

### End-to-End Inference

End-to-end tests exercise the complete inference path:

- Accept raw input in production format
- Run full preprocessing pipeline
- Execute model inference
- Apply postprocessing
- Return final predictions in expected format

These tests should use representative data samples from production or curated test datasets. Synthetic data often misses edge cases that real data contains.

End-to-end tests should verify:
- Predictions are numerically correct for known inputs
- Latency is within acceptable bounds
- Memory usage does not grow unbounded
- Error handling works for malformed inputs

### Multi-Model Pipeline Integration

Many ML systems chain multiple models: a classifier followed by specialized models, or an ensemble combining multiple predictions. Integration tests for these pipelines verify:

- Model outputs correctly feed into downstream models
- Confidence thresholds and routing logic work correctly
- Fallback paths function when upstream models fail
- Aggregation logic produces expected final outputs

### External Service Integration

ML systems often integrate with external services:

- Feature stores (Feast, Tecton)
- Model registries (MLflow, Weights and Biases)
- Monitoring services (Datadog, Prometheus)
- Logging infrastructure

Integration tests should verify these connections work, though they may use mocks or test instances rather than production services. The goal is to catch configuration issues and client library incompatibilities.

## Test Data Management

### Representative Test Data

Integration tests need data that exercises realistic code paths. Strategies for obtaining test data:

Sampled production data: Extract anonymized samples from production. This captures real-world patterns but requires careful handling of PII and sensitive data.

Curated test sets: Manually construct examples covering important cases: common inputs, edge cases, known failure modes, and boundary conditions.

Recorded request-response pairs: Log production inputs and outputs, then use these as golden test cases. This approach is powerful but requires versioning as models change.

### Test Data Versioning

Test data should be versioned alongside code. When models or preprocessing change, test expectations may need updating. Git LFS, DVC, or cloud storage with version tags can manage test data versioning.

Store test data expectations (expected outputs) separately from test inputs. This allows updating expectations when intentional model changes occur while keeping the input dataset stable.

### Avoiding Data Leakage in Tests

Integration test data should not overlap with training data. Using training examples as test cases gives false confidence: the model may have memorized those specific examples. Maintain strict separation between training, validation, and integration test datasets.

## Test Environment Strategy

### Local Development

Local integration tests should run quickly and not require external services. Use:

- Mocked external services
- SQLite instead of production databases
- Local model artifacts
- Subset of test data

Local tests catch obvious integration issues during development but cannot fully validate production behavior.

### CI/CD Pipeline

CI integration tests run on every commit or pull request. They should:

- Use production-like container environments
- Test with full test datasets
- Connect to test instances of external services (not mocks)
- Run full preprocessing and inference pipelines

CI tests are slower than local tests but more realistic. A 10-15 minute CI pipeline is reasonable for integration tests.

### Staging Environment

Staging integration tests run in an environment mirroring production:

- Same container images and configurations
- Same infrastructure (or close approximation)
- Production-like data volumes
- Real external service connections

Staging tests validate that deployment artifacts work in realistic conditions. They should run before production deployments.

## Implementing Integration Tests

### Test Fixtures

ML integration tests require fixtures that may be expensive to create:

- Trained models
- Preprocessing pipelines
- Feature store data
- Database state

Use fixture caching and sharing across tests where possible. pytest fixtures with session scope can load models once for all tests.

### Assertions

Integration test assertions should check:

- Numerical correctness within tolerance (use np.allclose or similar)
- Output shapes and types
- Prediction confidence ranges
- Processing time bounds
- Memory usage bounds

Avoid overly precise numerical assertions. Model outputs may vary slightly across hardware or library versions. Test that outputs are correct within acceptable tolerance.

### Test Organization

Organize integration tests by pipeline stage:

```
tests/
  integration/
    test_preprocessing.py
    test_feature_store.py
    test_model_loading.py
    test_inference.py
    test_end_to_end.py
```

This organization makes it easy to run subsets of integration tests and identify which component caused failures.

### Handling Flaky Tests

Integration tests can be flaky due to:

- Network issues with external services
- Resource contention in shared environments
- Non-deterministic model behavior
- Timing-dependent code

Strategies for reducing flakiness:

- Retry failed tests with backoff
- Use deterministic random seeds where possible
- Increase timeouts for slow operations
- Isolate tests from shared resources

Document known flaky tests and their causes. Unexplained flaky tests erode trust in the test suite.

## Performance Integration Testing

Beyond correctness, integration tests should verify performance characteristics:

### Latency Testing

Measure inference latency under realistic conditions:

- Cold start latency (first request after deployment)
- Warm latency (subsequent requests)
- Latency percentiles (p50, p95, p99)
- Latency under concurrent load

Set latency budgets and fail tests that exceed them. Catching latency regressions before production prevents user-facing issues.

### Throughput Testing

Test maximum throughput the system can handle:

- Requests per second at target latency
- Batch processing rate for batch inference
- Scaling behavior under increasing load

Throughput tests help size infrastructure and set autoscaling parameters.

### Memory Testing

Verify memory usage stays within bounds:

- Peak memory during inference
- Memory growth over many requests (check for leaks)
- GPU memory usage if applicable

Memory leaks may not appear in quick unit tests but surface during longer integration test runs.

## Common Pitfalls

### Testing Against Production Services

Integration tests should not run against production services. Use dedicated test environments or mocks. Production testing risks data corruption, quota exhaustion, and interference with real users.

### Ignoring Preprocessing

Many integration test suites test models in isolation, passing preprocessed tensors directly. This misses the critical preprocessing-model integration. Always test the full pipeline.

### Static Test Data

Test data that never changes becomes stale. Periodically refresh test data with recent production samples to catch drift-related issues.

### Insufficient Error Path Testing

Happy path tests are not enough. Test error handling:

- Malformed inputs
- Missing features
- Service unavailability
- Timeout behavior
- Partial failures in multi-model pipelines

Production systems encounter errors. Integration tests should verify errors are handled gracefully.
