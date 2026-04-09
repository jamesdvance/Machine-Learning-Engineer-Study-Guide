# Data Drift Detection

## Summary

Data drift refers to the change in the statistical distribution of input data over time relative to the data a model was trained on. Because ML models learn patterns from training data distributions, shifts in those distributions degrade model predictions even when the model itself has not changed. Data drift detection is the practice of continuously monitoring input features in production and raising alerts when the incoming data no longer resembles the training or reference data.

Data drift is one of the most common causes of silent model failure. Unlike a crashed service, a drifting model continues to return predictions, but those predictions become progressively less reliable. Detecting drift early allows teams to investigate root causes, retrain models, or apply corrective measures before business impact accumulates.

Key points to remember:

- Data drift is a change in the input feature distribution P(X), not in the relationship between inputs and outputs
- Covariate shift, prior probability shift, and concept drift are distinct types of distributional change; data drift detection primarily targets covariate shift
- Statistical tests such as the Kolmogorov-Smirnov test, Population Stability Index, and Jensen-Shannon divergence are the workhorses of drift detection
- Different data types require different detection strategies: numerical features use distribution-based tests, categorical features use frequency-based tests, and unstructured data requires embedding-based approaches
- Windowing strategy determines how reference and current data are compared and directly affects sensitivity and false alarm rate
- Feature-level monitoring catches granular shifts; model-level monitoring catches aggregate changes that affect predictions
- Drift detection is necessary but not sufficient on its own; it should be paired with concept drift monitoring and performance degradation tracking for complete model observability

## Why Data Drift Matters

A model trained on historical data encodes assumptions about feature distributions. When a fraud detection model learns that transactions above a certain amount are rare, it relies on that distributional fact. If the business expands to a new market where large transactions are common, the feature distribution shifts and the model's learned thresholds no longer apply.

Data drift can originate from many sources:

- Upstream pipeline changes: a data engineering team renames a column, changes a join order, or modifies a transformation
- Population shifts: the user base changes demographics, geography, or behavior
- Seasonality: holiday periods, fiscal quarters, or weather patterns alter feature distributions
- External events: regulatory changes, economic shifts, or global events disrupt established patterns
- Instrumentation changes: logging format updates, sensor calibration drift, or schema migrations

Without drift detection, teams typically discover degradation only through downstream business metrics, which may lag by days or weeks. Drift detection provides an early warning system that sits between data ingestion and model performance monitoring.

## Types of Distributional Shift

Understanding the taxonomy of distributional change clarifies what data drift detection covers and where its boundaries lie.

### Covariate Shift

Covariate shift occurs when the distribution of input features P(X) changes but the conditional distribution P(Y|X) remains the same. The relationship between features and target is preserved, but the model encounters inputs in regions of the feature space it rarely saw during training.

For example, a model trained on loan applications from urban areas deployed to rural areas may see different income distributions. The relationship between income and default risk might be the same, but the model is now operating in a less-confident region of the input space.

Data drift detection directly targets covariate shift. Most statistical tests compare the marginal distributions of individual features or the joint distribution of feature vectors.

### Prior Probability Shift

Prior probability shift occurs when P(Y) changes. The prevalence of the target class changes, but the feature distributions conditioned on the target remain stable. This is common in fraud detection where fraud rates fluctuate, or in medical diagnosis where disease prevalence varies by season.

Prior probability shift can sometimes be detected indirectly through feature drift when the features correlate with the target, but it is more precisely detected through label monitoring when labels are available.

### Concept Drift

Concept drift occurs when P(Y|X) changes: the same inputs should now produce different outputs. This is a fundamentally different problem from data drift and is covered in detail in the sibling chapter on [Concept Drift](../concept-drift/ReadMe.md). Concept drift cannot be detected by examining input distributions alone. It requires monitoring model outputs, predictions, or ground truth labels.

In practice, data drift and concept drift often co-occur. A shift in customer demographics may change both the input distribution and the relationship between features and outcomes. Comprehensive model monitoring should address both.

## Statistical Tests for Drift Detection

### Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (KS) test is a non-parametric test that compares two sample distributions by measuring the maximum distance between their empirical cumulative distribution functions (ECDFs). The KS statistic D is defined as the supremum of the absolute difference between the two ECDFs.

The KS test is widely used for numerical features because it makes no assumptions about the underlying distribution shape. A p-value below a chosen significance level (commonly 0.05) indicates that the two samples likely come from different distributions.

Strengths:

- Non-parametric; no distributional assumptions
- Sensitive to differences in location, scale, and shape
- Well-understood statistical properties
- Easy to implement and interpret

Limitations:

- Only applicable to univariate continuous distributions
- More sensitive to differences near the center of the distribution than at the tails
- With large sample sizes, statistically significant drift may not be practically significant
- Does not capture multivariate dependencies

### Population Stability Index

The Population Stability Index (PSI) is a measure borrowed from credit risk modeling that quantifies how much a distribution has shifted from a reference (expected) distribution. Features are binned, and PSI is calculated as the sum over bins of (Actual% - Expected%) * ln(Actual% / Expected%).

Common interpretation thresholds:

- PSI less than 0.1: no significant shift
- PSI between 0.1 and 0.25: moderate shift, investigate
- PSI greater than 0.25: significant shift, action required

PSI is popular in regulated industries because of its interpretability and established threshold guidelines. It works for both numerical features (after binning) and categorical features (using category frequencies directly).

Strengths:

- Symmetric awareness of both increases and decreases in bin proportions
- Produces a single scalar value that is easy to track over time
- Established threshold conventions in financial services
- Works for both numerical and categorical features

Limitations:

- Sensitive to binning strategy for numerical features
- Bins with zero counts require smoothing to avoid division by zero
- Does not capture distributional shape within bins
- Thresholds are conventions, not statistically derived critical values

### Chi-Squared Test

The chi-squared test compares observed frequencies against expected frequencies across categories. For drift detection, it compares the category distribution in the current window against the reference distribution. It is the natural choice for categorical features.

The test statistic sums (Observed - Expected)^2 / Expected across categories. A large test statistic (low p-value) indicates significant distributional difference.

Strengths:

- Standard test for categorical data
- Well-understood statistical properties
- Straightforward implementation

Limitations:

- Requires sufficient samples per category (typically at least 5 expected counts per bin)
- Sensitive to granularity; many low-frequency categories inflate the statistic
- Not suitable for continuous data without binning

### Jensen-Shannon Divergence

Jensen-Shannon (JS) divergence is a symmetric, bounded measure of the difference between two probability distributions. It is derived from the Kullback-Leibler (KL) divergence but fixes KL's asymmetry and its undefined behavior when one distribution has zero probability where the other does not.

JS divergence is calculated as the average of the KL divergence from each distribution to their mixture: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q).

JS divergence ranges from 0 (identical distributions) to 1 (when using base-2 logarithm). It is always finite and symmetric, making it well-suited for monitoring dashboards where values need to be compared across features.

Strengths:

- Symmetric and bounded
- No undefined values from zero-probability events
- Works for both discrete and continuous distributions (after binning or density estimation)
- Can be used as a distance metric (its square root is a true metric)

Limitations:

- Requires discretization for continuous features
- Sensitive to binning strategy
- No built-in p-value; thresholds must be determined empirically or through permutation testing

### Wasserstein Distance

The Wasserstein distance (also called the Earth Mover's Distance) measures the minimum cost of transforming one distribution into another, where cost is defined as the amount of probability mass times the distance it must be moved. For one-dimensional distributions, it simplifies to the integral of the absolute difference between the two CDFs.

The Wasserstein distance has a natural physical interpretation: it is the amount of "work" needed to reshape one distribution into another. Unlike KL or JS divergence, it accounts for the metric structure of the underlying space, meaning it captures the magnitude of the shift, not just whether a shift exists.

Strengths:

- Accounts for the geometry of the feature space
- Meaningful even when distributions have non-overlapping support
- Captures the magnitude of shift, not just its presence
- Well-suited for ordered or continuous features

Limitations:

- Computationally more expensive than KS or PSI for high-dimensional data
- No standard threshold conventions; must be calibrated per feature
- Sensitive to outliers in the tails

### Choosing the Right Test

The choice of test depends on the data type and operational requirements:

- For numerical features with moderate sample sizes, the KS test provides a solid default with well-understood p-values.
- For categorical features, the chi-squared test is the standard choice.
- For production dashboards where a single interpretable number is needed, PSI offers established conventions.
- For comparing distributions where the magnitude of shift matters, Wasserstein distance captures how far the distribution has moved.
- For general-purpose monitoring with bounded values, JS divergence provides a clean 0-to-1 metric.

In practice, teams often compute multiple metrics simultaneously. A feature might be flagged for investigation based on a KS test p-value while a dashboard displays PSI for executive reporting.

## Detecting Drift Across Data Types

### Numerical Features

Numerical features are the most straightforward case. The statistical tests described above apply directly. For each feature, the reference distribution is established from training data or a stable production period, and the current window of incoming data is compared.

Preprocessing considerations include:

- Normalizing features before comparison to account for scale differences
- Handling missing values consistently between reference and current data
- Deciding whether to compare raw values or transformed values (log-scaled, standardized)
- Monitoring quantile statistics (mean, median, standard deviation, percentiles) as a complement to full distributional tests

### Categorical Features

Categorical features require frequency-based tests. The chi-squared test and PSI (treating categories as bins) are standard. Additional considerations include:

- New categories appearing in production that were absent in training data; these are a strong drift signal
- Category collapse where rare categories become frequent or vice versa
- High-cardinality features (such as ZIP codes or product IDs) require grouping or embedding-based approaches
- Monitoring the proportion of unseen categories over time

### Text Data

Text data cannot be compared directly with univariate statistical tests. Common approaches include:

- Embedding-based monitoring: encode text inputs with a pretrained language model, then monitor drift in the embedding space using multivariate drift detectors or dimensionality-reduced representations
- Vocabulary monitoring: track the distribution of tokens, n-grams, or topics; new vocabulary appearing in production signals distributional change
- Length and structure statistics: monitor text length distributions, punctuation patterns, language distribution, or format changes as proxies for deeper drift
- Classifier-based detection: train a binary classifier to distinguish reference text from current text; high accuracy indicates drift

### Image Data

Image drift detection follows similar principles to text:

- Embedding-based monitoring: extract feature vectors from a pretrained vision model (such as a penultimate-layer representation) and monitor drift in the embedding space
- Low-level statistics: monitor pixel intensity distributions, image dimensions, channel statistics, or file sizes as coarse drift indicators
- Domain-specific features: for medical imaging, monitor tissue region proportions; for satellite imagery, monitor cloud cover percentages
- Autoencoder reconstruction error: train an autoencoder on reference images and monitor reconstruction error on incoming images; increasing error suggests out-of-distribution inputs

### Multivariate Drift

Monitoring features independently can miss drift that only manifests in feature interactions. A fraud model might depend on the relationship between transaction amount and time of day; both marginal distributions could be stable while their joint distribution shifts.

Approaches to multivariate drift detection include:

- Maximum Mean Discrepancy (MMD): a kernel-based test that compares distributions in a reproducing kernel Hilbert space, capturing complex multivariate dependencies
- Classifier-based detection: train a model to distinguish reference data from current data; high discriminative performance indicates drift
- Dimensionality reduction: project features into a lower-dimensional space (PCA, UMAP, t-SNE) and apply univariate tests to the projected components
- Monitoring model prediction distributions as an aggregate signal that implicitly captures multivariate feature interactions

## Windowing Strategies

The choice of how to define the "current" data window and the "reference" data baseline fundamentally affects drift detection sensitivity, latency, and false alarm rate.

### Fixed Reference Window

A fixed reference window uses the training data distribution (or a stable baseline period) as the permanent reference. All incoming data is compared against this fixed baseline.

This approach is simple and deterministic. It answers the question: "Has the data changed since the model was trained?" It is appropriate when the goal is to detect any deviation from training conditions.

The downside is that after a model is retrained on newer data, the reference window must be explicitly updated. Additionally, if the data has a legitimate gradual trend, drift will eventually be flagged even if the model has been adapted to handle it.

### Sliding Window

A sliding window compares recent data against slightly older data. For example, the most recent 7 days might be compared against the prior 30 days. The reference moves forward over time.

This approach detects sudden changes and is less sensitive to gradual trends. It answers the question: "Has something changed recently?" It is well-suited for detecting upstream pipeline breaks or sudden population shifts.

The disadvantage is that gradual drift may go undetected because both the reference and current windows drift together. A model could silently move far from its training distribution without triggering an alert.

### Adaptive Window

Adaptive windowing adjusts the window size based on detected change. The ADWIN (Adaptive Windowing) algorithm is the canonical example: it maintains a growing window and shrinks it when a statistically significant change is detected within the window.

Adaptive windows balance sensitivity and stability. They detect both sudden and gradual changes and automatically adjust to the rate of change in the data.

The trade-off is implementation complexity and the need to tune the algorithm's sensitivity parameters.

### Practical Window Configuration

In production systems, a common pattern combines multiple strategies:

- A fixed reference window compared against daily aggregates catches overall drift from training conditions
- A sliding window with a shorter horizon catches sudden shifts from upstream failures
- Alerts are tiered: the fixed reference comparison generates informational alerts, while the sliding window comparison generates urgent alerts for large changes

Window size should be chosen based on data volume. Each window needs enough samples for statistical tests to be reliable. For the KS test, a few hundred samples per window is generally sufficient; for multivariate tests, sample requirements grow with dimensionality.

## Setting Thresholds and Reducing False Alarms

One of the most challenging practical aspects of drift detection is setting thresholds that catch real drift without overwhelming on-call engineers with false positives.

### Calibrating Thresholds from Historical Data

Rather than using textbook p-value cutoffs, calibrate thresholds from the system's own history:

1. Compute drift statistics on historical data where the model was performing well, using the same windowing strategy that will be used in production
2. Observe the natural variability of drift statistics during normal operation
3. Set thresholds at a level that would have produced a manageable alert rate during this stable period (for example, no more than one alert per week per feature)

This empirical approach accounts for the specific data characteristics, sample sizes, and windowing strategy in use.

### Multiple Testing Correction

When monitoring many features simultaneously, the probability of at least one false alarm increases rapidly. If 100 features are each tested at the 0.05 significance level, approximately 5 features will trigger false alarms at any given check.

Apply corrections for multiple testing:

- Bonferroni correction: divide the significance threshold by the number of tests. Simple but conservative.
- Benjamini-Hochberg procedure: controls the false discovery rate rather than the family-wise error rate. Less conservative and often more practical.
- Aggregate scoring: combine individual feature drift scores into a single aggregate metric, reducing the number of tests to one.

### Tiered Alerting

Not all drift is equally urgent. Implement a tiered system:

- Warning level: drift detected in a small number of features or at moderate magnitude; log and monitor but do not page
- Alert level: drift detected across multiple features or at high magnitude; notify the team for investigation during business hours
- Critical level: massive drift suggesting a pipeline failure or data corruption; page on-call immediately

### Suppressing Noise

Additional strategies for reducing false alarms:

- Require drift to persist across multiple consecutive check intervals before alerting
- Weight features by their importance to the model (using feature importance or SHAP values) so that drift in low-importance features does not trigger alerts
- Exclude features with known seasonal patterns from standard drift checks and monitor them with season-aware baselines
- Use a cool-down period after an alert to prevent repeated notifications for the same ongoing drift event

## Feature-Level vs Model-Level Monitoring

### Feature-Level Monitoring

Feature-level monitoring tests each input feature individually for distributional change. It provides granular visibility into exactly which features are drifting and by how much.

Advantages:

- Directly actionable: when an alert fires, the engineer knows which feature to investigate
- Can distinguish upstream data issues (one feature broken) from population shifts (many features shifting)
- Enables root cause analysis by correlating drifting features with upstream pipelines or data sources

Disadvantages:

- Generates many parallel tests, increasing false alarm risk
- Misses multivariate drift that occurs in feature interactions
- Can overwhelm dashboards when the model has hundreds of features

### Model-Level Monitoring

Model-level monitoring tracks aggregate signals related to model behavior:

- Prediction distribution: monitor the distribution of model scores or predicted classes
- Confidence distribution: track how the model's confidence scores change over time
- Activation monitoring: for neural networks, monitor intermediate layer activations as a proxy for the model's internal representation of inputs

Advantages:

- A single aggregate signal that captures the combined effect of all feature changes
- Implicitly accounts for multivariate interactions
- Fewer tests and simpler alerting

Disadvantages:

- Does not pinpoint which features are responsible for the change
- A shift in predictions could be caused by data drift, concept drift, or a model bug, requiring further investigation to distinguish
- May miss drift in features that do not strongly influence predictions

### Combined Approach

The most effective monitoring strategy combines both levels:

- Model-level monitoring serves as the primary alert: if the prediction distribution shifts significantly, investigate
- Feature-level monitoring serves as the diagnostic tool: once a model-level alert fires, examine individual features to identify the root cause
- Feature importance weighting connects the two levels: focus feature-level monitoring on the features that most influence predictions

## Tools and Libraries

### Evidently AI

Evidently is an open-source Python library that generates drift detection reports and dashboards. It supports statistical tests for numerical and categorical features, calculates PSI, KS, chi-squared, and other metrics, and produces visual reports comparing reference and current data. Evidently integrates with common ML pipelines and can produce JSON-formatted test results for programmatic consumption. It also provides a monitoring UI for continuous tracking.

### NannyML

NannyML focuses on estimating model performance without ground truth labels, but it also provides robust data drift detection. Its drift detection module supports univariate methods (KS, chi-squared, JS divergence, Wasserstein, Hellinger distance) and multivariate methods (PCA reconstruction error, domain classifier). NannyML's distinguishing feature is its ability to correlate detected drift with estimated performance changes.

### Alibi Detect

Alibi Detect is an open-source library specializing in outlier, adversarial, and drift detection. It provides implementations of advanced drift detectors including MMD-based tests, classifier-based drift detection, Learned Kernel MMD, and context-aware drift detection. It supports tabular, text, and image data types. Alibi Detect is particularly strong for multivariate drift detection and unstructured data.

### Great Expectations

Great Expectations is primarily a data validation framework, but its expectation-based approach can be used for drift-adjacent monitoring. By defining expectations on feature distributions (expected ranges, value sets, null rates, distributional parameters), teams can catch schema changes and coarse distributional shifts as part of the data pipeline before data reaches the model. It complements rather than replaces dedicated drift detection tools.

### Whylogs and WhyLabs

Whylogs is an open-source library for data logging and profiling that creates compact statistical profiles of datasets. These profiles can be compared to detect drift without requiring access to raw data, which is valuable for privacy-sensitive applications. WhyLabs is the managed platform built on whylogs, providing dashboards, alerting, and integration with ML pipelines.

## Practical Pipeline: Setting Up Drift Monitoring in Production

### Step 1: Establish a Reference Profile

Before deploying drift monitoring, establish a reference dataset that represents the distribution the model was trained on. This is typically the training set or a held-out validation set. Compute and store summary statistics, histograms, and distributional profiles for every monitored feature.

Store the reference profile as a versioned artifact alongside the model. When the model is retrained, update the reference profile to match the new training data.

### Step 2: Instrument the Prediction Service

Add logging to the prediction service so that every prediction request records the input features. Store these in a feature log table or stream them to a message queue. Include metadata such as timestamps, request IDs, and model version.

Avoid logging predictions synchronously in the hot path if latency is critical. Instead, use asynchronous logging via a queue or a sidecar process.

### Step 3: Aggregate and Compute Drift Metrics

On a scheduled basis (hourly, daily, or per-batch depending on data volume), aggregate logged features into windows and compute drift statistics against the reference profile. Run the appropriate statistical tests for each feature type:

- KS test or Wasserstein distance for numerical features
- Chi-squared test or PSI for categorical features
- Embedding distance metrics for text or image features

Store the drift metrics in a time-series database for trend analysis.

### Step 4: Configure Alerting

Set up alerts based on calibrated thresholds:

- Connect drift metrics to an alerting system (such as PagerDuty, OpsGenie, or a Slack webhook)
- Apply multiple testing correction when monitoring many features
- Implement tiered alerting with different severity levels
- Include a cool-down period to avoid alert fatigue

### Step 5: Build Dashboards

Create dashboards that display:

- Per-feature drift metrics over time, with threshold lines
- Aggregate drift score across all features
- Prediction distribution compared to the reference
- A heatmap showing which features are drifting and when

Dashboards serve the investigation workflow. When an alert fires, the engineer should be able to open a dashboard and immediately see which features are responsible.

### Step 6: Define Response Procedures

Drift detection is only valuable if the team knows how to respond. Define runbooks for common scenarios:

- Single feature drift: investigate the upstream pipeline for that feature; check for schema changes, missing values, or transformation bugs
- Broad feature drift: investigate population changes; check if a new user segment, geography, or product line was introduced
- Prediction distribution drift: check model performance if labels are available; consider retraining
- Persistent drift with stable performance: update the reference profile to reflect the new normal; document the decision

### Step 7: Close the Loop

Integrate drift monitoring into the model lifecycle:

- When drift triggers a retrain, the retraining pipeline should automatically update the reference profile
- Track drift events in the model registry alongside model versions
- Use drift history to inform retraining schedules (retrain more frequently when drift is detected more often)
- Review drift alert history periodically to tune thresholds and reduce false alarm rates

## Connection to Related Topics

Data drift detection is one component of a broader model monitoring strategy. It works in concert with:

- **Concept Drift**: covered in the sibling chapter on [Concept Drift](../concept-drift/ReadMe.md), concept drift addresses changes in P(Y|X). Data drift detection catches input distribution changes; concept drift detection catches changes in the relationship between inputs and outputs. Both are needed because data drift does not always cause performance degradation (if the model is robust in the shifted region) and concept drift can occur without any change in input distributions.

- **Performance Degradation**: covered in the sibling chapter on [Performance Degradation](../performance-degradation/ReadMe.md), performance monitoring tracks actual model accuracy, precision, recall, or business metrics over time. Performance degradation is the ultimate signal that something has gone wrong, but it requires ground truth labels, which may be delayed or unavailable. Data drift detection serves as a leading indicator that does not require labels.

Together, these three monitoring capabilities form a defense-in-depth strategy: data drift detection provides early warning without labels, concept drift detection identifies when the learned relationship breaks down, and performance monitoring confirms actual impact on outcomes.
