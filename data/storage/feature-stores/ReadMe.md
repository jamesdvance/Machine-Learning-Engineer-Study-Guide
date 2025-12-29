# Feature Stores

## Summary

Feature stores are centralized repositories for storing, managing, and serving machine learning features. They solve the critical problem of training-serving skew by providing a single source of truth for feature definitions, ensuring that the same feature values used during training are available during inference. Feature stores handle the complexity of point-in-time correct joins, feature materialization, and low-latency serving.

Key points to remember:

- Dual-store architecture: offline store for training, online store for serving
- Point-in-time correctness prevents data leakage in training datasets
- Materialization syncs features from offline to online stores
- Feature definitions are versioned and documented centrally
- Reduces duplicate feature engineering across teams
- Feast is open-source and self-managed
- Tecton is fully managed with native streaming support
- Hopsworks provides a complete ML platform beyond just features

## The Feature Store Problem

### Training-Serving Skew

Without a feature store, features are often computed differently for training and serving:

```
Training Pipeline:
  Raw Data ’ SQL/Spark ’ Training Features ’ Model

Serving Pipeline:
  Real-time Data ’ Python/Java ’ Serving Features ’ Model
```

Problems:
- Different code paths produce different feature values
- Bugs discovered only after deployment
- Silent degradation of model performance

### Solved by Feature Stores

```
Feature Pipeline:
  Raw Data ’ Feature Engineering ’ Feature Store

Training:
  Feature Store (Offline) ’ Training Data ’ Model

Serving:
  Feature Store (Online) ’ Feature Vector ’ Model
```

Benefits:
- Single source of truth
- Consistent feature values
- Versioned definitions
- Reusable across models

## Core Concepts

### Feature Groups / Feature Views

Logical grouping of related features:

```python
# Example: customer transaction features
customer_features = FeatureView(
    name="customer_transactions",
    entities=["customer_id"],
    features=[
        Feature("total_purchases", Float),
        Feature("avg_order_value", Float),
        Feature("days_since_last_order", Int),
    ],
    source=transactions_table,
    ttl=timedelta(days=30),
)
```

### Entities

Objects that features describe (users, products, transactions):

```python
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Unique customer identifier"
)
```

Entities provide:
- Join keys for feature retrieval
- Organizational structure
- Documentation

### Offline Store

Historical feature storage for training:
- Typically data warehouse or lake (BigQuery, Snowflake, S3)
- Stores complete feature history
- Enables point-in-time correct joins
- Optimized for batch reads

### Online Store

Low-latency storage for serving:
- Key-value stores (Redis, DynamoDB)
- Stores latest feature values per entity
- Optimized for single-digit millisecond reads
- Populated via materialization

### Materialization

Process of syncing features from offline to online store:

```
Offline Store (Historical)
         “
    Materialization
         “
Online Store (Latest Values)
```

Modes:
- Full: Load all historical data
- Incremental: Load only new records
- Streaming: Continuous real-time updates

## Point-in-Time Correctness

### The Problem

When creating training data, you need features as they existed at the time of each training example:

```
Event: Customer churned on 2024-01-15
Features needed: As of 2024-01-14 (day before)
Wrong: Using features from 2024-01-20 (data leakage)
```

### The Solution

Feature stores perform point-in-time joins:

```python
# Entity dataframe with timestamps
entity_df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "event_timestamp": ["2024-01-14", "2024-01-15", "2024-01-16"],
    "label": [1, 0, 1]
})

# Get features as of each timestamp
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_purchases", ...]
)
```

For each row, features are retrieved as of `event_timestamp`, preventing future data from leaking into training.

## Feature Engineering Patterns

### Batch Features

Computed periodically from historical data:

```python
# Daily aggregation
@batch_feature_view(
    schedule=timedelta(days=1),
    sources=[transactions],
)
def daily_customer_stats(df):
    return df.groupby("customer_id").agg({
        "amount": ["sum", "mean", "count"],
    })
```

### Streaming Features

Computed from real-time event streams:

```python
# Real-time aggregations
@stream_feature_view(
    source=kafka_stream,
    aggregations=[
        Aggregation("amount", sum_, timedelta(hours=1)),
        Aggregation("amount", sum_, timedelta(days=1)),
    ]
)
def realtime_spend(events):
    return events.select("customer_id", "amount", "timestamp")
```

### On-Demand Features

Computed at request time:

```python
@on_demand_feature_view(
    sources=[customer_features, request_data],
)
def request_time_features(customer, request):
    return {
        "distance_from_home": haversine(
            customer["home_lat"], customer["home_lon"],
            request["lat"], request["lon"]
        )
    }
```

## Platform Comparison

### Architecture Comparison

| Aspect | Feast | Tecton | Hopsworks |
|--------|-------|--------|-----------|
| Hosting | Self-managed | Managed | Both |
| Open Source | Yes | No | Yes |
| Offline Store | Pluggable | Managed | HopsFS/External |
| Online Store | Pluggable | Managed | RonDB |
| Streaming | Limited | Native | Kafka/Flink |

### Feature Comparison

| Feature | Feast | Tecton | Hopsworks |
|---------|-------|--------|-----------|
| Point-in-time joins | Yes | Yes | Yes |
| Real-time features | Basic | Sub-second | Kafka-based |
| Transformations | On-demand | Full pipeline | Spark/Pandas |
| Model serving | No | No | Yes (KServe) |
| Vector store | No | No | Yes |
| Monitoring | Basic | Included | Included |

### Operational Comparison

| Aspect | Feast | Tecton | Hopsworks |
|--------|-------|--------|-----------|
| Setup complexity | Medium | Low | High |
| Maintenance burden | High | None | Medium |
| Cost | Infrastructure | Subscription | Infrastructure/Subscription |
| Scaling | Manual | Automatic | Manual/Auto |
| Support | Community | Enterprise | Both |

## Decision Framework

### Choose Feast When

- Budget constraints favor open-source
- Team has infrastructure expertise
- Simple batch feature serving sufficient
- Flexibility and customization needed
- Learning feature store concepts

### Choose Tecton When

- Operational simplicity is priority
- Real-time features with sub-second freshness needed
- Team lacks feature platform expertise
- Enterprise support required
- Budget allows managed service

### Choose Hopsworks When

- Complete ML platform needed (not just features)
- Self-hosting with full control required
- Vector database integration needed
- Complex streaming pipelines
- On-premises deployment required

## Best Practices

### Feature Naming

```python
# Good: descriptive with context
"customer_total_purchases_30d"
"product_avg_rating_all_time"
"user_session_count_7d"

# Avoid: ambiguous
"count"
"value"
"amount"
```

### Versioning

```python
# Version feature groups for breaking changes
customer_features_v1 = FeatureGroup(name="customer_features", version=1)
customer_features_v2 = FeatureGroup(name="customer_features", version=2)
```

### TTL Configuration

```python
# Short TTL for volatile features
session_features = FeatureView(ttl=timedelta(hours=1))

# Long TTL for stable features
profile_features = FeatureView(ttl=timedelta(days=30))
```

### Feature Services

Group features by model:

```python
fraud_model_features = FeatureService(
    name="fraud_detection_v2",
    features=[
        customer_features[["total_spend", "avg_order"]],
        transaction_features[["amount", "merchant_category"]],
        realtime_features,
    ]
)
```

Benefits:
- Document which features each model uses
- Version feature sets independently
- Simplify serving API

## Monitoring

### Feature Freshness

Track when features were last updated:
- Materialization lag
- Streaming pipeline delay
- Alert on stale features

### Data Quality

Monitor feature distributions:
- Min/max/mean statistics
- Null rates
- Distribution drift over time

### Serving Metrics

Track online store performance:
- Latency (p50, p95, p99)
- Throughput (QPS)
- Cache hit rates
- Error rates

## Integration Patterns

### Training Integration

```python
# Get features for training
training_df = store.get_historical_features(
    entity_df=labels_df,
    features=feature_service,
).to_pandas()

# Train model
model.fit(training_df[features], training_df[labels])
```

### Serving Integration

```python
# Real-time inference
@app.route("/predict", methods=["POST"])
def predict():
    entity = request.json
    features = store.get_online_features(
        entity_rows=[entity],
        features=feature_service,
    ).to_dict()

    prediction = model.predict([features])
    return jsonify({"prediction": prediction})
```

### Batch Scoring

```python
# Batch inference
all_customers = get_customer_ids()
features = store.get_historical_features(
    entity_df=pd.DataFrame({
        "customer_id": all_customers,
        "event_timestamp": [datetime.now()] * len(all_customers)
    }),
    features=feature_service,
).to_pandas()

predictions = model.predict(features)
```

## Further Reading

For detailed information on each platform, see:
- [Feast](feast/ReadMe.md)
- [Tecton](tecton/ReadMe.md)
- [Hopsworks](hopsworks/ReadMe.md)
