# Transformers4Rec

## Summary

Transformers4Rec is NVIDIA's open-source library for building transformer-based sequential and session-based recommendation models. Built on top of HuggingFace Transformers, it provides a modular framework for creating production-ready recommendation systems with state-of-the-art architectures. The library handles the complexity of feature preprocessing, model training, and inference while leveraging GPU acceleration through integration with NVIDIA's data processing tools.

Key points to remember:

- NVIDIA's library for transformer-based sequential recommendations
- Built on HuggingFace Transformers architecture
- Modular design: schema, features, model, training separated
- Supports XLNet, GPT-2, BERT, and custom transformer architectures
- Integration with NVTabular for GPU-accelerated preprocessing
- Handles both sequential and session-based recommendation
- Production-ready with Triton Inference Server deployment
- Automatic feature encoding (categorical, continuous, embeddings)
- Compared to custom implementations: faster development, production features

## Architecture Overview

```
+------------------+
|    Raw Data      |
+------------------+
        |
        v
+------------------+
|   NVTabular      |  <- GPU-accelerated preprocessing
|   (ETL/Features) |
+------------------+
        |
        v
+------------------+
|     Schema       |  <- Feature definitions
+------------------+
        |
        v
+------------------+
|  TabularSequence |  <- Feature encoding
|  Features        |
+------------------+
        |
        v
+------------------+
|   Transformer    |  <- XLNet, GPT-2, BERT, etc.
|     Model        |
+------------------+
        |
        v
+------------------+
|   Prediction     |  <- Next-item, classification
|      Head        |
+------------------+
```

## Installation

```bash
# Install core library
pip install transformers4rec[pytorch]

# With NVTabular for preprocessing
pip install nvtabular transformers4rec[pytorch]

# Full installation with all features
pip install transformers4rec[all]
```

## Basic Usage

### Data Preparation with NVTabular

```python
import nvtabular as nvt
from nvtabular import ops

# Define preprocessing workflow
workflow = nvt.Workflow(
    # Categorical encoding
    ['item_id', 'category_id'] >> ops.Categorify() >>
    # Group by session
    ops.Groupby(
        groupby_cols=['session_id'],
        aggs={
            'item_id': ['list'],
            'category_id': ['list'],
            'timestamp': ['list'],
        },
        name_sep='-'
    )
)

# Apply to data
dataset = nvt.Dataset(train_df)
workflow.fit(dataset)
train_dataset = workflow.transform(dataset)
```

### Define Schema

```python
from merlin.schema import Schema, Tags, ColumnSchema
from transformers4rec import schema

# Automatic schema from NVTabular
train_schema = train_dataset.schema

# Or define manually
schema = Schema([
    ColumnSchema(
        name='item_id-list',
        tags=[Tags.ITEM_ID, Tags.CATEGORICAL, Tags.LIST],
        dims=(None,),  # Variable length
    ),
    ColumnSchema(
        name='category_id-list',
        tags=[Tags.CATEGORICAL, Tags.LIST],
        dims=(None,),
    ),
])
```

### Build Model

```python
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

# Input module
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=20,
    continuous_projection=64,  # Project continuous features
    aggregation="concat",
)

# Transformer block (XLNet)
transformer_config = tr.XLNetConfig.build(
    d_model=64,
    n_head=4,
    n_layer=2,
    total_seq_length=20,
)
transformer = tr.TransformerBlock(transformer_config, masking="causal")

# Prediction head
prediction_task = tr.NextItemPredictionTask(
    weight_tying=True,  # Share embeddings
    metrics=[NDCGAt(top_ks=[10, 20]), RecallAt(top_ks=[10, 20])]
)

# Combine into model
model = transformer_config.to_torch_model(input_module, transformer, prediction_task)
```

### Training

```python
from transformers4rec.torch.trainer import Trainer

# Training arguments
training_args = tr.T4RecTrainingArguments(
    output_dir="./output",
    max_sequence_length=20,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=0.001,
    num_train_epochs=10,
    learning_rate_scheduler="linear",
    report_to=[],  # Disable wandb
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    schema=train_schema,
    compute_metrics=True,
)

# Train
trainer.train_dataset_or_path = train_dataset
trainer.eval_dataset_or_path = valid_dataset
trainer.train()
```

## Transformer Architectures

### XLNet (Recommended)

Permutation language modeling with causal attention:

```python
from transformers4rec.torch import XLNetConfig

xlnet_config = XLNetConfig.build(
    d_model=128,           # Hidden dimension
    n_head=4,              # Attention heads
    n_layer=2,             # Transformer layers
    total_seq_length=50,   # Max sequence length
    attn_type="bi",        # Bidirectional attention
)

transformer = tr.TransformerBlock(xlnet_config, masking="causal")
```

### GPT-2

Causal (autoregressive) language modeling:

```python
from transformers4rec.torch import GPT2Config

gpt2_config = GPT2Config.build(
    d_model=128,
    n_head=4,
    n_layer=2,
    total_seq_length=50,
)

transformer = tr.TransformerBlock(gpt2_config, masking="causal")
```

### BERT

Masked language modeling:

```python
from transformers4rec.torch import BertConfig

bert_config = BertConfig.build(
    d_model=128,
    n_head=4,
    n_layer=2,
    total_seq_length=50,
)

# Use MLM masking for BERT-style training
transformer = tr.TransformerBlock(bert_config, masking="mlm")
```

### Custom Transformer

```python
from transformers4rec.torch import TransformerConfig

custom_config = TransformerConfig(
    d_model=256,
    n_head=8,
    n_layer=4,
    hidden_act="gelu",
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    dropout=0.1,
    attention_dropout=0.1,
    total_seq_length=100,
)
```

## Feature Engineering

### Tabular Sequence Features

```python
from transformers4rec.torch import TabularSequenceFeatures

# From schema (automatic)
input_module = TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=50,
    continuous_projection=64,    # MLP for continuous
    d_output=128,                # Output dimension
    aggregation="concat",        # How to combine features
)

# Manual configuration
input_module = TabularSequenceFeatures.from_schema(
    schema,
    embedding_dims={
        'item_id-list': 128,
        'category_id-list': 32,
    },
    embedding_dim_default=64,
    max_sequence_length=50,
)
```

### Continuous Features

```python
# Project continuous features to embedding space
input_module = TabularSequenceFeatures.from_schema(
    schema,
    continuous_projection=64,  # Project to 64-dim
    continuous_soft_embeddings=True,  # Learnable projection
)

# Or use specific projection
from transformers4rec.torch.features.continuous import ContinuousFeatures

continuous = ContinuousFeatures.from_schema(
    schema,
    tags=[Tags.CONTINUOUS],
    soft_embedding_cardinalities={"price": 10},  # Discretize
)
```

### Pre-trained Embeddings

```python
import torch

# Load pre-trained item embeddings
pretrained_embeddings = torch.load("item_embeddings.pt")

input_module = TabularSequenceFeatures.from_schema(
    schema,
    embedding_dims={'item_id-list': 128},
    pre_trained_embeddings={'item_id-list': pretrained_embeddings},
    freeze_pre_trained=False,  # Fine-tune
)
```

## Prediction Tasks

### Next-Item Prediction

```python
from transformers4rec.torch import NextItemPredictionTask
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt, AvgPrecisionAt

prediction_task = NextItemPredictionTask(
    weight_tying=True,           # Share input/output embeddings
    item_embedding_table=None,   # Use input embeddings
    metrics=[
        NDCGAt(top_ks=[10, 20, 50]),
        RecallAt(top_ks=[10, 20, 50]),
        AvgPrecisionAt(top_ks=[10, 20]),
    ],
)
```

### Multi-Task Learning

```python
from transformers4rec.torch import (
    NextItemPredictionTask,
    BinaryClassificationTask,
    RegressionTask,
)

# Predict next item + purchase probability + dwell time
prediction_tasks = tr.PredictionTasks(
    NextItemPredictionTask(weight_tying=True),
    BinaryClassificationTask(
        target='purchased',
        task_name='purchase_prediction',
    ),
    RegressionTask(
        target='dwell_time',
        task_name='dwell_prediction',
    ),
)

model = transformer_config.to_torch_model(
    input_module,
    transformer,
    prediction_tasks
)
```

## Training Strategies

### Negative Sampling

```python
from transformers4rec.torch import NextItemPredictionTask

# In-batch negative sampling
prediction_task = NextItemPredictionTask(
    weight_tying=True,
    sampled_softmax=True,
    min_id=1,                    # Minimum item ID
    max_id=10000,                # Maximum item ID
)

# Or specify negative sampler
from transformers4rec.torch.utils.sampling import InBatchNegativesSampler

sampler = InBatchNegativesSampler(
    max_id=10000,
    min_id=1,
)
```

### Learning Rate Scheduling

```python
training_args = tr.T4RecTrainingArguments(
    output_dir="./output",
    learning_rate=0.001,
    learning_rate_scheduler="cosine",  # cosine, linear, constant
    warmup_steps=100,
    weight_decay=0.01,
)
```

### Mixed Precision Training

```python
training_args = tr.T4RecTrainingArguments(
    output_dir="./output",
    fp16=True,  # Enable mixed precision
    fp16_backend="amp",
)
```

## Evaluation

### Built-in Metrics

```python
from transformers4rec.torch.ranking_metric import (
    NDCGAt,
    RecallAt,
    AvgPrecisionAt,
    MRRAt,
)

metrics = [
    NDCGAt(top_ks=[5, 10, 20]),
    RecallAt(top_ks=[5, 10, 20]),
    MRRAt(top_ks=[10]),
]

# Use in prediction task
prediction_task = NextItemPredictionTask(metrics=metrics)
```

### Custom Evaluation

```python
def evaluate_model(model, eval_dataset, schema):
    model.eval()

    predictions = []
    targets = []

    for batch in eval_dataset:
        with torch.no_grad():
            output = model(batch)
            preds = output['predictions']
            targets.append(batch['target'])
            predictions.append(preds)

    # Compute custom metrics
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # Coverage
    unique_predictions = set(predictions.argmax(dim=-1).unique().tolist())
    coverage = len(unique_predictions) / n_items

    return {'coverage': coverage}
```

## Session-Based Recommendation

### Session Data Processing

```python
import nvtabular as nvt
from nvtabular import ops

# Group interactions into sessions
session_workflow = nvt.Workflow(
    ['item_id', 'category_id', 'price'] >>
    ops.Groupby(
        groupby_cols=['session_id'],
        aggs={
            'item_id': ['list'],
            'category_id': ['list'],
            'price': ['list'],
        },
    ) >>
    # Filter short sessions
    ops.Filter(lambda df: df['item_id-list'].list.len() >= 2)
)
```

### Session-Aware Model

```python
# Add session features
schema = Schema([
    ColumnSchema('item_id-list', tags=[Tags.ITEM_ID, Tags.LIST]),
    ColumnSchema('category_id-list', tags=[Tags.CATEGORICAL, Tags.LIST]),
    ColumnSchema('session_id', tags=[Tags.SESSION_ID]),
    ColumnSchema('session_length', tags=[Tags.CONTINUOUS]),
])

# Model with session context
input_module = TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=30,
    aggregation="concat",
)
```

## Production Deployment

### Export for Triton

```python
from transformers4rec.torch.utils.export import export_pytorch_model

# Export model
export_pytorch_model(
    model,
    schema,
    export_path="./model_repository/t4rec/1",
    max_sequence_length=50,
)

# Generate Triton config
config = """
name: "t4rec"
platform: "pytorch_libtorch"
max_batch_size: 256
input [
  {
    name: "item_id-list"
    data_type: TYPE_INT64
    dims: [-1]
  }
]
output [
  {
    name: "predictions"
    data_type: TYPE_FP32
    dims: [-1]
  }
]
"""
```

### Inference Pipeline

```python
from transformers4rec.torch import Model

# Load model
model = Model.load("./output/checkpoint-1000")
model.eval()

# Inference
def predict_next_items(session_items, top_k=10):
    with torch.no_grad():
        # Prepare input
        batch = {
            'item_id-list': torch.tensor([session_items])
        }

        # Forward pass
        output = model(batch)
        scores = output['predictions'].squeeze()

        # Get top-k
        top_scores, top_indices = torch.topk(scores, top_k)

        return list(zip(top_indices.tolist(), top_scores.tolist()))
```

## Example: E-commerce Session Recommendation

```python
import pandas as pd
import nvtabular as nvt
from nvtabular import ops
import transformers4rec.torch as tr
from merlin.schema import Tags

# 1. Prepare data
interactions = pd.read_parquet("interactions.parquet")

# 2. NVTabular preprocessing
workflow = nvt.Workflow(
    ['item_id', 'category_id', 'price'] >>
    ops.Categorify(freq_threshold=5) >>
    ops.Groupby(
        groupby_cols=['session_id'],
        aggs={
            'item_id': ['list'],
            'category_id': ['list'],
            'price': ['list', 'mean'],
        },
    ) >>
    ops.AddMetadata(tags=[Tags.LIST], columns=['item_id-list', 'category_id-list'])
)

dataset = nvt.Dataset(interactions)
workflow.fit(dataset)
processed = workflow.transform(dataset)
schema = processed.schema

# 3. Build model
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=30,
    continuous_projection=32,
)

xlnet_config = tr.XLNetConfig.build(
    d_model=128,
    n_head=4,
    n_layer=2,
    total_seq_length=30,
)

transformer = tr.TransformerBlock(xlnet_config, masking="causal")

prediction_task = tr.NextItemPredictionTask(
    weight_tying=True,
    metrics=[tr.ranking_metric.NDCGAt(top_ks=[10, 20])]
)

model = xlnet_config.to_torch_model(input_module, transformer, prediction_task)

# 4. Train
training_args = tr.T4RecTrainingArguments(
    output_dir="./ecommerce_model",
    per_device_train_batch_size=256,
    learning_rate=0.001,
    num_train_epochs=5,
)

trainer = tr.Trainer(
    model=model,
    args=training_args,
    schema=schema,
)

trainer.train_dataset_or_path = processed
trainer.train()
```

## Comparison with Other Libraries

| Feature | Transformers4Rec | RecBole | Custom PyTorch |
|---------|------------------|---------|----------------|
| GPU preprocessing | Yes (NVTabular) | No | No |
| HuggingFace integration | Yes | No | Manual |
| Production deployment | Triton ready | No | Manual |
| Feature engineering | Automatic | Manual | Manual |
| Multi-task learning | Built-in | Some | Manual |
| Learning curve | Medium | Medium | High |

## When to Use Transformers4Rec

Transformers4Rec is well-suited for:
- Production sequential recommendation systems
- When GPU preprocessing helps (large data)
- Teams familiar with HuggingFace
- Multi-task recommendation scenarios
- Triton deployment environments

Consider alternatives when:
- Simple models suffice
- No GPU available
- Need maximum customization
- Research/experimental settings
