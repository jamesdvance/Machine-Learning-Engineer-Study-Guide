# Text Classification

## Summary

Text classification is the task of assigning predefined categories to text documents. It is one of the most common NLP tasks, powering spam detection, content moderation, sentiment analysis, intent recognition, and document organization. The field has evolved from rule-based systems through statistical machine learning to transformer-based models that achieve near-human performance on many benchmarks.

Key points to remember:

- Multi-class: Each document belongs to exactly one class
- Multi-label: Documents can belong to multiple classes simultaneously
- Binary: Special case with two classes (positive/negative, spam/not spam)
- Approaches range from traditional ML (SVM, Naive Bayes) to transformers (BERT, RoBERTa)
- Transfer learning: Pre-trained language models dramatically reduce data requirements
- Supervised vs unsupervised: Classification requires labeled data; topic modeling discovers patterns
- Evaluation: Accuracy for balanced datasets, F1 score for imbalanced
- Zero-shot and few-shot: Modern models can classify with minimal or no training examples

## Task Variations

### By Label Structure

| Type | Description | Example |
|------|-------------|---------|
| Binary | Two mutually exclusive classes | Spam detection |
| Multi-class | Multiple mutually exclusive classes | News categorization |
| Multi-label | Multiple non-exclusive classes | Article tagging |
| Hierarchical | Classes organized in taxonomy | Product categorization |
| Ordinal | Ordered classes | Rating prediction (1-5 stars) |

### By Supervision

| Approach | Training Data | Example Task |
|----------|---------------|--------------|
| Supervised | Labeled examples | Intent classification |
| Unsupervised | No labels | Topic modeling |
| Semi-supervised | Limited labels + unlabeled data | Low-resource classification |
| Zero-shot | No task-specific examples | Classification via NLI |
| Few-shot | Very few examples per class (5-20) | SetFit, prompting |

## Comparison of Subtasks

### Sentiment Analysis

Determines emotional tone or opinion:

```
Input: "This product exceeded my expectations!"
Output: Positive

Characteristics:
- Usually binary or 3-5 classes
- Domain-dependent (movie reviews vs financial text)
- Challenges: sarcasm, negation, implicit sentiment
```

### Intent Classification

Identifies user purpose in conversational systems:

```
Input: "Book a flight to London tomorrow"
Output: book_flight

Characteristics:
- Many classes (often 50-200 intents)
- Short utterances
- Requires out-of-scope detection
- Often paired with slot filling/entity extraction
```

### Topic Modeling

Discovers latent thematic structure:

```
Input: Collection of news articles
Output: Topics (politics, sports, technology, etc.)

Characteristics:
- Unsupervised (no predefined labels)
- Number of topics often unknown
- Topics defined by word distributions
- Each document is mixture of topics
```

### When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| Analyzing customer feedback polarity | Sentiment Analysis |
| Building a chatbot NLU | Intent Classification |
| Organizing large document collection | Topic Modeling |
| Categorizing support tickets | Multi-class Classification |
| Tagging articles with multiple themes | Multi-label Classification |

## Approaches

### Traditional Machine Learning

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

pipeline.fit(train_texts, train_labels)
predictions = pipeline.predict(test_texts)
```

Pros:
- Fast training and inference
- Interpretable
- Works well with limited data
- Low computational requirements

Cons:
- Requires feature engineering
- Limited semantic understanding
- Struggles with rare words

### Deep Learning (Pre-Transformer)

```python
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        return self.fc(self.dropout(x))
```

Pros:
- Learns features automatically
- Captures local patterns well
- Faster than transformers

Cons:
- Requires more data than traditional ML
- Limited long-range dependencies (for CNNs)
- Less effective than transformers

### Transformer-Based

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_classes
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5
    ),
    train_dataset=train_dataset.map(tokenize, batched=True),
    eval_dataset=test_dataset.map(tokenize, batched=True)
)

trainer.train()
```

Pros:
- State-of-the-art accuracy
- Transfer learning reduces data requirements
- Captures long-range dependencies
- Pre-trained models widely available

Cons:
- Computationally expensive
- Requires GPU for efficient training
- May be overkill for simple tasks

### Zero-Shot Classification

No training examples required:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "The new smartphone features an improved camera system"
labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, labels)
print(f"Label: {result['labels'][0]}, Score: {result['scores'][0]:.3f}")
```

Pros:
- No labeled data needed
- Flexible; add new classes without retraining
- Fast to deploy

Cons:
- Lower accuracy than fine-tuned models
- Requires well-designed label names
- Slower inference than dedicated classifiers

## Handling Class Imbalance

Common in real-world classification:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute balanced weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

# Use in training
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = torch.tensor(
            [class_weight_dict[l.item()] for l in labels],
            device=logits.device
        )
        loss = F.cross_entropy(logits, labels, weight=weights)

        return (loss, outputs) if return_outputs else loss
```

Other strategies:
- Oversampling minority classes
- Undersampling majority classes
- SMOTE for synthetic examples
- Focal loss to focus on hard examples

## Multi-Label Classification

When documents can have multiple labels:

```python
from transformers import AutoModelForSequenceClassification
import torch

# Multi-label model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# Training uses BCEWithLogitsLoss automatically
# Prediction requires sigmoid + threshold
def predict_multilabel(model, tokenizer, text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities > threshold).squeeze().tolist()
    return [label for label, pred in zip(labels, predictions) if pred]
```

## Evaluation

### Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# For multi-class
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

# Detailed report
print(classification_report(y_true, y_pred, target_names=class_names))

# For multi-label
from sklearn.metrics import multilabel_confusion_matrix, hamming_loss

# Hamming loss: fraction of wrong labels
h_loss = hamming_loss(y_true, y_pred)
```

### Choosing Metrics

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classes | Accuracy |
| Imbalanced classes | Macro F1, Weighted F1 |
| Cost of false positives high | Precision |
| Cost of false negatives high | Recall |
| Multi-label | Hamming loss, Subset accuracy |
| Ranking quality | AUC-ROC |

## Best Practices

### Data Preparation

```python
def prepare_classification_data(texts, labels, test_size=0.2, val_size=0.1):
    """Split data with stratification."""
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=42
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
```

### Model Selection Guidelines

| Data Size | Classes | Recommended Approach |
|-----------|---------|---------------------|
| < 1,000 | Few | Traditional ML (SVM, NB) |
| 1,000-10,000 | Few-Medium | Fine-tuned BERT or distilled model |
| > 10,000 | Any | Fine-tuned BERT or custom training |
| Any | Very many (> 100) | Hierarchical or embedding-based |
| Limited labels | Any | Zero-shot or few-shot |

### Confidence Handling

```python
def classify_with_confidence(model, tokenizer, text, threshold=0.8):
    """Only return prediction if confident."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

    confidence, predicted_class = probabilities.max(dim=-1)

    if confidence.item() < threshold:
        return {
            "label": "uncertain",
            "confidence": confidence.item(),
            "top_predictions": get_top_k_predictions(probabilities, k=3)
        }

    return {
        "label": id2label[predicted_class.item()],
        "confidence": confidence.item()
    }
```

## Production Considerations

### Latency Optimization

```python
# Use distilled models for speed
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",  # 40% faster than BERT
    num_labels=num_classes
)

# Quantization for faster inference
import torch
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX export
from transformers import onnx
onnx.export(model, tokenizer, "model.onnx")
```

### Monitoring

```python
import logging
from datetime import datetime

def log_prediction(text, prediction, confidence, latency_ms):
    logging.info({
        "timestamp": datetime.utcnow().isoformat(),
        "text_length": len(text),
        "predicted_class": prediction,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "low_confidence": confidence < 0.7
    })

# Monitor for drift
def check_distribution_drift(recent_predictions, baseline_distribution):
    from scipy.stats import chi2_contingency
    # Compare recent class distribution to baseline
    observed = get_class_counts(recent_predictions)
    expected = baseline_distribution * len(recent_predictions)
    chi2, p_value, _, _ = chi2_contingency([observed, expected])
    if p_value < 0.05:
        alert("Distribution drift detected")
```

## Comparison Summary

| Task | Supervision | Labels | Typical Use Case |
|------|-------------|--------|------------------|
| Sentiment Analysis | Supervised | 2-5 predefined | Customer feedback |
| Intent Classification | Supervised | Many predefined | Chatbots |
| Topic Modeling | Unsupervised | Discovered | Document exploration |
| Multi-label | Supervised | Multiple per doc | Article tagging |
| Zero-shot | None | Dynamic | New domains |

Each approach serves different needs. Start simple (traditional ML or zero-shot) and add complexity only when needed for performance.
