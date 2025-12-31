# Transformer-Based NER

## Summary

Transformer-based NER leverages pre-trained language models like BERT, RoBERTa, and their variants to achieve state-of-the-art performance on named entity recognition tasks. By fine-tuning these models on NER datasets, they learn to classify each token into entity categories while benefiting from the contextual representations learned during pre-training.

Key points to remember:

- Token classification: Each token is classified into an entity tag (B-PER, I-ORG, O, etc.)
- Subword handling: Transformers use subword tokenization; strategies needed to align with word-level labels
- Fine-tuning: Start from pre-trained checkpoints and train classification head
- CRF optional: Adding CRF layer can improve sequence consistency
- Pre-trained NER models: Many fine-tuned models available on Hugging Face Hub
- Performance: Typically 90%+ F1 on standard benchmarks
- Computation: Requires GPU for efficient training and inference
- Label schemes: BIO (Begin-Inside-Outside) most common

## Architecture

### Token Classification Head

```
Input: "Tim Cook is Apple's CEO"

Tokenization: ["[CLS]", "Tim", "Cook", "is", "Apple", "'", "s", "CEO", "[SEP]"]

Transformer Encoder (BERT, RoBERTa, etc.)
     |
     v
Hidden States: [h_CLS, h_Tim, h_Cook, h_is, h_Apple, h_', h_s, h_CEO, h_SEP]
     |
     v
Classification Layer (Linear: hidden_size -> num_labels)
     |
     v
Predictions: [-, B-PER, I-PER, O, B-ORG, O, O, O, -]
             (Ignore special tokens)
```

### Label Scheme

| Tag | Meaning |
|-----|---------|
| O | Outside any entity |
| B-PER | Beginning of person entity |
| I-PER | Inside person entity |
| B-ORG | Beginning of organization entity |
| I-ORG | Inside organization entity |
| B-LOC | Beginning of location entity |
| I-LOC | Inside location entity |
| B-MISC | Beginning of miscellaneous entity |
| I-MISC | Inside miscellaneous entity |

Example:
```
Tim      Cook     is    the    CEO    of    Apple
B-PER    I-PER    O     O      O      O     B-ORG
```

## Implementation with Hugging Face

### Using Pre-trained NER Models

```python
from transformers import pipeline

# Load pre-trained NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"  # Aggregate subwords
)

text = "Tim Cook announced new products at Apple's headquarters in Cupertino."
entities = ner_pipeline(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.4f})")

# Output:
# Tim Cook: PER (0.9994)
# Apple: ORG (0.9987)
# Cupertino: LOC (0.9995)
```

### Popular Pre-trained Models

| Model | Base | Entities | F1 (CoNLL-03) |
|-------|------|----------|---------------|
| dslim/bert-base-NER | BERT | 4 types | 91.3 |
| dslim/bert-large-NER | BERT-large | 4 types | 92.1 |
| Jean-Baptiste/roberta-large-ner-english | RoBERTa-large | 4 types | 92.5 |
| xlm-roberta-large-finetuned-conll03-english | XLM-R | 4 types | 93.2 |
| flair/ner-english-large | Flair embeddings | 4 types | 94.1 |

### Fine-tuning for Custom NER

```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("conll2003")

# Define label mappings
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First subword
            else:
                label_ids.append(-100)  # Ignore subsequent subwords
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics
from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Subword Alignment Strategies

### Strategy 1: First Subword Only

Assign label only to first subword of each word:

```python
def align_labels_first_subword(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore non-first subwords
            previous_word_idx = word_idx

        labels.append(label_ids)
    return labels
```

### Strategy 2: All Subwords Same Label

All subwords get the same label:

```python
def align_labels_all_subwords(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])

        labels.append(label_ids)
    return labels
```

### Strategy 3: B-tag for First, I-tag for Rest

Convert B-tags to I-tags for subsequent subwords:

```python
def align_labels_bi_conversion(examples, tokenizer, label2id):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # Convert B-XXX to I-XXX for non-first subwords
                current_label = id2label[label[word_idx]]
                if current_label.startswith("B-"):
                    i_label = "I-" + current_label[2:]
                    label_ids.append(label2id.get(i_label, label[word_idx]))
                else:
                    label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    return labels
```

## Adding CRF Layer

CRF (Conditional Random Field) can improve sequence consistency:

```python
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel

class BertCRFForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # CRF loss
            mask = attention_mask.bool()
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return {"loss": loss, "logits": logits}
        else:
            # Decode best path
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            return {"logits": logits, "predictions": predictions}
```

## Inference Optimization

### Batch Inference

```python
from transformers import pipeline
import torch

# Create pipeline with GPU
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=0  # GPU
)

texts = [
    "Apple announced new products.",
    "Microsoft acquired GitHub.",
    "Elon Musk founded SpaceX.",
]

# Batch processing
results = ner_pipeline(texts, batch_size=32)

for text, entities in zip(texts, results):
    print(f"Text: {text}")
    for ent in entities:
        print(f"  {ent['word']}: {ent['entity_group']}")
```

### ONNX Export

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Export to ONNX
dummy_input = tokenizer("Test sentence", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "ner_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=12
)
```

### Quantization

```python
from transformers import AutoModelForTokenClassification
import torch

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "quantized_ner_model.pt")
```

## Multilingual NER

### XLM-RoBERTa Based

```python
from transformers import pipeline

# Multilingual NER
ner_pipeline = pipeline(
    "ner",
    model="Davlan/xlm-roberta-large-ner-hrl",
    aggregation_strategy="simple"
)

# Works across languages
texts = [
    "Apple announced new products in Cupertino.",  # English
    "Apple a annonce de nouveaux produits a Paris.",  # French
    "Apple hat neue Produkte in Berlin angekuendigt.",  # German
]

for text in texts:
    entities = ner_pipeline(text)
    print(f"{text[:50]}...")
    for ent in entities:
        print(f"  {ent['word']}: {ent['entity_group']}")
```

### Language-Specific Models

```python
# German NER
german_ner = pipeline("ner", model="flair/ner-german-large")

# French NER
french_ner = pipeline("ner", model="Jean-Baptiste/camembert-ner")

# Spanish NER
spanish_ner = pipeline("ner", model="mrm8488/bert-spanish-cased-finetuned-ner")

# Chinese NER
chinese_ner = pipeline("ner", model="ckiplab/bert-base-chinese-ner")
```

## Nested NER

Handle overlapping entities:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class NestedNERModel:
    """Handle nested entities by running multiple passes or using span-based approach."""

    def __init__(self, models):
        self.models = models  # Different models for different entity levels

    def predict(self, text, tokenizer):
        all_entities = []

        for model in self.models:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Extract entities and add to list
            entities = self._extract_entities(text, predictions, tokenizer)
            all_entities.extend(entities)

        return self._merge_entities(all_entities)

    def _merge_entities(self, entities):
        """Merge overlapping entities keeping both."""
        # Sort by start position
        entities.sort(key=lambda x: (x['start'], -x['end']))
        return entities
```

## Evaluation

### Using seqeval

```python
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

# True and predicted labels (list of lists)
y_true = [["O", "B-PER", "I-PER", "O", "B-ORG"]]
y_pred = [["O", "B-PER", "I-PER", "O", "B-ORG"]]

# F1 score
f1 = f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
print(f"F1: {f1:.4f}")

# Detailed report
report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
print(report)
```

### Evaluation with Hugging Face

```python
from datasets import load_metric
import numpy as np

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

## Best Practices

### Model Selection

| Scenario | Recommended Model |
|----------|------------------|
| English, accuracy priority | roberta-large-ner |
| English, speed priority | bert-base-NER |
| Multilingual | xlm-roberta-large-ner |
| Domain-specific | Fine-tune on domain data |
| Low resource | Few-shot with TARS or prompting |

### Training Tips

1. Use learning rate 2e-5 to 5e-5 for fine-tuning
2. Train for 3-5 epochs (more may overfit)
3. Use weight decay 0.01
4. Warm up learning rate for first 10% of steps
5. Evaluate every epoch and save best model
6. Use gradient accumulation for larger effective batch sizes

### Common Issues

**Subword tokenization misalignment:**
```python
# Always use is_split_into_words=True for pre-tokenized input
tokenizer(words, is_split_into_words=True)
```

**Class imbalance (O tag dominant):**
```python
# Use weighted loss
from torch.nn import CrossEntropyLoss

weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
loss_fn = CrossEntropyLoss(weight=weights, ignore_index=-100)
```

**Long sequences:**
```python
# Use sliding window for long documents
def process_long_document(text, tokenizer, model, max_length=512, stride=128):
    encodings = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True
    )
    # Process each chunk and merge results
```
