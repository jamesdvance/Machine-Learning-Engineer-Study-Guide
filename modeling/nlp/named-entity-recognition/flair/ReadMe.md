# Flair for Named Entity Recognition

## Summary

Flair is a PyTorch-based NLP framework developed at Zalando Research that introduced contextual string embeddings for sequence labeling tasks. It achieves state-of-the-art results on NER benchmarks by combining character-level language model embeddings with traditional word embeddings. Flair provides a simple API for using pre-trained models and training custom NER systems.

Key points to remember:

- Contextual string embeddings: Character-level language models capture morphology and context
- Stacked embeddings: Combine multiple embedding types (Flair, BERT, GloVe, etc.)
- Strong out-of-box performance on NER benchmarks
- Easy to stack pre-trained models from different sources
- Training API uses a unified SequenceTagger interface
- Supports pooled embeddings for document classification
- Handles multiple NER formats: BIO, BIOES, IOB
- Active development with regular model updates

## Installation and Basic Usage

### Setup

```python
# pip install flair

from flair.data import Sentence
from flair.models import SequenceTagger

# Load pre-trained NER model
tagger = SequenceTagger.load("flair/ner-english-large")

# Create sentence and predict
sentence = Sentence("Apple CEO Tim Cook announced new products in California.")
tagger.predict(sentence)

# Access entities
for entity in sentence.get_spans("ner"):
    print(f"{entity.text}: {entity.tag} ({entity.score:.4f})")

# Output:
# Apple: ORG (0.9985)
# Tim Cook: PER (0.9998)
# California: LOC (0.9995)
```

### Pre-trained Models

| Model | Language | Entities | F1 (CoNLL-03) |
|-------|----------|----------|---------------|
| ner-english | English | PER, LOC, ORG, MISC | 93.03 |
| ner-english-large | English | PER, LOC, ORG, MISC | 94.09 |
| ner-english-ontonotes-large | English | 18 types | 90.93 |
| ner-multi | Multilingual | PER, LOC, ORG, MISC | Varies |
| ner-german | German | PER, LOC, ORG, MISC | 88.27 |
| ner-dutch | Dutch | PER, LOC, ORG, MISC | 90.44 |

```python
# Load different models
tagger_ontonotes = SequenceTagger.load("flair/ner-english-ontonotes-large")
tagger_multi = SequenceTagger.load("flair/ner-multi")
tagger_german = SequenceTagger.load("flair/ner-german-large")
```

## Embedding Types

### Flair Embeddings (Contextual String Embeddings)

Character-level language model embeddings that capture context:

```python
from flair.embeddings import FlairEmbeddings

# Forward and backward language model embeddings
flair_forward = FlairEmbeddings("news-forward")
flair_backward = FlairEmbeddings("news-backward")

# Embed a sentence
sentence = Sentence("Machine learning is fascinating.")
flair_forward.embed(sentence)

# Access token embeddings
for token in sentence:
    print(f"{token.text}: {token.embedding.shape}")
```

### Stacked Embeddings

Combine multiple embedding types for best performance:

```python
from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    WordEmbeddings
)

# Stack Flair + GloVe embeddings
stacked_embeddings = StackedEmbeddings([
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
    WordEmbeddings("glove"),
])

sentence = Sentence("Apple is a technology company.")
stacked_embeddings.embed(sentence)

# Each token now has concatenated embeddings
print(f"Embedding dimension: {sentence[0].embedding.shape[0]}")
```

### Transformer Embeddings

Use BERT, RoBERTa, and other transformers:

```python
from flair.embeddings import TransformerWordEmbeddings

# BERT embeddings
bert_embeddings = TransformerWordEmbeddings(
    "bert-base-uncased",
    layers="-1,-2,-3,-4",  # Use last 4 layers
    layer_mean=True,       # Average across layers
    fine_tune=False
)

# RoBERTa embeddings
roberta_embeddings = TransformerWordEmbeddings(
    "roberta-base",
    layers="-1",
    fine_tune=True  # Fine-tune during training
)

# Stack with Flair
combined = StackedEmbeddings([
    TransformerWordEmbeddings("bert-base-uncased"),
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
])
```

## Batch Processing

### Processing Multiple Sentences

```python
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-english-large")

# Create sentences
texts = [
    "Apple is based in Cupertino.",
    "Microsoft acquired GitHub.",
    "Elon Musk founded SpaceX.",
]
sentences = [Sentence(text) for text in texts]

# Batch prediction (more efficient)
tagger.predict(sentences, mini_batch_size=32)

# Extract entities from all sentences
for sentence in sentences:
    print(f"Text: {sentence.text}")
    for entity in sentence.get_spans("ner"):
        print(f"  {entity.text}: {entity.tag}")
```

### Large-Scale Processing

```python
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

def process_large_corpus(texts, tagger, batch_size=32):
    """Process large corpus efficiently."""
    all_entities = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        sentences = [Sentence(t) for t in batch_texts]

        tagger.predict(sentences, mini_batch_size=batch_size)

        for sent in sentences:
            entities = [
                {
                    "text": ent.text,
                    "label": ent.tag,
                    "score": ent.score,
                    "start": ent.start_position,
                    "end": ent.end_position
                }
                for ent in sent.get_spans("ner")
            ]
            all_entities.append(entities)

    return all_entities

# Process
tagger = SequenceTagger.load("flair/ner-english-large")
entities = process_large_corpus(texts, tagger)
```

## Training Custom NER Models

### Data Format

Flair uses column-based format (one token per line):

```
George B-PER
Washington I-PER
was O
the O
first O
president O
of O
the O
United B-LOC
States I-LOC
. O

Apple B-ORG
is O
based O
in O
Cupertino B-LOC
. O
```

### Loading Data

```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# Define column format
columns = {0: "text", 1: "ner"}

# Load corpus from directory
corpus: Corpus = ColumnCorpus(
    data_folder="./data",
    column_format=columns,
    train_file="train.txt",
    dev_file="dev.txt",
    test_file="test.txt"
)

# Corpus statistics
print(f"Train: {len(corpus.train)}")
print(f"Dev: {len(corpus.dev)}")
print(f"Test: {len(corpus.test)}")

# View a sample
print(corpus.train[0].to_tagged_string("ner"))
```

### Training with Flair Embeddings

```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# Load corpus
columns = {0: "text", 1: "ner"}
corpus = ColumnCorpus("./data", columns, train_file="train.txt", dev_file="dev.txt", test_file="test.txt")

# Get tag dictionary
tag_dictionary = corpus.make_label_dictionary(label_type="ner")

# Create embeddings
embeddings = StackedEmbeddings([
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
    WordEmbeddings("glove"),
])

# Create tagger
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True,  # Use CRF for sequence labeling
    use_rnn=True,
    rnn_layers=1
)

# Create trainer
trainer = ModelTrainer(tagger, corpus)

# Train
trainer.train(
    base_path="./models/custom_ner",
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=150,
    embeddings_storage_mode="gpu"  # Keep embeddings on GPU
)
```

### Training with Transformers

```python
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# Transformer embeddings with fine-tuning
embeddings = TransformerWordEmbeddings(
    "bert-base-uncased",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True
)

# Create tagger
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True,
    use_rnn=False,  # Often skip RNN with transformers
    reproject_embeddings=False
)

# Train with lower learning rate for transformers
trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path="./models/bert_ner",
    learning_rate=5e-5,
    mini_batch_size=16,
    max_epochs=10,
    optimizer=torch.optim.AdamW,
    scheduler=OneCycleLR,
    embeddings_storage_mode="none"  # Recompute to save memory
)
```

### Few-Shot Training with TARS

```python
from flair.models import TARSTagger
from flair.data import Corpus, Sentence

# TARS: Task-Aware Representation of Sentences
# Supports zero-shot and few-shot NER

tars = TARSTagger.load("tars-ner")

# Define custom entity types
labels = ["product", "company", "technology"]
tars.add_and_switch_to_new_task(
    "custom task",
    label_dictionary=labels,
    label_type="ner"
)

# Zero-shot prediction
sentence = Sentence("The new iPhone uses machine learning.")
tars.predict(sentence)

for entity in sentence.get_spans("ner"):
    print(f"{entity.text}: {entity.tag}")
```

## Evaluation

### Model Evaluation

```python
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus

# Load trained model
tagger = SequenceTagger.load("./models/custom_ner/best-model.pt")

# Evaluate on test set
result = tagger.evaluate(corpus.test, gold_label_type="ner")

print(f"F1: {result.main_score:.4f}")
print(result.detailed_results)
```

### Per-Entity Analysis

```python
def evaluate_per_entity_type(tagger, test_sentences):
    """Detailed evaluation per entity type."""
    from collections import defaultdict

    scores = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sentence in test_sentences:
        gold_spans = sentence.get_spans("ner")
        gold_set = {(s.start_position, s.end_position, s.tag) for s in gold_spans}

        # Get predictions
        tagger.predict(sentence)
        pred_spans = sentence.get_spans("ner")
        pred_set = {(s.start_position, s.end_position, s.tag) for s in pred_spans}

        # Calculate per-type scores
        for span in pred_set:
            if span in gold_set:
                scores[span[2]]["tp"] += 1
            else:
                scores[span[2]]["fp"] += 1

        for span in gold_set:
            if span not in pred_set:
                scores[span[2]]["fn"] += 1

    # Print results
    for entity_type, counts in scores.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{entity_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
```

## Advanced Features

### Multi-Task Learning

Train on multiple NER datasets simultaneously:

```python
from flair.data import MultiCorpus
from flair.datasets import CONLL_03, WNUT_17

# Load multiple corpora
conll = CONLL_03()
wnut = WNUT_17()

# Combine into MultiCorpus
multi_corpus = MultiCorpus([conll, wnut])

# Train single model on combined data
# Useful when target domain is similar to multiple sources
```

### Document-Level Embeddings

```python
from flair.embeddings import DocumentPoolEmbeddings
from flair.data import Sentence

# Pool token embeddings to document level
document_embeddings = DocumentPoolEmbeddings([
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
])

sentence = Sentence("Apple announced new products.")
document_embeddings.embed(sentence)

# Access document-level embedding
doc_embedding = sentence.embedding
print(f"Document embedding shape: {doc_embedding.shape}")
```

### Relation Extraction

Flair also supports relation extraction:

```python
from flair.models import RelationExtractor
from flair.data import Sentence

# Load relation extractor
extractor = RelationExtractor.load("relations")

sentence = Sentence("Apple was founded by Steve Jobs in California.")
extractor.predict(sentence)

# Access relations
for relation in sentence.get_relations("relation"):
    print(f"{relation.first.text} -> {relation.tag} -> {relation.second.text}")
```

## Comparison with SpaCy

| Feature | Flair | SpaCy |
|---------|-------|-------|
| Primary approach | Contextual string embeddings | Statistical + transformer |
| Best accuracy | Slightly higher on benchmarks | Comparable |
| Speed | Slower | Faster |
| Memory usage | Higher | Lower |
| Ease of use | Simple API | Simple API |
| Production readiness | Good | Excellent |
| Ecosystem | Growing | Mature |
| Custom training | Straightforward | Straightforward |

Choose Flair when:
- Maximum accuracy is priority
- Research and experimentation
- Combining multiple embedding types
- Few-shot learning needed

Choose SpaCy when:
- Production speed matters
- Full NLP pipeline needed
- Memory constraints exist
- Mature ecosystem preferred

## Best Practices

### Embedding Selection

```python
# For best accuracy (slower)
embeddings = StackedEmbeddings([
    TransformerWordEmbeddings("xlm-roberta-large", fine_tune=True),
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
])

# For good balance (medium speed)
embeddings = StackedEmbeddings([
    TransformerWordEmbeddings("distilbert-base-uncased"),
    FlairEmbeddings("news-forward-fast"),
    FlairEmbeddings("news-backward-fast"),
])

# For speed (lower accuracy)
embeddings = StackedEmbeddings([
    WordEmbeddings("glove"),
    FlairEmbeddings("news-forward-fast"),
    FlairEmbeddings("news-backward-fast"),
])
```

### Memory Management

```python
import torch

# Clear GPU cache between batches
torch.cuda.empty_cache()

# Use smaller batch sizes for large models
trainer.train(
    mini_batch_size=8,
    embeddings_storage_mode="none"  # Don't cache embeddings
)

# Move model to CPU after loading
tagger = SequenceTagger.load("model.pt")
tagger = tagger.to(torch.device("cpu"))
```

### Saving and Loading

```python
# Save model
tagger.save("my_model.pt")

# Load model
loaded_tagger = SequenceTagger.load("my_model.pt")

# Export to ONNX (experimental)
# Requires additional setup
```
