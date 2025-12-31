# SpaCy for Named Entity Recognition

## Summary

SpaCy is an industrial-strength NLP library that provides fast, production-ready named entity recognition (NER). It offers pre-trained models for multiple languages, efficient pipelines for processing text at scale, and straightforward APIs for training custom NER models. SpaCy's architecture balances accuracy with speed, making it suitable for both prototyping and production deployment.

Key points to remember:

- Pre-trained models available in small, medium, and large variants
- Built-in entity types: PERSON, ORG, GPE, DATE, MONEY, and more
- Pipeline architecture: Tokenizer, Tagger, Parser, NER components
- Training custom NER: Use spacy train with annotated data in spaCy format
- Rule-based matching: EntityRuler for pattern-based entity extraction
- Efficient processing: Streams, batching, and multiprocessing support
- Integration: Works with transformers via spacy-transformers
- Model sizes: sm (fast, less accurate) to trf (slow, most accurate)

## Pre-trained Models

### Model Variants

| Model | Size | NER F1 | Speed | Use Case |
|-------|------|--------|-------|----------|
| en_core_web_sm | 12 MB | ~85% | Fastest | Quick prototyping |
| en_core_web_md | 40 MB | ~85% | Fast | General use |
| en_core_web_lg | 560 MB | ~86% | Medium | Better accuracy |
| en_core_web_trf | 440 MB | ~90% | Slow | Maximum accuracy |

### Installation and Loading

```python
# Install spaCy and model
# pip install spacy
# python -m spacy download en_core_web_lg

import spacy

# Load model
nlp = spacy.load("en_core_web_lg")

# Process text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Extract entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_} ({ent.start_char}-{ent.end_char})")

# Output:
# Apple: ORG (0-5)
# U.K.: GPE (27-31)
# $1 billion: MONEY (44-54)
```

### Built-in Entity Types

| Label | Description | Example |
|-------|-------------|---------|
| PERSON | People names | "Barack Obama" |
| ORG | Organizations | "Google", "United Nations" |
| GPE | Geopolitical entities | "France", "New York City" |
| LOC | Non-GPE locations | "Mount Everest", "Pacific Ocean" |
| DATE | Dates and periods | "January 2024", "next week" |
| TIME | Times | "3:00 PM", "morning" |
| MONEY | Monetary values | "$50", "20 euros" |
| PERCENT | Percentages | "25%", "fifty percent" |
| CARDINAL | Numerals | "100", "three million" |
| ORDINAL | Order expressions | "first", "3rd" |
| PRODUCT | Products | "iPhone", "Windows 10" |
| EVENT | Named events | "World Cup", "Olympics" |
| WORK_OF_ART | Titles of works | "Hamlet", "Mona Lisa" |
| LAW | Laws and documents | "First Amendment" |
| LANGUAGE | Languages | "English", "French" |

## Pipeline Processing

### Understanding the Pipeline

```python
import spacy

nlp = spacy.load("en_core_web_lg")

# View pipeline components
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

# Disable components for speed if only NER needed
nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])

# Or use enable for specific components
nlp = spacy.load("en_core_web_lg", enable=["ner"])
```

### Efficient Batch Processing

```python
import spacy

nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])

texts = [
    "Apple announced new products in California.",
    "Microsoft acquired GitHub for $7.5 billion.",
    "Elon Musk founded SpaceX in 2002.",
]

# Process in batches using nlp.pipe()
docs = list(nlp.pipe(texts, batch_size=50))

for doc in docs:
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)

# With progress tracking
from tqdm import tqdm

docs = list(tqdm(nlp.pipe(texts, batch_size=50), total=len(texts)))
```

### Multiprocessing

```python
import spacy

nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])

# Use multiple processes
docs = list(nlp.pipe(texts, n_process=4, batch_size=100))

# For very large datasets, process as generator
def process_large_dataset(texts, nlp, batch_size=1000):
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=-1):
        yield [(ent.text, ent.label_, ent.start_char, ent.end_char)
               for ent in doc.ents]
```

## Entity Information

### Accessing Entity Attributes

```python
doc = nlp("Apple CEO Tim Cook announced the iPhone in San Francisco.")

for ent in doc.ents:
    print(f"Text: {ent.text}")
    print(f"Label: {ent.label_} ({spacy.explain(ent.label_)})")
    print(f"Start: {ent.start_char}, End: {ent.end_char}")
    print(f"Token indices: {ent.start} to {ent.end}")
    print(f"Root: {ent.root.text}")
    print("---")
```

### Entity Context

```python
def get_entity_context(doc, entity, window=3):
    """Get surrounding context for an entity."""
    start = max(0, entity.start - window)
    end = min(len(doc), entity.end + window)

    left_context = doc[start:entity.start].text
    right_context = doc[entity.end:end].text

    return {
        "entity": entity.text,
        "label": entity.label_,
        "left_context": left_context,
        "right_context": right_context
    }

for ent in doc.ents:
    context = get_entity_context(doc, ent)
    print(context)
```

### Visualizing Entities

```python
from spacy import displacy

doc = nlp("Apple CEO Tim Cook announced new products at the Steve Jobs Theater.")

# Render in Jupyter notebook
displacy.render(doc, style="ent", jupyter=True)

# Save to HTML file
html = displacy.render(doc, style="ent", page=True)
with open("entities.html", "w") as f:
    f.write(html)

# Customize colors
colors = {"ORG": "#ff6b6b", "PERSON": "#4ecdc4", "GPE": "#45b7d1"}
options = {"colors": colors}
displacy.render(doc, style="ent", options=options)
```

## Rule-Based NER with EntityRuler

### Basic Pattern Matching

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_lg")

# Create EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define patterns
patterns = [
    {"label": "PRODUCT", "pattern": "iPhone"},
    {"label": "PRODUCT", "pattern": "MacBook Pro"},
    {"label": "ORG", "pattern": [{"LOWER": "openai"}]},
    {"label": "TECH", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
]

ruler.add_patterns(patterns)

doc = nlp("OpenAI released ChatGPT, which uses machine learning.")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### Advanced Patterns

```python
# Token-based patterns with attributes
patterns = [
    # Match any capitalized word followed by "Inc." or "Corp."
    {
        "label": "ORG",
        "pattern": [
            {"IS_TITLE": True},
            {"TEXT": {"IN": ["Inc.", "Corp.", "Ltd."]}}
        ]
    },

    # Match version numbers (e.g., "v2.0", "version 3.1")
    {
        "label": "VERSION",
        "pattern": [
            {"LOWER": {"IN": ["v", "version"]}},
            {"LIKE_NUM": True}
        ]
    },

    # Match email-like patterns
    {
        "label": "EMAIL",
        "pattern": [{"LIKE_EMAIL": True}]
    },

    # Match URLs
    {
        "label": "URL",
        "pattern": [{"LIKE_URL": True}]
    },

    # Match with regex
    {
        "label": "PHONE",
        "pattern": [{"TEXT": {"REGEX": r"\d{3}-\d{3}-\d{4}"}}]
    }
]

ruler.add_patterns(patterns)
```

### Combining Rules with Statistical NER

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_lg")

# Add ruler BEFORE statistical NER to override
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Add patterns for domain-specific entities
patterns = [
    {"label": "DRUG", "pattern": "aspirin"},
    {"label": "DRUG", "pattern": "ibuprofen"},
    {"label": "DISEASE", "pattern": "diabetes"},
]
ruler.add_patterns(patterns)

# Or add AFTER to supplement (not override)
# ruler = nlp.add_pipe("entity_ruler", after="ner")
```

## Training Custom NER Models

### Data Format

SpaCy v3 uses .spacy binary format:

```python
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

# Training data in standard format
TRAIN_DATA = [
    ("Apple is a technology company.", {"entities": [(0, 5, "ORG")]}),
    ("Tim Cook is the CEO.", {"entities": [(0, 8, "PERSON")]}),
    ("They are based in Cupertino.", {"entities": [(18, 27, "GPE")]}),
]

# Convert to DocBin
doc_bin = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

# Save to file
doc_bin.to_disk("./train.spacy")
```

### Configuration File

Create `config.cfg`:

```ini
[paths]
train = "./train.spacy"
dev = "./dev.spacy"

[system]
gpu_allocator = null

[nlp]
lang = "en"
pipeline = ["tok2vec", "ner"]
batch_size = 1000

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 2500, 2500, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.embed.width}

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
size = 1000

[training.optimizer]
@optimizers = "Adam.v1"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
```

### Training Command

```bash
# Initialize config
python -m spacy init config config.cfg --lang en --pipeline ner

# Fill in defaults
python -m spacy init fill-config config.cfg config_full.cfg

# Train model
python -m spacy train config_full.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy
```

### Training with Python API

```python
import spacy
from spacy.training import Example
import random

nlp = spacy.blank("en")

# Add NER component
ner = nlp.add_pipe("ner")

# Add labels
for _, annotations in TRAIN_DATA:
    for start, end, label in annotations["entities"]:
        ner.add_label(label)

# Training loop
optimizer = nlp.begin_training()

for epoch in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)

    print(f"Epoch {epoch}, Losses: {losses}")

# Save model
nlp.to_disk("./custom_ner_model")
```

## Transfer Learning with Transformers

### Using spacy-transformers

```python
# pip install spacy-transformers
# python -m spacy download en_core_web_trf

import spacy

# Load transformer-based model
nlp = spacy.load("en_core_web_trf")

doc = nlp("Apple CEO Tim Cook announced new products.")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### Fine-tuning Transformer NER

Config for transformer-based training:

```ini
[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "roberta-base"
tokenizer_config = {"use_fast": true}

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.ner.model.tok2vec.pooling]
@layers = "reduce_mean.v1"
```

## Evaluation

### Built-in Scoring

```python
from spacy.scorer import Scorer
from spacy.training import Example

def evaluate(nlp, examples):
    scorer = Scorer()

    for input_, annotations in examples:
        doc = nlp.make_doc(input_)
        example = Example.from_dict(doc, annotations)
        pred = nlp(input_)
        example.predicted = pred
        scores = scorer.score([example])

    return scores

scores = evaluate(nlp, test_data)
print(f"NER Precision: {scores['ents_p']:.2f}")
print(f"NER Recall: {scores['ents_r']:.2f}")
print(f"NER F1: {scores['ents_f']:.2f}")
```

### Per-Entity Evaluation

```python
def evaluate_per_entity(nlp, test_data):
    """Evaluate NER performance per entity type."""
    from collections import defaultdict

    entity_scores = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, annotations in test_data:
        doc = nlp(text)
        pred_ents = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold_ents = {tuple(e) for e in annotations["entities"]}

        for ent in pred_ents:
            if ent in gold_ents:
                entity_scores[ent[2]]["tp"] += 1
            else:
                entity_scores[ent[2]]["fp"] += 1

        for ent in gold_ents:
            if ent not in pred_ents:
                entity_scores[ent[2]]["fn"] += 1

    # Calculate metrics per entity
    for label, counts in entity_scores.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{label}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

    return entity_scores
```

## Best Practices

### Model Selection

| Scenario | Recommended Model |
|----------|------------------|
| Quick prototyping | en_core_web_sm |
| Production (speed priority) | en_core_web_md |
| Production (accuracy priority) | en_core_web_lg |
| Maximum accuracy | en_core_web_trf |
| Domain-specific | Fine-tuned custom |

### Performance Optimization

```python
import spacy

# Disable unnecessary components
nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

# Use blank model with only NER for inference
nlp = spacy.load("en_core_web_lg")
nlp_ner_only = spacy.blank("en")
nlp_ner_only.add_pipe("ner", source=nlp)

# Increase batch size for throughput
docs = list(nlp.pipe(texts, batch_size=1000))

# Use GPU if available
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
```

### Error Analysis

```python
def analyze_errors(nlp, test_data):
    """Categorize NER errors."""
    errors = {
        "false_positives": [],  # Predicted but not in gold
        "false_negatives": [],  # In gold but not predicted
        "wrong_type": [],       # Correct span, wrong label
        "partial_match": []     # Overlapping but not exact
    }

    for text, annotations in test_data:
        doc = nlp(text)
        pred_ents = [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]
        gold_ents = [(s, e, l, text[s:e]) for s, e, l in annotations["entities"]]

        pred_spans = {(s, e) for s, e, _, _ in pred_ents}
        gold_spans = {(s, e) for s, e, _, _ in gold_ents}

        for ent in pred_ents:
            span = (ent[0], ent[1])
            if span not in gold_spans:
                # Check for overlaps
                overlaps = [g for g in gold_ents if not (ent[1] <= g[0] or ent[0] >= g[1])]
                if overlaps:
                    errors["partial_match"].append((ent, overlaps))
                else:
                    errors["false_positives"].append(ent)
            else:
                # Check label match
                gold_label = next(g[2] for g in gold_ents if (g[0], g[1]) == span)
                if ent[2] != gold_label:
                    errors["wrong_type"].append((ent, gold_label))

        for ent in gold_ents:
            if (ent[0], ent[1]) not in pred_spans:
                errors["false_negatives"].append(ent)

    return errors
```
