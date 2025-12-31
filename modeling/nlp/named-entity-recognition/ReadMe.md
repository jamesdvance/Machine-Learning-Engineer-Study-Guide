# Named Entity Recognition

## Summary

Named Entity Recognition (NER) is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, and more. NER is a foundational NLP task that enables downstream applications like information extraction, question answering, knowledge graph construction, and search enhancement.

Key points to remember:

- Sequence labeling task: Classify each token into entity categories
- Standard entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, etc.
- Label schemes: BIO (Begin-Inside-Outside), BIOES, IOB2
- Approaches: Rule-based, CRF, BiLSTM-CRF, Transformer-based
- Pre-trained models widely available for common entity types
- Custom training needed for domain-specific entities (medical, legal, financial)
- Evaluation: Entity-level precision, recall, F1 using seqeval library
- Challenges: Nested entities, entity boundary detection, rare entities

## Entity Types

### Standard Entity Categories

| Type | Description | Examples |
|------|-------------|----------|
| PERSON (PER) | People's names | "Barack Obama", "Marie Curie" |
| ORGANIZATION (ORG) | Companies, agencies, institutions | "Google", "United Nations" |
| LOCATION (LOC) | Physical locations, landmarks | "Mount Everest", "Pacific Ocean" |
| GPE | Geopolitical entities | "United States", "Paris" |
| DATE | Dates and time periods | "January 2024", "next week" |
| TIME | Times of day | "3:00 PM", "morning" |
| MONEY | Monetary values | "$50 million", "20 euros" |
| PERCENT | Percentages | "25%", "fifty percent" |
| PRODUCT | Products | "iPhone", "Windows 11" |
| EVENT | Named events | "World Cup", "Olympics" |

### Domain-Specific Entities

| Domain | Custom Entities |
|--------|-----------------|
| Medical | DISEASE, DRUG, SYMPTOM, TREATMENT |
| Legal | CASE, STATUTE, COURT, JUDGE |
| Financial | TICKER, EXCHANGE, FINANCIAL_METRIC |
| Scientific | CHEMICAL, GENE, PROTEIN, SPECIES |

## Label Schemes

### BIO (Begin-Inside-Outside)

Most common scheme:

```
Tim     Cook    is    Apple's   CEO
B-PER   I-PER   O     B-ORG     O
```

- B-XXX: Beginning of entity type XXX
- I-XXX: Inside (continuation) of entity type XXX
- O: Outside any entity

### BIOES (BIOLU)

More granular scheme:

```
Tim     Cook    is    Apple's   CEO
B-PER   E-PER   O     S-ORG     O
```

- B: Beginning
- I: Inside
- O: Outside
- E: End of entity
- S: Single-token entity

### IOB2

Variant where B is used at every entity start:

```
Tim     Cook    is    Apple's   CEO    of    Microsoft
B-PER   I-PER   O     B-ORG     O      O     B-ORG
```

## Tool Comparison

### SpaCy

Industrial-strength NLP library with fast, production-ready NER.

```python
import spacy
nlp = spacy.load("en_core_web_lg")
doc = nlp("Apple CEO Tim Cook announced new products.")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

**Pros:**
- Fast inference
- Production-ready
- Extensive pipeline (tokenizer, POS, dependencies, NER)
- Rule-based matching (EntityRuler)

**Cons:**
- Less flexible for research
- Smaller model variety than Hugging Face

### Flair

Research-oriented library with contextual string embeddings.

```python
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-large")
sentence = Sentence("Apple CEO Tim Cook announced new products.")
tagger.predict(sentence)
```

**Pros:**
- State-of-the-art accuracy
- Stacked embeddings (combine multiple types)
- Few-shot learning with TARS

**Cons:**
- Slower than SpaCy
- Higher memory usage

### Hugging Face Transformers

Flexible framework for transformer-based NER.

```python
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
entities = ner("Apple CEO Tim Cook announced new products.")
```

**Pros:**
- Largest model selection
- Easy fine-tuning
- State-of-the-art performance

**Cons:**
- Requires GPU for speed
- Subword alignment complexity

### Comparison Matrix

| Feature | SpaCy | Flair | Transformers |
|---------|-------|-------|--------------|
| Speed | Fast | Medium | Medium-Slow |
| Accuracy | Good | Excellent | Excellent |
| Ease of use | High | High | Medium |
| Custom training | Good | Good | Excellent |
| Model variety | Limited | Good | Extensive |
| Production ready | Excellent | Good | Good |
| Memory | Low | High | Medium |
| GPU required | No | Recommended | Recommended |

## When to Use Each Tool

**Choose SpaCy when:**
- Speed is priority
- Full NLP pipeline needed
- Production deployment
- Limited compute resources

**Choose Flair when:**
- Maximum accuracy needed
- Research/experimentation
- Combining embedding types
- Few-shot scenarios

**Choose Transformers when:**
- Fine-tuning on custom data
- Using specific pre-trained model
- Need flexibility
- Multilingual requirements

## Training Custom NER

### Data Preparation

Common formats:
- CoNLL: Tab-separated tokens and labels
- JSON/JSONL: Structured with spans
- SpaCy format: DocBin binary

```
# CoNLL format
Apple   B-ORG
CEO     O
Tim     B-PER
Cook    I-PER
announced   O
new     O
products    O
.       O
```

### Training Considerations

| Factor | Recommendation |
|--------|----------------|
| Data size | Minimum 1,000 examples per entity type |
| Class balance | Ensure all entity types represented |
| Label quality | Consistent annotation guidelines |
| Negative examples | Include sentences without entities |
| Context variety | Diverse sentence structures |

### Fine-tuning Steps

1. Prepare annotated data in required format
2. Split into train/dev/test (80/10/10)
3. Choose base model appropriate for domain
4. Configure training parameters (lr, epochs, batch size)
5. Train with early stopping based on dev F1
6. Evaluate on held-out test set
7. Error analysis and iteration

## Evaluation

### Metrics

```python
from seqeval.metrics import classification_report, f1_score

y_true = [["O", "B-PER", "I-PER", "O", "B-ORG"]]
y_pred = [["O", "B-PER", "I-PER", "O", "B-ORG"]]

# Overall F1
f1 = f1_score(y_true, y_pred)

# Detailed report
print(classification_report(y_true, y_pred))
```

### Evaluation Modes

| Mode | Description |
|------|-------------|
| Strict | Exact match (start, end, type) required |
| Partial | Overlapping spans count as partial match |
| Type-only | Correct type regardless of boundaries |

### Common Benchmarks

| Dataset | Language | Entities | Size |
|---------|----------|----------|------|
| CoNLL-2003 | English | 4 types | 22K sentences |
| OntoNotes 5.0 | English | 18 types | 77K sentences |
| WNUT-17 | English (Twitter) | 6 types | 5K tweets |
| MultiCoNER | Multilingual | 6 types | 9 languages |

## Challenges and Solutions

### Nested Entities

Entities can overlap:

```
"Bank of America" contains:
- "Bank of America" (ORG)
- "America" (LOC)
```

Solutions:
- Layered models (run NER at different granularities)
- Span-based approaches
- Multi-label token classification

### Rare Entities

Low-frequency entity types have poor recall.

Solutions:
- Oversampling rare entity examples
- Data augmentation
- Few-shot learning (TARS, SetFit)
- External knowledge (gazetteers, knowledge bases)

### Entity Boundary Detection

Difficulty determining where entity starts/ends.

Solutions:
- CRF layer for sequence consistency
- BIOES scheme (explicit end markers)
- Span-based rather than token-based

### Domain Adaptation

General models perform poorly on specialized text.

Solutions:
- Fine-tune on domain data
- Domain-specific pre-training
- Entity lists/gazetteers
- Active learning

## Best Practices

### Annotation Guidelines

1. Define clear entity types with examples and edge cases
2. Handle ambiguous cases consistently
3. Train annotators and measure inter-annotator agreement
4. Regular annotation review sessions

### Model Selection

```
Start here:
    |
    v
Need speed? ---Yes---> SpaCy (en_core_web_lg)
    |
    No
    v
Need maximum accuracy? ---Yes---> Flair or fine-tuned Transformer
    |
    No
    v
Standard entities sufficient? ---Yes---> Pre-trained model
    |
    No
    v
Fine-tune on domain data
```

### Production Deployment

1. Choose model size based on latency requirements
2. Batch processing for throughput
3. GPU for transformer models
4. Monitor entity distribution for drift
5. Regular retraining on new data
6. Confidence thresholds for low-quality predictions

## Integration with Downstream Tasks

NER feeds into:

- **Information Extraction**: Extract relationships between entities
- **Question Answering**: Identify answer spans
- **Knowledge Graphs**: Populate nodes and edges
- **Search**: Entity-based filtering and boosting
- **Summarization**: Ensure key entities are preserved
- **Machine Translation**: Preserve entity translations
