# Relation Extraction

## Summary

Relation extraction (RE) is the task of identifying and classifying semantic relationships between entities in text. Given a sentence with identified entities, RE determines how those entities are connected, such as "works_for", "located_in", or "founded_by" relationships. It is essential for knowledge graph construction, question answering, and structured information retrieval from unstructured text.

Key points to remember:

- Pipeline approach: Typically follows NER (first identify entities, then extract relations)
- Joint extraction: Modern approaches extract entities and relations simultaneously
- Relation types: Predefined schema (closed) or open information extraction (open)
- Sentence-level vs document-level: Most work focuses on within-sentence relations
- Distant supervision: Use knowledge bases to automatically label training data
- Neural approaches: BERT-based models achieve state-of-the-art performance
- Output format: Triplets (subject, relation, object) or (head_entity, relation, tail_entity)
- Challenges: Overlapping relations, long-range dependencies, implicit relations

## Task Definition

### Input and Output

```
Input: "Tim Cook, who leads Apple, announced the new iPhone in Cupertino."

Entities:
- Tim Cook (PERSON)
- Apple (ORG)
- iPhone (PRODUCT)
- Cupertino (LOC)

Output Relations:
- (Tim Cook, CEO_of, Apple)
- (Apple, headquartered_in, Cupertino)
- (iPhone, produced_by, Apple)
```

### Relation Types

Common relation categories:

| Category | Examples |
|----------|----------|
| Employment | works_for, founded_by, CEO_of |
| Location | located_in, headquartered_in, born_in |
| Family | spouse_of, parent_of, sibling_of |
| Affiliation | member_of, part_of, subsidiary_of |
| Temporal | founded_in, died_in, occurred_on |
| Creation | authored_by, directed_by, invented_by |

## Approaches

### Pipeline Approach

Extract entities first, then classify relations:

```python
from transformers import pipeline

# Step 1: NER
ner = pipeline("ner", aggregation_strategy="simple")
text = "Elon Musk founded SpaceX in 2002."
entities = ner(text)
# [{'entity_group': 'PER', 'word': 'Elon Musk', ...},
#  {'entity_group': 'ORG', 'word': 'SpaceX', ...},
#  {'entity_group': 'DATE', 'word': '2002', ...}]

# Step 2: For each entity pair, classify relation
def classify_relation(text, entity1, entity2, model):
    # Mark entities in text
    marked_text = mark_entities(text, entity1, entity2)
    # Classify relation type
    return model.predict(marked_text)
```

### Entity Marking Strategies

Different ways to indicate entity positions to the model:

```python
# Strategy 1: Special tokens
"[E1] Elon Musk [/E1] founded [E2] SpaceX [/E2] in 2002."

# Strategy 2: Entity type markers
"<PER> Elon Musk </PER> founded <ORG> SpaceX </ORG> in 2002."

# Strategy 3: Positional markers
"@ Elon Musk @ founded # SpaceX # in 2002."

# Strategy 4: Typed markers with position
"<S:PER> Elon Musk </S:PER> founded <O:ORG> SpaceX </O:ORG> in 2002."
```

### BERT for Relation Extraction

```python
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class BertForRelationExtraction(nn.Module):
    def __init__(self, model_name, num_relations, entity_marker="typed"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # Classification head
        hidden_size = self.bert.config.hidden_size
        if entity_marker == "entity_start":
            # Concatenate start positions of both entities
            self.classifier = nn.Linear(hidden_size * 2, num_relations)
        else:
            # Use [CLS] token
            self.classifier = nn.Linear(hidden_size, num_relations)

        self.entity_marker = entity_marker

    def forward(self, input_ids, attention_mask, entity1_pos=None, entity2_pos=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        if self.entity_marker == "entity_start" and entity1_pos is not None:
            # Get representations at entity start positions
            batch_size = input_ids.size(0)
            e1_repr = sequence_output[range(batch_size), entity1_pos]
            e2_repr = sequence_output[range(batch_size), entity2_pos]
            pooled = torch.cat([e1_repr, e2_repr], dim=-1)
        else:
            # Use [CLS] token
            pooled = outputs.pooler_output

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits
```

### Joint Entity and Relation Extraction

Extract entities and relations in one model:

```python
import torch
import torch.nn as nn
from transformers import BertModel

class JointERModel(nn.Module):
    def __init__(self, model_name, num_entity_types, num_relation_types):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Entity extraction head (sequence labeling)
        self.entity_classifier = nn.Linear(hidden_size, num_entity_types)

        # Relation extraction head (for each entity pair)
        self.relation_classifier = nn.Linear(hidden_size * 2, num_relation_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Entity predictions
        entity_logits = self.entity_classifier(sequence_output)

        # For relation extraction, enumerate entity pairs
        # (Simplified - production would use predicted entities)
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Create pairwise representations
        # Each position i paired with each position j
        expanded_i = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        expanded_j = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pair_repr = torch.cat([expanded_i, expanded_j], dim=-1)

        # Relation predictions for all pairs
        relation_logits = self.relation_classifier(pair_repr)

        return entity_logits, relation_logits
```

## Using Pre-trained Models

### Hugging Face RE Models

```python
from transformers import pipeline

# Load relation extraction model
re_pipeline = pipeline(
    "text-classification",
    model="Babelscape/rebel-large"
)

text = "Elon Musk is the CEO of Tesla and SpaceX."
result = re_pipeline(text)
```

### REBEL: Relation Extraction By End-to-end Language Generation

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")

def extract_relations(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
        num_return_sequences=1
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return parse_rebel_output(decoded)

def parse_rebel_output(text):
    """Parse REBEL's triplet format."""
    triplets = []
    # REBEL outputs: <triplet> subject <subj> relation <obj> object
    # Parse accordingly
    return triplets

text = "Barack Obama was born in Honolulu and served as the 44th President."
relations = extract_relations(text)
```

### OpenNRE

```python
import opennre

# Load pre-trained model
model = opennre.get_model('wiki80_bert_softmax')

# Extract relation
result = model.infer({
    'text': 'Bill Gates founded Microsoft.',
    'h': {'pos': (0, 10)},   # head entity position
    't': {'pos': (19, 28)}   # tail entity position
})

print(f"Relation: {result[0]}, Confidence: {result[1]:.4f}")
```

## Distant Supervision

Automatically generate training data using knowledge bases:

```python
def distant_supervision(text, knowledge_base):
    """
    Label sentences using knowledge base facts.

    If sentence contains two entities with known relation,
    assume sentence expresses that relation.
    """
    entities = extract_entities(text)
    training_examples = []

    for e1, e2 in itertools.combinations(entities, 2):
        # Check if relation exists in KB
        relation = knowledge_base.get_relation(e1, e2)

        if relation:
            training_examples.append({
                'text': text,
                'entity1': e1,
                'entity2': e2,
                'relation': relation
            })
        else:
            # Negative example (no known relation)
            training_examples.append({
                'text': text,
                'entity1': e1,
                'entity2': e2,
                'relation': 'no_relation'
            })

    return training_examples

# Example with Wikidata
from SPARQLWrapper import SPARQLWrapper

def query_wikidata_relation(entity1, entity2):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?relation WHERE {{
        wd:{entity1} ?relation wd:{entity2} .
    }}
    """
    sparql.setQuery(query)
    results = sparql.query().convert()
    return results
```

### Noise Reduction in Distant Supervision

Distant supervision is noisy (wrong labels). Mitigation strategies:

```python
# Multi-instance learning
# Group sentences mentioning same entity pair
# Assume at least one expresses the relation

class MultiInstanceBag:
    def __init__(self, entity_pair, relation, sentences):
        self.entity_pair = entity_pair
        self.relation = relation
        self.sentences = sentences

    def aggregate_prediction(self, model):
        # Predict for all sentences in bag
        predictions = [model.predict(s) for s in self.sentences]

        # Aggregate (e.g., max, attention-weighted)
        return max(predictions, key=lambda x: x['confidence'])
```

## Document-Level Relation Extraction

Relations spanning multiple sentences:

```python
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class DocREModel(nn.Module):
    """Document-level relation extraction with cross-sentence reasoning."""

    def __init__(self, model_name, num_relations):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Entity-level attention
        self.entity_attention = nn.MultiheadAttention(hidden_size, num_heads=8)

        # Relation classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations)
        )

    def forward(self, input_ids, attention_mask, entity_positions):
        # Encode full document
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Get entity representations (average of entity tokens)
        entity_reprs = self.get_entity_representations(hidden_states, entity_positions)

        # Cross-entity attention for document-level reasoning
        entity_reprs, _ = self.entity_attention(
            entity_reprs, entity_reprs, entity_reprs
        )

        # Classify relations between all entity pairs
        num_entities = entity_reprs.size(1)
        relations = []

        for i in range(num_entities):
            for j in range(num_entities):
                if i != j:
                    pair_repr = torch.cat([entity_reprs[:, i], entity_reprs[:, j]], dim=-1)
                    relation_logits = self.classifier(pair_repr)
                    relations.append((i, j, relation_logits))

        return relations
```

## Evaluation

### Metrics

```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_relation_extraction(predictions, ground_truth):
    """
    Evaluate RE performance.

    predictions: List of (subject, relation, object) tuples
    ground_truth: List of (subject, relation, object) tuples
    """
    # Exact match
    pred_set = set(predictions)
    gold_set = set(ground_truth)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Per-relation evaluation
def evaluate_per_relation(predictions, ground_truth, relation_types):
    results = {}
    for relation in relation_types:
        pred_rel = [p for p in predictions if p[1] == relation]
        gold_rel = [g for g in ground_truth if g[1] == relation]
        results[relation] = evaluate_relation_extraction(pred_rel, gold_rel)
    return results
```

### Common Benchmarks

| Dataset | Domain | Relations | Instances | Notes |
|---------|--------|-----------|-----------|-------|
| TACRED | News | 42 | 106K | Sentence-level, TAC KBP |
| SemEval-2010 Task 8 | General | 19 | 10K | Semantic relations |
| DocRED | Wikipedia | 96 | 63K | Document-level |
| NYT | News | 24 | 56K | Distant supervision |
| FewRel | Wikipedia | 100 | 70K | Few-shot learning |

## Challenges and Solutions

### Overlapping Relations

Multiple relations for same entity pair:

```
"Steve Jobs co-founded Apple and later became its CEO."

Entity pair (Steve Jobs, Apple):
- founded_by
- CEO_of
```

Solution: Multi-label classification instead of multi-class

```python
class MultiLabelREModel(nn.Module):
    def __init__(self, hidden_size, num_relations):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_relations)
        # Use BCEWithLogitsLoss for multi-label

    def forward(self, x):
        return self.classifier(x)  # Apply sigmoid at inference
```

### No Relation (NA) Class

Most entity pairs have no relation:

```python
# Handle class imbalance
from torch.nn import CrossEntropyLoss

# Weight NA class lower
weights = torch.ones(num_relations)
weights[na_class_idx] = 0.1  # Reduce weight of NA class
loss_fn = CrossEntropyLoss(weight=weights)

# Or use focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

### Long-Range Dependencies

Entities far apart in text:

```python
# Use longer context models
from transformers import LongformerModel

class LongDocREModel(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        # Can handle up to 4096 tokens
```

## Open Information Extraction

Extract relations without predefined schema:

```python
# Using OpenIE tools
import subprocess

def stanford_openie(text):
    """Extract open-domain triplets using Stanford OpenIE."""
    # Requires Stanford CoreNLP
    result = subprocess.run(
        ['java', '-mx4g', '-cp', '*', 'edu.stanford.nlp.naturalli.OpenIE'],
        input=text,
        capture_output=True,
        text=True
    )
    return parse_openie_output(result.stdout)

# Example output:
# "Barack Obama was born in Hawaii"
# -> (Barack Obama, was born in, Hawaii)
```

### Neural Open IE

```python
from transformers import pipeline

# Use generative model for open extraction
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def extract_open_relations(text):
    prompt = f"""Extract all relationships from the following text as triplets.
    Format: (subject, relation, object)

    Text: {text}

    Triplets:"""

    result = generator(prompt, max_length=200)
    return parse_triplets(result[0]['generated_text'])
```

## Best Practices

### Data Preparation

```python
def prepare_re_data(text, entities, relations):
    """Prepare data for relation extraction training."""
    examples = []

    for rel in relations:
        # Get entity positions
        e1_pos = find_entity_position(text, rel['subject'])
        e2_pos = find_entity_position(text, rel['object'])

        # Mark entities in text
        marked_text = insert_entity_markers(text, e1_pos, e2_pos)

        examples.append({
            'text': marked_text,
            'label': rel['relation'],
            'e1_pos': e1_pos,
            'e2_pos': e2_pos
        })

    # Add negative examples (entity pairs with no relation)
    for e1, e2 in get_entity_pairs_without_relation(entities, relations):
        marked_text = insert_entity_markers(text, e1['pos'], e2['pos'])
        examples.append({
            'text': marked_text,
            'label': 'no_relation',
            'e1_pos': e1['pos'],
            'e2_pos': e2['pos']
        })

    return examples
```

### Model Selection

| Scenario | Recommended Approach |
|----------|---------------------|
| Standard benchmark | Fine-tuned BERT with entity markers |
| Document-level | Longformer or hierarchical model |
| Few training examples | Few-shot learning (FewRel-style) |
| Open domain | Generative models (T5, REBEL) |
| Production speed | Distilled models or pipeline approach |

### Error Analysis

```python
def analyze_re_errors(predictions, ground_truth):
    """Categorize relation extraction errors."""
    errors = {
        'false_positive': [],      # Predicted but wrong
        'false_negative': [],      # Missed relation
        'wrong_direction': [],     # Correct relation, wrong entity order
        'similar_relation': []     # Confused with related relation type
    }

    pred_dict = {(p[0], p[2]): p[1] for p in predictions}
    gold_dict = {(g[0], g[2]): g[1] for g in ground_truth}

    for (s, o), rel in pred_dict.items():
        if (s, o) in gold_dict:
            if gold_dict[(s, o)] != rel:
                errors['similar_relation'].append(((s, rel, o), gold_dict[(s, o)]))
        elif (o, s) in gold_dict:
            errors['wrong_direction'].append((s, rel, o))
        else:
            errors['false_positive'].append((s, rel, o))

    for (s, o), rel in gold_dict.items():
        if (s, o) not in pred_dict and (o, s) not in pred_dict:
            errors['false_negative'].append((s, rel, o))

    return errors
```
