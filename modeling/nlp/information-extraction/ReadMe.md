# Information Extraction

## Summary

Information Extraction (IE) is the task of automatically extracting structured information from unstructured text. It transforms free-form text into structured data that can be stored in databases, used for knowledge graph construction, or queried programmatically. IE encompasses several related tasks including named entity recognition, relation extraction, and event extraction, which together enable comprehensive understanding of text content.

Key points to remember:

- IE pipeline: NER identifies entities, RE finds relationships, EE captures events
- Structured output: Transforms text into queryable facts and knowledge
- Knowledge graphs: IE populates nodes (entities) and edges (relations)
- Domain adaptation: General models often need fine-tuning for specialized text
- Joint vs pipeline: Trade-off between error propagation and model complexity
- Evaluation: Entity-level F1, relation triplet accuracy, event argument matching
- Applications: Question answering, search, business intelligence, scientific discovery

## IE Pipeline

### Traditional Pipeline

```
Raw Text
    |
    v
+------------------+
| Named Entity     |  Identify: PERSON, ORG, LOC, DATE, etc.
| Recognition      |
+------------------+
    |
    v
+------------------+
| Coreference      |  Link mentions: "Apple" = "the company" = "it"
| Resolution       |
+------------------+
    |
    v
+------------------+
| Relation         |  Extract: (Tim Cook, CEO_of, Apple)
| Extraction       |
+------------------+
    |
    v
+------------------+
| Event            |  Detect: Acquisition(buyer=X, target=Y, price=Z)
| Extraction       |
+------------------+
    |
    v
Structured Knowledge
```

### Example Transformation

```
Input Text:
"Apple CEO Tim Cook announced on Tuesday that the company acquired
Beats Electronics for $3 billion. The deal was finalized in Cupertino."

Extracted Information:

Entities:
- Apple (ORG)
- Tim Cook (PERSON)
- Tuesday (DATE)
- Beats Electronics (ORG)
- $3 billion (MONEY)
- Cupertino (LOC)

Relations:
- (Tim Cook, CEO_of, Apple)
- (Apple, acquired, Beats Electronics)
- (Apple, headquartered_in, Cupertino)

Events:
- Announcement(speaker=Tim Cook, time=Tuesday)
- Acquisition(buyer=Apple, target=Beats Electronics, price=$3 billion, place=Cupertino)
```

## Comparison: Relation vs Event Extraction

| Aspect | Relation Extraction | Event Extraction |
|--------|---------------------|------------------|
| Focus | Static relationships between entities | Dynamic occurrences with participants |
| Output | Binary/n-ary relations | Events with typed arguments |
| Temporal | Usually atemporal | Inherently temporal |
| Trigger | No explicit trigger needed | Requires trigger identification |
| Complexity | Simpler structure | More complex structure |
| Example | (Jobs, founded, Apple) | Founding(founder=Jobs, org=Apple, date=1976) |

### When to Use Each

**Relation Extraction:**
- Building knowledge graphs with entity relationships
- Extracting facts from encyclopedic text
- Populating structured databases
- Simple binary relationships sufficient

**Event Extraction:**
- News monitoring and analysis
- Temporal reasoning required
- Multiple participants with distinct roles
- Actions and occurrences matter

## Joint Extraction Approaches

### Benefits of Joint Modeling

1. Reduced error propagation (NER errors don't cascade)
2. Entity and relation information inform each other
3. Shared representations improve efficiency
4. Better handling of overlapping entities/relations

### Joint Entity and Relation Extraction

```python
from transformers import AutoModel
import torch.nn as nn

class JointERModel(nn.Module):
    """Extract entities and relations jointly."""

    def __init__(self, encoder_name, num_entity_types, num_relation_types):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        # Entity head (token classification)
        self.entity_head = nn.Linear(hidden, num_entity_types)

        # Relation head (span pair classification)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_relation_types)
        )

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids, attention_mask).last_hidden_state

        # Entity predictions (per token)
        entity_logits = self.entity_head(encoded)

        # Relation predictions (per entity pair) - computed after entity detection
        return entity_logits
```

### End-to-End with Generation

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class GenerativeIE:
    """Use seq2seq for end-to-end information extraction."""

    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def extract(self, text):
        prompt = f"""Extract all entities, relations, and events from:
{text}

Format:
Entities: [entity1 (type), entity2 (type), ...]
Relations: [(subject, relation, object), ...]
Events: [Event_Type(arg1=value1, arg2=value2), ...]
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Knowledge Graph Construction

### From IE to Knowledge Graph

```python
class KnowledgeGraphBuilder:
    def __init__(self):
        self.entities = {}  # entity_id -> entity_info
        self.relations = []  # list of (head, relation, tail) triplets

    def add_from_ie_output(self, ie_output):
        """Add extracted information to knowledge graph."""
        # Add entities
        for entity in ie_output['entities']:
            entity_id = self._normalize_entity(entity['text'])
            self.entities[entity_id] = {
                'text': entity['text'],
                'type': entity['type'],
                'mentions': self.entities.get(entity_id, {}).get('mentions', []) + [entity]
            }

        # Add relations
        for relation in ie_output['relations']:
            head_id = self._normalize_entity(relation['subject'])
            tail_id = self._normalize_entity(relation['object'])
            self.relations.append((head_id, relation['relation'], tail_id))

    def to_rdf(self):
        """Export as RDF triples."""
        triples = []
        for head, rel, tail in self.relations:
            triples.append(f"<{head}> <{rel}> <{tail}> .")
        return "\n".join(triples)

    def to_neo4j(self):
        """Generate Cypher queries for Neo4j."""
        queries = []
        for entity_id, info in self.entities.items():
            queries.append(f"MERGE (n:{info['type']} {{id: '{entity_id}', name: '{info['text']}'}})")
        for head, rel, tail in self.relations:
            queries.append(f"MATCH (a {{id: '{head}'}}), (b {{id: '{tail}'}}) MERGE (a)-[:{rel}]->(b)")
        return queries
```

## Open Information Extraction

Extract information without predefined schema:

```python
class OpenIE:
    """Schema-free information extraction."""

    def extract_triplets(self, text):
        """
        Extract (subject, predicate, object) triplets without predefined relations.

        Example:
        "Einstein developed the theory of relativity"
        -> (Einstein, developed, the theory of relativity)
        """
        # Using dependency parsing
        import spacy
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)

        triplets = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Find subject
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    # Find object
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]

                    for subj in subjects:
                        for obj in objects:
                            triplets.append({
                                'subject': self._get_span(subj),
                                'predicate': token.text,
                                'object': self._get_span(obj)
                            })

        return triplets
```

### Comparison: Closed vs Open IE

| Aspect | Closed IE | Open IE |
|--------|-----------|---------|
| Schema | Predefined relations | No schema |
| Coverage | Limited to known types | Any relation |
| Precision | Higher | Lower |
| Consistency | Standardized output | Variable |
| Use case | Specific applications | Exploration, QA |

## Evaluation Strategies

### End-to-End Evaluation

```python
def evaluate_ie_pipeline(predictions, ground_truth):
    """Evaluate complete IE pipeline."""

    results = {
        'entity': evaluate_entities(predictions['entities'], ground_truth['entities']),
        'relation': evaluate_relations(predictions['relations'], ground_truth['relations']),
        'event': evaluate_events(predictions['events'], ground_truth['events'])
    }

    # Joint evaluation (relation correct only if both entities correct)
    results['strict_relation'] = evaluate_strict_relations(predictions, ground_truth)

    return results

def evaluate_entities(pred_entities, gold_entities):
    """Entity evaluation with exact and partial matching."""
    exact_matches = 0
    partial_matches = 0

    pred_spans = {(e['start'], e['end'], e['type']) for e in pred_entities}
    gold_spans = {(e['start'], e['end'], e['type']) for e in gold_entities}

    exact_matches = len(pred_spans & gold_spans)

    # Partial: overlapping spans with correct type
    for pred in pred_entities:
        for gold in gold_entities:
            if pred['type'] == gold['type']:
                if spans_overlap(pred, gold) and (pred['start'], pred['end'], pred['type']) not in gold_spans:
                    partial_matches += 1

    return {
        'exact_precision': exact_matches / len(pred_entities) if pred_entities else 0,
        'exact_recall': exact_matches / len(gold_entities) if gold_entities else 0,
        'partial_matches': partial_matches
    }
```

## Domain Adaptation

### Challenges by Domain

| Domain | Challenges | Solutions |
|--------|------------|-----------|
| Biomedical | Complex nomenclature, nested entities | Domain-specific models (BioBERT) |
| Legal | Long documents, cross-references | Document-level models |
| Financial | Numerical reasoning, temporal | Custom entity types |
| Scientific | Technical terms, abbreviations | Specialized vocabularies |

### Transfer Learning Approach

```python
def adapt_ie_to_domain(base_model, domain_data, domain_name):
    """Adapt general IE model to specific domain."""

    # Option 1: Continue pre-training on domain text
    domain_pretrained = continue_pretraining(
        base_model,
        domain_corpus,
        epochs=3
    )

    # Option 2: Fine-tune on domain-specific IE annotations
    domain_finetuned = finetune_ie(
        domain_pretrained,
        domain_data['train'],
        domain_data['dev']
    )

    # Option 3: Few-shot with domain examples
    if len(domain_data['train']) < 100:
        return few_shot_adaptation(base_model, domain_data['train'])

    return domain_finetuned
```

## Production Considerations

### Pipeline vs Joint

| Factor | Pipeline | Joint |
|--------|----------|-------|
| Modularity | Easy to update components | Single model to maintain |
| Error handling | Can recover from component errors | All-or-nothing |
| Interpretability | Intermediate outputs available | Black box |
| Latency | Multiple inference passes | Single pass |
| Training | Separate datasets per task | Requires joint annotations |

### Scaling Strategies

```python
class ScalableIEPipeline:
    def __init__(self):
        self.ner_model = load_ner_model()
        self.re_model = load_re_model()
        self.cache = {}

    def process_batch(self, documents, batch_size=32):
        """Process documents in batches for efficiency."""
        all_results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Batch NER
            ner_results = self.ner_model.predict_batch(batch)

            # Batch RE for all entity pairs
            re_inputs = self._prepare_re_inputs(batch, ner_results)
            re_results = self.re_model.predict_batch(re_inputs)

            # Combine results
            batch_results = self._combine_results(ner_results, re_results)
            all_results.extend(batch_results)

        return all_results

    def process_with_caching(self, document, doc_id):
        """Cache expensive computations."""
        if doc_id in self.cache:
            return self.cache[doc_id]

        result = self.process_batch([document])[0]
        self.cache[doc_id] = result
        return result
```

## Best Practices

### Annotation Guidelines

1. Clear entity boundary rules
2. Relation directionality conventions
3. Handling of nested and overlapping structures
4. Treatment of implicit information
5. Inter-annotator agreement metrics

### Quality Assurance

```python
def validate_ie_output(ie_result):
    """Validate extracted information for consistency."""
    issues = []

    # Check entity validity
    for entity in ie_result['entities']:
        if entity['end'] <= entity['start']:
            issues.append(f"Invalid entity span: {entity}")

    # Check relation validity
    for relation in ie_result['relations']:
        # Ensure entities exist
        entity_texts = [e['text'] for e in ie_result['entities']]
        if relation['subject'] not in entity_texts:
            issues.append(f"Relation subject not in entities: {relation}")
        if relation['object'] not in entity_texts:
            issues.append(f"Relation object not in entities: {relation}")

    # Check event validity
    for event in ie_result['events']:
        if not event.get('trigger'):
            issues.append(f"Event missing trigger: {event}")

    return issues
```

### Choosing the Right Approach

```
Start here:
    |
    v
Predefined schema? ---No---> Open IE
    |
    Yes
    v
Joint annotations available? ---No---> Pipeline approach
    |
    Yes
    v
Complex dependencies? ---Yes---> Joint model
    |
    No
    v
Pipeline (more modular)
```
