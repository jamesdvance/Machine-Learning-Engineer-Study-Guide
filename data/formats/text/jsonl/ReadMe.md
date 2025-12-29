# JSONL (JSON Lines)

## Summary

JSONL (JSON Lines), also known as newline-delimited JSON (NDJSON), is a text format where each line is a valid JSON object. This simple modification to JSON enables streaming processing, efficient appending, and parallel processing of large datasets. JSONL has become the standard format for ML training data, log files, and data pipelines.

Key points to remember:

- Each line is a complete, valid JSON object
- No commas between lines, no enclosing array brackets
- Enables streaming: process line-by-line without loading entire file
- Append-friendly: add new records without rewriting file
- Easily parallelizable: split file by lines
- Standard for OpenAI fine-tuning, Hugging Face datasets, and many ML tools
- Compared to JSON arrays, JSONL is streamable and append-friendly
- Compared to CSV, JSONL supports nested structures and mixed types

## Format

### Structure

```jsonl
{"id": 1, "name": "Alice", "scores": [95, 87]}
{"id": 2, "name": "Bob", "scores": [82, 91]}
{"id": 3, "name": "Carol", "scores": [88, 94]}
```

Rules:
- One JSON object per line
- Lines separated by `\n` (newline)
- No trailing commas
- No enclosing brackets
- Each line must be valid JSON independently

### Comparison with JSON Array

```json
// JSON Array (single document)
[
  {"id": 1, "name": "Alice"},
  {"id": 2, "name": "Bob"},
  {"id": 3, "name": "Carol"}
]
```

```jsonl
// JSONL (streaming format)
{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Carol"}
```

## Python Usage

### Reading JSONL

```python
import json

# Read all lines
def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Stream lines (memory efficient)
def stream_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                yield json.loads(line)

# Usage
for record in stream_jsonl("data.jsonl"):
    process(record)
```

### Writing JSONL

```python
import json

def write_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Append to existing file
def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### Using orjson (Faster)

```python
import orjson

def stream_jsonl_fast(path: str):
    with open(path, "rb") as f:  # Binary mode for orjson
        for line in f:
            if line.strip():
                yield orjson.loads(line)

def write_jsonl_fast(path: str, records: list[dict]):
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")
```

### Pandas Integration

```python
import pandas as pd

# Read JSONL to DataFrame
df = pd.read_json("data.jsonl", lines=True)

# Write DataFrame to JSONL
df.to_json("output.jsonl", orient="records", lines=True)

# Chunked reading for large files
chunks = pd.read_json("large.jsonl", lines=True, chunksize=10000)
for chunk in chunks:
    process(chunk)
```

### With Compression

```python
import gzip
import json

# Read gzipped JSONL
def read_jsonl_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# Write gzipped JSONL
def write_jsonl_gz(path: str, records: list[dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
```

## ML Applications

### OpenAI Fine-Tuning Format

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

```python
# Prepare fine-tuning data
def create_fine_tuning_example(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

examples = [
    create_fine_tuning_example("Summarize this:", "Here is the summary..."),
    create_fine_tuning_example("Translate to French:", "Voici la traduction..."),
]

write_jsonl("fine_tuning_data.jsonl", examples)
```

### Hugging Face Datasets

```python
from datasets import load_dataset

# Load JSONL file
dataset = load_dataset("json", data_files="train.jsonl")

# Load multiple splits
dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl"
})

# Save as JSONL
dataset["train"].to_json("output.jsonl")
```

### Training Data Format

```jsonl
{"text": "The quick brown fox", "label": "positive"}
{"text": "A slow gray elephant", "label": "neutral"}
{"text": "The angry red bird", "label": "negative"}
```

```python
# Load training data
def load_training_data(path: str):
    texts, labels = [], []
    for record in stream_jsonl(path):
        texts.append(record["text"])
        labels.append(record["label"])
    return texts, labels
```

### Embedding Storage

```jsonl
{"id": "doc1", "text": "Document content...", "embedding": [0.1, 0.2, 0.3, ...]}
{"id": "doc2", "text": "Another document...", "embedding": [0.4, 0.5, 0.6, ...]}
```

### Experiment Logging

```jsonl
{"epoch": 1, "loss": 2.34, "accuracy": 0.65, "timestamp": "2024-01-15T10:00:00"}
{"epoch": 2, "loss": 1.89, "accuracy": 0.72, "timestamp": "2024-01-15T10:15:00"}
{"epoch": 3, "loss": 1.45, "accuracy": 0.78, "timestamp": "2024-01-15T10:30:00"}
```

## Parallel Processing

### Split by Lines

```bash
# Split large JSONL file into chunks
split -l 100000 large.jsonl chunk_

# Process chunks in parallel
parallel python process.py {} ::: chunk_*
```

### Python Multiprocessing

```python
from multiprocessing import Pool
import json

def process_chunk(lines: list[str]) -> list[dict]:
    return [process(json.loads(line)) for line in lines]

def parallel_process_jsonl(path: str, num_workers: int = 4):
    with open(path) as f:
        lines = f.readlines()

    chunk_size = len(lines) // num_workers
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]

    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    return [item for sublist in results for item in sublist]
```

### Line-Based Sharding

```python
def shard_jsonl(input_path: str, num_shards: int):
    """Split JSONL into multiple shards."""
    writers = [
        open(f"shard_{i}.jsonl", "w")
        for i in range(num_shards)
    ]

    try:
        with open(input_path) as f:
            for i, line in enumerate(f):
                writers[i % num_shards].write(line)
    finally:
        for w in writers:
            w.close()
```

## Validation

### Schema Validation

```python
from jsonschema import validate, ValidationError
import json

record_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "text": {"type": "string"},
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]}
    },
    "required": ["id", "text", "label"]
}

def validate_jsonl(path: str, schema: dict):
    errors = []
    for i, record in enumerate(stream_jsonl(path), 1):
        try:
            validate(record, schema)
        except ValidationError as e:
            errors.append({"line": i, "error": e.message})
    return errors
```

### Data Quality Checks

```python
def check_jsonl_quality(path: str) -> dict:
    stats = {
        "total_lines": 0,
        "empty_lines": 0,
        "parse_errors": 0,
        "valid_records": 0,
    }

    with open(path) as f:
        for i, line in enumerate(f, 1):
            stats["total_lines"] += 1

            if not line.strip():
                stats["empty_lines"] += 1
                continue

            try:
                json.loads(line)
                stats["valid_records"] += 1
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                print(f"Parse error on line {i}")

    return stats
```

## Command Line Tools

### jq (Recommended)

```bash
# Pretty print first record
head -1 data.jsonl | jq .

# Extract field from all records
cat data.jsonl | jq -r '.name'

# Filter records
cat data.jsonl | jq -c 'select(.score > 0.9)'

# Transform records
cat data.jsonl | jq -c '{id, text}'

# Count records
wc -l data.jsonl
```

### Standard Unix Tools

```bash
# First N records
head -n 10 data.jsonl

# Last N records
tail -n 10 data.jsonl

# Random sample
shuf -n 1000 data.jsonl > sample.jsonl

# Count by field value
cat data.jsonl | jq -r '.label' | sort | uniq -c

# Concatenate files
cat part1.jsonl part2.jsonl > combined.jsonl
```

## Comparison with Alternatives

### JSONL vs JSON Array

| Aspect | JSONL | JSON Array |
|--------|-------|------------|
| Streaming | Yes | No |
| Append | Easy | Requires rewrite |
| Parallel processing | Simple | Complex |
| Partial read | Yes | No |
| File corruption | Affects one line | Affects entire file |
| Compression | Good | Similar |

### JSONL vs CSV

| Aspect | JSONL | CSV |
|--------|-------|-----|
| Nested data | Yes | No |
| Mixed types | Yes | No (all strings) |
| Schema | Implicit | Header row |
| Human readable | Good | Good |
| Parsing complexity | Higher | Lower |
| Size | Larger | Smaller |
| Column selection | Parse all | Can skip |

### JSONL vs Parquet

| Aspect | JSONL | Parquet |
|--------|-------|---------|
| Human readable | Yes | No (binary) |
| Compression | Moderate | Excellent |
| Column selection | No | Yes |
| Schema | Optional | Required |
| Streaming write | Yes | No |
| Ecosystem | Universal | Analytics-focused |

## Best Practices

### Consistent Schema

```python
# Good: consistent structure
{"id": 1, "name": "Alice", "score": 0.95}
{"id": 2, "name": "Bob", "score": 0.87}

# Avoid: varying structure
{"id": 1, "name": "Alice", "score": 0.95}
{"id": 2, "user_name": "Bob"}  # Different key
```

### Include Metadata

```jsonl
{"_meta": {"version": "1.0", "created": "2024-01-15"}}
{"id": 1, "text": "First record..."}
{"id": 2, "text": "Second record..."}
```

### Handle Empty Lines

```python
# Robust reading
for line in f:
    line = line.strip()
    if not line:
        continue
    record = json.loads(line)
```

### Use Compression for Large Files

```python
# .jsonl.gz is common convention
with gzip.open("data.jsonl.gz", "wt") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
```

## When to Use JSONL

JSONL is well-suited for:
- ML training datasets
- Log files and event streams
- Data pipeline intermediates
- Fine-tuning data (OpenAI, etc.)
- Streaming data processing
- Append-heavy workloads

Consider alternatives when:
- Columnar access patterns (use Parquet)
- Schema enforcement needed (use Avro)
- Simple tabular data (use CSV)
- Single document (use JSON)
- Binary data (use appropriate format)
