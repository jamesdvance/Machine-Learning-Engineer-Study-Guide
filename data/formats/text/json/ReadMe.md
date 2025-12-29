# JSON

## Summary

JSON (JavaScript Object Notation) is a lightweight, human-readable data interchange format that has become the de facto standard for web APIs and configuration files. Its simplicity and universal support across programming languages make it ubiquitous in ML pipelines for configuration, metadata, API responses, and small-to-medium datasets.

Key points to remember:

- Human-readable text format with key-value pairs and arrays
- Native support in virtually every programming language
- Schema-less by default, though JSON Schema can add validation
- Not optimized for large datasets (no compression, verbose)
- Single JSON objects must fit in memory for parsing
- Use JSONL (JSON Lines) for streaming large datasets
- Compared to XML, JSON is more compact and easier to parse
- Compared to Parquet/Avro, JSON lacks schema enforcement and compression

## Structure

### Data Types

JSON supports six primitive types:

```json
{
  "string": "Hello, World",
  "number": 42,
  "float": 3.14159,
  "boolean": true,
  "null": null,
  "array": [1, 2, 3],
  "object": {"nested": "value"}
}
```

Type details:
- Strings: UTF-8 encoded, double-quoted
- Numbers: Integer or floating-point (no distinction)
- Booleans: `true` or `false` (lowercase)
- Null: `null` (lowercase)
- Arrays: Ordered lists of any type
- Objects: Unordered key-value maps

### Nesting

JSON supports arbitrary nesting:

```json
{
  "user": {
    "name": "Alice",
    "addresses": [
      {
        "type": "home",
        "city": "Seattle",
        "coordinates": {
          "lat": 47.6062,
          "lon": -122.3321
        }
      }
    ]
  }
}
```

## Python Usage

### Reading JSON

```python
import json

# From string
data = json.loads('{"name": "Alice", "age": 30}')

# From file
with open("data.json", "r") as f:
    data = json.load(f)

# With encoding
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

### Writing JSON

```python
import json

data = {"name": "Alice", "scores": [95, 87, 92]}

# To string
json_str = json.dumps(data)

# Pretty-printed
json_str = json.dumps(data, indent=2)

# To file
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)

# Handle non-ASCII
json.dumps(data, ensure_ascii=False)
```

### Common Options

```python
json.dumps(data,
    indent=2,              # Pretty print with indentation
    sort_keys=True,        # Alphabetically sort keys
    ensure_ascii=False,    # Allow non-ASCII characters
    default=str,           # Convert non-serializable to string
    separators=(",", ":"), # Compact output (no spaces)
)
```

### Custom Serialization

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

# Custom encoder
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

# Usage
data = {"timestamp": datetime.now()}
json.dumps(data, cls=CustomEncoder)

# Dataclasses
@dataclass
class User:
    name: str
    age: int

user = User("Alice", 30)
json.dumps(asdict(user))
```

### Custom Deserialization

```python
import json
from datetime import datetime

def object_hook(d):
    for key, value in d.items():
        if key.endswith("_at") and isinstance(value, str):
            try:
                d[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    return d

data = json.loads(
    '{"created_at": "2024-01-15T10:30:00"}',
    object_hook=object_hook
)
# data["created_at"] is now a datetime object
```

## Performance Considerations

### Standard Library Limitations

Python's `json` module is pure Python and relatively slow:

```python
import json
import time

# Benchmark
data = [{"id": i, "value": f"item_{i}"} for i in range(100000)]

start = time.time()
json_str = json.dumps(data)
print(f"json.dumps: {time.time() - start:.3f}s")
```

### Faster Alternatives

**orjson (Recommended)**
```python
import orjson

# 10x faster than stdlib
data = orjson.loads(json_bytes)
json_bytes = orjson.dumps(data)

# Returns bytes, not str
json_str = orjson.dumps(data).decode("utf-8")

# Options
orjson.dumps(data, option=orjson.OPT_INDENT_2)
orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
```

**ujson**
```python
import ujson

# 3-5x faster than stdlib
data = ujson.loads(json_str)
json_str = ujson.dumps(data)
```

**Benchmark Comparison**
```
Library     | Serialize | Deserialize
------------|-----------|------------
json        | 1.0x      | 1.0x
ujson       | 3-5x      | 2-4x
orjson      | 6-10x     | 3-6x
```

### Memory Efficiency

JSON parsing loads entire document into memory:

```python
# Problem: large file loads entirely into memory
with open("large.json") as f:
    data = json.load(f)  # Memory spike

# Solution: use JSONL for large datasets
# Or stream with ijson
import ijson

with open("large.json", "rb") as f:
    for item in ijson.items(f, "items.item"):
        process(item)  # Streaming, low memory
```

## JSON Schema

### Validation

```python
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
    "additionalProperties": False
}

data = {"name": "Alice", "age": 30}

try:
    validate(data, schema)
    print("Valid!")
except ValidationError as e:
    print(f"Invalid: {e.message}")
```

### Common Schema Patterns

```json
{
  "type": "object",
  "properties": {
    "id": {"type": "integer"},
    "name": {"type": "string"},
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1
    },
    "metadata": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "status": {
      "type": "string",
      "enum": ["active", "inactive", "pending"]
    }
  },
  "required": ["id", "name"]
}
```

## ML Applications

### Configuration Files

```json
{
  "model": {
    "type": "transformer",
    "hidden_size": 768,
    "num_layers": 12,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100
  },
  "data": {
    "train_path": "data/train.jsonl",
    "val_path": "data/val.jsonl"
  }
}
```

### Model Metadata

```json
{
  "model_name": "sentiment-classifier-v2",
  "version": "2.1.0",
  "created_at": "2024-01-15T10:30:00Z",
  "metrics": {
    "accuracy": 0.923,
    "f1_score": 0.918
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "training_data": {
    "samples": 50000,
    "checksum": "sha256:abc123..."
  }
}
```

### API Responses

```python
# ML inference API response
{
    "model": "gpt-4",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Response text here"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150
    }
}
```

## Limitations

### No Native Types

JSON lacks support for:
- Dates/timestamps (use ISO 8601 strings)
- Binary data (use Base64 encoding)
- Comments (not valid JSON)
- Trailing commas (not valid JSON)
- NaN/Infinity (not valid JSON)

```python
import json
import math

# Problem: NaN/Infinity not supported
json.dumps({"value": float("nan")})  # Raises ValueError

# Solution: allow_nan or convert to null
json.dumps({"value": float("nan")}, allow_nan=True)  # Non-standard
json.dumps({"value": None if math.isnan(x) else x})  # Standard-compliant
```

### Verbosity

JSON is more verbose than binary formats:

```
Format     | Size (100k records)
-----------|--------------------
JSON       | 15 MB
Gzipped JSON | 2 MB
Parquet    | 1.5 MB
```

### No Streaming

Standard JSON requires parsing entire document:

```json
{
  "data": [
    {"id": 1, ...},
    {"id": 2, ...},
    // Must read entire array before processing
  ]
}
```

Solution: Use JSONL for streaming (one JSON object per line).

## Best Practices

### Use Consistent Key Naming

```python
# Good: snake_case (Python convention)
{"user_id": 123, "created_at": "2024-01-15"}

# Also acceptable: camelCase (JavaScript convention)
{"userId": 123, "createdAt": "2024-01-15"}

# Avoid: mixing conventions
{"user_id": 123, "createdAt": "2024-01-15"}
```

### Validate Input

```python
import json
from jsonschema import validate

def parse_config(path: str, schema: dict) -> dict:
    with open(path) as f:
        config = json.load(f)
    validate(config, schema)
    return config
```

### Handle Encoding

```python
# Always specify encoding
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle BOM in some files
import codecs
with codecs.open("data.json", "r", "utf-8-sig") as f:
    data = json.load(f)
```

### Use JSONL for Large Datasets

```python
# Instead of one large JSON array
# Use JSONL for streaming
with open("data.jsonl", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
```

## When to Use JSON

JSON is well-suited for:
- Configuration files
- API request/response payloads
- Model metadata and experiment tracking
- Small datasets (< 100MB)
- Human-readable data exchange
- Web application data

Consider alternatives when:
- Large datasets (use Parquet, JSONL)
- Schema enforcement needed (Avro, Protobuf)
- Binary data (use appropriate binary format)
- High-performance serialization (use orjson or binary formats)
- Streaming required (use JSONL)
