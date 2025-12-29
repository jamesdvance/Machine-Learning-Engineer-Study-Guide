# Text Data Formats

## Summary

Text-based data formats store information as human-readable character sequences. They offer universal readability and easy debugging at the cost of storage efficiency and processing speed. ML engineers use text formats extensively for configuration, data interchange, training data, and interoperability with external systems, while graduating to binary formats (Parquet, Arrow) for production data pipelines.

Key points to remember:

- Human-readable: can inspect and debug with any text editor
- Universal support: every language and tool can process text
- No native type information: parsing required for numbers, dates, etc.
- Larger file sizes: verbose compared to binary formats
- Slower I/O: more bytes to read and parsing overhead
- CSV: simplest tabular format, universal but no nesting
- JSON: hierarchical, web-standard, single-document
- JSONL: streaming JSON, ML training data standard
- XML: enterprise/scientific, schema validation, verbose

## Format Comparison

### Structure Overview

| Format | Structure | Type Support | Schema | Streaming |
|--------|-----------|--------------|--------|-----------|
| CSV | Flat table | None (strings) | Header only | Yes |
| JSON | Hierarchical | Basic (6 types) | JSON Schema | No |
| JSONL | Line-delimited | Basic (6 types) | Per-line | Yes |
| XML | Hierarchical | Via XSD | XSD, DTD | With iterparse |

### Feature Matrix

| Feature | CSV | JSON | JSONL | XML |
|---------|-----|------|-------|-----|
| Human readable | High | High | High | Medium |
| Nested data | No | Yes | Yes | Yes |
| Comments | No* | No | No | Yes |
| Streaming | Yes | No | Yes | Partial |
| Append-friendly | Yes | No | Yes | No |
| Schema validation | No | Optional | Optional | Yes (XSD) |
| Namespaces | No | No | No | Yes |
| Attributes | No | No | No | Yes |

*Some tools support # comments, not part of standard

### Python Libraries

| Format | Standard Library | Recommended |
|--------|------------------|-------------|
| CSV | `csv` | `pandas` |
| JSON | `json` | `orjson` |
| JSONL | `json` (per line) | `orjson` |
| XML | `xml.etree.ElementTree` | `lxml` |

## Use Cases

### CSV

Best for:
- Tabular data export/import
- Spreadsheet interoperability
- Simple data exploration
- Universal tool support

```csv
id,name,value,category
1,Widget,19.99,electronics
2,Gadget,29.99,tools
```

```python
import pandas as pd
df = pd.read_csv("data.csv")
```

### JSON

Best for:
- Configuration files
- API responses
- Metadata storage
- Single-document interchange

```json
{
  "model": {
    "type": "transformer",
    "hidden_size": 768
  },
  "training": {
    "epochs": 100,
    "batch_size": 32
  }
}
```

```python
import json
with open("config.json") as f:
    config = json.load(f)
```

### JSONL

Best for:
- ML training data
- Log files
- Streaming data
- Large dataset processing

```jsonl
{"prompt": "Summarize:", "completion": "Summary here..."}
{"prompt": "Translate:", "completion": "Translation here..."}
```

```python
import json
def stream_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)
```

### XML

Best for:
- Enterprise integrations
- Scientific formats (NLM, GML)
- Schema-validated documents
- Legacy system interoperability

```xml
<model id="1">
    <name>RandomForest</name>
    <parameters>
        <n_estimators>100</n_estimators>
    </parameters>
</model>
```

```python
from lxml import etree
tree = etree.parse("model.xml")
```

## Performance Characteristics

### File Size (100k records)

```
Format          | Size    | Gzipped
----------------|---------|--------
CSV             | 12 MB   | 1.5 MB
JSON            | 18 MB   | 2.0 MB
JSONL           | 15 MB   | 1.8 MB
XML             | 25 MB   | 2.5 MB
Parquet (ref)   | 1.5 MB  | N/A
```

### Parse Speed (Relative)

```
Format          | Read    | Write
----------------|---------|-------
CSV (pandas)    | 1.0x    | 1.0x
JSON (stdlib)   | 0.8x    | 0.8x
JSON (orjson)   | 6.0x    | 8.0x
JSONL (orjson)  | 6.0x    | 8.0x
XML (lxml)      | 1.2x    | 1.0x
Parquet (ref)   | 5.0x    | 3.0x
```

## Decision Framework

### Choose CSV When

- Simple tabular data
- Spreadsheet users need access
- Universal tool compatibility required
- Quick data exploration
- Version control of data files

### Choose JSON When

- Hierarchical/nested configuration
- API request/response payloads
- Single-document interchange
- Web application data
- Human-editable config files

### Choose JSONL When

- ML training datasets
- Streaming data processing
- Append-heavy workloads
- Log file storage
- Large dataset processing

### Choose XML When

- Enterprise system integration
- Scientific data formats
- Schema validation critical
- Mixed content (text + markup)
- Legacy system interoperability

### Consider Binary Formats When

- Large datasets (> 100MB)
- Repeated access patterns
- Type preservation critical
- Performance-sensitive pipelines
- Columnar access patterns

## Common Patterns

### Configuration Loading

```python
import json
from pathlib import Path

def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    elif p.suffix == ".csv":
        import pandas as pd
        return pd.read_csv(p).to_dict(orient="records")
    elif p.suffix == ".xml":
        from lxml import etree
        # Parse to dict...
    raise ValueError(f"Unknown format: {p.suffix}")
```

### Format Conversion

```python
import pandas as pd
import json

# CSV to JSONL
df = pd.read_csv("data.csv")
df.to_json("data.jsonl", orient="records", lines=True)

# JSONL to CSV
df = pd.read_json("data.jsonl", lines=True)
df.to_csv("data.csv", index=False)

# JSON to CSV (if array of objects)
df = pd.read_json("data.json")
df.to_csv("data.csv", index=False)
```

### Streaming Large Files

```python
import json
import pandas as pd

# Stream CSV
for chunk in pd.read_csv("large.csv", chunksize=10000):
    process(chunk)

# Stream JSONL
def stream_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# Stream XML
import xml.etree.ElementTree as ET
for event, elem in ET.iterparse("large.xml"):
    if elem.tag == "record":
        process(elem)
        elem.clear()
```

## Best Practices

### Encoding

Always specify UTF-8 explicitly:

```python
# CSV
pd.read_csv("data.csv", encoding="utf-8")

# JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# XML
from lxml import etree
tree = etree.parse("data.xml", parser=etree.XMLParser(encoding="UTF-8"))
```

### Validation

Validate data after reading:

```python
# Schema validation for JSON
from jsonschema import validate
validate(data, schema)

# Schema validation for XML
from lxml import etree
schema = etree.XMLSchema(etree.parse("schema.xsd"))
schema.validate(doc)

# Data validation for CSV
import pandera as pa
schema.validate(df)
```

### Compression

Use compression for large files:

```python
# CSV
df.to_csv("data.csv.gz", compression="gzip")

# JSONL
import gzip
with gzip.open("data.jsonl.gz", "wt") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
```

## Further Reading

For detailed information on each format, see:
- [JSON](json/ReadMe.md)
- [JSONL](jsonl/ReadMe.md)
- [XML](xml/ReadMe.md)
- [CSV](csv/ReadMe.md)
