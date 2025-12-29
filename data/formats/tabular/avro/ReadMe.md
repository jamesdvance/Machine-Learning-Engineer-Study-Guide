# Apache Avro

## Summary

Apache Avro is a row-based data serialization format that excels at schema evolution and streaming data. Unlike columnar formats (Parquet, ORC), Avro stores data by row, making it ideal for write-heavy workloads, message streaming, and scenarios where schema changes frequently. Avro is the standard format for Kafka messages and is widely used in microservices communication and event-driven architectures.

Key points to remember:

- Row-based format: stores complete records sequentially
- Schema stored with data: self-describing files
- Excellent schema evolution: backward and forward compatibility
- Standard for Kafka message serialization
- Compact binary encoding with rich type system
- fastavro is the recommended Python library (10x faster than avro-python3)
- Schema Registry integration for centralized schema management
- No code generation required (unlike Protocol Buffers)
- Compared to Parquet: Avro better for streaming, Parquet better for analytics
- Compared to JSON: Avro is binary, compact, and schema-enforced

## Schema Definition

### JSON Schema Format

```json
{
  "type": "record",
  "name": "User",
  "namespace": "com.example",
  "doc": "User record for analytics",
  "fields": [
    {"name": "id", "type": "long"},
    {"name": "name", "type": "string"},
    {"name": "email", "type": ["null", "string"], "default": null},
    {"name": "age", "type": ["null", "int"], "default": null},
    {"name": "created_at", "type": {"type": "long", "logicalType": "timestamp-millis"}}
  ]
}
```

### Primitive Types

```json
{
  "type": "record",
  "name": "AllTypes",
  "fields": [
    {"name": "null_field", "type": "null"},
    {"name": "bool_field", "type": "boolean"},
    {"name": "int_field", "type": "int"},           // 32-bit signed
    {"name": "long_field", "type": "long"},         // 64-bit signed
    {"name": "float_field", "type": "float"},       // 32-bit IEEE 754
    {"name": "double_field", "type": "double"},     // 64-bit IEEE 754
    {"name": "bytes_field", "type": "bytes"},       // Arbitrary bytes
    {"name": "string_field", "type": "string"}      // UTF-8 string
  ]
}
```

### Complex Types

```json
{
  "type": "record",
  "name": "ComplexExample",
  "fields": [
    // Array
    {"name": "tags", "type": {"type": "array", "items": "string"}},

    // Map
    {"name": "metadata", "type": {"type": "map", "values": "string"}},

    // Enum
    {"name": "status", "type": {
      "type": "enum",
      "name": "Status",
      "symbols": ["ACTIVE", "INACTIVE", "PENDING"]
    }},

    // Fixed (fixed-size bytes)
    {"name": "uuid", "type": {"type": "fixed", "name": "UUID", "size": 16}},

    // Nested record
    {"name": "address", "type": {
      "type": "record",
      "name": "Address",
      "fields": [
        {"name": "street", "type": "string"},
        {"name": "city", "type": "string"}
      ]
    }},

    // Union (nullable)
    {"name": "optional_field", "type": ["null", "string"], "default": null}
  ]
}
```

### Logical Types

```json
{
  "type": "record",
  "name": "LogicalTypes",
  "fields": [
    // Decimal (arbitrary precision)
    {"name": "price", "type": {
      "type": "bytes",
      "logicalType": "decimal",
      "precision": 10,
      "scale": 2
    }},

    // Date (days since Unix epoch)
    {"name": "birth_date", "type": {"type": "int", "logicalType": "date"}},

    // Time (milliseconds since midnight)
    {"name": "start_time", "type": {"type": "int", "logicalType": "time-millis"}},

    // Timestamp (milliseconds since epoch)
    {"name": "created_at", "type": {"type": "long", "logicalType": "timestamp-millis"}},

    // UUID
    {"name": "id", "type": {"type": "string", "logicalType": "uuid"}}
  ]
}
```

## Python Usage (fastavro)

### Installation

```bash
pip install fastavro
```

### Writing Avro Files

```python
from fastavro import writer, parse_schema

# Define schema
schema = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": ["null", "string"], "default": None},
    ]
}
parsed_schema = parse_schema(schema)

# Write records
records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": None},
]

with open("users.avro", "wb") as f:
    writer(f, parsed_schema, records)
```

### Reading Avro Files

```python
from fastavro import reader

# Read all records
with open("users.avro", "rb") as f:
    avro_reader = reader(f)
    for record in avro_reader:
        print(record)

# Read schema from file
with open("users.avro", "rb") as f:
    avro_reader = reader(f)
    print(avro_reader.writer_schema)
```

### Appending to Files

```python
from fastavro import writer, reader, parse_schema

# Read existing schema
with open("users.avro", "rb") as f:
    avro_reader = reader(f)
    schema = avro_reader.writer_schema

# Append new records
new_records = [{"id": 3, "name": "Carol", "email": "carol@example.com"}]

with open("users.avro", "a+b") as f:
    writer(f, parse_schema(schema), new_records)
```

### Schemaless Operations

```python
from fastavro import schemaless_writer, schemaless_reader, parse_schema
import io

schema = parse_schema({
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
    ]
})

# Write single record without header (for Kafka)
buffer = io.BytesIO()
schemaless_writer(buffer, schema, {"id": 1, "name": "Alice"})
data = buffer.getvalue()

# Read schemaless record
buffer = io.BytesIO(data)
record = schemaless_reader(buffer, schema)
```

### Compression

```python
from fastavro import writer, parse_schema

# Supported codecs: null, deflate, bzip2, snappy, zstandard, lz4, xz
with open("users.avro", "wb") as f:
    writer(f, parsed_schema, records, codec="snappy")

# With compression level
with open("users.avro", "wb") as f:
    writer(f, parsed_schema, records, codec="deflate", codec_compression_level=9)
```

## Schema Evolution

### Compatibility Rules

**Backward Compatible** (new schema can read old data):
- Add fields with defaults
- Remove fields

**Forward Compatible** (old schema can read new data):
- Remove fields
- Add fields with defaults

**Full Compatible** (both):
- Add/remove optional fields with defaults

### Adding Fields

```python
# Original schema
schema_v1 = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
    ]
}

# Evolved schema (backward compatible)
schema_v2 = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": ["null", "string"], "default": None},  # New field
    ]
}

# Read old data with new schema
from fastavro import reader, parse_schema

with open("users_v1.avro", "rb") as f:
    avro_reader = reader(f, reader_schema=parse_schema(schema_v2))
    for record in avro_reader:
        print(record)  # email will be None for old records
```

### Removing Fields

```python
# Schema with field to remove
schema_v1 = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "deprecated_field", "type": ["null", "string"], "default": None},
    ]
}

# Evolved schema (forward compatible)
schema_v2 = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        # deprecated_field removed
    ]
}

# Old readers will ignore the missing field
```

### Schema Validation

```python
from fastavro.schema import parse_schema, fullname
from fastavro.validation import validate

schema = parse_schema({
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
    ]
})

record = {"id": 1, "name": "Alice"}

# Validate record against schema
try:
    validate(record, schema)
    print("Valid")
except Exception as e:
    print(f"Invalid: {e}")
```

## Kafka Integration

### Producer with Schema Registry

```python
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

# Schema Registry client
schema_registry = SchemaRegistryClient({"url": "http://localhost:8081"})

# Avro serializer
schema_str = """
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"}
    ]
}
"""
avro_serializer = AvroSerializer(schema_registry, schema_str)

# Producer
producer = Producer({"bootstrap.servers": "localhost:9092"})

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()}")

# Send message
user = {"id": 1, "name": "Alice"}
producer.produce(
    topic="users",
    value=avro_serializer(user, None),
    on_delivery=delivery_report
)
producer.flush()
```

### Consumer with Schema Registry

```python
from confluent_kafka import Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

# Schema Registry client
schema_registry = SchemaRegistryClient({"url": "http://localhost:8081"})

# Avro deserializer
avro_deserializer = AvroDeserializer(schema_registry)

# Consumer
consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "my-group",
    "auto.offset.reset": "earliest"
})
consumer.subscribe(["users"])

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Error: {msg.error()}")
            continue

        user = avro_deserializer(msg.value(), None)
        print(f"Received: {user}")
finally:
    consumer.close()
```

### Schemaless Kafka (with fastavro)

```python
from fastavro import schemaless_writer, schemaless_reader, parse_schema
from confluent_kafka import Producer, Consumer
import io

schema = parse_schema({
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
    ]
})

# Serialize
def serialize(record):
    buffer = io.BytesIO()
    schemaless_writer(buffer, schema, record)
    return buffer.getvalue()

# Deserialize
def deserialize(data):
    buffer = io.BytesIO(data)
    return schemaless_reader(buffer, schema)
```

## Spark Integration

### Reading Avro

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
    .getOrCreate()

# Read Avro file
df = spark.read.format("avro").load("path/to/data.avro")

# With schema (optional)
df = spark.read \
    .format("avro") \
    .option("avroSchema", schema_json) \
    .load("path/to/data.avro")
```

### Writing Avro

```python
# Write Avro file
df.write.format("avro").save("output/data.avro")

# With compression
df.write \
    .format("avro") \
    .option("compression", "snappy") \
    .save("output/data.avro")

# With partitioning
df.write \
    .format("avro") \
    .partitionBy("year", "month") \
    .save("output/partitioned/")
```

### Streaming with Kafka

```python
from pyspark.sql import SparkSession
from pyspark.sql.avro.functions import from_avro, to_avro

spark = SparkSession.builder.getOrCreate()

schema_str = """
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"}
    ]
}
"""

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "users") \
    .load()

# Deserialize Avro
parsed = df.select(from_avro(df.value, schema_str).alias("user"))

# Process
processed = parsed.select("user.id", "user.name")

# Write to Kafka with Avro
output = processed.select(to_avro(struct("*")).alias("value"))
output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "processed-users") \
    .start()
```

## File Structure

### Container File Format

```
Avro Container File
   Header
      Magic bytes ("Obj1")
      File metadata (schema, codec)
      Sync marker (16 random bytes)
   Data Block 1
      Object count
      Serialized objects (compressed)
      Sync marker
   Data Block 2
      Object count
      Serialized objects (compressed)
      Sync marker
   ...
```

**Header**
- Magic bytes identify Avro format
- Metadata includes schema JSON
- Sync marker for block boundaries

**Data Blocks**
- Multiple records per block
- Compressed as unit
- Sync markers enable splitting

### Object Container Properties

```python
from fastavro import writer, parse_schema

# Configure block size
with open("users.avro", "wb") as f:
    writer(
        f,
        parsed_schema,
        records,
        codec="snappy",
        sync_interval=16000,  # Bytes per block (default 16KB)
    )
```

## Performance Optimization

### fastavro vs avro-python3

```python
# fastavro is ~10x faster than avro-python3
# Install: pip install fastavro

# fastavro uses C extensions for speed
from fastavro import reader, writer  # Fast

# avro-python3 is pure Python
from avro.datafile import DataFileReader  # Slower
```

### Batch Processing

```python
from fastavro import reader
import pandas as pd

# Read to DataFrame efficiently
with open("data.avro", "rb") as f:
    records = list(reader(f))
df = pd.DataFrame(records)

# Or iterate for memory efficiency
with open("large.avro", "rb") as f:
    for record in reader(f):
        process(record)
```

### Compression Selection

```
Codec     | Compression | Speed    | Use Case
----------|-------------|----------|----------
null      | None        | Fastest  | Pre-compressed
snappy    | Moderate    | Fast     | Real-time streaming
deflate   | High        | Slow     | Archival
zstandard | High        | Moderate | Best balance
lz4       | Low         | Very fast| Speed priority
```

## Comparison with Alternatives

### Avro vs Parquet

| Aspect | Avro | Parquet |
|--------|------|---------|
| Layout | Row-based | Columnar |
| Best for | Streaming, writes | Analytics, reads |
| Schema evolution | Excellent | Good |
| Compression | Per-block | Per-column |
| Splitting | Easy | Complex |
| Column selection | Read all | Efficient pruning |

### Avro vs Protocol Buffers

| Aspect | Avro | Protocol Buffers |
|--------|------|------------------|
| Schema | JSON, in file | .proto, compiled |
| Code generation | Optional | Required |
| Schema evolution | Excellent | Good |
| Dynamic typing | Yes | Limited |
| File format | Container | Messages only |

### Avro vs JSON

| Aspect | Avro | JSON |
|--------|------|------|
| Format | Binary | Text |
| Size | Compact | Verbose |
| Schema | Enforced | None |
| Parsing | Fast | Slower |
| Human readable | No | Yes |

## Best Practices

### Schema Design

```python
# Good: optional fields with defaults
{"name": "email", "type": ["null", "string"], "default": None}

# Good: use logical types
{"name": "timestamp", "type": {"type": "long", "logicalType": "timestamp-millis"}}

# Good: namespace for organization
{
    "type": "record",
    "namespace": "com.example.users",
    "name": "User",
    ...
}

# Avoid: required fields that might be missing
{"name": "optional_data", "type": "string"}  # Bad if data might be null
```

### Schema Registry

```python
# Always use Schema Registry in production Kafka
# - Centralized schema management
# - Compatibility checking
# - Version tracking

from confluent_kafka.schema_registry import SchemaRegistryClient

client = SchemaRegistryClient({"url": "http://schema-registry:8081"})

# Register schema
schema_id = client.register_schema("users-value", schema)

# Get schema by ID
schema = client.get_schema(schema_id)
```

### File Naming

```python
# Include version and timestamp
"users_v1_20240115.avro"

# Or use partitioned directories
"users/year=2024/month=01/data.avro"
```

## When to Use Avro

Avro is well-suited for:
- Kafka message serialization
- Event-driven architectures
- Microservices communication
- Write-heavy workloads
- Frequently evolving schemas
- RPC protocols
- Hadoop ecosystem integration

Consider alternatives when:
- Analytical queries (use Parquet)
- Human-readable format needed (use JSON)
- Maximum compression needed (use columnar formats)
- Static schemas with code generation (use Protocol Buffers)
