# Data Formats for Machine Learning

## Summary

Data formats determine how information is stored, transmitted, and processed throughout the ML lifecycle. Choosing the right format affects storage costs, processing speed, and interoperability across tools and frameworks. ML engineers must balance human readability, compression efficiency, type safety, and ecosystem compatibility.

Key points to remember:

- Text formats (CSV, JSON, JSONL) offer universal readability at the cost of efficiency
- Binary columnar formats (Parquet, ORC) provide 5-10x compression and fast analytical queries
- Row-based formats (Avro) excel for streaming and write-heavy workloads
- ML-specific formats (TFRecord, Safetensors, ONNX) optimize for training, storage, and deployment
- Image formats trade off between compression and quality preservation
- Video codecs significantly impact both storage and decode performance
- Format choice should align with access patterns: analytical vs streaming, read-heavy vs write-heavy

## Format Categories

### By Data Type

| Category | Formats | Primary Use |
|----------|---------|-------------|
| Text | CSV, JSON, JSONL, XML | Configuration, interchange, small datasets |
| Tabular Binary | Parquet, ORC, Avro | Large datasets, data lakes, analytics |
| ML-Specific | TFRecord, Safetensors, ONNX, HDF5 | Training data, model weights, deployment |
| Image | JPEG, PNG, WebP, TIFF | Vision ML training and inference |
| Video | H.264, H.265, VP9, AV1 | Video understanding, generation |

### By Access Pattern

| Pattern | Recommended Formats |
|---------|---------------------|
| Configuration | JSON, YAML |
| Streaming/Append | JSONL, Avro |
| Analytics | Parquet, ORC |
| Random Access | Parquet (columnar), HDF5 |
| Model Weights | Safetensors |
| Model Deployment | ONNX, TensorRT |
| Training Data | TFRecord (TensorFlow), Parquet (general) |

## Text Formats

Text formats store data as human-readable character sequences. They provide universal compatibility and easy debugging at the cost of storage efficiency.

### Comparison

| Format | Structure | Best For |
|--------|-----------|----------|
| CSV | Flat table | Simple tabular data, spreadsheet export |
| JSON | Hierarchical | Configuration, API payloads, single documents |
| JSONL | Line-delimited | ML training data, streaming, logs |
| XML | Hierarchical with schema | Enterprise integration, scientific formats |

### When to Use Text Formats

- Small datasets (under 100 MB)
- Configuration files
- API interchange
- Human inspection and debugging
- Version control of data
- Cross-platform compatibility

### When to Avoid Text Formats

- Large datasets (use Parquet instead)
- Performance-critical pipelines
- Type preservation requirements
- Columnar access patterns

## Tabular Binary Formats

Binary formats sacrifice human readability for dramatic improvements in storage and processing efficiency. Columnar formats (Parquet, ORC) store data by column for analytical workloads, while row-based formats (Avro) optimize for streaming.

### Comparison

| Feature | Parquet | ORC | Avro |
|---------|---------|-----|------|
| Layout | Columnar | Columnar | Row-based |
| Compression | Excellent | Excellent | Good |
| Column Pruning | Yes | Yes | No |
| Predicate Pushdown | Yes | Yes | No |
| Schema Evolution | Limited | Limited | Excellent |
| Streaming | Poor | Poor | Excellent |

### Decision Framework

**Choose Parquet when:**
- Analytical queries and aggregations
- Data lake storage (default choice)
- ML training data
- Broad ecosystem support needed

**Choose ORC when:**
- Hive-centric Hadoop environment
- Native ACID transactions (Hive tables)
- Bloom filter indexing needed

**Choose Avro when:**
- Kafka message serialization
- Event streaming architectures
- Frequent schema changes
- Write-heavy workloads

### Performance Comparison

Typical file sizes for 100M rows, 10 columns:

| Format | Uncompressed | Zstd Compressed |
|--------|--------------|-----------------|
| CSV | 5.0 GB | 0.9 GB |
| Parquet | 0.8 GB | 0.4 GB |
| ORC | 0.7 GB | 0.4 GB |
| Avro | 2.0 GB | 0.7 GB |

Parquet and ORC achieve 5-10x compression over text formats while enabling column-selective reads that further reduce I/O.

## ML-Specific Formats

ML-specific formats address the unique requirements of machine learning workflows: efficient data loading, secure weight storage, and cross-framework interoperability.

### Format Purposes

| Format | Purpose | Stage |
|--------|---------|-------|
| TFRecord | Training data serialization | Training (TensorFlow) |
| Safetensors | Model weight storage | Storage, sharing |
| ONNX | Model interchange | Deployment |
| HDF5 | Large array storage | Research, legacy |

### Security Considerations

| Format | Security | Notes |
|--------|----------|-------|
| Safetensors | Safe | No code execution possible |
| ONNX | Safe | Protobuf-based |
| TFRecord | Safe | Protobuf-based |
| Pickle (.pt, .bin) | Unsafe | Arbitrary code execution |

Use Safetensors for all model weight storage and sharing. Avoid loading pickle files from untrusted sources.

### Decision Framework

**For training data:**
- TensorFlow: TFRecord
- PyTorch: Raw files with DataLoader, or HDF5 for large arrays
- General: Parquet with PyArrow

**For model weights:**
- Modern workflows: Safetensors
- Hugging Face Hub: Safetensors (default)
- Legacy Keras: HDF5 (.h5)

**For deployment:**
- Cross-framework: ONNX
- Maximum performance: ONNX with runtime optimization
- Edge devices: ONNX, TFLite

## Image Formats

Image format choice affects storage costs, training throughput, and potentially model quality due to compression artifacts.

### Comparison

| Format | Compression | Transparency | Best For |
|--------|-------------|--------------|----------|
| JPEG | Lossy | No | Photographs, natural images |
| PNG | Lossless | Yes | Labels, masks, graphics |
| WebP | Both | Yes | Modern workflows, web |
| TIFF | Both | Yes | Scientific, medical, geospatial |

### Decision Framework

**Choose JPEG when:**
- Photographs and natural images
- File size is priority
- No transparency needed
- Perfect pixel accuracy not required

**Choose PNG when:**
- Segmentation masks and labels
- Transparency required
- Graphics, text, screenshots
- Lossless quality essential

**Choose WebP when:**
- Best of both worlds needed
- Modern tooling available
- Web delivery

**Choose TIFF when:**
- Scientific or medical imaging
- High bit depth (16-bit, 32-bit float)
- GeoTIFF for geospatial data

### ML Considerations

- Train on consistent compression levels to avoid learning artifacts
- Use lossless formats for ground truth labels
- Consider compression as data augmentation for robustness
- Account for decode time in data loading pipelines

## Video Formats

Video codecs trade off between compression efficiency, encode speed, and decode compatibility. The choice significantly impacts both storage costs and training throughput.

### Comparison

| Codec | Compression | Encode Speed | HW Decode Support |
|-------|-------------|--------------|-------------------|
| H.264 | Baseline | Fast | Universal |
| H.265 | +40-50% | Slow | Widespread |
| VP9 | +30-50% | Moderate | Good |
| AV1 | +50-60% | Very slow | Growing |

### Decision Framework

**Choose H.264 when:**
- Maximum compatibility needed
- Real-time encoding required
- Hardware decode essential

**Choose H.265 when:**
- 4K/HDR content
- Good compression, reasonable encode time
- Apple ecosystem

**Choose VP9 when:**
- Royalty-free requirement
- YouTube/web delivery
- Good balance of efficiency and support

**Choose AV1 when:**
- Best compression essential
- Encode time is acceptable
- Royalty-free requirement

### Video in ML Workflows

For training on video data:

1. Use hardware decode (NVIDIA DALI, Decord GPU) for throughput
2. Consider extracting frames for random-access-heavy workloads
3. Standardize format during data ingestion
4. Account for GOP structure when random frame access needed

## Format Selection by Use Case

### Data Lake Storage

```
Raw Layer: Avro (streaming) or JSON (batch)
    |
    v
Curated Layer: Parquet with Iceberg/Delta Lake
    |
    v
Serving Layer: Parquet or optimized serving format
```

### Model Development

```
Training Data: TFRecord (TF) / Parquet (general)
    |
    v
Checkpoints: Safetensors (weights) + JSON (config)
    |
    v
Final Model: Safetensors (weights) + ONNX (deployment)
```

### Vision Pipeline

```
Source: JPEG/PNG (original quality)
    |
    v
Training: WebP (consistent compression) or TFRecord
    |
    v
Labels: PNG (lossless, segmentation masks)
```

### Video Pipeline

```
Source: Varied formats
    |
    v
Standardized: H.264 or VP9 (consistent quality)
    |
    v
Training: Hardware-decoded frames
```

## Format Conversion

### Text to Binary

```python
import pandas as pd

# CSV to Parquet
df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", compression="zstd")

# JSON to Parquet
df = pd.read_json("data.json")
df.to_parquet("data.parquet")

# JSONL to Parquet
df = pd.read_json("data.jsonl", lines=True)
df.to_parquet("data.parquet")
```

### Image Format Conversion

```python
from PIL import Image

# Convert to WebP
img = Image.open("image.png")
img.save("image.webp", "WEBP", quality=85)

# Convert to JPEG (for photos)
img = Image.open("photo.png")
img.convert("RGB").save("photo.jpg", "JPEG", quality=85)
```

### Model Format Conversion

```python
# Pickle to Safetensors
import torch
from safetensors.torch import save_file

state_dict = torch.load("model.pt")
save_file(state_dict, "model.safetensors")

# PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Best Practices

### Compression Selection

| Use Case | Recommended Compression |
|----------|------------------------|
| Cold storage | Zstd (best ratio) |
| Hot/streaming | Snappy (fast) |
| Archive | Zstd level 19 |
| Default | Zstd level 3 |

### File Sizing

- Target 128 MB - 1 GB files for distributed processing
- Avoid many small files (metadata overhead)
- Use compaction for streaming ingestion

### Schema Management

- Define explicit schemas for binary formats
- Use Schema Registry for Kafka/Avro
- Version schemas for evolution
- Document breaking changes

### Quality vs Size Trade-offs

| Priority | Image | Video | Tabular |
|----------|-------|-------|---------|
| Quality | PNG/TIFF | H.264 CRF 18 | Parquet uncompressed |
| Balanced | WebP 85 | H.264 CRF 23 | Parquet zstd |
| Size | WebP 70 | AV1 CRF 35 | Parquet zstd-19 |

## Common Pitfalls

1. **Using CSV for large data**: Parquet is 5-10x smaller and faster
2. **Pickle for model weights**: Security risk; use Safetensors
3. **Inconsistent image compression**: Can affect model training
4. **Ignoring decode overhead**: Matters for training throughput
5. **Wrong format for access pattern**: Columnar for analytics, row for streaming

## Further Reading

For detailed information on each format category, see:

- [Text Data Formats](text/ReadMe.md)
- [Tabular Data Formats](tabular/ReadMe.md)
- [ML-Specific Formats](ml-specific/ReadMe.md)
- [Image Formats](image/ReadMe.md)
- [Video Formats](video/ReadMe.md)
