# CSV (Comma-Separated Values)

## Summary

CSV (Comma-Separated Values) is a plain-text tabular format where each line represents a row and values are separated by delimiters (typically commas). Its simplicity and universal support make it the most common format for sharing datasets, but it lacks type information, schema enforcement, and efficient storage. ML engineers use CSV extensively for data exploration, model input/output, and interoperability, while graduating to more efficient formats (Parquet, Arrow) for production workloads.

Key points to remember:

- Plain text format: human-readable but verbose and inefficient
- No native type information: everything is a string until parsed
- Universal support: every tool reads CSV
- Pandas is the standard Python library for CSV data manipulation
- `usecols` and `dtype` parameters dramatically improve read performance
- Chunked reading enables processing files larger than memory
- Quoting handles fields containing delimiters or newlines
- Encoding issues (UTF-8 vs Latin-1) are a common pain point
- Compared to Parquet: CSV is 5-10x larger and slower but universally readable
- Compared to JSON: CSV is simpler for tabular data but lacks nested structures

## Format Specification

### Basic Structure

```csv
name,age,score,city
Alice,30,95.5,New York
Bob,25,87.3,"Los Angeles"
Carol,35,92.1,Chicago
```

Rules:
- First row typically contains headers
- Each subsequent row is a record
- Values separated by delimiter (comma by default)
- Line endings: LF (`\n`) or CRLF (`\r\n`)
- Quoted fields: use double quotes for values containing delimiter, newlines, or quotes
- Escape quotes: double them (`""`)

### Quoting and Escaping

```csv
name,description,price
Widget,"A useful, everyday item",19.99
Gadget,"Contains ""special"" characters",29.99
Device,"Multi-line
description here",39.99
```

Quoting rules:
- Fields containing delimiter must be quoted
- Fields containing newlines must be quoted
- Quotes within quoted fields are doubled
- Leading/trailing whitespace is significant

### Common Variants

| Variant | Delimiter | Use Case |
|---------|-----------|----------|
| CSV | `,` | Standard |
| TSV | `\t` (tab) | Avoids comma issues |
| SSV | `;` | European locale (comma as decimal) |
| PSV | `|` (pipe) | Database exports |

## Python Standard Library

### Reading CSV

```python
import csv

# Basic reading
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        print(row)  # List of strings

# Dictionary reader (access by column name)
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])
```

### Writing CSV

```python
import csv

data = [
    ["name", "age", "score"],
    ["Alice", 30, 95.5],
    ["Bob", 25, 87.3],
]

# Basic writing
with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Dictionary writer
with open("output.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["name", "age", "score"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"name": "Alice", "age": 30, "score": 95.5})
```

### Handling Variants

```python
import csv

# Tab-separated
with open("data.tsv", "r") as f:
    reader = csv.reader(f, delimiter="\t")

# Semicolon-separated
with open("data.csv", "r") as f:
    reader = csv.reader(f, delimiter=";")

# Custom quoting
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
```

## Pandas Usage

### Reading CSV

```python
import pandas as pd

# Basic read
df = pd.read_csv("data.csv")

# With type specification (faster, more memory efficient)
df = pd.read_csv("data.csv", dtype={
    "id": "int32",
    "name": "string",
    "price": "float32",
    "category": "category",
})

# Select columns (faster parsing)
df = pd.read_csv("data.csv", usecols=["name", "age", "score"])

# Parse dates
df = pd.read_csv("data.csv", parse_dates=["date", "timestamp"])

# Handle missing values
df = pd.read_csv("data.csv", na_values=["", "NA", "null", "-"])
```

### Common Parameters

```python
df = pd.read_csv(
    "data.csv",
    sep=",",                    # Delimiter
    header=0,                   # Row number for headers (None if no header)
    names=["a", "b", "c"],      # Column names (if no header)
    index_col="id",             # Column to use as index
    usecols=["a", "b"],         # Columns to read (faster)
    dtype={"a": "int32"},       # Column types (faster)
    parse_dates=["date"],       # Date columns
    na_values=["NA", "null"],   # Missing value markers
    encoding="utf-8",           # File encoding
    skiprows=5,                 # Skip initial rows
    nrows=1000,                 # Read only first N rows
    skipfooter=3,               # Skip footer rows (requires engine="python")
    comment="#",                # Skip lines starting with character
)
```

### Writing CSV

```python
import pandas as pd

# Basic write
df.to_csv("output.csv", index=False)

# With options
df.to_csv(
    "output.csv",
    index=False,                # Don't write index
    columns=["a", "b", "c"],    # Subset of columns
    header=True,                # Include header row
    sep=",",                    # Delimiter
    na_rep="",                  # String for missing values
    float_format="%.2f",        # Float formatting
    date_format="%Y-%m-%d",     # Date formatting
    encoding="utf-8",           # Encoding
    compression="gzip",         # Compress output (.csv.gz)
)
```

## Large File Processing

### Chunked Reading

```python
import pandas as pd

# Process in chunks (memory efficient)
chunk_size = 100_000
chunks = pd.read_csv("large.csv", chunksize=chunk_size)

for chunk in chunks:
    # Process each chunk
    process(chunk)

# Aggregate results
results = []
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    result = chunk.groupby("category")["sales"].sum()
    results.append(result)

final = pd.concat(results).groupby(level=0).sum()
```

### Memory Optimization

```python
import pandas as pd

# Specify dtypes to reduce memory
df = pd.read_csv("data.csv", dtype={
    "id": "int32",              # Instead of int64
    "amount": "float32",        # Instead of float64
    "category": "category",     # Categorical for repeated strings
    "flag": "bool",             # Boolean
})

# Use category for low-cardinality strings
df["status"] = df["status"].astype("category")

# Check memory usage
print(df.memory_usage(deep=True))
```

### Parallel Processing

```python
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import glob

def process_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.groupby("category")["value"].sum()

# Process multiple files in parallel
files = glob.glob("data/*.csv")
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_file, files))

combined = pd.concat(results)
```

## Encoding Handling

### Common Encodings

```python
import pandas as pd

# UTF-8 (default, recommended)
df = pd.read_csv("data.csv", encoding="utf-8")

# Latin-1 (Windows/legacy)
df = pd.read_csv("data.csv", encoding="latin-1")

# UTF-8 with BOM
df = pd.read_csv("data.csv", encoding="utf-8-sig")

# Detect encoding
import chardet

with open("data.csv", "rb") as f:
    result = chardet.detect(f.read(10000))
    encoding = result["encoding"]

df = pd.read_csv("data.csv", encoding=encoding)
```

### Handling Errors

```python
import pandas as pd

# Replace undecodable characters
df = pd.read_csv("data.csv", encoding="utf-8", encoding_errors="replace")

# Skip lines with encoding errors
df = pd.read_csv("data.csv", on_bad_lines="skip")

# Custom error handling
df = pd.read_csv("data.csv", on_bad_lines="warn")
```

## Data Quality

### Missing Values

```python
import pandas as pd

# Recognize various null representations
df = pd.read_csv("data.csv", na_values=[
    "", "NA", "N/A", "null", "NULL", "None", "-", ".", "NaN"
])

# Per-column NA values
df = pd.read_csv("data.csv", na_values={
    "age": ["-1", "unknown"],
    "score": ["NA", "pending"],
})

# Check for missing values
print(df.isnull().sum())
```

### Data Validation

```python
import pandas as pd

def validate_csv(path: str) -> dict:
    df = pd.read_csv(path)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

# Schema validation with pandera
import pandera as pa

schema = pa.DataFrameSchema({
    "id": pa.Column(int, unique=True),
    "name": pa.Column(str, nullable=False),
    "age": pa.Column(int, checks=pa.Check.in_range(0, 120)),
    "email": pa.Column(str, checks=pa.Check.str_matches(r".+@.+\..+")),
})

df = pd.read_csv("data.csv")
validated_df = schema.validate(df)
```

## ML Applications

### Feature Engineering

```python
import pandas as pd

# Load training data
df = pd.read_csv("train.csv")

# Common preprocessing
X = df.drop(columns=["target"])
y = df["target"]

# Feature selection by dtype
numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
```

### Train/Val/Test Splits

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv("data.csv")

# Split and save
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
```

### Kaggle-Style Data

```python
import pandas as pd

# Load competition data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Make predictions
predictions = model.predict(test[features])

# Create submission
submission = sample_submission.copy()
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
```

## Performance Optimization

### Read Performance

```python
import pandas as pd

# Fastest: specify dtypes and use only needed columns
df = pd.read_csv(
    "data.csv",
    usecols=["col1", "col2", "col3"],
    dtype={"col1": "int32", "col2": "float32", "col3": "category"},
)

# PyArrow backend (Pandas 2.0+)
df = pd.read_csv("data.csv", engine="pyarrow")

# Low memory mode for very large files
df = pd.read_csv("data.csv", low_memory=True)
```

### Benchmark Comparison

```
Operation       | Default | With dtype | With usecols | Both
----------------|---------|------------|--------------|------
Read 1M rows    | 3.2s    | 1.8s       | 1.5s         | 0.9s
Memory usage    | 800MB   | 400MB      | 300MB        | 150MB
```

### When to Convert to Other Formats

```python
import pandas as pd

# Read CSV once, save as Parquet for repeated use
df = pd.read_csv("large_data.csv")
df.to_parquet("large_data.parquet")

# Subsequent reads are 10x faster
df = pd.read_parquet("large_data.parquet")
```

## Compression

### Reading Compressed Files

```python
import pandas as pd

# Automatic detection by extension
df = pd.read_csv("data.csv.gz")      # gzip
df = pd.read_csv("data.csv.bz2")     # bzip2
df = pd.read_csv("data.csv.xz")      # xz
df = pd.read_csv("data.csv.zip")     # zip

# Explicit compression
df = pd.read_csv("data.csv", compression="gzip")
```

### Writing Compressed Files

```python
import pandas as pd

df.to_csv("data.csv.gz", index=False, compression="gzip")
df.to_csv("data.csv.bz2", index=False, compression="bz2")

# With options
df.to_csv("data.csv.gz", compression={
    "method": "gzip",
    "compresslevel": 9,
})
```

## Comparison with Alternatives

### CSV vs Parquet

| Aspect | CSV | Parquet |
|--------|-----|---------|
| Human readable | Yes | No |
| File size | Large | 5-10x smaller |
| Read speed | Slow | Fast |
| Type preservation | No | Yes |
| Column selection | Read all | Read subset |
| Ecosystem | Universal | Analytics/ML |

### CSV vs JSON/JSONL

| Aspect | CSV | JSON/JSONL |
|--------|-----|------------|
| Structure | Flat table | Nested objects |
| Schema | Header row | None/Implicit |
| Streaming | Possible | JSONL only |
| Mixed types | Per column | Per field |
| ML training | Common | Fine-tuning |

### CSV vs Excel

| Aspect | CSV | Excel |
|--------|-----|-------|
| Format | Text | Binary |
| Multiple sheets | No | Yes |
| Formatting | None | Rich |
| File size | Smaller | Larger |
| Version control | Easy | Difficult |
| Programmatic | Simple | Complex |

## Best Practices

### Always Specify Encoding

```python
# Good: explicit encoding
df = pd.read_csv("data.csv", encoding="utf-8")

# Avoid: implicit encoding (platform-dependent)
df = pd.read_csv("data.csv")
```

### Use dtype for Known Schemas

```python
# Good: explicit types
df = pd.read_csv("data.csv", dtype={
    "id": "int32",
    "price": "float32",
    "category": "category",
})

# Avoid: let pandas infer (slower, more memory)
df = pd.read_csv("data.csv")
```

### Handle Headers Consistently

```python
# With header
df = pd.read_csv("data.csv", header=0)

# Without header
df = pd.read_csv("data.csv", header=None, names=["a", "b", "c"])

# Skip metadata rows
df = pd.read_csv("data.csv", skiprows=3, header=0)
```

### Validate Data After Reading

```python
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Check expected columns
    required = {"id", "name", "value"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    # Check for unexpected nulls
    assert not df["id"].isnull().any(), "Null IDs found"

    # Check data types
    assert pd.api.types.is_numeric_dtype(df["value"]), "Value must be numeric"

    return df
```

## When to Use CSV

CSV is well-suited for:
- Data exploration and quick analysis
- Sharing data with non-technical users
- Interoperability with any tool or system
- Small to medium datasets (< 100MB)
- Version control of data files
- Export/import with spreadsheet applications

Consider alternatives when:
- Large datasets (use Parquet)
- Repeated reads of same data (convert to Parquet)
- Type preservation critical (use Parquet or Arrow)
- Nested data structures (use JSON)
- High-performance pipelines (use binary formats)
- Schema enforcement needed (use Avro)
