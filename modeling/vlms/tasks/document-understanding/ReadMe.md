# Document Understanding

## Summary

Document understanding is the task of extracting structured information from document images. Unlike plain text processing, document understanding must interpret visual layout, typography, tables, and graphics alongside textual content. Documents range from forms and invoices to academic papers and handwritten notes, each presenting unique challenges.

Key points to remember:

- Combines OCR, layout analysis, and semantic understanding
- Must preserve document structure: headings, paragraphs, tables, lists
- Layout matters: visual position conveys meaning (headers, sidebars, footnotes)
- Key-value extraction is a common subtask (invoices, forms)
- Table extraction requires understanding row/column structure
- Multi-page documents need cross-page reasoning
- Specialized models (LayoutLM, Donut) outperform general VLMs on structured extraction
- VLMs excel at flexible querying and complex reasoning about documents

## Document Types

### Structured Documents

Consistent layouts with predictable fields:

**Forms**: Application forms, surveys, registration documents
- Fixed field positions
- Mix of printed and handwritten content
- Checkboxes, signatures, dates

**Invoices and Receipts**: Financial documents
- Key-value pairs (vendor, date, total)
- Line items with quantities and prices
- Variable layouts across vendors

**ID Documents**: Passports, licenses, cards
- Standardized layouts per document type
- Security features and formatting

### Semi-Structured Documents

Consistent patterns with some variation:

**Business Documents**: Contracts, reports, letters
- Headings and paragraphs
- Tables and figures
- Headers and footers

**Academic Papers**: Research papers, articles
- Abstract, sections, references
- Equations, figures, tables
- Multi-column layouts

### Unstructured Documents

Highly variable formats:

**Handwritten Notes**: Free-form handwriting
- Variable quality and legibility
- No standard structure

**Historical Documents**: Aged or degraded documents
- Damaged or faded content
- Archaic fonts and layouts

## Core Capabilities

### Layout Analysis

Understanding document structure:

```
Document Page
    |
+---+---+---+
|   |   |   |
| Header    |
|           |
+-----------+
|     |     |
|Col1 |Col2 |  <- Multi-column
|     |     |
+-----------+
|  Table    |
|  Footer   |
+-----------+
```

**Segmentation**: Identifying regions (text blocks, tables, figures).

**Reading Order**: Determining correct sequence for multi-column layouts.

**Hierarchical Structure**: Headings, subheadings, paragraphs.

### Text Extraction (OCR)

Converting image text to machine-readable format:

- Printed text recognition
- Handwriting recognition
- Multi-language support
- Handling rotated or skewed text

### Information Extraction

Deriving structured data from documents:

**Key-Value Extraction**:
```
Input: Invoice image
Output: {
    "vendor": "Acme Corp",
    "date": "2024-01-15",
    "total": "$1,234.56",
    "items": [...]
}
```

**Table Extraction**:
```
Input: Table image
Output: [
    ["Header1", "Header2", "Header3"],
    ["Row1Col1", "Row1Col2", "Row1Col3"],
    ...
]
```

**Entity Recognition**:
- Dates, amounts, addresses
- Names, organizations
- Document-specific entities

## Approaches

### Traditional Pipeline

Sequential processing stages:

```
Document Image
    |
    v
OCR (Tesseract, etc.)
    |
    v
Layout Analysis
    |
    v
Entity Extraction (NER, regex)
    |
    v
Structured Output
```

**Pros**: Interpretable, modular
**Cons**: Error propagation, no joint optimization

### Layout-Aware Models

Models that jointly understand text and layout:

**LayoutLM Family**:
- LayoutLM: Adds 2D position embeddings to BERT
- LayoutLMv2: Adds visual features
- LayoutLMv3: Unified multimodal architecture

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_entity_types
)

# Process document
encoding = processor(
    image,
    words,
    boxes=bounding_boxes,
    return_tensors="pt"
)
outputs = model(**encoding)
```

### End-to-End Models

Models that skip explicit OCR:

**Donut (Document Understanding Transformer)**:
- No OCR required
- Generates text directly from image
- JSON output for structured extraction

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Generate structured output
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
result = processor.decode(outputs[0])
```

### VLM-Based Understanding

Using general VLMs for document tasks:

```python
# GPT-4V for invoice extraction
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """Extract information from this invoice as JSON:
{
    "vendor": "...",
    "invoice_number": "...",
    "date": "...",
    "items": [{"description": "...", "quantity": ..., "price": ...}],
    "subtotal": "...",
    "tax": "...",
    "total": "..."
}"""},
            {"type": "image_url", "image_url": {"url": invoice_url}}
        ]
    }]
)
```

## Benchmarks

### DocVQA

Visual question answering on documents:
- 50K questions on 12K document images
- Tests reading comprehension on documents
- Answers derived from document content

### FUNSD

Form understanding:
- 199 fully annotated forms
- Entity extraction and linking
- Small but challenging dataset

### CORD

Receipt understanding:
- 1000 Indonesian receipts
- 30 entity types
- Tests real-world robustness

### RVL-CDIP

Document classification:
- 400K document images
- 16 categories (letter, form, invoice, etc.)
- Tests document type recognition

### TableBank

Table detection and recognition:
- 417K tables from Word and LaTeX
- Detection and structure recognition tasks

## Implementation Patterns

### Key-Value Extraction Pipeline

```python
class DocumentExtractor:
    def __init__(self, model_name="gpt-4o"):
        self.client = OpenAI()
        self.model = model_name

    def extract(self, image_path, schema):
        """Extract structured data based on schema."""
        base64_image = encode_image(image_path)

        prompt = f"""Extract the following fields from this document:
{json.dumps(schema, indent=2)}

Return a JSON object with these fields. Use null for fields not found."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            }],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
```

### Table Extraction

```python
def extract_table(image, client):
    """Extract table as structured data."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": """Extract the table from this image.
Return as a JSON array where each element is a row (also an array).
The first row should be headers.
Example: [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]"""},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("table", [])
```

### Multi-Page Processing

```python
def process_multipage_document(pages, client):
    """Process multi-page document with context."""
    # First pass: summarize each page
    page_summaries = []
    for i, page in enumerate(pages):
        summary = summarize_page(page, client)
        page_summaries.append(summary)

    # Second pass: extract with full context
    full_context = "\n".join([
        f"Page {i+1}: {s}" for i, s in enumerate(page_summaries)
    ])

    results = []
    for i, page in enumerate(pages):
        extraction = extract_with_context(
            page,
            context=full_context,
            page_number=i+1,
            client=client
        )
        results.append(extraction)

    return merge_results(results)
```

## Challenges

### Layout Complexity

- Multi-column layouts require correct reading order
- Floating elements (figures, sidebars) interrupt flow
- Tables span pages or wrap awkwardly

**Mitigation**:
- Use layout-aware models (LayoutLM)
- Segment documents before processing
- Provide explicit layout instructions

### Poor Image Quality

- Scanned documents with artifacts
- Faded or damaged historical documents
- Low resolution or compression artifacts

**Mitigation**:
- Image preprocessing (deskewing, enhancement)
- Models robust to noise
- Multiple attempts with different preprocessing

### Handwriting

- Variable handwriting styles
- Mixed printed and handwritten content
- Abbreviations and shorthand

**Mitigation**:
- Specialized handwriting models
- Human-in-the-loop for low confidence
- Training on diverse handwriting samples

### Domain Vocabulary

- Technical terms, abbreviations
- Industry-specific formats
- Multi-language documents

**Mitigation**:
- Domain-specific fine-tuning
- Custom entity dictionaries
- Multi-language models

## Best Practices

### Schema Definition

Define clear extraction schemas:

```python
invoice_schema = {
    "vendor_name": "string",
    "vendor_address": "string",
    "invoice_number": "string",
    "invoice_date": "date (YYYY-MM-DD)",
    "due_date": "date (YYYY-MM-DD)",
    "line_items": [{
        "description": "string",
        "quantity": "number",
        "unit_price": "number",
        "total": "number"
    }],
    "subtotal": "number",
    "tax_rate": "number",
    "tax_amount": "number",
    "total": "number"
}
```

### Validation

Validate extracted data:

```python
def validate_invoice(extracted):
    errors = []

    # Check required fields
    required = ["vendor_name", "invoice_number", "total"]
    for field in required:
        if not extracted.get(field):
            errors.append(f"Missing required field: {field}")

    # Validate calculations
    computed_total = sum(
        item.get("total", 0) for item in extracted.get("line_items", [])
    )
    if abs(computed_total - extracted.get("subtotal", 0)) > 0.01:
        errors.append("Line item totals don't match subtotal")

    # Validate date formats
    for date_field in ["invoice_date", "due_date"]:
        if extracted.get(date_field):
            try:
                datetime.strptime(extracted[date_field], "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date format: {date_field}")

    return errors
```

### Confidence Scoring

Track extraction confidence:

```python
prompt = """Extract fields from this document.
For each field, also provide a confidence level (high/medium/low).

Return as:
{
    "fields": {
        "vendor": {"value": "...", "confidence": "high"},
        ...
    }
}"""
```

### Human-in-the-Loop

Route uncertain extractions for review:

```python
def process_with_review(document, threshold=0.8):
    extraction = extract(document)

    low_confidence_fields = [
        field for field, data in extraction.items()
        if data.get("confidence") == "low"
    ]

    if low_confidence_fields:
        return {
            "status": "needs_review",
            "extraction": extraction,
            "review_fields": low_confidence_fields
        }

    return {"status": "complete", "extraction": extraction}
```

## Model Selection

| Use Case | Recommended | Notes |
|----------|-------------|-------|
| Fixed-format forms | LayoutLM fine-tuned | Consistent structure |
| Variable invoices | VLM (GPT-4V, Gemini) | Handles format variation |
| Table extraction | Table-specific models | Specialized performance |
| General queries | VLM | Flexible questioning |
| High volume | Donut, LayoutLM | Self-hosted, fast |
| Complex reasoning | GPT-4V, Gemini | Cross-reference ability |
