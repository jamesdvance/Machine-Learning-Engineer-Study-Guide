# XML (Extensible Markup Language)

## Summary

XML (Extensible Markup Language) is a hierarchical, human-readable markup format that provides strict structure through schemas and namespaces. While largely superseded by JSON for web APIs, XML remains prevalent in enterprise systems, scientific data formats, configuration files, and legacy integrations. ML engineers encounter XML in configuration files (Maven, Ant), scientific datasets (bioinformatics, GIS), and when integrating with enterprise systems.

Key points to remember:

- Hierarchical document format with elements, attributes, and text content
- Schema validation (XSD, DTD) enables strict type checking
- Namespaces prevent element name collisions in complex documents
- More verbose than JSON but supports comments and mixed content
- Python's ElementTree provides basic parsing; lxml offers full XPath and validation
- iterparse enables streaming for large files without loading into memory
- Compared to JSON, XML is more verbose but offers schema validation
- Still common in enterprise systems, scientific formats (NLM, GML), and config files

## Structure

### Basic Elements

```xml
<?xml version="1.0" encoding="UTF-8"?>
<document>
    <element attribute="value">Text content</element>
    <parent>
        <child>Nested element</child>
    </parent>
    <!-- This is a comment -->
    <empty-element />
</document>
```

Components:
- Declaration: `<?xml version="1.0"?>` specifies XML version and encoding
- Elements: `<tag>content</tag>` with opening and closing tags
- Attributes: `attribute="value"` within opening tags
- Text content: Between opening and closing tags
- Comments: `<!-- comment -->`
- Empty elements: `<tag />` self-closing

### Namespaces

Prevent element name collisions in complex documents:

```xml
<?xml version="1.0"?>
<root xmlns:ml="http://example.com/ml"
      xmlns:data="http://example.com/data">
    <ml:model type="classifier">
        <ml:parameters>
            <data:feature name="age" />
            <data:feature name="income" />
        </ml:parameters>
    </ml:model>
</root>
```

- Default namespace: `xmlns="http://example.com/default"`
- Prefixed namespace: `xmlns:prefix="http://example.com/ns"`
- Elements expand to `{namespace}localname` internally

### Schema (XSD)

Define structure and types:

```xml
<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xs:element name="model">
        <xs:complexType>
            <xs:sequence>
                <xs:element name="name" type="xs:string"/>
                <xs:element name="version" type="xs:decimal"/>
                <xs:element name="accuracy" type="xs:float"/>
            </xs:sequence>
            <xs:attribute name="id" type="xs:integer" use="required"/>
        </xs:complexType>
    </xs:element>
</xs:schema>
```

Benefits:
- Type validation (string, integer, float, date, etc.)
- Structure enforcement (required elements, sequence)
- Documentation of expected format
- Interoperability guarantees

## Python Usage

### ElementTree (Standard Library)

```python
import xml.etree.ElementTree as ET

# Parse from file
tree = ET.parse("data.xml")
root = tree.getroot()

# Parse from string
root = ET.fromstring('<root><item>value</item></root>')

# Navigate tree
for child in root:
    print(child.tag, child.attrib, child.text)

# Find elements
element = root.find("item")           # First match
elements = root.findall("item")        # All matches
elements = root.findall(".//item")     # Recursive search

# Access attributes and text
print(element.get("attribute"))
print(element.text)
```

### Writing XML

```python
import xml.etree.ElementTree as ET

# Create elements
root = ET.Element("models")
model = ET.SubElement(root, "model", id="1")
name = ET.SubElement(model, "name")
name.text = "RandomForest"
accuracy = ET.SubElement(model, "accuracy")
accuracy.text = "0.95"

# Pretty print (Python 3.9+)
ET.indent(root)

# Write to file
tree = ET.ElementTree(root)
tree.write("output.xml", encoding="unicode", xml_declaration=True)

# Write to string
xml_string = ET.tostring(root, encoding="unicode")
```

### lxml (Recommended)

```python
from lxml import etree

# Parse with lxml (same API, more features)
tree = etree.parse("data.xml")
root = tree.getroot()

# Full XPath support
results = root.xpath("//model[@accuracy > 0.9]/name/text()")

# Parent access (not in ElementTree)
parent = element.getparent()

# Pretty printing
xml_bytes = etree.tostring(root, pretty_print=True, encoding="unicode")
```

### Namespace Handling

```python
import xml.etree.ElementTree as ET

xml_data = '''
<root xmlns:ml="http://example.com/ml">
    <ml:model>
        <ml:name>Classifier</ml:name>
    </ml:model>
</root>
'''

root = ET.fromstring(xml_data)

# Define namespace mapping
ns = {"ml": "http://example.com/ml"}

# Find with namespace
model = root.find("ml:model", ns)
name = root.find("ml:model/ml:name", ns)

# Or use full namespace URI
model = root.find("{http://example.com/ml}model")
```

### Large File Processing

```python
import xml.etree.ElementTree as ET

# Streaming parse with iterparse
def process_large_xml(path: str):
    for event, elem in ET.iterparse(path, events=["end"]):
        if elem.tag == "record":
            # Process record
            yield {
                "id": elem.get("id"),
                "value": elem.find("value").text
            }
            # Clear element to free memory
            elem.clear()

# Process without loading entire file
for record in process_large_xml("large_data.xml"):
    process(record)
```

### lxml Streaming

```python
from lxml import etree

# More efficient streaming with lxml
def stream_large_xml(path: str):
    context = etree.iterparse(path, events=["end"], tag="record")
    for event, elem in context:
        yield {
            "id": elem.get("id"),
            "value": elem.findtext("value")
        }
        elem.clear()
        # Also clear parent references
        while elem.getprevious() is not None:
            del elem.getparent()[0]
```

## Schema Validation

### XSD Validation with lxml

```python
from lxml import etree

# Load schema
with open("schema.xsd") as f:
    schema_doc = etree.parse(f)
schema = etree.XMLSchema(schema_doc)

# Validate document
doc = etree.parse("data.xml")
if schema.validate(doc):
    print("Valid")
else:
    for error in schema.error_log:
        print(f"Line {error.line}: {error.message}")

# Validate during parse
parser = etree.XMLParser(schema=schema)
try:
    doc = etree.parse("data.xml", parser)
except etree.XMLSyntaxError as e:
    print(f"Validation failed: {e}")
```

### DTD Validation

```python
from lxml import etree

# Validate against DTD
dtd = etree.DTD(open("schema.dtd"))
doc = etree.parse("data.xml")
if dtd.validate(doc):
    print("Valid")
```

## XPath Queries

### ElementTree (Limited)

```python
import xml.etree.ElementTree as ET

root = ET.parse("data.xml").getroot()

# Basic XPath support
root.find(".")           # Current element
root.find("model")       # Direct child
root.findall(".//model") # All descendants
root.find("model[@id]")  # With attribute
root.find("model[@id='1']")  # Attribute value
root.find("model[1]")    # Position (1-indexed)
```

### lxml (Full XPath)

```python
from lxml import etree

root = etree.parse("data.xml").getroot()

# Full XPath 1.0 support
root.xpath("//model[accuracy > 0.9]")
root.xpath("//model[contains(name, 'Forest')]")
root.xpath("count(//model)")
root.xpath("sum(//model/accuracy)")
root.xpath("//model[position() <= 3]")

# With namespaces
ns = {"ml": "http://example.com/ml"}
root.xpath("//ml:model", namespaces=ns)

# Functions and predicates
root.xpath("//model[last()]")
root.xpath("//model[accuracy = max(//model/accuracy)]")
```

## XSLT Transformations

```python
from lxml import etree

# Load stylesheet
xslt_doc = etree.parse("transform.xslt")
transform = etree.XSLT(xslt_doc)

# Apply transformation
doc = etree.parse("data.xml")
result = transform(doc)

# Get result as string
output = str(result)
```

Example XSLT:

```xml
<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/models">
        <html>
            <body>
                <table>
                    <xsl:for-each select="model">
                        <tr>
                            <td><xsl:value-of select="name"/></td>
                            <td><xsl:value-of select="accuracy"/></td>
                        </tr>
                    </xsl:for-each>
                </table>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>
```

## ML Applications

### Configuration Files

```xml
<?xml version="1.0"?>
<experiment>
    <model type="random_forest">
        <n_estimators>100</n_estimators>
        <max_depth>10</max_depth>
        <min_samples_split>5</min_samples_split>
    </model>
    <training>
        <epochs>100</epochs>
        <batch_size>32</batch_size>
        <learning_rate>0.001</learning_rate>
    </training>
    <data>
        <train_path>data/train.csv</train_path>
        <val_path>data/val.csv</val_path>
    </data>
</experiment>
```

```python
def load_config(path: str) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    return {
        "model": {
            "type": root.find("model").get("type"),
            "n_estimators": int(root.findtext("model/n_estimators")),
            "max_depth": int(root.findtext("model/max_depth")),
        },
        "training": {
            "epochs": int(root.findtext("training/epochs")),
            "batch_size": int(root.findtext("training/batch_size")),
        }
    }
```

### Scientific Data Formats

Many scientific domains use XML-based formats:

**NLM (National Library of Medicine)**
```xml
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>ML for Drug Discovery</article-title>
            </title-group>
            <abstract>
                <p>Abstract text here...</p>
            </abstract>
        </article-meta>
    </front>
    <body>
        <!-- Article content -->
    </body>
</article>
```

**GML (Geography Markup Language)**
```xml
<gml:Point srsName="EPSG:4326">
    <gml:coordinates>-122.4194,37.7749</gml:coordinates>
</gml:Point>
```

### Maven/Ant Build Files

ML projects using Java tooling:

```xml
<!-- pom.xml -->
<project>
    <dependencies>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-mllib_2.12</artifactId>
            <version>3.5.0</version>
        </dependency>
    </dependencies>
</project>
```

## Performance Considerations

### Library Comparison

```
Library      | Parse Speed | Memory | Features
-------------|-------------|--------|----------
ElementTree  | Moderate    | Good   | Basic XPath
lxml         | Fast        | Good   | Full XPath, validation
minidom      | Slow        | High   | DOM API
```

### Optimization Tips

```python
# Use iterparse for large files
for event, elem in ET.iterparse(path):
    process(elem)
    elem.clear()  # Free memory

# Parse only needed events
ET.iterparse(path, events=["start", "end"])

# Use lxml for speed
from lxml import etree  # 5-10x faster than ElementTree

# Cache compiled XPath expressions
find_models = etree.XPath("//model[@accuracy > $min_acc]")
results = find_models(root, min_acc=0.9)
```

## Comparison with Alternatives

### XML vs JSON

| Aspect | XML | JSON |
|--------|-----|------|
| Verbosity | High | Low |
| Comments | Yes | No |
| Schemas | XSD, DTD | JSON Schema |
| Namespaces | Native | None |
| Mixed content | Yes | No |
| Attributes | Yes | No |
| Web APIs | Legacy | Standard |

### XML vs YAML

| Aspect | XML | YAML |
|--------|-----|------|
| Human readable | Moderate | High |
| Comments | Yes | Yes |
| Complexity | High | Low |
| Type support | Via schema | Native |
| Anchors/refs | No | Yes |
| Config files | Enterprise | Modern apps |

## Security Considerations

### XXE (XML External Entity) Prevention

```python
from lxml import etree

# Safe parser configuration
parser = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    dtd_validation=False,
    load_dtd=False,
)

# Parse safely
doc = etree.parse("untrusted.xml", parser)
```

### Billion Laughs Attack Prevention

```python
# Limit entity expansion
parser = etree.XMLParser(huge_tree=False)

# Or use defusedxml library
import defusedxml.ElementTree as DefusedET
root = DefusedET.parse("untrusted.xml")
```

## Best Practices

### Consistent Structure

```python
# Good: clear hierarchy
<model>
    <name>RandomForest</name>
    <parameters>
        <n_estimators>100</n_estimators>
    </parameters>
</model>

# Avoid: mixing elements and attributes inconsistently
<model name="RandomForest" n_estimators="100">
    <max_depth>10</max_depth>
</model>
```

### Use Namespaces for Complex Documents

```xml
<!-- Avoid collisions in multi-source documents -->
<root xmlns:model="http://example.com/model"
      xmlns:data="http://example.com/data">
    <model:config>...</model:config>
    <data:source>...</data:source>
</root>
```

### Schema Validation in Production

```python
def load_validated_config(path: str, schema_path: str) -> etree._Element:
    schema = etree.XMLSchema(etree.parse(schema_path))
    parser = etree.XMLParser(schema=schema)
    return etree.parse(path, parser).getroot()
```

## When to Use XML

XML is well-suited for:
- Enterprise system integrations
- Scientific data formats (NLM, GML, TEI)
- Configuration requiring schema validation
- Documents with mixed content (text and markup)
- Legacy system interoperability
- Build systems (Maven, Ant)

Consider alternatives when:
- Building web APIs (use JSON)
- Simple configuration (use YAML or TOML)
- Data interchange (use JSON or Parquet)
- High-performance serialization (use Protocol Buffers)
- Modern applications without legacy constraints
