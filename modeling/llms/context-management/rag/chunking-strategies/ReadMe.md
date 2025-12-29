# Chunking Strategies for RAG

## Summary

Chunking is the process of splitting documents into smaller pieces for retrieval-augmented generation (RAG). The quality of retrieval depends heavily on how documents are chunked: too large and retrieval becomes imprecise, too small and context is lost. Effective chunking strategies consider document structure, semantic boundaries, and the embedding model's context window. The choice between fixed-size, semantic, recursive, or document-aware chunking depends on document type, retrieval requirements, and computational constraints.

Key points to remember:

- Chunk size affects retrieval precision: Smaller for specific facts, larger for context
- Overlap prevents boundary artifacts: 10-20% overlap is typical
- Semantic chunking respects content boundaries: Better coherence than fixed-size
- Recursive chunking handles diverse documents: Falls back through separators
- Document-aware chunking uses structure: Headings, code blocks, paragraphs
- Chunk for your embedding model: Match context window and training
- Test empirically: Optimal chunking depends on use case

## The Chunking Trade-off

### Size Considerations

```
Small chunks (100-200 tokens):
+ Precise retrieval
+ Less noise in context
- May lack necessary context
- More chunks to store and search

Large chunks (1000+ tokens):
+ Full context preserved
+ Fewer chunks overall
- Retrieval less precise
- May include irrelevant content
- May exceed embedding model limits
```

### Finding the Right Size

| Use Case | Recommended Size | Rationale |
|----------|-----------------|-----------|
| FAQ/Support | 200-500 tokens | Questions need specific answers |
| Documentation | 500-1000 tokens | Need surrounding context |
| Legal/Contracts | 1000-2000 tokens | Clauses need full context |
| Code | Function/class level | Logical boundaries matter |
| Research papers | Section-based | Structure carries meaning |

## Fixed-Size Chunking

### Basic Implementation

```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Find a good break point (sentence end, newline)
        if end < len(text):
            # Look for natural break within last 20% of chunk
            search_start = int(end - chunk_size * 0.2)
            break_chars = ['. ', '.\n', '\n\n', '\n']

            best_break = end
            for char in break_chars:
                pos = text.rfind(char, search_start, end)
                if pos != -1:
                    best_break = pos + len(char)
                    break

            end = best_break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks
```

### Token-Based Chunking

```python
from tiktoken import encoding_for_model

def token_chunk(text, model="gpt-3.5-turbo", chunk_tokens=500, overlap_tokens=50):
    """Chunk by token count (more accurate for LLMs)."""
    enc = encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens_slice))
        start = end - overlap_tokens

    return chunks
```

## Semantic Chunking

### Sentence-Based

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_chunk(text, max_sentences=5, overlap_sentences=1):
    """Chunk by sentences, respecting semantic boundaries."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap_sentences):
        chunk_sents = sentences[i:i + max_sentences]
        chunks.append(' '.join(chunk_sents))

    return chunks
```

### Paragraph-Based

```python
def paragraph_chunk(text, max_paragraphs=3, overlap=1):
    """Chunk by paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs - overlap):
        chunk_paras = paragraphs[i:i + max_paragraphs]
        chunks.append('\n\n'.join(chunk_paras))

    return chunks
```

### Embedding-Based Semantic Chunking

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(text, model, similarity_threshold=0.5, max_chunk_size=1000):
    """Chunk based on semantic similarity between sentences."""
    sentences = split_into_sentences(text)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        # Compare to current chunk's centroid
        similarity = cosine_similarity(current_embedding, embeddings[i])

        # Check if should continue chunk
        total_len = sum(len(s) for s in current_chunk) + len(sentences[i])

        if similarity >= similarity_threshold and total_len < max_chunk_size:
            current_chunk.append(sentences[i])
            # Update centroid
            current_embedding = np.mean(embeddings[i-len(current_chunk)+1:i+1], axis=0)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

## Recursive Chunking

### LangChain-Style Recursive Splitter

```python
class RecursiveChunker:
    """Recursively split using multiple separators."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separators in order of preference
        self.separators = [
            "\n\n",      # Paragraphs
            "\n",        # Lines
            ". ",        # Sentences
            ", ",        # Clauses
            " ",         # Words
            ""           # Characters (fallback)
        ]

    def split(self, text):
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text, separators):
        """Split text using separators recursively."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            chunks = [text[i:i+self.chunk_size]
                     for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        else:
            splits = text.split(separator)
            chunks = []
            current_chunk = ""

            for split in splits:
                test_chunk = current_chunk + separator + split if current_chunk else split

                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # If split itself is too large, recurse
                    if len(split) > self.chunk_size:
                        chunks.extend(self._split_recursive(split, remaining_separators))
                        current_chunk = ""
                    else:
                        current_chunk = split

            if current_chunk:
                chunks.append(current_chunk)

        return chunks
```

## Document-Aware Chunking

### Markdown Chunking

```python
import re

def markdown_chunk(text, max_size=1000):
    """Chunk markdown respecting structure."""
    chunks = []

    # Split by headers
    header_pattern = r'^(#{1,6})\s+(.+)$'
    sections = re.split(r'\n(?=#{1,6}\s)', text)

    for section in sections:
        if len(section) <= max_size:
            chunks.append(section)
        else:
            # Further split by code blocks, paragraphs
            subsections = split_by_structure(section, max_size)
            chunks.extend(subsections)

    return chunks


def split_by_structure(text, max_size):
    """Split section by structural elements."""
    parts = []

    # Protect code blocks
    code_pattern = r'```[\s\S]*?```'
    code_blocks = re.findall(code_pattern, text)
    text_without_code = re.sub(code_pattern, '<<<CODE>>>', text)

    # Split by paragraphs
    paragraphs = text_without_code.split('\n\n')

    current = ""
    code_idx = 0

    for para in paragraphs:
        if para == '<<<CODE>>>':
            para = code_blocks[code_idx]
            code_idx += 1

        if len(current) + len(para) <= max_size:
            current += '\n\n' + para if current else para
        else:
            if current:
                parts.append(current)
            current = para

    if current:
        parts.append(current)

    return parts
```

### Code Chunking

```python
import ast

def python_code_chunk(code):
    """Chunk Python code by logical units."""
    chunks = []

    try:
        tree = ast.parse(code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Extract function/class with docstring
                start = node.lineno - 1
                end = node.end_lineno
                chunk = '\n'.join(code.split('\n')[start:end])
                chunks.append({
                    'type': type(node).__name__,
                    'name': node.name,
                    'content': chunk
                })
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Group imports
                pass

    except SyntaxError:
        # Fall back to simpler splitting
        chunks = fixed_size_chunk(code)

    return chunks
```

### HTML/Document Chunking

```python
from bs4 import BeautifulSoup

def html_chunk(html, max_size=1000):
    """Chunk HTML preserving structure."""
    soup = BeautifulSoup(html, 'html.parser')
    chunks = []

    # Extract meaningful sections
    for element in soup.find_all(['article', 'section', 'div', 'p', 'h1', 'h2', 'h3']):
        text = element.get_text(strip=True)

        if len(text) > max_size:
            # Further split large sections
            sub_chunks = fixed_size_chunk(text, chunk_size=max_size)
            chunks.extend(sub_chunks)
        elif text:
            chunks.append(text)

    return chunks
```

## Chunking with Metadata

### Rich Chunks

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            'content': self.content,
            **self.metadata
        }


def chunk_with_metadata(document, chunker):
    """Create chunks with provenance metadata."""
    raw_chunks = chunker.split(document['content'])

    chunks = []
    for i, content in enumerate(raw_chunks):
        chunk = Chunk(
            content=content,
            metadata={
                'source': document['source'],
                'chunk_index': i,
                'total_chunks': len(raw_chunks),
                'document_title': document.get('title', ''),
                'chunk_size': len(content),
                'created_at': datetime.now().isoformat()
            }
        )
        chunks.append(chunk)

    return chunks
```

### Hierarchical Chunking

```python
def hierarchical_chunk(text, levels=['section', 'paragraph', 'sentence']):
    """Create multi-level chunking for hierarchical retrieval."""
    result = {'text': text, 'children': []}

    if not levels:
        return result

    current_level = levels[0]
    remaining_levels = levels[1:]

    if current_level == 'section':
        parts = re.split(r'\n#{1,3}\s', text)
    elif current_level == 'paragraph':
        parts = text.split('\n\n')
    else:  # sentence
        parts = split_into_sentences(text)

    for part in parts:
        if part.strip():
            child = hierarchical_chunk(part, remaining_levels)
            result['children'].append(child)

    return result
```

## Choosing a Strategy

### Decision Guide

```python
def recommend_chunking(document_type, retrieval_task, avg_doc_size):
    """Recommend chunking strategy based on use case."""

    if document_type == 'code':
        return {
            'strategy': 'document_aware',
            'unit': 'function/class',
            'fallback': 'recursive'
        }

    if document_type == 'markdown' or document_type == 'documentation':
        return {
            'strategy': 'document_aware',
            'unit': 'section',
            'chunk_size': 800,
            'overlap': 100
        }

    if retrieval_task == 'qa':
        return {
            'strategy': 'semantic',
            'chunk_size': 400,
            'overlap': 50
        }

    if retrieval_task == 'summarization':
        return {
            'strategy': 'recursive',
            'chunk_size': 1000,
            'overlap': 200
        }

    # Default
    return {
        'strategy': 'recursive',
        'chunk_size': 500,
        'overlap': 100
    }
```

## Key Takeaways

1. **Match chunk size to task**: Smaller for precision, larger for context.

2. **Use overlap**: 10-20% prevents losing information at boundaries.

3. **Respect document structure**: Markdown sections, code functions, HTML elements.

4. **Semantic > fixed-size**: When coherence matters, chunk by meaning.

5. **Recursive is versatile**: Falls back through separators for diverse documents.

6. **Include metadata**: Source, position, and structure aid retrieval.

7. **Test empirically**: The best strategy depends on your specific use case.
