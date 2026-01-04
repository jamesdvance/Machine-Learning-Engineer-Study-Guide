# OCR (Optical Character Recognition)

## Summary

Optical Character Recognition converts images of text into machine-readable characters. While OCR has existed for decades, modern VLMs have expanded its scope from simple text extraction to understanding text in context. Today's OCR ranges from specialized engines optimized for speed and accuracy on clean documents to VLM-based approaches that handle complex scenes, multiple languages, and contextual interpretation.

Key points to remember:

- Traditional OCR optimized for documents: fast, accurate on clean input
- Scene text OCR handles text in photographs: signs, products, storefronts
- Handwriting recognition remains challenging, especially cursive
- VLMs can perform OCR while also understanding context and answering questions
- Specialized OCR engines (Tesseract, PaddleOCR) often faster and more accurate for bulk processing
- VLMs better for complex scenes, mixed content, or when semantic understanding is needed
- Output includes text, bounding boxes, and confidence scores
- Post-processing (spell check, formatting) often needed for production use

## OCR Categories

### Document OCR

Recognizing text in scanned documents:

**Characteristics**:
- High contrast (black text on white)
- Standard fonts
- Structured layouts
- Clean backgrounds

**Challenges**:
- Skew and rotation
- Scanning artifacts
- Multi-column layouts
- Mixed font sizes

### Scene Text Recognition

Recognizing text in natural images:

**Characteristics**:
- Variable backgrounds
- Perspective distortion
- Artistic fonts
- Partial occlusion

**Examples**:
- Street signs
- Product labels
- Storefront names
- Vehicle plates

### Handwriting Recognition

Recognizing handwritten text:

**Characteristics**:
- Variable styles
- Connected letters (cursive)
- Inconsistent spacing
- Personal abbreviations

**Types**:
- Printed handwriting (easier)
- Cursive script (harder)
- Historical manuscripts (specialized)

## Traditional OCR Pipeline

### Standard Pipeline

```
Image
  |
Preprocessing (deskew, binarize)
  |
Layout Analysis (detect text regions)
  |
Line Segmentation
  |
Character/Word Recognition
  |
Post-processing (spell check, formatting)
  |
Output Text
```

### Preprocessing

Prepare images for recognition:

```python
import cv2
import numpy as np

def preprocess_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deskew
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), borderValue=255)

    # Binarize
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary
```

### Layout Analysis

Detect text regions before recognition:

```python
def detect_text_regions(image):
    # Find contours
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter to likely text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.1 < aspect_ratio < 10 and w * h > 100:
            text_regions.append((x, y, w, h))

    return text_regions
```

## OCR Engines

### Tesseract

Open-source OCR engine maintained by Google:

```python
import pytesseract
from PIL import Image

# Basic usage
text = pytesseract.image_to_string(Image.open('document.png'))

# With bounding boxes
data = pytesseract.image_to_data(
    Image.open('document.png'),
    output_type=pytesseract.Output.DICT
)

# Specify language
text = pytesseract.image_to_string(
    image,
    lang='eng+fra'  # English and French
)

# Page segmentation modes
# PSM 1: Auto with OSD
# PSM 3: Fully automatic (default)
# PSM 6: Uniform block of text
# PSM 7: Single text line
# PSM 11: Sparse text
text = pytesseract.image_to_string(image, config='--psm 6')
```

**Strengths**: Free, supports 100+ languages, customizable
**Weaknesses**: Slower than commercial options, struggles with complex layouts

### PaddleOCR

Open-source multilingual OCR:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr('document.png')

for line in result[0]:
    bbox, (text, confidence) = line
    print(f"{text} ({confidence:.2f})")
```

**Strengths**: Fast, excellent for Asian languages, good accuracy
**Weaknesses**: Larger models, some setup complexity

### EasyOCR

Simple Python OCR library:

```python
import easyocr

reader = easyocr.Reader(['en', 'fr'])
results = reader.readtext('image.png')

for (bbox, text, confidence) in results:
    print(f"{text}: {confidence:.2f}")
```

**Strengths**: Easy to use, good accuracy, GPU support
**Weaknesses**: Limited language coverage compared to Tesseract

### Cloud OCR Services

**Google Cloud Vision**:
```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()
with open('document.png', 'rb') as f:
    content = f.read()

image = vision.Image(content=content)
response = client.text_detection(image=image)
text = response.text_annotations[0].description
```

**AWS Textract**:
```python
import boto3

client = boto3.client('textract')
with open('document.png', 'rb') as f:
    response = client.detect_document_text(Document={'Bytes': f.read()})

for block in response['Blocks']:
    if block['BlockType'] == 'LINE':
        print(block['Text'])
```

**Azure Computer Vision**:
```python
from azure.cognitiveservices.vision.computervision import ComputerVisionClient

client = ComputerVisionClient(endpoint, credentials)
result = client.read_in_stream(image_stream, raw=True)
```

## VLM-Based OCR

### Using VLMs for Text Extraction

VLMs can extract and understand text:

```python
# GPT-4V OCR
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image. Preserve the original formatting and layout as much as possible."},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)
text = response.choices[0].message.content
```

### Contextual Understanding

VLMs understand text meaning, not just characters:

```python
# Understanding context
prompt = """Look at this sign in the image.
1. What does the text say exactly?
2. What does it mean in context?
3. Is this a warning, advertisement, or information?"""
```

### Handling Difficult Text

VLMs can handle challenging OCR scenarios:

```python
# Stylized or artistic text
prompt = "Read the stylized text in this logo, even if it uses unusual fonts or letter forms."

# Partially obscured text
prompt = "Read the text on this sign. Some letters may be partially hidden. Infer missing letters if possible."

# Handwritten text
prompt = "Transcribe the handwritten text in this note. Indicate any words you're uncertain about with [?]."
```

## Comparison: Traditional vs VLM OCR

| Aspect | Traditional OCR | VLM-Based OCR |
|--------|-----------------|---------------|
| Speed | Fast (100+ pages/sec) | Slow (seconds per image) |
| Cost | Free or low | API costs per image |
| Accuracy (clean docs) | Excellent | Good |
| Accuracy (scene text) | Moderate | Excellent |
| Handwriting | Limited | Better |
| Context understanding | None | Yes |
| Bounding boxes | Yes | Sometimes |
| Batch processing | Efficient | Expensive |
| Custom training | Possible | Limited |

### When to Use Each

**Use Traditional OCR**:
- High volume document processing
- Clean, well-formatted documents
- Need for bounding boxes
- Cost-sensitive applications
- Consistent document types

**Use VLM-Based OCR**:
- Complex scene text
- Need contextual understanding
- Mixed content (text + images)
- Handwritten or stylized text
- Answering questions about text

## Scene Text Recognition

### CRAFT + Recognition

Detection followed by recognition:

```python
# Using CRAFT for detection
from craft_text_detector import Craft

craft = Craft(output_dir='output', cuda=True)
prediction = craft.detect_text(image)

# Then recognize each detected region
for bbox in prediction['boxes']:
    cropped = crop_region(image, bbox)
    text = recognize(cropped)
```

### End-to-End Models

Single model for detection and recognition:

**ASTER**: Attention-based sequence-to-sequence
**CRNN**: CNN + RNN for sequence recognition
**TrOCR**: Transformer-based OCR

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Post-Processing

### Spell Correction

Fix OCR errors using language models:

```python
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_ocr_text(text):
    words = text.split()
    corrected = []
    for word in words:
        if word.lower() in spell:
            corrected.append(word)
        else:
            correction = spell.correction(word.lower())
            if correction:
                # Preserve original case
                if word.isupper():
                    corrected.append(correction.upper())
                elif word[0].isupper():
                    corrected.append(correction.capitalize())
                else:
                    corrected.append(correction)
            else:
                corrected.append(word)
    return ' '.join(corrected)
```

### Layout Reconstruction

Reconstruct document structure:

```python
def reconstruct_layout(ocr_results):
    """Group OCR results by lines and paragraphs."""
    # Sort by y coordinate
    sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])

    lines = []
    current_line = []
    current_y = None
    threshold = 10  # Pixel threshold for same line

    for result in sorted_results:
        y = result['bbox'][1]
        if current_y is None or abs(y - current_y) < threshold:
            current_line.append(result)
            current_y = y
        else:
            if current_line:
                # Sort line by x coordinate
                current_line.sort(key=lambda x: x['bbox'][0])
                lines.append(' '.join([r['text'] for r in current_line]))
            current_line = [result]
            current_y = y

    if current_line:
        current_line.sort(key=lambda x: x['bbox'][0])
        lines.append(' '.join([r['text'] for r in current_line]))

    return '\n'.join(lines)
```

### Confidence Filtering

Filter low-confidence results:

```python
def filter_by_confidence(results, threshold=0.8):
    return [r for r in results if r['confidence'] >= threshold]
```

## Challenges and Solutions

### Low Resolution Images

**Problem**: Blurry or low-resolution text
**Solutions**:
- Image upscaling (ESRGAN, Real-ESRGAN)
- Super-resolution preprocessing
- Lower confidence thresholds

### Rotated and Skewed Text

**Problem**: Text at angles
**Solutions**:
- Automatic orientation detection
- Deskewing algorithms
- Use models with rotation invariance

### Multi-Language Documents

**Problem**: Documents with multiple languages
**Solutions**:
- Specify multiple languages to OCR engine
- Language detection per region
- Multilingual models

### Dense Text

**Problem**: Very small or tightly packed text
**Solutions**:
- Higher resolution input
- Smaller detection windows
- Region-by-region processing

## Best Practices

### Image Preparation

```python
def prepare_image(image_path, target_dpi=300):
    """Prepare image for optimal OCR."""
    from PIL import Image

    img = Image.open(image_path)

    # Ensure sufficient resolution
    if img.info.get('dpi', (72, 72))[0] < target_dpi:
        scale = target_dpi / img.info.get('dpi', (72, 72))[0]
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img
```

### Quality Assessment

Assess OCR quality:

```python
def assess_ocr_quality(results):
    """Assess overall OCR quality."""
    if not results:
        return {"quality": "failed", "score": 0}

    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    low_confidence_count = sum(1 for c in confidences if c < 0.8)

    if avg_confidence > 0.95 and low_confidence_count == 0:
        quality = "excellent"
    elif avg_confidence > 0.85:
        quality = "good"
    elif avg_confidence > 0.70:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "quality": quality,
        "score": avg_confidence,
        "low_confidence_words": low_confidence_count
    }
```

### Choosing an Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Bulk document scanning | Tesseract or PaddleOCR |
| Enterprise documents | Cloud services (AWS, Google, Azure) |
| Scene text in photos | VLM or specialized models |
| Handwritten notes | VLM or handwriting-specific models |
| Real-time mobile | On-device models (MLKit, Vision) |
| Multi-language | PaddleOCR or cloud services |
| Historical documents | Specialized + VLM fallback |
