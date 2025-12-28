# H.264 / AVC Video Codec

## Summary

H.264, also known as AVC (Advanced Video Coding) or MPEG-4 Part 10, is the most widely deployed video codec in history. Released in 2003, it remains the de facto standard for video delivery due to universal hardware support, excellent compression efficiency, and broad compatibility. For machine learning practitioners, H.264 is frequently encountered in training datasets, surveillance footage, and any video content from the past two decades.

Key points to remember:

- Near-universal hardware decode support (GPUs, mobile devices, embedded systems)
- Excellent balance of compression efficiency and decode complexity
- Multiple profiles and levels for different use cases
- Lossy compression with quality/bitrate trade-offs
- Container formats: MP4, MKV, MOV, AVI, TS
- Common file extensions: .mp4, .m4v, .mkv, .mov

Profiles (complexity/feature tiers):

- Baseline: Low complexity, no B-frames, mobile/video conferencing
- Main: B-frames enabled, broadcast quality
- High: 8x8 transforms, better compression, HD content
- High 10: 10-bit color depth
- High 4:2:2/4:4:4: Professional color sampling

When to use H.264:

- Maximum compatibility across devices and platforms
- Hardware-accelerated encoding/decoding is essential
- Existing pipelines built around H.264
- Real-time encoding with limited compute
- Legacy system integration

When to consider alternatives:

- Better compression needed (use H.265 or AV1)
- Royalty-free requirement (use VP9 or AV1)
- Cutting-edge quality at low bitrates (use AV1)

---

## Understanding H.264 Compression

H.264 achieves compression through a sophisticated pipeline that exploits spatial and temporal redundancy in video content.

### Core Compression Techniques

**Intra-Frame Prediction**: Within a single frame, pixels are predicted from neighboring pixels. Multiple prediction modes (for 4x4, 8x8, and 16x16 blocks) allow the encoder to choose the best match.

**Inter-Frame Prediction**: Most compression comes from predicting frames based on previous (and future) frames. Motion estimation finds similar blocks in reference frames; only the difference (residual) is encoded.

**Frame Types**:
- I-frames (Intra): Complete frames, no dependencies. Largest size, used for seeking.
- P-frames (Predicted): Reference previous frames. Medium size.
- B-frames (Bidirectional): Reference both past and future frames. Smallest size, best compression.

**Transform and Quantization**: Residual data undergoes integer DCT, then quantization discards less important frequency components. The Quantization Parameter (QP) controls this trade-off: lower QP means higher quality and larger files.

**Entropy Coding**: Two options:
- CAVLC (Context-Adaptive Variable-Length Coding): Simpler, faster, used in Baseline profile
- CABAC (Context-Adaptive Binary Arithmetic Coding): 10-15% better compression, higher complexity, used in Main/High profiles

### GOP Structure

A Group of Pictures (GOP) defines the pattern of I, P, and B frames. Common structures:

- **IBBPBBPBBPBB...** (GOP size 12): Good compression, standard broadcast
- **IPPPPPPP...** (no B-frames): Lower latency, easier editing
- **IIII...** (all I-frames): Maximum editability, minimal compression

Longer GOPs improve compression but:
- Increase seek time (must decode from nearest I-frame)
- Make frame-accurate editing difficult
- Increase memory requirements for decode

For ML training, consider using shorter GOPs or all I-frames for random access to individual frames.

## H.264 in Machine Learning Workflows

### Decoding for Training

Video frames must be decoded before use in ML pipelines. Common approaches:

**OpenCV**:
```python
import cv2

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame is BGR numpy array, shape (H, W, 3)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**PyAV (FFmpeg bindings)**:
```python
import av

container = av.open("video.mp4")
for frame in container.decode(video=0):
    # Convert to numpy array
    array = frame.to_ndarray(format="rgb24")
```

**Decord (optimized for ML)**:
```python
from decord import VideoReader, cpu

vr = VideoReader("video.mp4", ctx=cpu(0))

# Random access to specific frames
frame_42 = vr[42].asnumpy()  # Shape: (H, W, 3)

# Batch access
frames = vr.get_batch([0, 10, 20, 30]).asnumpy()
```

### Hardware-Accelerated Decoding

For large-scale video processing, hardware decode is essential:

**NVIDIA GPU Decoding**:
```python
from decord import VideoReader, gpu

# Use GPU for decoding
vr = VideoReader("video.mp4", ctx=gpu(0))
```

**NVIDIA DALI**:
```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def video_pipeline():
    video = fn.readers.video(
        device="gpu",
        filenames=["video.mp4"],
        sequence_length=16,
        random_shuffle=True
    )
    return video
```

### Frame Extraction Considerations

**Seek Accuracy**: Seeking to arbitrary frames requires decoding from the nearest I-frame. For random frame sampling:
- Extract all frames once and cache as images
- Use a video format with frequent I-frames
- Accept the overhead of sequential decode with skipping

**Temporal Sampling**: For video understanding tasks, sample frames systematically:
```python
import numpy as np
from decord import VideoReader

vr = VideoReader("video.mp4")
total_frames = len(vr)

# Uniform sampling of 16 frames
indices = np.linspace(0, total_frames - 1, 16, dtype=int)
frames = vr.get_batch(indices).asnumpy()

# Random sampling
indices = np.random.choice(total_frames, 16, replace=False)
indices.sort()  # Sequential access is faster
frames = vr.get_batch(indices).asnumpy()
```

### Encoding Model Outputs

When saving video predictions or visualizations:

```python
import cv2

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))

for frame in predictions:
    # frame should be BGR, uint8
    out.write(frame)

out.release()
```

With FFmpeg (more control):
```python
import subprocess

process = subprocess.Popen([
    'ffmpeg',
    '-y',  # Overwrite output
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}',
    '-pix_fmt', 'rgb24',
    '-r', '30',
    '-i', '-',  # Read from stdin
    '-c:v', 'libx264',
    '-preset', 'medium',
    '-crf', '23',
    '-pix_fmt', 'yuv420p',
    'output.mp4'
], stdin=subprocess.PIPE)

for frame in predictions:
    process.stdin.write(frame.tobytes())

process.stdin.close()
process.wait()
```

## Encoding Parameters

### Rate Control

**CRF (Constant Rate Factor)**: Quality-based encoding. Lower values = higher quality.
- CRF 18: Visually lossless
- CRF 23: Default, good quality
- CRF 28: Lower quality, smaller files
- CRF 51: Minimum quality

**CBR (Constant Bitrate)**: Fixed bitrate for streaming.

**VBR (Variable Bitrate)**: Variable bitrate within constraints, better quality per bit.

### Presets

Presets trade encoding speed for compression efficiency:

| Preset | Speed | File Size | Use Case |
|--------|-------|-----------|----------|
| ultrafast | Fastest | Largest | Real-time, testing |
| superfast | Very fast | Very large | Low-latency streaming |
| veryfast | Fast | Large | Fast encoding |
| faster | Moderate-fast | Moderate-large | General use |
| fast | Moderate | Moderate | General use |
| medium | Balanced | Balanced | Default |
| slow | Slow | Small | Quality focus |
| slower | Very slow | Smaller | Archival |
| veryslow | Slowest | Smallest | Maximum compression |

For ML preprocessing where encoding is a one-time cost, use slower presets for better compression.

### Color Space Considerations

H.264 typically uses YUV color space with chroma subsampling:

- **YUV 4:2:0**: Standard, chrominance at quarter resolution. Default for consumer video.
- **YUV 4:2:2**: Half horizontal chroma resolution. Professional/broadcast.
- **YUV 4:4:4**: Full chroma resolution. Requires High 4:4:4 profile.

When extracting frames for ML:
```python
# PyAV can decode directly to RGB
frame.to_ndarray(format="rgb24")

# OpenCV decodes to BGR by default
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

## Performance Considerations

### Decode Throughput

H.264 decode is highly optimized. Typical throughput on modern hardware:

| Platform | Approximate Throughput |
|----------|------------------------|
| CPU (single core) | 100-300 fps (1080p) |
| CPU (multi-core) | 500+ fps (1080p) |
| GPU (NVDEC) | 1000+ fps (1080p) |

For training pipelines processing thousands of videos, GPU decode prevents data loading from becoming a bottleneck.

### Memory Usage

Video decoding requires buffering for B-frame prediction and reference frames:
- Baseline profile: 2-3 reference frames
- Main/High profile: Up to 16 reference frames

Each reference frame at 1080p requires ~6 MB (YUV 4:2:0). High-resolution or high-reference-count videos can consume significant memory.

### Bitrate vs Quality Trade-offs

For a given visual quality level, bitrate requirements vary significantly with content:

| Content Type | Typical Bitrate (1080p, good quality) |
|--------------|--------------------------------------|
| Static/low motion | 2-4 Mbps |
| Medium motion | 4-8 Mbps |
| High motion/sports | 8-15 Mbps |
| Complex scenes | 10-20 Mbps |

High bitrate generally means less compression artifacts but larger files. Consider your storage and bandwidth constraints.

## Comparison with Other Codecs

### H.264 vs H.265 (HEVC)

| Aspect | H.264 | H.265 |
|--------|-------|-------|
| Compression | Baseline | ~40-50% smaller at same quality |
| Decode complexity | Lower | Higher |
| Hardware support | Universal | Widespread but not universal |
| Encode speed | Faster | Slower |
| Licensing | Established | More complex |

Use H.265 when you need better compression and can guarantee decode support.

### H.264 vs VP9

| Aspect | H.264 | VP9 |
|--------|-------|-----|
| Compression | Baseline | ~30-40% smaller |
| Royalties | Yes | Royalty-free |
| Browser support | Universal | Good (all modern browsers) |
| Hardware support | Universal | Good but not universal |

VP9 is a solid royalty-free alternative with good compression.

### H.264 vs AV1

| Aspect | H.264 | AV1 |
|--------|-------|-----|
| Compression | Baseline | ~50% smaller |
| Encode speed | Fast | Very slow |
| Decode complexity | Low | Higher |
| Royalties | Yes | Royalty-free |
| Hardware support | Universal | Growing |

AV1 offers the best compression but with significant encoding cost.

## Practical Recommendations

1. **Dataset Preparation**: For training datasets, consider re-encoding to consistent parameters (resolution, framerate, codec settings) to reduce variability.

2. **Storage Optimization**: Use CRF-based encoding at quality levels appropriate for your task. CRF 23-28 is often sufficient for training data.

3. **Random Access**: If you need random frame access, use shorter GOPs or extract frames to images.

4. **Hardware Decode**: Always use hardware-accelerated decoding for large-scale processing.

5. **Color Accuracy**: Be aware of YUV-to-RGB conversion and potential color space issues when precise color matching matters.
