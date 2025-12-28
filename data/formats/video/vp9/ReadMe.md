# VP9 Video Codec

## Summary

VP9 is a royalty-free video codec developed by Google, released in 2013 as the successor to VP8. It achieves compression efficiency comparable to H.265/HEVC while being completely open-source and free from licensing fees. VP9 is the primary codec for YouTube and is widely supported in web browsers, making it common in web-sourced video datasets. For ML practitioners, VP9 offers a strong balance of compression, compatibility, and freedom from patent concerns.

Key points to remember:

- Royalty-free and open-source (BSD license)
- Compression efficiency similar to H.265 (30-50% better than H.264)
- Excellent browser support (Chrome, Firefox, Edge, Safari on macOS 11+)
- Native support for 10-bit and 12-bit color depths
- Hardware decode support widespread but not universal
- Container formats: WebM (primary), MP4, MKV

Profiles:

- Profile 0: 8-bit 4:2:0, most common
- Profile 1: 8-bit 4:2:2 and 4:4:4
- Profile 2: 10-bit and 12-bit 4:2:0, HDR content
- Profile 3: 10-bit and 12-bit 4:2:2 and 4:4:4

When to use VP9:

- Web delivery without licensing concerns
- YouTube or web-scraped video content (likely already VP9)
- Cross-platform compatibility with royalty-free requirement
- Good compression without H.265 licensing complexity
- Browser-based video playback

When to consider alternatives:

- Maximum compatibility with older devices (use H.264)
- Best possible compression (use AV1)
- Real-time encoding with limited CPU (use H.264)
- Hardware encode required (support varies)

---

## VP9 Technical Overview

VP9 uses a similar block-based hybrid coding approach to H.264 and H.265 but with different design choices that achieve comparable compression.

### Block Structure

**Superblocks**: VP9 uses superblocks up to 64x64 pixels, similar to H.265's CTUs. These can be subdivided recursively to 4x4 blocks.

**Block Partitioning**: Supports square and rectangular partitions, allowing adaptation to content. Non-square partitions can be more efficient for edges and directional content.

### Prediction

**Intra Prediction**: 10 modes including directional predictions at various angles, DC, true motion, and horizontal/vertical.

**Inter Prediction**: Advanced motion vector prediction with multiple reference frames. Supports compound prediction (blending two predictions).

**Reference Frames**: Up to 8 reference frames with flexible selection per block. This allows efficient coding of complex motion.

### Transforms and Quantization

- Transform sizes from 4x4 to 32x32
- Hybrid transforms using DCT and ADST (Asymmetric Discrete Sine Transform)
- Adaptive quantization with segment-based quality variation

### Entropy Coding

Uses boolean arithmetic coding throughout, similar to CABAC but with some different context models. Frame-level updates to probability models improve efficiency.

## VP9 in Machine Learning Workflows

### Decoding VP9 Video

VP9 is well-supported in Python video libraries:

**OpenCV**:
```python
import cv2

cap = cv2.VideoCapture("video.webm")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame is BGR numpy array
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**PyAV**:
```python
import av

container = av.open("video.webm")
for frame in container.decode(video=0):
    array = frame.to_ndarray(format="rgb24")
```

**Decord**:
```python
from decord import VideoReader, cpu, gpu

# CPU decode
vr = VideoReader("video.webm", ctx=cpu(0))

# GPU decode if supported
vr = VideoReader("video.webm", ctx=gpu(0))

frames = vr.get_batch([0, 30, 60]).asnumpy()
```

### YouTube and Web Video Datasets

Many ML video datasets are sourced from YouTube, which delivers VP9 as the default codec. When downloading YouTube videos:

**yt-dlp (recommended)**:
```bash
# Download best VP9 video + audio
yt-dlp -f "bestvideo[vcodec^=vp9]+bestaudio" URL

# Download specific resolution
yt-dlp -f "bestvideo[height<=720][vcodec^=vp9]+bestaudio" URL

# List available formats
yt-dlp -F URL
```

**Programmatic access**:
```python
import yt_dlp

ydl_opts = {
    'format': 'bestvideo[vcodec^=vp9][height<=1080]+bestaudio/best',
    'outtmpl': '%(id)s.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['VIDEO_URL'])
```

### Hardware Decode Support

VP9 hardware decode is widely but not universally supported:

| Platform | Support Status |
|----------|---------------|
| NVIDIA (Pascal+) | Full support |
| AMD (Polaris+) | Full support |
| Intel (Kaby Lake+) | Full support |
| Apple M1/M2 | Full support |
| Apple Intel | Limited/No |
| Older GPUs | CPU fallback |

Check support and enable hardware decode:
```bash
# FFmpeg with NVIDIA hardware decode
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input.webm ...

# FFmpeg with VAAPI (Linux Intel/AMD)
ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -i input.webm ...
```

### Encoding VP9 Video

VP9 encoding is relatively slow but produces excellent results:

**Basic encoding**:
```bash
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 output.webm
```

**Two-pass encoding for better quality**:
```bash
# Pass 1
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 2M -pass 1 -f webm /dev/null

# Pass 2
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 2M -pass 2 output.webm
```

**Python with subprocess**:
```python
import subprocess

def encode_vp9(input_path, output_path, crf=30):
    subprocess.run([
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libvpx-vp9',
        '-crf', str(crf),
        '-b:v', '0',  # Constant quality mode
        '-row-mt', '1',  # Enable row-based multithreading
        output_path
    ], check=True)
```

## Encoding Parameters

### Quality Settings

VP9 uses a CRF-like quality parameter (quantizer):

| CRF/Q Value | Quality Level | Typical Use |
|-------------|---------------|-------------|
| 15-20 | High quality | Archival, source |
| 20-30 | Good quality | General use |
| 30-40 | Medium quality | Web delivery |
| 40-50 | Lower quality | Bandwidth-limited |

Note: VP9's quality scale differs from H.264/H.265. Values are generally higher for equivalent visual quality.

### Speed vs Quality Trade-off

The `-cpu-used` parameter controls encoding speed:

| Value | Speed | Compression |
|-------|-------|-------------|
| 0 | Slowest | Best |
| 1-2 | Slow | Very good |
| 3-4 | Medium | Good |
| 5-6 | Fast | Moderate |
| 7-8 | Very fast | Lower |

For ML preprocessing where encoding is a one-time cost:
```bash
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 25 -b:v 0 -cpu-used 1 output.webm
```

### Multithreading

VP9 encoding benefits significantly from multithreading:

```bash
# Enable row-based multithreading
ffmpeg -i input.mp4 -c:v libvpx-vp9 -row-mt 1 -threads 8 output.webm

# Tile-based parallelism for larger resolutions
ffmpeg -i input_4k.mp4 -c:v libvpx-vp9 -tile-columns 2 -tile-rows 1 output.webm
```

## WebM Container Format

VP9 is primarily used with the WebM container, which is a subset of Matroska (MKV).

### WebM Characteristics

- Designed for web delivery
- Supports VP8, VP9, AV1 video codecs
- Supports Vorbis and Opus audio codecs
- Royalty-free like the codecs it contains
- Good browser support

### Working with WebM

```python
import av

# Reading WebM
container = av.open("video.webm")
video_stream = container.streams.video[0]
print(f"Codec: {video_stream.codec_context.name}")
print(f"Resolution: {video_stream.width}x{video_stream.height}")

# Writing WebM
output = av.open("output.webm", "w")
stream = output.add_stream("libvpx-vp9", rate=30)
stream.width = 1920
stream.height = 1080
stream.pix_fmt = "yuv420p"

for frame in frames:
    packet = stream.encode(frame)
    output.mux(packet)

output.close()
```

## Performance Considerations

### Decode Performance

VP9 decode is moderately complex, between H.264 and H.265:

| Platform | Approximate Throughput (1080p) |
|----------|-------------------------------|
| CPU (1 core) | 100-200 fps |
| CPU (multi-core) | 400-600 fps |
| GPU (hardware) | 800+ fps |

For large-scale ML training, hardware decode is recommended but CPU decode is often acceptable for VP9.

### Encode Performance

VP9 encoding is slower than H.264 but faster than AV1:

| Preset (cpu-used) | Relative Speed | Notes |
|-------------------|----------------|-------|
| 0 | 1x | Reference quality |
| 4 | 5-8x | Good balance |
| 8 | 15-20x | Fast, lower quality |

Two-pass encoding improves quality but doubles encode time.

### Memory Usage

VP9 decoding requires moderate memory for reference frames and internal buffers. Typical requirements:
- 1080p: 100-200 MB
- 4K: 400-800 MB

This is comparable to H.264 and less than H.265.

## Comparison with Other Codecs

### VP9 vs H.264

| Aspect | VP9 | H.264 |
|--------|-----|-------|
| Compression | 30-50% smaller | Baseline |
| Royalties | Free | Yes |
| Encode speed | Slower | Faster |
| Hardware support | Good | Universal |
| Browser support | Excellent | Excellent |

VP9 is a good H.264 replacement when royalty-free is important.

### VP9 vs H.265

| Aspect | VP9 | H.265 |
|--------|-----|-------|
| Compression | Similar | Similar |
| Royalties | Free | Complex licensing |
| Browser support | Excellent | Limited |
| Hardware support | Good | Widespread |
| HDR support | Good | Excellent |

VP9 and H.265 are roughly equivalent in compression, but VP9 has better browser support and simpler licensing.

### VP9 vs AV1

| Aspect | VP9 | AV1 |
|--------|-----|-----|
| Compression | Baseline | 20-30% smaller |
| Encode speed | Moderate | Very slow |
| Hardware decode | Widespread | Growing |
| Maturity | Established | Newer |

VP9 is more practical when AV1's encoding cost is prohibitive.

## Practical Recommendations

### For Web-Sourced Datasets

1. **YouTube Content**: Expect VP9 encoding. Download in native VP9 to avoid transcoding losses.

2. **Quality Selection**: For training data, prefer higher quality VP9 variants when available.

3. **Format Preservation**: Keep videos in WebM/VP9 rather than transcoding unless necessary.

### For Dataset Preparation

1. **Standardization**: When combining sources, consider standardizing to VP9 for consistency.

2. **Quality Settings**: CRF 25-30 provides good quality for training data with reasonable file sizes.

3. **Multithreading**: Always enable row-based multithreading for encoding.

### For Browser-Based Applications

1. **Universal Support**: VP9 works in all modern browsers.

2. **Adaptive Streaming**: VP9 is well-suited for DASH/HLS adaptive streaming.

3. **WebM Container**: Use WebM for maximum browser compatibility.

### For Training Pipelines

1. **Hardware Decode**: Use GPU decode when available for better throughput.

2. **CPU Fallback**: VP9 CPU decode is acceptable for moderate-scale pipelines.

3. **Frame Extraction**: For random access, consider extracting to images if decode overhead is significant.

## Google and Open Web Video

VP9 was developed by Google as part of the WebM project, continuing the effort started with VP8. Key motivations:

- Royalty-free alternative to H.264/H.265
- Open-source implementation (libvpx)
- Web-optimized for HTML5 video
- Foundation for the Alliance for Open Media (AV1)

This open approach has made VP9 the dominant codec for web video delivery, particularly through YouTube's massive scale. For ML practitioners, this means VP9 is likely the format of any web-scraped video content.
