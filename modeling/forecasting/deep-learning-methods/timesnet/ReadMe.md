# TimesNet

## Summary

TimesNet is a general-purpose deep learning architecture for time series analysis that transforms 1D time series into 2D representations by exploiting multi-periodicity. By reshaping sequences based on discovered periods and applying 2D convolutions, TimesNet captures both intra-period and inter-period patterns simultaneously. This approach achieves strong results across five major time series tasks: forecasting, imputation, classification, anomaly detection, and short-term forecasting.

Key points to remember:

- TimesNet converts 1D time series to 2D tensors based on discovered periodicities
- Uses FFT to automatically discover dominant periods in the data
- Applies 2D convolutions (Inception blocks) to capture multi-scale patterns
- Designed as a general backbone for multiple time series tasks
- Particularly effective when data has clear periodic structure
- Achieves competitive results on long-term forecasting benchmarks
- Available in the Time-Series-Library and neuralforecast implementations

## When to Use TimesNet

TimesNet is appropriate when:

- Data has periodic patterns (daily, weekly, yearly cycles)
- You need a single model architecture for multiple tasks
- Long-term forecasting is required
- Multi-scale pattern capture is important
- You want an alternative to Transformer-based approaches
- Standard RNN/Transformer methods underperform

Consider alternatives when:

- Data lacks clear periodicity (consider Transformers or N-BEATS)
- Interpretability is paramount (use TFT or classical methods)
- Computational resources are limited (use simpler methods)
- Very short sequences with no periodic structure

## Core Concept: 1D to 2D Transformation

### The Key Insight

Time series are inherently 1D, but periodicity creates structure that can be better captured in 2D. TimesNet:

1. Discovers dominant periods using FFT
2. Reshapes the 1D sequence into 2D based on period length
3. Applies 2D convolutions to capture:
   - **Intra-period patterns**: Variations within one cycle
   - **Inter-period patterns**: Variations across cycles

```
1D Time Series:  [x_1, x_2, x_3, ..., x_T]
                          |
                    (period = p)
                          |
                          v
2D Tensor:        [[x_1,   x_2,   ..., x_p  ],
                   [x_p+1, x_p+2, ..., x_2p ],
                   [...                     ],
                   [x_T-p, x_T-p+1,..., x_T ]]

Shape: (num_periods, period_length)
```

### Period Discovery via FFT

TimesNet uses Fast Fourier Transform to find dominant frequencies:

```python
# Conceptual period discovery
def discover_periods(x, top_k=3):
    # FFT to frequency domain
    fft_result = torch.fft.rfft(x, dim=1)
    amplitude = torch.abs(fft_result)

    # Find top-k frequencies (excluding DC component)
    _, top_indices = torch.topk(amplitude[:, 1:], k=top_k, dim=1)

    # Convert frequency indices to periods
    periods = (x.shape[1] / (top_indices + 1)).int()
    return periods
```

The model discovers multiple periods (e.g., daily, weekly) and processes each separately.

## Architecture

### Overall Structure

```
Input Time Series (B, T, C)
         |
         v
+-------------------+
| Embedding Layer   |  (Linear projection to d_model dimensions)
+-------------------+
         |
         v
+-------------------+
| TimesBlock x N    |  (Stacked blocks)
+-------------------+
         |
         v
+-------------------+
| Projection Head   |  (Task-specific output)
+-------------------+
         |
         v
Output (forecasts, class labels, etc.)
```

### TimesBlock

Each TimesBlock contains:

```
Input (B, T, d_model)
         |
    +----+----+
    |         |
    v         v
FFT Period    |
Discovery     |
    |         |
    v         |
Reshape to    |
2D (B, p, f, d)|
    |         |
    v         |
Inception     |
2D Conv       |
    |         |
    v         |
Reshape back  |
to 1D         |
    |         |
    v         |
Adaptive      |
Aggregation   |
    |         |
    +----+----+
         |
         v
Residual Connection + LayerNorm
         |
         v
Feed-Forward Network
         |
         v
Output (B, T, d_model)
```

### Inception Block

TimesNet uses Inception-style 2D convolutions with multiple kernel sizes:

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=2)
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out_pool = self.maxpool(x)
        return out1 + out3 + out5 + out_pool
```

### Adaptive Aggregation

When multiple periods are discovered, TimesNet aggregates them with learned weights:

```python
def aggregate_periods(period_features, period_weights):
    # period_features: list of (B, T, d_model) for each period
    # period_weights: learned importance weights from FFT amplitudes

    weights = F.softmax(period_weights, dim=-1)
    aggregated = sum(w * f for w, f in zip(weights, period_features))
    return aggregated
```

## Implementation

### Using Time-Series-Library

```python
# Clone and install
# git clone https://github.com/thuml/Time-Series-Library
# cd Time-Series-Library

from models import TimesNet
import torch

# Model configuration
configs = {
    'seq_len': 96,           # Input sequence length
    'label_len': 48,         # Start token length for decoder
    'pred_len': 96,          # Prediction length
    'enc_in': 7,             # Encoder input size (number of features)
    'd_model': 64,           # Model dimension
    'd_ff': 64,              # Feed-forward dimension
    'e_layers': 2,           # Number of encoder layers
    'top_k': 3,              # Top-k periods to consider
    'num_kernels': 6,        # Number of inception kernels
    'dropout': 0.1,
}

model = TimesNet.Model(configs)

# Input shape: (batch, seq_len, features)
x = torch.randn(32, 96, 7)

# Forward pass
output = model(x)  # Shape: (32, pred_len, features)
```

### Using NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
import pandas as pd

# Prepare data (long format)
df = pd.DataFrame({
    'unique_id': ['series_1'] * len(values),
    'ds': dates,
    'y': values
})

# Configure model
model = TimesNet(
    h=96,                    # Forecast horizon
    input_size=96,           # Input sequence length
    hidden_size=64,          # Hidden dimension
    conv_hidden_size=64,     # Convolution hidden size
    n_layers=2,              # Number of TimesBlocks
    top_k=3,                 # Number of periods
    num_kernels=6,           # Inception kernels
    max_steps=1000,
    learning_rate=1e-3,
)

# Train and predict
nf = NeuralForecast(models=[model], freq='H')
nf.fit(df=df)
forecasts = nf.predict()
```

### Training Loop (Manual)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup
model = TimesNet.Model(configs)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
model.train()
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward
        outputs = model(batch_x)

        # Loss on prediction horizon
        loss = criterion(outputs, batch_y)

        # Backward
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Hyperparameters

### Key Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| d_model | 32-128 | Model dimension, capacity |
| e_layers | 2-4 | Depth, more layers capture complex patterns |
| top_k | 2-5 | Number of periods to consider |
| num_kernels | 4-8 | Inception kernel variety |
| d_ff | 64-256 | Feed-forward dimension |
| dropout | 0.0-0.3 | Regularization |

### Sequence Length Guidelines

```python
# Input length should capture multiple periods
# Rule of thumb: at least 2x the longest period

# Hourly data with daily and weekly patterns
seq_len = 7 * 24 * 2  # 2 weeks = 336 hours

# Daily data with weekly and monthly patterns
seq_len = 30 * 3  # 3 months = 90 days
```

### Period Selection

```python
# top_k determines how many periods to consider
# More periods = more computation but captures more patterns

# Simple daily patterns only
top_k = 1

# Daily + weekly
top_k = 2

# Daily + weekly + monthly
top_k = 3
```

## Multi-Task Capabilities

TimesNet can be adapted for multiple tasks by changing the output head:

### Long-Term Forecasting

```python
class ForecastingHead(nn.Module):
    def __init__(self, d_model, pred_len, c_out):
        self.projection = nn.Linear(d_model, c_out)
        self.pred_len = pred_len

    def forward(self, x):
        # x: (B, T, d_model)
        # Take last pred_len steps and project
        return self.projection(x[:, -self.pred_len:, :])
```

### Imputation

```python
class ImputationHead(nn.Module):
    def __init__(self, d_model, c_out):
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x):
        # Reconstruct full sequence
        return self.projection(x)
```

### Classification

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        self.projection = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Global pooling + classification
        pooled = x.mean(dim=1)  # (B, d_model)
        return self.projection(pooled)
```

### Anomaly Detection

```python
class AnomalyHead(nn.Module):
    def __init__(self, d_model, c_out):
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x):
        # Reconstruct and compare to input
        return self.projection(x)

    def detect(self, x, x_reconstructed):
        # Anomaly score based on reconstruction error
        return torch.mean((x - x_reconstructed) ** 2, dim=-1)
```

## Common Pitfalls

### Pitfall 1: No Clear Periodicity

**Symptom**: Model struggles to find meaningful periods.

**Solution**: TimesNet works best with periodic data. For non-periodic series, consider N-BEATS or Transformers.

### Pitfall 2: Too Short Input Sequence

**Symptom**: Period discovery fails or captures noise.

**Solution**: Input must be long enough to contain multiple complete cycles of the longest period.

### Pitfall 3: Incorrect top_k

**Symptom**: Missing important patterns or capturing spurious frequencies.

**Solution**: Analyze data periodicity first (e.g., with ACF) to choose appropriate top_k.

### Pitfall 4: Memory Issues with Long Sequences

**Symptom**: Out of memory errors.

**Solution**: The 2D reshape increases memory usage. Reduce batch size or sequence length.

## TimesNet vs Alternatives

| Aspect | TimesNet | TFT | N-BEATS | PatchTST |
|--------|----------|-----|---------|----------|
| Core mechanism | 2D Conv on periods | Attention | FC blocks | Patched attention |
| Periodicity handling | Explicit | Implicit | Implicit | Implicit |
| Covariates | Limited | Excellent | No | Limited |
| Interpretability | Medium | High | Medium | Low |
| Multi-task | Designed for | Forecasting | Forecasting | Multi-task |
| Long sequences | Good | Moderate | Good | Excellent |

**Guidance**:
- TimesNet for clearly periodic data across multiple tasks
- TFT for forecasting with rich covariates and interpretability
- N-BEATS for pure univariate forecasting accuracy
- PatchTST for very long sequences with patched attention

## Benchmarks

### Long-Term Forecasting (ETT Datasets)

TimesNet shows competitive performance on standard benchmarks:

| Dataset | Horizon | TimesNet MSE | Autoformer MSE | FEDformer MSE |
|---------|---------|--------------|----------------|---------------|
| ETTh1 | 96 | 0.384 | 0.435 | 0.376 |
| ETTh1 | 336 | 0.434 | 0.456 | 0.420 |
| ETTm1 | 96 | 0.338 | 0.505 | 0.379 |
| ETTm1 | 336 | 0.381 | 0.459 | 0.426 |

Note: Results vary by implementation and hyperparameters.

### Other Tasks

TimesNet demonstrates versatility:
- **Imputation**: Competitive with specialized imputation methods
- **Classification**: Strong on UCR archive
- **Anomaly Detection**: Effective on SMD, MSL, SMAP datasets

## Practical Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Simulated data with multiple periodicities
# Daily data with weekly (7-day) and monthly (30-day) patterns
T = 365 * 2  # 2 years
t = torch.arange(T, dtype=torch.float32)

# Create signal with multiple periods
signal = (
    torch.sin(2 * torch.pi * t / 7) +       # Weekly
    0.5 * torch.sin(2 * torch.pi * t / 30) + # Monthly
    0.1 * torch.randn(T)                     # Noise
)

# Prepare sequences
seq_len = 96
pred_len = 24
X, Y = [], []
for i in range(len(signal) - seq_len - pred_len):
    X.append(signal[i:i+seq_len])
    Y.append(signal[i+seq_len:i+seq_len+pred_len])

X = torch.stack(X).unsqueeze(-1)  # (N, seq_len, 1)
Y = torch.stack(Y).unsqueeze(-1)  # (N, pred_len, 1)

# Split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# DataLoader
train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=32,
    shuffle=True
)

# Model (using Time-Series-Library style config)
class SimpleTimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.top_k = top_k

        self.embed = nn.Linear(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # Embed
        x = self.embed(x)
        # Encode
        x = self.encoder(x)
        # Project last pred_len steps
        x = x[:, -self.pred_len:, :]
        return self.projection(x)

model = SimpleTimesNet(seq_len, pred_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
for epoch in range(50):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, Y_test)
    print(f"Test MSE: {test_loss.item():.4f}")
```

## Key Takeaways

1. **Period discovery is key**: TimesNet's strength comes from explicitly finding and exploiting periodicities via FFT.

2. **2D convolutions for 1D data**: Reshaping time series into 2D based on periods enables powerful convolutional processing.

3. **Multi-task versatility**: The same backbone works for forecasting, imputation, classification, and anomaly detection.

4. **Requires periodic data**: Without clear periodicities, the 2D transformation provides less benefit.

5. **Computationally moderate**: More efficient than full Transformer attention for long sequences.

6. **Multiple periods**: Using top-k periods captures multi-scale temporal patterns.

7. **Benchmark competitive**: Achieves strong results on standard forecasting benchmarks, especially for periodic series.
