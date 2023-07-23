# Quantization

Quantization refers to lowering the precision of the data types used to store neural network weights in order to save memory

## Quantization Methods

### Post-Training Quantization
Weights of an already-trained model are converted to lower precision without requiring retraining. Creates performance degradation. 

### Quantization-Aware Training
Incorporates weight conversion process during the pre-training or fine-tuning stage, which improves model performoance compared to PQT. But is computationally expensive and requires representative training data. 

## Floating Point Storage
Uses n bits to store a numerical value. These n bits are paritioned into three distinct components. 
1. Sign - 0 indicates positive, 1 indicates negative
2. Exponent - 
3. Significand - significant digits of the number

### Common Floating Point Data Types
* FP32 - uses 32 bits (4 bytes) to represent a number. One for the sign, 8 for the exponent, and 23 for the significand. 
* FP16 - 16 bits. One for sign, five for exponent, ten for significand
* BF16 - 16 bit format with one bit for the sign, eight for the exponent, seven for significand. This expands how many numbers it can represent (the representative range) which decreases underflow and overflow risk.

FP32 is often called 'full precision' (4 bytes) while FP16 and BF16 are 'half precision' (2 bytes).

### Integer Data Types
INT8 can store weights in one byte. It can store 2^8 = 256 different values

### 8-Bit Quantization Methods

#### Absolute Minimum Quantization
Original number is divided by the absolute maximum value (the max of the abs) of the tensor and multiplied by a scaling factor of 127 and rounded to map the inputs into the range [-127,127]. To retrieve the original FP32 (dequantization) the INT8 number is divided by the quantization factor (absolute maximum value of the tensor / 127). Considered *symmetric* quantization because its symmetric around 0.


#### Zero Point Quantization
Inputs are first scaled by the total range of values (127 - -128 = 255). Then the distribution is divided by the range (max(Tensor)-min(Tensor)). This distribution is then shifted by the zero-point to map it onto the range [-128, 127]. 
* Scale: 255 / (max(X)-min(X))
* Zeropoint: -round(scale*min(X)) - 128
* Quantize: round(scale * X + zeropoint)
* Dequantize: (Xquant - zeropoint)/scale
Considered *assymetric* because it is not symmetric ([-128, 127])

#### GTPQ


## Resources
[1] [TDS - Intro To Weight Quantization](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)