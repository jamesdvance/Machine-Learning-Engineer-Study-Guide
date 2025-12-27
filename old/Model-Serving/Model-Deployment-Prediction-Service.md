<<<<<<< HEAD
# Model Deployment And Prediction Service
Chapter 7, Designing Machine Learning Systems

## Model Compression
Making models smaller provides lower latency and lower memory and disk space needed

### Low-Rank Factorization
* Compresses higher-rank tensors into lower rank tensors
* E.g. from K x K x C -> K x K x 1 + 1 x 1 x C
* So far has been found useful in convolutional filters

### Knowledge distillation
* A small model is trained to mimic a large model or ensembles
* The new model is called the student and the old model is called the teacher
* E.g. Bert -> DistilBERT (40% reduction in size, 60% faster, 97% of accuracy retained)
* Can work despite architectural differences (e.g. GBDT on Transformer) but not always

### Pruning
* Remove sections of a decision tree or neural network where 
* For neural networks its more common to set less-useful parameters to 0 than to remove their nodes from the network
* Some suggest retraining on the sparse model after pruning a neural network by sestting weights to 0

### Quantization
* Reduce parameter's sizes by reducing representation bits
* 32 bits is the standard default
* Using 16 bits to represent a float is called half precision
* Integers only take 8  bits to represent (known as 'fixed-point')
* 1-bit representation for each weight is an extreme use case explored by Xnor.ai
* Reduces memory and computation speed (low level addition is done bit by bit)
* Allows increasing batch size
* Rounding to achieve quantization can lead to errors (overflow/underflow) and degrading performance
* Most major libraries have capability built in
* Can be during training (quantization aware training) or post training
* Quantization during training lets you use less memory, so can train larger models on same hardware
* Low-precision training is supported by Tensor Cores by NVIDIA. Google TPUs also support 16-bit training
* Fixed-point inference is a standard in industry. Some edge devices only support fixed-point inference
=======
# Model Deployment And Prediction Service
Chapter 7, Designing Machine Learning Systems

## Model Compression
Making models smaller provides lower latency and lower memory and disk space needed

### Low-Rank Factorization
* Compresses higher-rank tensors into lower rank tensors
* E.g. from K x K x C -> K x K x 1 + 1 x 1 x C
* So far has been found useful in convolutional filters

### Knowledge distillation
* A small model is trained to mimic a large model or ensembles
* The new model is called the student and the old model is called the teacher
* E.g. Bert -> DistilBERT (40% reduction in size, 60% faster, 97% of accuracy retained)
* Can work despite architectural differences (e.g. GBDT on Transformer) but not always

### Pruning
* Remove sections of a decision tree or neural network where 
* For neural networks its more common to set less-useful parameters to 0 than to remove their nodes from the network
* Some suggest retraining on the sparse model after pruning a neural network by sestting weights to 0

### Quantization
* Reduce parameter's sizes by reducing representation bits
* 32 bits is the standard default
* Using 16 bits to represent a float is called half precision
* Integers only take 8  bits to represent (known as 'fixed-point')
* 1-bit representation for each weight is an extreme use case explored by Xnor.ai
* Reduces memory and computation speed (low level addition is done bit by bit)
* Allows increasing batch size
* Rounding to achieve quantization can lead to errors (overflow/underflow) and degrading performance
* Most major libraries have capability built in
* Can be during training (quantization aware training) or post training
* Quantization during training lets you use less memory, so can train larger models on same hardware
* Low-precision training is supported by Tensor Cores by NVIDIA. Google TPUs also support 16-bit training
* Fixed-point inference is a standard in industry. Some edge devices only support fixed-point inference

## ML On Cloud
* 


## ML On Edge
>>>>>>> 456eae7 (Saving work so far)
