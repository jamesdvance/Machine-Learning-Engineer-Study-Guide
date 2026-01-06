# Keras

## Summary

Keras is a high-level neural network API designed for fast experimentation. Originally built on top of TensorFlow, Keras 3 now supports multiple backends including TensorFlow, JAX, and PyTorch. Its simple, consistent interface makes it ideal for rapid prototyping while still supporting production deployment and distributed training.

Key points to remember:

- High-level API for building and training neural networks
- Backend-agnostic: TensorFlow, JAX, or PyTorch
- Sequential and Functional APIs for model building
- Built-in training loop with fit/evaluate/predict
- Callbacks for customization during training
- Distribution strategies for multi-GPU/TPU training
- Seamless integration with TensorFlow ecosystem
- Extensive pre-built layers and models

## Core APIs

### Sequential API

```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Functional API

```python
import keras
from keras import layers

# Define inputs
inputs = keras.Input(shape=(784,))

# Build the graph
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing API

```python
import keras
from keras import layers

class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.classifier = layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

model = MyModel()
```

## Training

### Basic Training

```python
# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)
```

### With Datasets

```python
import tensorflow as tf

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Train
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### Custom Training Loop

```python
import keras
import tensorflow as tf

model = MyModel()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss:.4f}")
```

## Callbacks

### Built-in Callbacks

```python
from keras import callbacks

callback_list = [
    # Save best model
    callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    ),

    # Early stopping
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    # Learning rate reduction
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),

    # TensorBoard logging
    callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    ),

    # CSV logging
    callbacks.CSVLogger('training.log')
]

model.fit(
    train_data,
    epochs=100,
    validation_data=val_data,
    callbacks=callback_list
)
```

### Custom Callbacks

```python
from keras import callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")

    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch}: loss={logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        print("Training complete!")
```

## Distributed Training

### MirroredStrategy (Multi-GPU)

```python
import tensorflow as tf
import keras

# Create strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Build and compile under strategy scope
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train as normal (distributed automatically)
model.fit(train_dataset, epochs=10)
```

### TPUStrategy

```python
import tensorflow as tf

# Connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=10)
```

### MultiWorkerMirroredStrategy

```python
import tensorflow as tf
import json
import os

# Set TF_CONFIG environment variable
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['worker0:port', 'worker1:port']
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(...)

model.fit(train_dataset, epochs=10)
```

## Mixed Precision

### Enabling Mixed Precision

```python
import keras

# Enable globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Build model (layers automatically use FP16)
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10)  # Output in FP32 for stability
])

# For the output layer, you may want explicit FP32
outputs = keras.layers.Activation('softmax', dtype='float32')(x)
```

### With XLA Compilation

```python
import keras

# Enable XLA
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    jit_compile=True  # Enable XLA
)
```

## Custom Layers

### Simple Custom Layer

```python
import keras
from keras import layers

class MyDenseLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Layer with Multiple Outputs

```python
class MultiOutputLayer(layers.Layer):
    def __init__(self, units1, units2):
        super().__init__()
        self.dense1 = layers.Dense(units1)
        self.dense2 = layers.Dense(units2)

    def call(self, inputs):
        return self.dense1(inputs), self.dense2(inputs)
```

## Custom Training

### Custom Loss Function

```python
import keras
import tensorflow as tf

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Or as a class
class CustomLoss(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1):
        super().__init__()
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.reduce_mean(tf.abs(y_pred))
        return mse + self.regularization_factor * reg

model.compile(optimizer='adam', loss=CustomLoss(0.1))
```

### Custom Metrics

```python
import keras
import tensorflow as tf

class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + 1e-7)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score()])
```

## Model Saving and Loading

### Native Keras Format

```python
# Save entire model
model.save('model.keras')

# Load model
loaded_model = keras.models.load_model('model.keras')
```

### Weights Only

```python
# Save weights
model.save_weights('weights.weights.h5')

# Load weights (model must exist)
model.load_weights('weights.weights.h5')
```

### SavedModel Format

```python
# Save as TensorFlow SavedModel
model.export('saved_model')

# Load
loaded = tf.saved_model.load('saved_model')
```

### ONNX Export

```python
import tf2onnx

# Convert to ONNX
spec = (tf.TensorSpec((None, 784), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

## Keras 3 Multi-Backend

### Backend Selection

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'  # or 'tensorflow' or 'torch'

import keras
```

### Backend-Agnostic Code

```python
import keras
from keras import ops  # Backend-agnostic operations

class MyLayer(keras.layers.Layer):
    def call(self, inputs):
        # Use keras.ops instead of tf/jax/torch ops
        return ops.relu(ops.matmul(inputs, self.w))
```

### PyTorch Backend

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import torch

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10)
])

# Model parameters are torch tensors
for param in model.parameters():
    print(type(param))  # <class 'torch.nn.Parameter'>
```

## Pre-trained Models

### Using Pre-trained Models

```python
from keras.applications import ResNet50, VGG16, EfficientNetB0

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add custom head
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

### Fine-tuning

```python
# After training the head, unfreeze some layers
base_model.trainable = True

# Freeze all but last N layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)
```

## Data Augmentation

### Built-in Augmentation Layers

```python
from keras import layers

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# Use in model
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])
```

## Best Practices

1. **Use Functional API**: More flexible than Sequential
2. **Enable mixed precision**: For faster training
3. **Use tf.data**: For efficient data loading
4. **Apply callbacks**: For monitoring and early stopping
5. **Use pre-trained models**: Transfer learning is powerful
6. **Compile with jit_compile**: For XLA optimization
7. **Monitor with TensorBoard**: Built-in callback support

## Debugging

```python
# Print model summary
model.summary()

# Check layer outputs
for layer in model.layers:
    print(layer.name, layer.output_shape)

# Eager execution for debugging
tf.config.run_functions_eagerly(True)

# Debug mode
keras.config.disable_traceback_filtering()
```
