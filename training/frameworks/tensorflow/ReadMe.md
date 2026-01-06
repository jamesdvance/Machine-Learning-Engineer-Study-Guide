# TensorFlow

## Summary

TensorFlow is a comprehensive open-source platform for machine learning developed by Google. It provides a complete ecosystem for building, training, and deploying ML models at scale. With its production-focused tools, distribution strategies, and deployment options (TF Serving, TF Lite, TF.js), TensorFlow excels in enterprise environments and production pipelines.

Key points to remember:

- End-to-end ML platform with training and deployment tools
- Eager execution by default (TF 2.x), graph mode with @tf.function
- Keras is the high-level API (now multi-backend in Keras 3)
- Distribution strategies for multi-GPU/TPU training
- tf.data for efficient input pipelines
- TensorBoard for visualization and debugging
- TF Serving for production model serving
- Strong mobile/edge support via TF Lite

## Core Concepts

### Tensors and Operations

```python
import tensorflow as tf

# Create tensors
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.Variable([[1.0, 2.0], [3.0, 4.0]])  # Trainable

# Operations
z = tf.matmul(x, y)
z = tf.nn.relu(z)

# NumPy interop
numpy_array = z.numpy()
```

### Automatic Differentiation

```python
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

# Compute gradient: dy/dx = 2x = 6
grad = tape.gradient(y, x)
print(grad)  # tf.Tensor(6.0, ...)
```

### GradientTape for Training

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function  # Compile to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## tf.function and Graphs

### Graph Compilation

```python
import tensorflow as tf

@tf.function
def compute(x):
    return tf.reduce_sum(x ** 2)

# First call traces and compiles
result = compute(tf.constant([1.0, 2.0, 3.0]))

# Subsequent calls use cached graph
result = compute(tf.constant([4.0, 5.0, 6.0]))
```

### Input Signatures

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32)
])
def predict(x):
    return model(x)

# Ensures consistent tracing
```

### Autograph

```python
@tf.function
def conditional(x):
    # Python control flow converted to tf.cond
    if tf.reduce_sum(x) > 0:
        return x * 2
    else:
        return x * 3

# Loops converted to tf.while_loop
@tf.function
def loop_example(n):
    result = 0
    for i in tf.range(n):
        result += i
    return result
```

## Data Input with tf.data

### Basic Pipeline

```python
import tensorflow as tf

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Apply transformations
dataset = (dataset
    .shuffle(buffer_size=10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE))

# Iterate
for batch_x, batch_y in dataset:
    train_step(batch_x, batch_y)
```

### Performance Optimization

```python
# Parallel loading and preprocessing
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=4,
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### TFRecord Format

```python
# Writing TFRecords
def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for image, label in dataset:
        writer.write(serialize_example(image, label))

# Reading TFRecords
def parse_fn(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

dataset = tf.data.TFRecordDataset('data.tfrecord')
dataset = dataset.map(parse_fn)
```

## Distribution Strategies

### MirroredStrategy (Multi-GPU)

```python
import tensorflow as tf

# Automatically uses all available GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training is automatically distributed
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
        'worker': ['worker0:12345', 'worker1:12345']
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

# Each worker runs this script
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
    model.compile(...)

model.fit(train_dataset, epochs=10)
```

### ParameterServerStrategy

```python
import tensorflow as tf

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

if cluster_resolver.task_type == 'ps':
    # Parameter server
    server = tf.distribute.Server(cluster_resolver.cluster_spec())
    server.join()
else:
    # Worker
    strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
    with strategy.scope():
        model = build_model()
```

## Mixed Precision

### Enabling Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Set policy globally
mixed_precision.set_global_policy('mixed_float16')

# Build model (automatically uses FP16 for compute, FP32 for variables)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Loss scaling is automatic
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### With Custom Training Loop

```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## Checkpointing and Saving

### Keras Model Save

```python
# Save entire model
model.save('my_model.keras')

# Load model
loaded_model = tf.keras.models.load_model('my_model.keras')
```

### SavedModel Format

```python
# Save as SavedModel (for serving)
tf.saved_model.save(model, 'saved_model_dir')

# Load
loaded = tf.saved_model.load('saved_model_dir')
```

### Checkpointing

```python
import tensorflow as tf

checkpoint = tf.train.Checkpoint(
    model=model,
    optimizer=optimizer,
    step=tf.Variable(0)
)

manager = tf.train.CheckpointManager(
    checkpoint,
    directory='./checkpoints',
    max_to_keep=3
)

# Save
manager.save()

# Restore
checkpoint.restore(manager.latest_checkpoint)
```

### Keras Callbacks

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'model_{epoch}.keras',
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3
    )
]

model.fit(train_data, epochs=100, callbacks=callbacks)
```

## TensorBoard

### Basic Logging

```python
import tensorflow as tf
from datetime import datetime

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

### Custom Logging

```python
import tensorflow as tf

writer = tf.summary.create_file_writer('logs/custom')

for step in range(1000):
    loss = train_step(batch)

    with writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.histogram('weights', model.layers[0].weights[0], step=step)
        tf.summary.image('sample', image_batch, step=step)
```

### Profiling

```python
import tensorflow as tf

# Profile training
tf.profiler.experimental.start('logs/profiler')
model.fit(train_data, epochs=1)
tf.profiler.experimental.stop()

# Or use callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    profile_batch='10,20'  # Profile batches 10-20
)
```

## Custom Training

### Custom Layer

```python
import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
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

### Custom Model

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = tf.nn.dropout(x, rate=0.5)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

### Custom Loss

```python
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Or as class
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1):
        super().__init__()
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        reg = self.regularization_factor * tf.reduce_mean(tf.abs(y_pred))
        return mse + reg

model.compile(loss=CustomLoss(0.1))
```

## Production Deployment

### TensorFlow Serving

```python
# Save model for serving
tf.saved_model.save(model, 'saved_model/1')

# Serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def serving_fn(x):
    return {'predictions': model(x)}

tf.saved_model.save(
    model,
    'saved_model/1',
    signatures={'serving_default': serving_fn}
)
```

```bash
# Start TF Serving
docker run -p 8501:8501 \
    -v "$(pwd)/saved_model:/models/my_model" \
    -e MODEL_NAME=my_model \
    tensorflow/serving
```

### TensorFlow Lite

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Run inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
```

## Debugging

### Eager Execution

```python
# Default in TF 2.x
tf.executing_eagerly()  # True

# Force eager for debugging
tf.config.run_functions_eagerly(True)

# Debug inside tf.function
@tf.function
def my_func(x):
    tf.print("Debug:", x)  # Works in graph mode
    return x * 2
```

### Debugging Tools

```python
# Check for NaN
tf.debugging.enable_check_numerics()

# Assert shapes
tf.debugging.assert_shapes([
    (x, ('N', 784)),
    (y, ('N', 10))
])

# Trace execution
tf.summary.trace_on(graph=True, profiler=True)
result = model(x)
tf.summary.trace_export(name="trace", step=0, profiler_outdir='logs')
```

## Best Practices

1. **Use tf.data**: For efficient data loading
2. **Use @tf.function**: For graph compilation
3. **Use distribution strategies**: For multi-GPU/TPU
4. **Enable mixed precision**: For faster training
5. **Use TensorBoard**: For monitoring
6. **Use SavedModel**: For deployment
7. **Profile regularly**: Find bottlenecks
8. **Use checkpoints**: Save training progress
