import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
import gzip
import struct
import tensorflowjs as tfjs

startTime = time.time()

# Check if TensorFlow is using a GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the following GPU(s):")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(gpu)
else:
    print("TensorFlow is not using a GPU.")
    input("Press enter to continue")

# Directory structure for MNIST data
mnist_dir = './data/MNIST/raw'

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)  # skip the magic number and number of labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load train and test data
x_train = load_mnist_images(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'))
y_train = load_mnist_labels(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'))
x_test = load_mnist_images(os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz'))
y_test = load_mnist_labels(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz'))

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D vectors
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    layers.Dropout(0.2),  # Dropout layer to reduce overfitting
    layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc:.4f}")
print(f"Elapsed time: {time.time()-startTime}")

saveModel = True
if saveModel:
    # Save the model in TensorFlow.js format
    tfjs_target_dir = './model/web_model'
    tfjs.converters.save_keras_model(model, tfjs_target_dir)

    print(f"Model saved to {tfjs_target_dir}")
