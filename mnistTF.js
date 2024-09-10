import * as tf from '@tensorflow/tfjs-node-gpu';  // Use TensorFlow.js with GPU
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';

// Check if GPU is available
if (tf.ENV.get('WEBGL_RENDER_FLOAT32_CAPABLE')) {
  console.log('TensorFlow.js is using WebGL for GPU acceleration.');
} else {
  console.log('TensorFlow.js is not using GPU. Press enter to continue.');
}

// Directory structure for MNIST data
const mnistDir = './data/MNIST/raw';

function loadMNISTImages(filename) {
  const buffer = zlib.gunzipSync(fs.readFileSync(filename));  // Decompress the .gz file
  const magic = buffer.readUInt32BE(0);
  const numImages = buffer.readUInt32BE(4);
  const rows = buffer.readUInt32BE(8);
  const cols = buffer.readUInt32BE(12);
  const imageData = buffer.slice(16);

  const images = new Float32Array(numImages * rows * cols);
  for (let i = 0; i < imageData.length; i++) {
    images[i] = imageData[i] / 255.0;  // Normalize to 0-1
  }

  return tf.tensor4d(images, [numImages, rows, cols, 1]);  // Shape (numImages, 28, 28, 1)
}
function loadMNISTLabels(filename) {
    const buffer = zlib.gunzipSync(fs.readFileSync(filename));  // Decompress the .gz file
    const labels = buffer.slice(8);  // Skip the magic number
    return tf.tensor1d(labels, 'int32').toFloat();  // Convert to float32 tensor
  }
  

// Load train and test data
const xTrain = loadMNISTImages(path.join(mnistDir, 'train-images-idx3-ubyte.gz'));
const yTrain = loadMNISTLabels(path.join(mnistDir, 'train-labels-idx1-ubyte.gz'));
const xTest = loadMNISTImages(path.join(mnistDir, 't10k-images-idx3-ubyte.gz'));
const yTest = loadMNISTLabels(path.join(mnistDir, 't10k-labels-idx1-ubyte.gz'));

// Build the model
const model = tf.sequential();
model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));  // Flatten input
model.add(tf.layers.dense({units: 128, activation: 'relu'}));  // Fully connected layer
model.add(tf.layers.dropout({rate: 0.2}));  // Dropout layer
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));  // Output layer

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// Train the model
async function trainModel() {
  await model.fit(xTrain, yTrain, {
    epochs: 10,
    validationData: [xTest, yTest],
  });

  // Evaluate the model
  const evalResult = model.evaluate(xTest, yTest);
  const testAcc = evalResult[1].dataSync();
  console.log(`Test accuracy: ${testAcc}`);

  // Save the model to TensorFlow.js format
  const tfjsTargetDir = './model/web_model';
  await model.save(`file://${tfjsTargetDir}`);
  console.log(`Model saved to ${tfjsTargetDir}`);
}

// Start training
trainModel();
