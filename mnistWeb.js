let model, xTest, yTest, randomImage;

// Function to draw the MNIST image on canvas
function drawMNISTImage(imageTensor) {
    const canvas = document.getElementById('mnistImage');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(28, 28);

    const data = imageTensor.dataSync();

    for (let i = 0; i < data.length; i++) {
        const value = Math.floor(data[i] * 255);
        imageData.data[i * 4] = value;        // Red channel
        imageData.data[i * 4 + 1] = value;    // Green channel
        imageData.data[i * 4 + 2] = value;    // Blue channel
        imageData.data[i * 4 + 3] = 255;      // Alpha channel (fully opaque)
    }

    ctx.putImageData(imageData, 0, 0);
}

// Function to pick a random MNIST image from the test set
function pickRandomImage() {
    const randomIndex = Math.floor(Math.random() * xTest.shape[0]);
    randomImage = xTest.slice([randomIndex, 0, 0, 0], [1, 28, 28, 1]);
    drawMNISTImage(randomImage.flatten());
}

// Function to predict the digit of the selected image
async function predictDigit() {
    const prediction = model.predict(randomImage);
    const predictedDigit = prediction.argMax(1).dataSync()[0];
    const predictoutput = document.getElementById('predictoutput');
    predictoutput.textContent = `Predicted digit: ${predictedDigit}`;
}

// Function to load and train the MNIST model
async function trainModel() {
    tf.setBackend('webgl');
    await tf.ready(); // Ensure that TensorFlow.js is fully initialized with the WebGL backend.

    const output = document.getElementById('output');
    output.textContent = "";

    // Check if WebGL backend (GPU) is available
    if (tf.engine().backendName === 'webgl') {
        output.textContent += 'GPU available. Using WebGL backend.\n';
        const gl = tf.backend().getGPGPUContext().gl;
        const rendererInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const gpuName = rendererInfo ? gl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown GPU';
        output.textContent += `GPU Name: ${gpuName}\n`;
    } else {
        output.textContent += 'No GPU available. Falling back to CPU.\n';
    }

    // URLs for MNIST data (in your server or a similar source)
    const prefix = './data/MNIST/raw/';
    const trainImagesUrl = prefix + 'train-images-idx3-ubyte.gz'; // Change to your path
    const trainLabelsUrl = prefix + 'train-labels-idx1-ubyte.gz'; // Change to your path
    const testImagesUrl = prefix + 't10k-images-idx3-ubyte.gz'; // Change to your path
    const testLabelsUrl = prefix + 't10k-labels-idx1-ubyte.gz'; // Change to your path

    output.textContent += 'Loading data...\n';

    // Load data
    const xTrain = await loadMNISTImages(trainImagesUrl);
    const yTrain = await loadMNISTLabels(trainLabelsUrl);
    xTest = await loadMNISTImages(testImagesUrl);
    yTest = await loadMNISTLabels(testLabelsUrl);

    output.textContent += 'Data loaded.\n';

    // Build the model
    model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    output.textContent += 'Training model...\n';

    // Train the model
    await model.fit(xTrain, yTrain, {
        epochs: 5,
        validationData: [xTest, yTest],
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                output.textContent += `Epoch ${epoch + 1} - Loss: ${logs.loss.toFixed(4)}, Accuracy: ${(logs.acc * 100).toFixed(2)}%\n`;
            }
        }
    });

    output.textContent += 'Model training complete.\n';

    // Show the "Random" and "Predict" buttons after training
    document.getElementById('randomBtn').style.display = 'inline';
    document.getElementById('predictBtn').style.display = 'inline';

    // Pick an initial random image
    pickRandomImage();
}

// Load MNIST images function (unchanged)
async function loadMNISTImages(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const data = pako.ungzip(new Uint8Array(arrayBuffer));

    const numImages = new DataView(data.buffer).getUint32(4);
    const rows = new DataView(data.buffer).getUint32(8);
    const cols = new DataView(data.buffer).getUint32(12);

    const images = new Float32Array(numImages * rows * cols);
    for (let i = 0, j = 16; i < images.length; i++, j++) {
        images[i] = data[j] / 255.0; // Normalize to 0-1
    }

    return tf.tensor4d(images, [numImages, rows, cols, 1]);
}

// Load MNIST labels function (unchanged)
async function loadMNISTLabels(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const data = pako.ungzip(new Uint8Array(arrayBuffer));

    const labels = new Uint8Array(data.slice(8)); // Skip the magic number
    return tf.tensor1d(labels, 'int32').toFloat();
}

// Add event listeners for buttons
document.getElementById('trainBtn').addEventListener('click', trainModel);
document.getElementById('randomBtn').addEventListener('click', pickRandomImage);
document.getElementById('predictBtn').addEventListener('click', predictDigit);
