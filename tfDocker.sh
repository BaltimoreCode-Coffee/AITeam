docker run -u $(id -u):$(id -g) --gpus all --net=host -v $(pwd):/workspace -it tensorflow/tensorflow:latest bash
