docker run -u $(id -u):$(id -g) --gpus all --net=host -v $(pwd):/workspace -w /workspace -it pytorch/pytorch:latest bash
