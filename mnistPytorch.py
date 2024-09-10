import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import time

start_time = time.time()

# Check if PyTorch is using a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"PyTorch is using the following GPU(s): {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is not using a GPU.")
    input("Press enter to continue")

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Save the dataset as NumPy binary files (optional)
if not os.path.exists('mnist_train.npz'):
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    np.savez('mnist_train.npz', x_train=x_train, y_train=y_train)
    np.savez('mnist_test.npz', x_test=x_test, y_test=y_test)

# Define the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)  # Fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)   # Dropout to reduce overfitting
        self.fc2 = nn.Linear(128, 10)    # Output layer for 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = NeuralNet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
