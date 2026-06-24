import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cnn_model import SimpleCNN


torch.manual_seed(42)
np.random.seed(42)

# Check which GPU is available
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(f"Using device: {device}")

BATCH_SIZE = 32
EPOCHS = 5


def train():
    # Prepare the data
    transform = transforms.Compose([
    	transforms.Resize((64, 64)),
   	transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Create the CNN model
    model = SimpleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_number, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = running_correct / running_total

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {train_loss:.4f}, "
            f"Accuracy: {train_accuracy:.4f}"
        )

    print("Finished Training")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cifar10_cnn.pt")
    print("Model saved to models/cifar10_cnn.pt")


if __name__ == "__main__":
    train()