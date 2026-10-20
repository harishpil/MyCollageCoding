import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# Convert each image into a PyTorch tensor
transform = transforms.ToTensor()

# Download and load the training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download and load the test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Load data in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


# Create the model
model = MNISTClassifier()

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 5 epochs
epochs = 5

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 1. Clear previous gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)

        # 3. Calculate loss
        loss = loss_fn(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # Track loss and training accuracy
        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    # print(
    #     f"Epoch [{epoch + 1}/{epochs}] "
    #     f"Loss: {avg_loss:.4f} "
    #     f"Accuracy: {accuracy:.2f}%"
    # )


# Evaluate the model on unseen test data
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

# print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Save the trained model weights
torch.save(model.state_dict(), "mnist_model.pth")
# print("Model saved as mnist_model.pth")


# Load the saved model weights into a fresh model
loaded_model = MNISTClassifier()
loaded_model.load_state_dict(
    torch.load("mnist_model.pth", map_location="cpu", weights_only=True)
)
loaded_model.eval()

# Get one image from the test dataset
image, true_label = test_dataset[0]

# Predict without calculating gradients
with torch.no_grad():
    output = loaded_model(image.unsqueeze(0))
    predicted_label = output.argmax(dim=1).item()

# print("\nSingle-image prediction:")
# print(f"Actual digit: {true_label}")
# print(f"Predicted digit: {predicted_label}")


import matplotlib.pyplot as plt

# Display the image and prediction
plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Actual: {true_label} | Predicted: {predicted_label}"
)
plt.axis("off")
plt.show()


