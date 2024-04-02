import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import mobilenet_v2
import torch.optim as optim

import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label


class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True).features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 2)  # Assuming 2 classes

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def plot_training_results(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(model, train_loader, optimizer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / total}")

    return losses, accuracies


# Setup data loader
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Update these paths
csv_file = "augmented_images/image_labels.csv"
img_dir = "augmented_images/"

dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model and optimizer
model = PrototypicalNetwork().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start training
losses, accuracies = train(model, train_loader, optimizer, epochs=20)

# Plot the training results
plot_training_results(losses, accuracies)
