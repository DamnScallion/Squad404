import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim

# Dataset class for loading data
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image names and corresponding labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample image.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Prototypical Networks class with Global Average Pooling
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer

    def forward(self, support_images, query_images):
        z_support = self.backbone(support_images)
        z_support = self.global_avg_pool(z_support)
        z_support = z_support.view(z_support.size(0), -1)  # Flatten the tensor

        z_query = self.backbone(query_images)
        z_query = self.global_avg_pool(z_query)
        z_query = z_query.view(z_query.size(0), -1)  # Flatten the tensor

        return z_support, z_query

def compute_prototypes(support_embeddings, support_labels, n_classes):
    prototypes = [support_embeddings[support_labels == i].mean(0) for i in range(n_classes)]
    prototypes = torch.stack(prototypes)
    return prototypes

def euclidean_dist(x, y):
    return torch.cdist(x, y)

# Training function with accuracy calculation
def train_prototypical_network(model, train_loader, optimizer, num_epochs=20, n_support=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_preds = 0  # Track the number of correct predictions
        total_preds = 0    # Track the total number of predictions

        for images, labels in train_loader:
            optimizer.zero_grad()
            n_classes = torch.unique(labels).size(0)
            support_indices = list(range(0, n_support)) + list(range(len(labels)//2, len(labels)//2 + n_support))
            query_indices = list(range(n_support, len(labels)//2)) + list(range(len(labels)//2 + n_support, len(labels)))
            support_images, query_images = images[support_indices], images[query_indices]
            support_labels, query_labels = labels[support_indices], labels[query_indices]

            # Forward pass
            z_support, z_query = model(support_images, query_images)
            prototypes = compute_prototypes(z_support, support_labels, n_classes)

            # Calculate distances and compute loss
            dists = euclidean_dist(z_query, prototypes)
            log_p_y = torch.log_softmax(-dists, dim=1)
            loss = -log_p_y.gather(1, query_labels.view(-1, 1)).squeeze().mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_labels = torch.max(log_p_y, 1)
            correct_preds += (predicted_labels == query_labels).sum().item()
            total_preds += query_labels.size(0)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds  # Calculate accuracy for this epoch
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

# Initialize model, dataloader, and optimizer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageDataset(csv_file='augmented_images/image_labels.csv', img_dir='augmented_images/', transform=transform)
train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
model = PrototypicalNetwork(backbone)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start training
train_prototypical_network(model, train_loader, optimizer, num_epochs=20, n_support=5)
