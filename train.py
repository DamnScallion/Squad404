import argparse
import os
from typing import Any
from PIL import Image

from common import load_image_labels, load_single_image, save_model

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from easyfsl.samplers import TaskSampler
from easyfsl.utils import sliding_average
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torchvision.models import MobileNet_V2_Weights



########################################################################################################################
# NOTE: Model Implementation
########################################################################################################################


# Prototypical network implementation based on https://colab.research.google.com/drive/1TPL2e3v8zcDK00ABqH3R0XXNJtJnLBCd?usp=sharing#scrollTo=UW5Rxifk7Kru
class Prototypical(nn.Module):
    #Initialise model with resNet as base CNN and flattening fc layer
    def __init__(self):
        super(Prototypical, self).__init__()
        
        # Uncomment below to use CBAM attention model
        # self.baseCNN = CBAMAttentionMN((224, 224))
        
        # self.baseCNN = models.mobilenet_v2(pretrained=True)
        self.baseCNN = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.support_images= []
        self.support_labels = []
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        support_features = self.baseCNN.forward(support_images)
        query_features = self.baseCNN.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        num_classes = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        prototype = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(num_classes)])

        # Compute the euclidean distance from queries to prototypes
        scores = -(torch.cdist(query_features, prototype))

        return scores



def augment_data(images, labels, augmentations_per_image):
    augmented_images = []
    augmented_labels = []
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    
    for img, label in zip(images, labels):
        for _ in range(augmentations_per_image):
            augmented_img = augmentations(img)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)
    
    augmented_images.extend(images)
    augmented_labels.extend(labels)
    
    return augmented_images, augmented_labels


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_labels(self):
        return self.labels
    
class ChannelAttention(nn.Module):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(filters, filters // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(filters // ratio, filters, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        avg_out = self.shared_mlp(self.avg_pool(inputs).view(inputs.size(0), -1)).view(inputs.size(0), -1, 1, 1)
        max_out = self.shared_mlp(self.max_pool(inputs).view(inputs.size(0), -1)).view(inputs.size(0), -1, 1, 1)
        attention = self.sigmoid(avg_out + max_out)
        return inputs * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv2d(x))
        return inputs * attention

class CBAMAttentionMN(nn.Module):
    def __init__(self, input_shape):
        super(CBAMAttentionMN, self).__init__()
        # self.baseModel = models.mobilenet_v2(pretrained=True).features
        self.baseModel = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.baseModel.trainable = False

        # Assume input channels to channel attention is 1280 because it's the output of MobileNetV2
        self.channel_attention = ChannelAttention(filters=1280, ratio=8)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        self.flatten = nn.Flatten()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Avg Pooling
        self.dense1 = nn.Linear(1280, 256)  # Adjust depending on the output size after pooling
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        print("inputs", inputs.shape)
        x = self.baseModel(inputs)
        print("x1", x.shape)
        x = self.channel_attention(x)
        print("x2", x.shape)
        x = self.spatial_attention(x)
        print("x3", x.shape)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        print("x4", x.shape)
        x = F.relu(self.dense1(x))
        print("x5", x.shape)
        x = self.dropout(x)
        x = self.dense2(x)
        output = self.sigmoid(x)
        return output


def episodic_evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    f1_scores = []

    with torch.no_grad():
        for _, (support_images, support_labels, query_images, query_labels, _) in enumerate(data_loader):
            # Forward pass through the model
            scores = model(support_images, support_labels, query_images)

            # Compute loss
            loss = criterion(scores, query_labels)
            total_loss += loss.item()

            # Convert scores to predictions
            _, preds = torch.max(scores, 1)

            # Compute F1 score
            f1 = f1_score(query_labels.cpu().numpy(), preds.cpu().numpy(), average='binary')
            f1_scores.append(f1)

    avg_loss = total_loss / len(data_loader)
    avg_f1 = np.mean(f1_scores)
    return avg_loss, avg_f1



        
########################################################################################################################
# NOTE: Template Code
########################################################################################################################
def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args


def load_train_resources(resource_dir: str = 'resources') -> Any:
    """
    Load any resources (i.e. pre-trained models, data files, etc) here.
    Make sure to submit the resources required for your algorithms in the sub-folder 'resources'
    :param resource_dir: the relative directory from train.py where resources are kept.
    :return: TBD
    """
    # raise RuntimeError(
    #     "load_train_resources() not implement. If you have no pre-trained models you can comment this out.")


def train(images: [Image], labels: [str], output_dir: str) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.
    
    # Converting labels to ints
    labels = [1 if label == "Yes" else 0 for label in labels]

    # Augmenting dataset
    augmented_images, augmented_labels = augment_data(images, labels, 10)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(augmented_images, augmented_labels, test_size=0.2, random_state=88)
    
    # Defining prototypical parameters
    N_WAY = 2 # Num classes
    N_SHOT = 5 # Images per class
    N_QUERY = 5 # Num query images
    N_EVALUATION_TASKS = 100
    
    # Pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading data
    train_data = CustomDataset(X_train, y_train, transform=transform)
    test_data = CustomDataset(X_test, y_test, transform=transform)

    # Sampler object that dynamically creates episodes of support and query sets
    train_sampler = TaskSampler(train_data, N_WAY, N_SHOT, N_QUERY, N_EVALUATION_TASKS)
    
    # Dataloader object feeds episodes generated by the sampler to the model
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    # Initialising model
    model = Prototypical()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training step function
    def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:
        # CLear gradients form previous step
        optimizer.zero_grad()
        
        # Outputs scores (distances) of each query image 
        classification_scores = model(
            support_images, support_labels, query_images
        )
        print("support images shape", support_images.shape)
        print("support labels shape", support_labels.shape)
        print("query images shape", query_images.shape)

        # Calculate loss 
        loss = criterion(classification_scores, query_labels)
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        # Convert scores to predictions
        _, preds = torch.max(classification_scores, 1)
        query_labels_np = query_labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Calculate F1 score
        f1 = f1_score(query_labels_np, preds_np, average='binary')

        # Return loss and f1 score 
        return loss.item(), f1
    

    log_update_frequency = 10

    all_loss = []
    all_f1 = []
    model.train()
    
    # This is for showing a progress bar
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        # Looping over each episode generated by train loader
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            loss_value, f1 = fit(support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)
            all_f1.append(f1)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(
                    loss=sliding_average(all_loss, log_update_frequency),
                    f1=np.mean(all_f1[-log_update_frequency:])
                )
    
    # Defining testing prototypical parameters
    N_WAY_TEST = 2 # Num classes
    N_SHOT_TEST = 2 # Images per class
    N_QUERY_TEST = 2 # Num query images
    N_EVALUATION_TASKS_TEST = 10

    # NOTE: Make sure Number of test_sampler at least has (N_SHOT_TEST + N_QUERY_TEST) samples

    # Setup for episodic evaluation on test data
    test_sampler = TaskSampler(test_data, N_WAY_TEST, N_SHOT_TEST, N_QUERY_TEST, N_EVALUATION_TASKS_TEST)
    test_loader = DataLoader(
        test_data,
        batch_sampler=test_sampler,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # Episodic evaluation
    test_loss, test_f1 = episodic_evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test F1 Score: {test_f1}")
    
    return model


def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """

    # load pre-trained models or resources at this stage.
    # load_train_resources()

    # load label file
    labels_file_path = os.path.join(train_input_dir, train_labels_file_name)
    df_labels = load_image_labels(labels_file_path)
    print(df_labels)

    # load in images and labels
    train_images = []
    train_labels = []
    # Now iterate through every record and load in the image data files
    # Given the small number of data samples, iterrows is not a big performance issue.
    for index, row in df_labels.iterrows():
        try:
            filename = row['Filename']
            label = row[target_column_name]

            print(f"Loading image file: {filename}")
            image_file_path = os.path.join(train_input_dir, filename)
            image = load_single_image(image_file_path)

            train_labels.append(label)
            train_images.append(image)
        except Exception as ex:
            print(f"Error loading {index}: {filename} due to {ex}")
    print(f"Loaded {len(train_labels)} training images and labels")

    # Create the output directory and don't error if it already exists.
    os.makedirs(train_output_dir, exist_ok=True)

    # train a model for this task
    model = train(train_images, train_labels, train_output_dir)
    
    # Save the images and labels to use as the support set at prediction
    model.support_images = train_images
    model.support_labels = train_labels

    # save model
    save_model(model, target_column_name, train_output_dir)


if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)

########################################################################################################################
