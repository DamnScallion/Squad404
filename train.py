import argparse
import os
from typing import Any
from PIL import Image

from common import load_image_labels, load_single_image, save_model

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average
from tqdm import tqdm


########################################################################################################################
def augment_data(images, labels, augmentations_per_image):
    augmented_images = []
    augmented_labels = []
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # Add more transformations as needed
    ])
    
    for img, label in zip(images, labels):
        for _ in range(augmentations_per_image):
            augmented_img = augmentations(img)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)
    
    # Don't forget to include the original images too
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
        # Return the labels of all samples in the dataset
        return self.labels

    
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores



        
        
    

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
    print(labels)
    labels = [1 if label == "Yes" else 0 for label in labels]
    print(labels)

    augmented_images, augmented_labels = augment_data(images, labels, 10)

    X_train, X_test, y_train, y_test = train_test_split(augmented_images, augmented_labels, test_size=0.2, random_state=88)
    
    N_WAY = 2 # Number of classes in a task
    N_SHOT = 5 # Number of images per class in the support set
    N_QUERY = 7 # Number of images per class in the query set
    N_EVALUATION_TASKS = 100
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = CustomDataset(X_train, y_train, transform=transform)
    test_data = CustomDataset(X_test, y_test, transform=transform)
    print(test_data)


    train_sampler = TaskSampler(
        train_data,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_EVALUATION_TASKS
    )
    
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    
    convolutional_network = models.resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()

    model = PrototypicalNetworks(convolutional_network)
    (
        example_support_images,
        example_support_labels,
        example_query_images,
        example_query_labels,
        example_class_ids,
    ) = next(iter(train_loader))
    model.eval()
    criterion = nn.functional.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:
        optimizer.zero_grad()
        classification_scores = model(
            support_images, support_labels, query_images
        )

        loss = criterion(classification_scores, query_labels)
        loss.backward()
        optimizer.step()

        return loss.item()
    log_update_frequency = 10

    all_loss = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
    
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
