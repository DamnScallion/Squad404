import argparse
import os
from typing import Any
from PIL import Image

from common import load_image_labels, load_single_image, save_model, Prototypical

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from easyfsl.samplers import TaskSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fn


########################################################################################################################
# NOTE: Set the device based on CUDA availability
########################################################################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



########################################################################################################################
# NOTE: Helpr Function
########################################################################################################################
class SaltAndPepperNoise(object):
    """Apply salt and pepper noise to an image."""
    def __init__(self, amount=0.004):
        self.amount = amount

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Image with salt and pepper noise applied.
        """
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Applying pepper noise, Pepper corresponds to 0
        mask_pepper = np.random.choice([0, 1], size=img_array.shape[:2], p=[self.amount/2, 1-self.amount/2])
        img_array[mask_pepper == 0] = 0
        
        # Applying salt noise, Salt corresponds to 255
        mask_salt = np.random.choice([0, 1], size=img_array.shape[:2], p=[self.amount/2, 1-self.amount/2])
        img_array[mask_salt == 0] = 255

        # Convert numpy array back to PIL Image
        return Fn.to_pil_image(img_array)


def augment_data(images, labels, augmentations_per_image):
    """
    Augments a list of images using specified transformations to increase the dataset size and variability,
    which can help improve model generalization during training.

    :param images: List of images (PIL Image format) to be augmented.
    :param labels: Corresponding labels for the images; these labels are replicated with their respective images.
    :param augmentations_per_image: Number of augmented versions to create per image.
    :return: Tuple of lists (augmented_images, augmented_labels) containing the original and augmented images and their labels.
    """
    augmented_images = []
    augmented_labels = []
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomApply([SaltAndPepperNoise(0.004)], p=0.5), # Applying salt and pepper noise with a 0.4% amount
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
    """
    Custom dataset class that handles loading images and their corresponding labels for training or evaluation.
    This dataset supports optional transformations that can be applied to images on-the-fly during data loading.

    :param images: List of images to be included in the dataset.
    :param labels: List of labels corresponding to the images.
    :param transform: Optional transform (or composed transforms) to be applied to each image when fetched.
    """
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


def prepare_data(images, labels, transform, augmentations_per_image, n_way, n_shot, n_query, n_evaluation_tasks):
    """
    Prepares training and testing data loaders.

    :param images: List of images to be used for training.
    :param labels: Corresponding labels for the images.
    :param transform: Transformation to be applied on images.
    :param augmentations_per_image: Number of augmented versions to create per image.
    :param n_way: Number of classes per episode.
    :param n_shot: Number of support examples per class in each episode.
    :param n_query: Number of query examples per episode.
    :param n_evaluation_tasks: Number of different tasks to evaluate on.
    :return: Train and test data loaders.
    """
    #Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=88)

    # Augmenting dataset
    X_train, y_train = augment_data(images, labels, augmentations_per_image)
    X_test, y_test = augment_data(images, labels, augmentations_per_image - 5)  # fewer augmentations for test

    # Loading data
    train_data = CustomDataset(X_train, y_train, transform)
    test_data = CustomDataset(X_test, y_test, transform)

    # Sampler object that dynamically creates episodes of support and query sets
    train_sampler = TaskSampler(train_data, n_way, n_shot, n_query, n_evaluation_tasks)
    test_sampler = TaskSampler(test_data, n_way, n_shot, n_query, n_evaluation_tasks // 10)  # fewer episodes for test

    # Dataloader object feeds episodes generated by the sampler to the model
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=train_sampler.episodic_collate_fn)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, collate_fn=test_sampler.episodic_collate_fn)

    return train_loader, test_loader


def train_episode(model, optimizer, scheduler, criterion, support_images, support_labels, query_images, query_labels):
    """
    Executes one training episode.

    :param model: The model to be trained.
    :param optimizer: Optimizer for the model.
    :param scheduler: Learning rate scheduler.
    :param criterion: Loss function to use.
    :param support_images: Support images for the episode.
    :param support_labels: Labels for the support images.
    :param query_images: Query images for the episode.
    :param query_labels: Labels for the query images.
    :return: Loss and F1 score for the episode.
    """
    # CLear gradients form previous step
    optimizer.zero_grad()

    # Outputs scores (distances) of each query image 
    classification_scores = model(support_images, support_labels, query_images)
    
    # Calculate loss 
    loss = criterion(classification_scores, query_labels)

    # Compute gradients
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Convert scores to predictions
    _, preds = torch.max(classification_scores, 1)
    f1 = f1_score(query_labels.cpu().numpy(), preds.cpu().numpy(), average='binary')
    return loss.item(), f1


def run_training_loop(model, train_loader, optimizer, scheduler, criterion, log_update_frequency=10):
    """
    Runs the training loop over all episodes.

    :param model: The model to train.
    :param train_loader: DataLoader for the training data.
    :param optimizer: Optimizer for the model.
    :param scheduler: Learning rate scheduler.
    :param criterion: Loss function.
    :param log_update_frequency: Frequency of logging information.
    :return: Lists of losses and F1 scores.
    """
    model.train()
    all_loss = []
    all_f1 = []

    # This is for showing a progress bar
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            support_images = support_images.to(DEVICE)
            support_labels = support_labels.to(DEVICE)
            query_images = query_images.to(DEVICE)
            query_labels = query_labels.to(DEVICE)

            loss, f1 = train_episode(model, optimizer, scheduler, criterion, support_images, support_labels, query_images, query_labels)
            all_loss.append(loss)
            all_f1.append(f1)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(
                    loss=np.mean(all_loss[-log_update_frequency:]), 
                    f1=np.mean(all_f1[-log_update_frequency:])
                )

    return all_loss, all_f1


def episodic_evaluate(model, data_loader, criterion):
    """
    Evaluates the model's performance on a given dataset using episodic tasks, computing average loss and F1 score.

    :param model: The model to be evaluated, set in evaluation mode to disable dropout or batch normalization effects.
    :param data_loader: DataLoader containing the episodic tasks for evaluation. Each episode consists of support and query sets.
    :param criterion: The loss function used to evaluate the model's performance on the query set.
    :return: avg_loss (float), representing the average loss over all episodes; avg_f1 (float), representing the average F1 score over all episodes.
    """
    model.eval()
    total_loss = 0.0
    f1_scores = []

    with torch.no_grad():
        for _, (support_images, support_labels, query_images, query_labels, _) in enumerate(data_loader):
            support_images = support_images.to(DEVICE)
            support_labels = support_labels.to(DEVICE)
            query_images = query_images.to(DEVICE)
            query_labels = query_labels.to(DEVICE)

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


def plot_training_metrics(train_loss: list[float], train_f1: list[float], output_dir: str) -> None:
    """
    Plots the training loss and F1 score from a training session on two subplots,
    saving the resulting figure to the specified output directory.

    :param train_loss: a list of floats, representing the training loss recorded after each batch or epoch.
    :param train_f1: a list of floats, representing the F1 scores recorded after each batch or epoch.
    :param output_dir: the directory where the plot image will be saved.
    """
    # Creating a figure with two subplots
    _, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting training loss
    axs[0].plot(train_loss, label='Training Loss', color='tab:blue', alpha=0.6)
    axs[0].set_title('Training Loss over Time')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plotting F1 score
    axs[1].plot(train_f1, label='Training F1 Score', color='tab:green', alpha=0.6)
    axs[1].set_title('Training F1 Score over Time')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('F1 Score')
    axs[1].legend()

    # Layout adjustment
    plt.tight_layout()

    # Saving the figure
    file_path = f"{output_dir}/training_records.png"
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")

    # Close the plot to free up memory
    plt.close()  



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


def train(images: list[Image.Image], labels: list[str], output_dir: str) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # Converting labels to ints
    labels = [1 if label == "Yes" else 0 for label in labels]

    # Pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Defining prototypical parameters
    N_WAY = 2 # Num classes
    N_SHOT = 5 # Images per class
    N_QUERY = 8 # Num query images
    N_EVALUATION_TASKS = 100

    # Number of augmented versions to create per image
    N_AUGMENT_PER_IMG = 15

    # Prepares training and testing data loaders
    train_loader, test_loader = prepare_data(images, labels, transform, N_AUGMENT_PER_IMG, N_WAY, N_SHOT, N_QUERY, N_EVALUATION_TASKS)

    # Initialising model
    model = Prototypical().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Initialize learning rate scheduler and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training step function
    all_loss, all_f1 = run_training_loop(model, train_loader, optimizer, scheduler, criterion)

    # Plot training metrics
    plot_training_metrics(all_loss, all_f1, output_dir)

    # Test data evaluation
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
