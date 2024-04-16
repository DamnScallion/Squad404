from typing import Any

import pandas as pd
from PIL import Image
import torch
import os

import torch
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights
import torch.nn.functional as F

########################################################################################################################
# NOTE: Set the device based on CUDA availability
########################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



########################################################################################################################
# NOTE: Model Implementation
########################################################################################################################

# Prototypical network implementation based on https://colab.research.google.com/drive/1TPL2e3v8zcDK00ABqH3R0XXNJtJnLBCd?usp=sharing#scrollTo=UW5Rxifk7Kru
class Prototypical(nn.Module):
    #Initialise model with resNet as base CNN and flattening fc layer
    def __init__(self):
        super(Prototypical, self).__init__()
        
        # Uncomment below to use CBAM attention model
        self.baseCNN = CBAMAttentionMN((224, 224))
        
        # self.baseCNN = models.mobilenet_v2(pretrained=True)
        # self.baseCNN = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # self.baseCNN = models.resnet18(weights=ResNet18_Weights.DEFAULT)
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
        # print("inputs", inputs.shape)
        x = self.baseModel(inputs)
        # print("x1", x.shape)
        x = self.channel_attention(x)
        # print("x2", x.shape)
        x = self.spatial_attention(x)
        # print("x3", x.shape)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        # print("x4", x.shape)
        x = F.relu(self.dense1(x))
        # print("x5", x.shape)
        x = self.dropout(x)
        x = self.dense2(x)
        output = self.sigmoid(x)
        return output



########################################################################################################################
# Data Loading functions
########################################################################################################################
def load_image_labels(labels_file_path: str):
    """
    Loads the labels from CSV file.

    :param labels_file_path: CSV file containing the image and labels.
    :return: Pandas DataFrame
    """
    df = pd.read_csv(labels_file_path)
    return df


def load_predict_image_names(predict_image_list_file: str) -> [str]:
    """
    Reads a text file with one image file name per line and returns a list of files
    :param predict_image_list_file: text file containing the image names
    :return list of file names:
    """
    with open(predict_image_list_file, 'r') as file:
        lines = file.readlines()
    # Remove trailing newline characters if needed
    lines = [line.rstrip('\n') for line in lines]
    return lines


def load_single_image(image_file_path: str) -> Image:
    """
    Load the image.

    NOTE: you can optionally do some initial image manipulation or transformation here.

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    # Load the image
    image = Image.open(image_file_path)

    # Convert the image to RGB format
    image = image.convert('RGB')

    # The following are examples on how you might manipulate the image.
    # See full documentation on Pillow (PIL): https://pillow.readthedocs.io/en/stable/

    # To make the image 50% smaller
    # Determine image dimensions
    # width, height = image.size
    # new_width = int(width * 0.50)
    # new_height = int(height * 0.50)
    # image = image.resize((new_width, new_height))

    # To crop the image
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # image = image.crop((left, upper, right, lower))

    # To view an image
    # image.show()

    # Return either the pixels as array - image_array
    # To convert to a NumPy array
    # image_array = np.asarray(image)
    # return image_array

    # or return the image
    return image


########################################################################################################################
# Model Loading and Saving Functions
########################################################################################################################

def save_model(model: Any, target: str, output_dir: str):
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation.

    Common Deep Learning Model File Formats are:

        SavedModel (TensorFlow)
        Pros: Framework-agnostic format, can be deployed in various environments. Contains a complete model representation.
        Cons: Can be somewhat larger in file size.

        HDF5 (.h5) (Keras)
        Pros: Hierarchical structure, good for storing model architecture and weights. Common in Keras.
        Cons: Primarily tied to the Keras/TensorFlow ecosystem.

        ONNX (Open Neural Network Exchange)
        Pros: Framework-agnostic format aimed at improving model portability.
        Cons: May not support all operations for every framework.

        Pickle (.pkl) (Python)
        Pros: Easy to save and load Python objects (including models).
        Cons: Less portable across languages and environments. Potential security concerns.

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param model: the model that you want to save.
    :param target: the target value - can be useful to name the model file for the target it is intended for
    :param output_dir: the output directory to same one or more model files.
    """
    # TODO: implement your model saving code here
    # model_dir = os.path.join(output_dir, f'model_{target}')
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the model path with a .pth extension for PyTorch
    model_path = os.path.join(output_dir, f'model_{target}.pth')

    # Save the model
    torch.save(model, model_path)



def load_model(trained_model_dir: str, target_column_name: str) -> Any:
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation and should mirror save_model()

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param trained_model_dir: the directory where the model file(s) are saved.
    :param target_column_name: the target value - can be useful to name the model file for the target it is intended for
    :returns: the model
    """
    model_path = os.path.join(trained_model_dir, f'model_{target_column_name}.pth')
    print(f"Loaded model from {model_path}")

    # model = torch.load(model_path)

    # Load the model with map_location to ensure compatibility across different devices
    model = torch.load(model_path, map_location=device)

    return model
