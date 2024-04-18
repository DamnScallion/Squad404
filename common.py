from typing import Any

import pandas as pd
from PIL import Image
import torch
import os

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

########################################################################################################################
# NOTE: Set the device based on CUDA availability
########################################################################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



########################################################################################################################
# NOTE: Model Implementation
########################################################################################################################

# Prototypical network implementation based on https://colab.research.google.com/drive/1TPL2e3v8zcDK00ABqH3R0XXNJtJnLBCd?usp=sharing#scrollTo=UW5Rxifk7Kru
class Prototypical(nn.Module):
    """
    A Prototypical Network model for few-shot learning which uses a CNN as the backbone for feature extraction,
    and computes prototypes to classify query images based on the minimum Euclidean distance to support set prototypes.

    :param None: Inherits from nn.Module, does not require parameters to be passed explicitly during initialization.
    """
    def __init__(self):
        """
        Initialise model with our attention model as base CNN and flattening fc layer
        """
        super(Prototypical, self).__init__()
        ########################################################################################
        # NOTE: Choose your base CNN to test. Our attention-based model CBAMAttentionMN is the best.
        ########################################################################################
        self.baseCNN = CBAMAttentionMN((224, 224))
        # self.baseCNN = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # self.baseCNN = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.baseCNN = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # self.baseCNN = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        ########################################################################################
        self.support_images= []
        self.support_labels = []
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model which computes the similarity between query images and class prototypes defined by support images.

        :param support_images: Tensor containing images of the support set.
        :param support_labels: Tensor containing labels of the support set.
        :param query_images: Tensor containing images to be classified.
        :return: Tensor of scores representing the distances of each query image to the class prototypes.
        """
        support_features = self.baseCNN.forward(support_images)
        query_features = self.baseCNN.forward(query_images)

        num_classes = len(torch.unique(support_labels))
        prototype = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(num_classes)])

        # Compute the euclidean distance to create prototypes
        scores = -(torch.cdist(query_features, prototype))

        return scores


# Attention module code based on https://github.com/EscVM/EscVM_YT/blob/master/Notebooks/0%20-%20TF2.X%20Tutorials/tf_2_visual_attention.ipynb
class ChannelAttention(nn.Module):
    """
    Channel attention module which emphasizes informative features by learning to use global spatial information
    of the channels in convolutional neural networks.

    :param filters: Number of filters in the previous convolution layer.
    :param ratio: Reduction ratio for dimensionality reduction in the attention mechanism.
    """
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
        """
        Forward pass of the channel attention module.

        :param inputs: Input tensor with feature maps from the previous layer.
        :return: Tensor, where attention has been applied to the input feature maps.
        """
        avg_out = self.shared_mlp(self.avg_pool(inputs).view(inputs.size(0), -1)).view(inputs.size(0), -1, 1, 1)
        max_out = self.shared_mlp(self.max_pool(inputs).view(inputs.size(0), -1)).view(inputs.size(0), -1, 1, 1)
        attention = self.sigmoid(avg_out + max_out)
        return inputs * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention module that highlights salient features in spatial dimensions of the input feature maps,
    improving the focus on relevant parts of the input data.

    :param kernel_size: Size of the convolution kernel used for creating the attention map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward pass of the spatial attention module.

        :param inputs: Input tensor with feature maps from the previous layer.
        :return: Tensor, where attention has been applied spatially to the input feature maps.
        """
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv2d(x))
        return inputs * attention


class CBAMAttentionMN(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) MobileNet model, an attention-enhanced network using both channel
    and spatial attention mechanisms to improve feature representation in convolutional neural networks.

    :param input_shape: Expected input shape for the model, used to configure layers.
    """
    def __init__(self, input_shape):
        super(CBAMAttentionMN, self).__init__()
        # Path to the locally saved weights
        weights_path = 'resources/pretrained/mobilenet_v2-7ebf99e0.pth'

        # Initialize MobileNetV2 without pre-trained weights
        model = models.mobilenet_v2(weights=None)

        # Load the weights from a local file instead of using MobileNet_V2_Weights
        model_weights = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(model_weights)

        # We are using only the features of MobileNetV2
        self.baseModel = model.features
        
        # Set the base model layers are not trainable (if needed)
        # for param in self.baseModel.parameters():
        #     param.requires_grad = False

        self.channel_attention = ChannelAttention(filters=1280, ratio=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        self.flatten = nn.Flatten()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(1280, 256)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward pass of the CBAM-enhanced MobileNet model.

        :param inputs: Input tensor containing the data to be processed.
        :return: Output tensor after applying both channel and spatial attention mechanisms.
        """
        x = self.baseModel(inputs)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
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

    # Load the model with map_location to ensure compatibility across different devices
    model = torch.load(model_path, map_location=DEVICE)

    return model
