import argparse
import os
from typing import Any

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from sklearn.metrics import f1_score

from common import load_model, load_predict_image_names, load_single_image

########################################################################################################################
# NOTE: Set the device based on CUDA availability
########################################################################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



########################################################################################################################
# NOTE: Helper Function
########################################################################################################################
def transform_image_to_tensor(image: Image) -> torch.Tensor:
    """
    Transforms an image into a tensor that's ready for model input, performing resizing, conversion to tensor, 
    and normalization using ImageNet standards.

    :param image: the PIL Image to be transformed.
    :return: a transformed tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform(image)


def calculate_acc_and_f1(predict_data_image_dir, target_column_name, df_predictions):
    """
    Calculate both the accuracy and the F1 score of predictions against the ground truth labels,
    and print them in a single line.

    :param predict_data_image_dir: Directory containing the ground truth CSV file.
    :param target_column_name: The name of the prediction column.
    :param df_predictions: DataFrame containing the prediction results.
    :return: None
    """
    # Read ground truth labels csv
    ground_truth_path = os.path.join(predict_data_image_dir, f"{target_column_name} - True Labels.csv")
    df_ground_truth = pd.read_csv(ground_truth_path)

    # Merge predictions with ground truth labels
    merged_df = pd.merge(df_predictions, df_ground_truth, on='Filename', suffixes=('_pred', '_true'))

    # Calculate prediction accuracy
    accuracy = (merged_df[f'{target_column_name}_pred'] == merged_df[f'{target_column_name}_true']).mean()

    # Calculate the overall F1 score
    f1 = f1_score(merged_df[f'{target_column_name}_true'], merged_df[f'{target_column_name}_pred'], average='binary', pos_label='Yes')

    # Print both metrics in one line
    print(f"{target_column_name} Prediction Accuracy: {accuracy:.2%}, Overall F1 score: {f1:.2f}")

######################################################################################################################
# NOTE: Template Code
########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args


def predict(model: torch.nn.Module, image: Image) -> str:
    """
    Generate a prediction for a single image using the model, returning a label of 'Yes' or 'No'

    IMPORTANT: The return value should ONLY be either a 'Yes' or 'No' (Case sensitive)

    :param model: the model to use.
    :param image: the image file to predict.
    :return: the label ('Yes' or 'No)
    """
    support_images = torch.stack([transform_image_to_tensor(image) for image in model.support_images]).to(DEVICE)
    support_labels = torch.tensor([1 if label == "Yes" else 0 for label in model.support_labels]).to(DEVICE)

    # # Apply the transformations to the image
    image_tensor = transform_image_to_tensor(image).unsqueeze(0).to(DEVICE) # Add batch dimension and move to device

    with torch.no_grad():
        # Forward pass
        scores = model(support_images, support_labels, image_tensor)

        # Convert the scores to predictions
        _, preds = torch.max(scores, 1)
        predicted_label = 'Yes' if preds.item() == 1 else 'No'

    return predicted_label


def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """

    # load pre-trained models or resources at this stage.
    # model = load_model(trained_model_dir, target_column_name)
    model = load_model(trained_model_dir, target_column_name).to(DEVICE)

    # Load in the image list
    image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
    image_filenames = load_predict_image_names(image_list_file)

    # Iterate through the image list to generate predictions
    predictions = []
    for filename in image_filenames:
        try:
            image_path = os.path.join(predict_data_image_dir, filename)
            image = load_single_image(image_path)
            label = predict(model, image)
            predictions.append(label)
        except Exception as ex:
            print(f"Error generating prediction for {filename} due to {ex}")
            predictions.append("Error")

    df_predictions = pd.DataFrame({'Filename': image_filenames, target_column_name: predictions})

    ########################################################################################################
    # NOTE: Simulate tutor evaluation. Call the function to caculate prediction accuracy and overall F1 Score.
    ########################################################################################################
    # calculate_acc_and_f1(predict_data_image_dir, target_column_name, df_predictions)
    ########################################################################################################

    os.makedirs(os.path.dirname(predicts_output_csv), exist_ok=True)

    # Finally, write out the predictions to CSV
    df_predictions.to_csv(predicts_output_csv, index=False)

    print(f"Predict output has been saved in {predicts_output_csv}\n\n")


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
