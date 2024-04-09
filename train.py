import argparse
import os
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape, Lambda
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tensorflow.keras import losses
from tensorflow.keras import backend as K

from common import load_image_labels, load_single_image, save_model

########################################################################################################################
# NOTE: Helper function
########################################################################################################################
def add_random_noise(image):
    """
    Adds random Salt Pepper noise to an image.
    
    :param image: Input image.
    :return: Image with added Salt Pepper noise.
    """
    salt_vs_pepper = 0.5
    amount = 0.004
    num_pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
    
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], :] = 0
    return image


def create_data_augmentation_generator(images_np, labels_np):
    """
    Creates a data generator with on-the-fly data augmentation.
    
    :param images_np: Numpy array of images.
    :param labels_np: Numpy array of labels corresponding to the images.
    :return: A data generator yielding batches of augmented images and labels.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=add_random_noise,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='constant'
    )
    
    return datagen.flow(
        x=images_np,
        y=labels_np,
        batch_size=64
    )

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = tf.keras.layers.Dense(
            self.filters//self.ratio,
            activation='relu', kernel_initializer='he_normal', 
            use_bias=True, 
            bias_initializer='zeros'
        )
        self.shared_layer_two = tf.keras.layers.Dense(
            self.filters,
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )

    def call(self, inputs):
        # AvgPool
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        # MaxPool
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1,1,self.filters))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        attention = tf.keras.layers.Add()([avg_pool,max_pool])
        attention = tf.keras.layers.Activation('sigmoid')(attention)
        return tf.keras.layers.Multiply()([inputs, attention])


#Code from https://github.com/EscVM/EscVM_YT/blob/master/Notebooks/0%20-%20TF2.X%20Tutorials/tf_2_visual_attention.ipynb
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(
            filters = 1,
            kernel_size = self.kernel_size,
            strides = 1,
            padding = 'same',
            activation = 'sigmoid',
            kernel_initializer = 'he_normal',
            use_bias = False,
        )

    def call(self, inputs):
        # AvgPool
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
        # MaxPool
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)
        attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
        attention = self.conv2d(attention)
        return tf.keras.layers.multiply([inputs, attention]) 
    
    
class CBAMAttentionMN(keras.models.Model):
    def __init__(self, input_shape):
        super(CBAMAttentionMN, self).__init__()
        #taking base model from MobileNetv2 and freezing layers
        self.baseModel = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        self.baseModel.trainable = False

        #attention layers
        self.channel_attention = ChannelAttention(1280, 8)
        self.spatial_attention = SpatialAttention(7)
        # fully connected layers with dropout added
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(1, activation='sigmoid')

    
    def call(self, inputs):
        x = self.baseModel(inputs)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        output = self.dense2(x)
        return output

def proto_dist(x):
    x = K.l2_normalize(x)
    pred_dist = tf.reduce_sum(x ** 2, axis=1, keepdims=True)
    feature_dist = tf.reduce_sum(x ** 2, axis=1, keepdims=True)
    dot = tf.matmul(pred_dist, tf.transpose(pred_dist))
    dist = tf.sqrt(pred_dist + tf.transpose(feature_dist) - 2 * dot)
    return dist 

def label_matrix(x):
    shape = tf.shape(x)[0]
    y = tf.transpose(x)
    matrix = tf.tile(x, (1, shape))
    matrix2 = tf.tile(y, (shape, 1))
    comparison_result = tf.not_equal(matrix, matrix2)
    return tf.cast(comparison_result, tf.float32)

def ProtoLoss(y_true, y_pred):
    proto_dists = (proto_dist)(y_pred)
    label_matrixs = (label_matrix)(y_true)
    distance_matrix = K.abs(proto_dists - label_matrixs)
    # Compute the loss value
    loss = K.mean(distance_matrix, axis=-1)

    return loss


def create_model(base):
    """
    Creates a model based on a given base pretrained CNN architecture.

    This function initializes a pretrained CNN without its top layer, appends custom layers on top,
    and compiles the model with a binary crossentropy loss and adam optimizer. The base model is frozen to
    prevent its weights from being updated during training.

    :param base: A function reference to a pretrained model class from keras.applications (e.g., MobileNetV2, ResNet50).
    :return: A compiled keras Model instance ready for training.
    """
    base_model = base(input_shape=(224, 224, 3), include_top=False)
    # base_model.trainable = False  # Freeze the base model
    
    # Append custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    proto = Dense(1024, activation='relu', name='output2')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid', name='output')(x)

    # Create and compile the model
    model = Model(inputs=base_model.input, outputs=[predictions, proto])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics={'output': ['accuracy']})
    model.compile(optimizer='adam', loss={'output2': ProtoLoss, 'output': 'binary_crossentropy'}, metrics={'output': ['accuracy']})
    return model


########################################################################################################################
# NOTE: Template code
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
    raise RuntimeError(
        "load_train_resources() not implement. If you have no pre-trained models you can comment this out.")


def train(images: [Image], labels: [str], output_dir: str) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of images (PIL Image format).
    :param labels: the list of training labels (str or 0,1).
    :param output_dir: the directory to write logs, stats, etc., along the way.
    :return: model: the trained model.
    """
    # Convert images and labels to numpy arrays
    images_np = np.array([img_to_array(image.resize((224, 224))) for image in images])
    labels_np = np.array(labels).astype(int)
    
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    
    for train_index, val_index in kf.split(images_np, labels_np):
        print(f"train: {train_index}, val: {val_index}")
    
    # Your choice of models to test
    CNN_models_to_test = [MobileNetV2, ResNet50]
    
    best_model = None
    best_f1_score = -1
    model_f1_scores = {}

    # Create a non-augmented data generator for validation data
    validation_datagen = ImageDataGenerator(rescale=1./255)

    for CNN_base_model in CNN_models_to_test:
        fold_no = 1
        fold_f1_scores = []
        for train_index, val_index in kf.split(images_np, labels_np):
            print(f"Training with {CNN_base_model.__name__}, Fold {fold_no}")
            X_train, X_val = images_np[train_index], images_np[val_index]
            y_train, y_val = labels_np[train_index], labels_np[val_index]
            
            # Data Augmentation Generator for training data
            train_generator = create_data_augmentation_generator(X_train, y_train)
            
            # Non-augmented data generator for validation data
            val_generator = validation_datagen.flow(
                X_val,
                y_val,
                batch_size=16,
            )
            
            # Model creation
            model = create_model(CNN_base_model)
            
            #========================
            #TESTING FOR MOBILENET WITH CBAM ATTENTION MODEL
            #Uncomment below and comment 'model = create_model(CNN_base_model)' to use
            #========================
            # model = CBAMAttentionMN(input_shape=(224, 224, 3))
            # x = Input(shape=(224, 224, 3))
            # model = keras.models.Model(inputs=[x], outputs = model.call(x))

            # lr = 1e-4
            # model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=["accuracy"])
            #========================
            
            
            # Training
            model.fit(train_generator, validation_data=val_generator, epochs=10)

            # Predict on the validation set
            val_predictions = model.predict(val_generator)[0]
            # print(val_predictions)
            val_predictions = [1 if x > 0.5 else 0 for x in val_predictions]


            # Calculate the F1 score
            f1 = f1_score(y_val, val_predictions)
            fold_f1_scores.append(f1)

            fold_no += 1

        # Calculate the average F1 score across all folds for the current model
        avg_f1 = np.mean(fold_f1_scores)
        model_f1_scores[CNN_base_model.__name__] = avg_f1

        # Update the best model if the current model has a better average F1 score
        if avg_f1 > best_f1_score:
            best_f1_score = avg_f1
            best_model = model

    for model_name, score in model_f1_scores.items():
        print(f"Average F1 Score for {model_name}: {score}")
    
    return best_model



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

    # Convert labels value to binary
    df_labels[target_column_name] = df_labels[target_column_name].map({'Yes': '1', 'No': '0'}) 

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
