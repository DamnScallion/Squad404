########################################################################################################################
# NOTE: No need to run preprocessing.py, unless you want to test it. 
# If u did, remember to remove the generate folder 'augmented_images' before pushing your code to the repo.
########################################################################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def add_random_noise(image):
    salt_vs_pepper = 0.5
    amount = 0.004
    num_pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
    
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], :] = 0
    return image


# read csv file and scaled label to 1, 0
# df = pd.read_csv('Data - Needs Respray - 2024-03-26/Labels-NeedsRespray-2024-03-26.csv')
# df['Needs Respray'] = df['Needs Respray'].map({'Yes': '1', 'No': '0'})
# df = pd.read_csv('Data - Is Epic Intro 2024-03-25/Labels-IsEpicIntro-2024-03-25.csv')
# df['Is Epic'] = df['Is Epic'].map({'Yes': '1', 'No': '0'})  
df = pd.read_csv('Data - Is GenAI - 2024-03-25/Labels-IsGenAI-2024-03-25.csv')
df['Is GenAI'] = df['Is GenAI'].map({'Yes': '1', 'No': '0'})

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    brightness_range=(0.8, 1),
    preprocessing_function = add_random_noise
)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    # directory='Data - Needs Respray - 2024-03-26',
    # directory='Data - Is Epic Intro 2024-03-25',
    directory='Data - Is GenAI - 2024-03-25',
    x_col='Filename',
    # y_col='Needs Respray',
    # y_col='Is Epic',
    y_col='Is GenAI',
    class_mode='raw',
    target_size=(224, 224),  # img size
    batch_size=32,
    shuffle=False,      # do not shuffle while each iteration generate augmented data
)

save_dir = 'resources/augmented_data/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_label_pairs = []

for i in range(8):  
    imgs, labels = next(train_generator)  
    for j, (img, label) in enumerate(zip(imgs, labels)):
        filename = f"aug_{i}_{j}.png"
        filepath = os.path.join(save_dir, filename)
        plt.imsave(filepath, img)
        image_label_pairs.append({'filename': filename, 'label': label})

df_image_labels = pd.DataFrame(image_label_pairs)

csv_filepath = os.path.join(save_dir, 'image_labels.csv')
df_image_labels.to_csv(csv_filepath, index=False)

print(f"Image labels saved to {csv_filepath}")
