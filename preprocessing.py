from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import matplotlib.pyplot as plt

# read csv file and scaled label to 1, 0
df = pd.read_csv('NeedsRespray/Labels-NeedsRespray-2024-03-26.csv')
df['Needs Respray'] = df['Needs Respray'].map({'Yes': 1, 'No': 0})  


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0
)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='NeedsRespray/images',
    x_col='Filename',
    y_col='Needs Respray',
    class_mode='raw',
    batch_size=32,
    shuffle=False,      # do not shuffle while each iteration generate augmented data
)

save_dir = 'augmented_images/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_label_pairs = []

for i in range(5):  
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
