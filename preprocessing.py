from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# 假设 df 是包含图片文件名和标签的 DataFrame
df = pd.read_csv('NeedsRespray/Labels-NeedsRespray-2024-03-26.csv')
df['Needs Respray'] = df['Needs Respray'].map({'Yes': 1, 'No': 0})  # 假设您已经这样转换了标签

# 初始化数据增强器
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 创建数据生成器
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='NeedsRespray/images',  # 图片存储路径
    x_col='Filename',
    y_col='Needs Respray',
    class_mode='raw',
    target_size=(150, 150),  # 图片目标大小
    batch_size=32,
    save_to_dir='augmented_images',  # 指定保存增强图片的目录
    save_prefix='aug_',  # 保存的图片名前缀
    save_format='jpeg'  # 保存的图片格式
)

saved_images_labels = []

import os

save_dir = 'augmented_images'  # 指定保存增强图片的目录

for i in range(5):  # 假设我们想要迭代5次生成器
    imgs, labels = next(train_generator)  # 这会生成并保存下一批图片

    # 对于每个标签，记录文件名和标签
    for index, label in enumerate(labels):
        # 构建增强后的图片文件名
        filename = f"aug_{i}_{index}.jpeg"  # 根据实际保存逻辑调整格式
        saved_images_labels.append({'Filename': filename, 'Label': label})

df_saved_labels = pd.DataFrame(saved_images_labels)

# 保存 DataFrame 到 CSV 文件
labels_csv_path = os.path.join(save_dir, 'augmented_images_labels.csv')
df_saved_labels.to_csv(labels_csv_path, index=False)
