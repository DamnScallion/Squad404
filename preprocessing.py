from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import matplotlib.pyplot as plt

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
)

save_dir = 'augmented_images/'

# 检查目录是否存在，如果不存在，则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 初始化一个列表来收集文件名和标签
image_label_pairs = []

for i in range(5):  # 假设我们想要迭代5次生成器
    imgs, labels = next(train_generator)  # 生成一批图像和标签
    for j, (img, label) in enumerate(zip(imgs, labels)):
        # 构建文件名，包括批次号、图像索引和标签
        filename = f"aug_{i}_{j}_label_{label}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 保存图像
        plt.imsave(filepath, img)
        
        # 将文件名和标签添加到列表
        image_label_pairs.append({'filename': filename, 'label': label})

# 将收集到的文件名和标签转换为 DataFrame
df_image_labels = pd.DataFrame(image_label_pairs)

# 保存 DataFrame 到 CSV 文件
csv_filepath = os.path.join(save_dir, 'image_labels.csv')
df_image_labels.to_csv(csv_filepath, index=False)

print(f"Image labels saved to {csv_filepath}")
