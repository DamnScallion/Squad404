import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 准备数据
csv_file = "resources/augmented_data/image_labels.csv"
img_dir = "resources/augmented_data/"

# 读取CSV文件
df = pd.read_csv(csv_file)

# 定义数据增强策略
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

# 使用flow_from_dataframe来准备训练数据
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True
)

# 构建模型
def build_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.02),  # 加入Dropout层，丢弃率设置为0.5
        Dense(2, activation='softmax')
    ])
    return model

model = build_model()

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_generator, epochs=25)

# 绘制训练结果
def plot_training_results(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_results(history)
