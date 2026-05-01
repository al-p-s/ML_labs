import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Нормализация — пиксели из 0-255 в 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Добавляем размерность канала (28,28) → (28,28,1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Метки в one-hot (7 → [0,0,0,0,0,0,0,1,0,0])
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)

# Архитектура CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_oh, epochs=10, batch_size=64,
                    validation_data=(X_test, y_test_oh), verbose=1)

# Оценка
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f'\nТочность на тесте: {test_acc:.4f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.tight_layout()
plt.savefig('cnn_training.png', dpi=120)
plt.show()

# Визуализация фильтров первого слоя
filters = model.layers[0].get_weights()[0]  # shape (3,3,1,32)
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Фильтры первого свёрточного слоя')
plt.savefig('filters.png', dpi=120)
plt.show()

# Визуализация feature maps
sample = X_test[[0, 45, 116, 200]]  # одно изображение
feature_model = keras.Model(inputs=model.inputs,
                            outputs=[model.layers[0].output, model.layers[2].output])
fmap1, fmap2 = feature_model.predict(sample)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fmap1[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature maps первого слоя')
plt.savefig('feature_maps_1.png', dpi=120)
plt.show()

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(fmap2[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature maps второго слоя')
plt.savefig('feature_maps_2.png', dpi=120)
plt.show()
