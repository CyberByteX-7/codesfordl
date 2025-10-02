import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
# This is the corrected import statement
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter

def generate_synthetic_pneumonia_data(num_samples=2000, img_size=128):
    print(f"Generating {num_samples} synthetic X-ray images for demonstration...")
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        label = np.random.randint(0, 2)
        labels[i] = label
        image = np.random.rand(img_size, img_size) * 0.2

        # Add horizontal lines to simulate rib cage
        for r in range(8):
            y_pos = int(img_size * (r * 0.1 + 0.1))
            image[y_pos-1:y_pos+1, :] += np.random.uniform(0.05, 0.1)

        # Add pneumonia-like patches for label 1
        if label == 1:
            num_patches = np.random.randint(2, 5)
            for _ in range(num_patches):
                patch = np.zeros((img_size, img_size))
                center_x = np.random.randint(int(img_size*0.2), int(img_size*0.8))
                center_y = np.random.randint(int(img_size*0.2), int(img_size*0.8))
                size = np.random.randint(int(img_size*0.1), int(img_size*0.25))
                patch[center_y-size//2:center_y+size//2, center_x-size//2:center_x+size//2] = 1
                blurred_patch = gaussian_filter(patch.astype(float), sigma=np.random.uniform(5, 10))
                image += blurred_patch * np.random.uniform(0.2, 0.5)

        image = np.clip(image, 0, 1)
        images[i] = np.dstack([image, image, image])

    print("Synthetic data generation complete.")
    return images, labels

X, y = generate_synthetic_pneumonia_data()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(128,128,3)),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nValidation Accuracy: {val_acc*100:.2f}%')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training History')
plt.show()

def display_predictions(model, images, labels, num_samples=10):
    class_names = ['NORMAL', 'PNEUMONIA']
    indices = np.random.choice(len(images), num_samples)

    plt.figure(figsize=(15, 8))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[index])
        plt.xticks([])
        plt.yticks([])

        img_array = np.expand_dims(images[index], axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]

        pred_label = 1 if prediction > 0.5 else 0
        true_label_name = class_names[labels[index]]
        pred_label_name = class_names[pred_label]

        color = 'green' if pred_label == labels[index] else 'red'
        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}", color=color)

    plt.tight_layout()
    plt.show()

print("\nDisplaying sample predictions from validation set...")
display_predictions(model, X_val, y_val)
