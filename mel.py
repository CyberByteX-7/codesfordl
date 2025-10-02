import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter

IMG_SIZE = 128
NUM_CLASSES = 2
CLASS_NAMES = ['Benign', 'Melanoma']
NUM_SAMPLES = 2000

def generate_synthetic_melanoma_data(num_samples=NUM_SAMPLES, img_size=IMG_SIZE):
    print(f"Generating {num_samples} synthetic lesion images...")
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=num_samples)

    for i in range(num_samples):
        skin_color = np.array([234, 192, 134]) / 255.0 + np.random.normal(0, 0.05, 3)
        img = np.ones((img_size, img_size, 3)) * skin_color

        label = labels[i]
        center_x, center_y = np.random.randint(img_size*0.3, img_size*0.7, 2)

        Y, X = np.ogrid[:img_size, :img_size]

        if label == 0:
            axis_x = np.random.uniform(0.1, 0.2) * img_size
            axis_y = axis_x * np.random.uniform(0.9, 1.1)
            dist = np.sqrt(((X - center_x) / axis_x) ** 2 + ((Y - center_y) / axis_y) ** 2)
            mask = dist <= 1
            lesion_color = np.array([101, 67, 33]) / 255.0 + np.random.normal(0, 0.03, 3)
            img[mask] = lesion_color
            img[mask] += np.random.normal(0, 0.02, img[mask].shape)

        else:
            mask = np.zeros((img_size, img_size), dtype=bool)
            for _ in range(np.random.randint(2, 4)):
                cx_offset = np.random.randint(-img_size*0.1, img_size*0.1)
                cy_offset = np.random.randint(-img_size*0.1, img_size*0.1)
                ax = np.random.uniform(0.08, 0.18) * img_size
                ay = np.random.uniform(0.08, 0.18) * img_size
                dist = np.sqrt(((X - (center_x + cx_offset)) / ax) ** 2 + ((Y - (center_y + cy_offset)) / ay) ** 2)
                mask = mask | (dist <= 1)

            noisy_mask = mask.astype(float) + np.random.uniform(-0.4, 0.4, size=(img_size,img_size))
            mask = noisy_mask > 0.5

            base_color = np.array([80, 40, 20]) / 255.0
            img[mask] = base_color

            for _ in range(np.random.randint(3, 6)):
                patch_mask = np.zeros((img_size, img_size), dtype=bool)
                if np.sum(mask) > 0:
                    px, py = np.argwhere(mask)[np.random.randint(np.sum(mask))]
                    p_size = np.random.uniform(0.02, 0.08) * img_size
                    dist = np.sqrt((X - py) ** 2 + (Y - px) ** 2)
                    patch_mask[dist < p_size] = True

                    patch_colors = [
                        np.array([10, 5, 5]) / 255.0,
                        np.array([139, 0, 0]) / 255.0
                    ]
                    patch_color = patch_colors[np.random.randint(0, len(patch_colors))]
                    img[mask & patch_mask] = patch_color

        images[i] = np.clip(gaussian_filter(img, sigma=0.5), 0, 1)

    print("Synthetic data generation complete.")
    return images, labels

X, y = generate_synthetic_melanoma_data()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
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
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.suptitle('Model Training History')
plt.show()

def display_predictions(model, images, labels, num_samples=10):
    indices = np.random.choice(len(images), num_samples, replace=False)

    plt.figure(figsize=(15, 8))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[index])
        plt.xticks([])
        plt.yticks([])

        img_array = np.expand_dims(images[index], axis=0)
        prediction_score = model.predict(img_array, verbose=0)[0][0]

        pred_label_index = 1 if prediction_score > 0.5 else 0
        true_label_index = labels[index]

        pred_label_name = CLASS_NAMES[pred_label_index]
        true_label_name = CLASS_NAMES[true_label_index]

        color = 'green' if pred_label_index == true_label_index else 'red'
        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}", color=color)

    plt.tight_layout()
    plt.show()

print("\nDisplaying sample predictions from validation set...")
display_predictions(model, X_val, y_val)
