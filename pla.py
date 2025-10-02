import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy.ndimage import rotate

IMG_SIZE = 128
NUM_CLASSES = 3
CLASS_NAMES = ['Healthy', 'Rust', 'Scab']
NUM_SAMPLES = 1500

def generate_synthetic_plant_data(num_samples=NUM_SAMPLES, img_size=IMG_SIZE):
    print(f"Generating {num_samples} synthetic leaf images...")
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=num_samples)

    for i in range(num_samples):
        img = np.zeros((img_size, img_size, 3))
        leaf_color = [0.1, np.random.uniform(0.4, 0.6), 0.1]
        img[:, :, :] = [0.8, 0.9, 0.8]

        center_x, center_y = img_size // 2, img_size // 2
        axis_x, axis_y = np.random.randint(img_size // 4, img_size // 2), np.random.randint(img_size // 3, img_size // 2)
        Y, X = np.ogrid[:img_size, :img_size]
        dist_from_center = np.sqrt(((X - center_x) / axis_x) ** 2 + ((Y - center_y) / axis_y) ** 2)
        leaf_mask = dist_from_center <= 1

        img[leaf_mask] = leaf_color
        img[leaf_mask] += np.random.normal(0, 0.05, img[leaf_mask].shape)

        label = labels[i]
        if label != 0:
            num_spots = np.random.randint(10, 30)
            if label == 1:
                spot_color = [0.6, 0.3, 0.05]
                spot_size_range = (1, 5)
            else:
                spot_color = [0.2, 0.2, 0.2]
                spot_size_range = (2, 7)

            for _ in range(num_spots):
                spot_coords = np.argwhere(leaf_mask)
                spot_center_y, spot_center_x = spot_coords[np.random.randint(len(spot_coords))]
                spot_size = np.random.randint(*spot_size_range)

                sY, sX = np.ogrid[:img_size, :img_size]
                spot_mask = ((sX - spot_center_x) ** 2 + (sY - spot_center_y) ** 2) < spot_size**2
                img[spot_mask] = spot_color

        angle = np.random.uniform(-45, 45)
        img = rotate(img, angle, reshape=False, mode='nearest')

        images[i] = np.clip(img, 0, 1)

    print("Synthetic data generation complete.")
    return images, labels

X, y = generate_synthetic_plant_data()

y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
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
        prediction = model.predict(img_array, verbose=0)

        pred_label_index = np.argmax(prediction)
        true_label_index = np.argmax(labels[index])

        pred_label_name = CLASS_NAMES[pred_label_index]
        true_label_name = CLASS_NAMES[true_label_index]

        color = 'green' if pred_label_index == true_label_index else 'red'
        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}", color=color)

    plt.tight_layout()
    plt.show()

print("\nDisplaying sample predictions from validation set...")
display_predictions(model, X_val, y_val)
