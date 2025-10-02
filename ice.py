import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
# This is the corrected import statement
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam

def generate_synthetic_iceberg_data(num_samples=1604):
    print(f"Generating {num_samples} synthetic images for demonstration...")
    data_list = []
    for i in range(num_samples):
        is_iceberg = np.random.randint(0, 2)
        band_1 = np.random.normal(loc=-20, scale=6, size=(75, 75))
        band_2 = band_1 + np.random.normal(loc=0, scale=3, size=(75, 75))
        if is_iceberg:
            center_x, center_y = np.random.randint(20, 55, size=2)
            size_x, size_y = np.random.randint(15, 30, size=2)
            Y, X = np.ogrid[:75, :75]
            dist_from_center = np.sqrt(((X - center_x)/size_x)**2 + ((Y - center_y)/size_y)**2)
            mask = dist_from_center <= 1
            noisy_mask = mask.astype(float) + np.random.uniform(-0.4, 0.4, size=(75,75))
            mask = noisy_mask > 0.5

            brightness = np.random.uniform(20, 30)
            band_1[mask] += brightness
            band_2[mask] += brightness * np.random.uniform(0.9, 1.1)
        else:
            center_x, center_y = np.random.randint(15, 60, size=2)
            width, height = np.random.randint(2, 4), np.random.randint(8, 18)
            if np.random.rand() > 0.5:
                width, height = height, width
            x1 = max(0, center_x - width // 2)
            x2 = min(75, center_x + width // 2)
            y1 = max(0, center_y - height // 2)
            y2 = min(75, center_y + height // 2)
            brightness = np.random.uniform(10, 20)
            band_1[y1:y2, x1:x2] += brightness
            band_2[y1:y2, x1:x2] += brightness * np.random.uniform(0.9, 1.1)
        data_list.append({
            'id': f'synthetic_{i}',
            'band_1': band_1.flatten().tolist(),
            'band_2': band_2.flatten().tolist(),
            'is_iceberg': is_iceberg
        })
    print("Synthetic data generation complete.")
    return pd.DataFrame(data_list)

data = generate_synthetic_iceberg_data()

def process_band(band):
    return np.array(band).reshape(75, 75)

images = []
for i, row in data.iterrows():
    band_1 = process_band(row['band_1'])
    band_2 = process_band(row['band_2'])
    img = np.dstack((band_1, band_2, (band_1 + band_2) / 2))
    images.append(img)

X = np.array(images, dtype=np.float32)
y = data['is_iceberg'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(75,75,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
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
    epochs=25,
    batch_size=32
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nValidation Accuracy: {val_acc*100:.2f}%')

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
