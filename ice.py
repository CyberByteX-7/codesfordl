import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
NUM_SAMPLES, IMG_SIZE = 2000, 75
print(f"Generating {NUM_SAMPLES} synthetic images...")
X = np.zeros((NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
y = np.zeros(NUM_SAMPLES, dtype=np.int32)

for i in range(NUM_SAMPLES):
    is_iceberg = np.random.randint(0, 2)
    band_1 = np.random.normal(loc=-20, scale=6, size=(IMG_SIZE, IMG_SIZE))
    band_2 = band_1 + np.random.normal(loc=0, scale=3, size=(IMG_SIZE, IMG_SIZE))
    if is_iceberg:
        center_x, center_y = np.random.randint(20, 55, size=2)
        size_x, size_y = np.random.randint(15, 30, size=2)
        Y_grid, X_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = np.sqrt(((X_grid - center_x)/size_x)**2 + ((Y_grid - center_y)/size_y)**2) <= 1
        mask = ((mask.astype(float) + np.random.uniform(-0.4, 0.4, mask.shape)) > 0.5)
        brightness = np.random.uniform(20, 30)
        band_1[mask] += brightness
        band_2[mask] += brightness * np.random.uniform(0.9, 1.1)
    else:
        center_x, center_y = np.random.randint(15, 60, size=2)
        w, h = np.random.randint(2, 4), np.random.randint(8, 18)
        if np.random.rand() > 0.5: w, h = h, w
        x1, x2 = max(0, center_x-w//2), min(IMG_SIZE, center_x+w//2)
        y1, y2 = max(0, center_y-h//2), min(IMG_SIZE, center_y+h//2)
        brightness = np.random.uniform(10, 20)
        band_1[y1:y2, x1:x2] += brightness
        band_2[y1:y2, x1:x2] += brightness * np.random.uniform(0.9, 1.1)
    X[i] = np.dstack((band_1, band_2, (band_1 + band_2) / 2))
    y[i] = is_iceberg
print("Data generation complete.")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
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
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=callbacks)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nFinal Validation Accuracy: {val_acc*100:.2f}%')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy'); plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.suptitle('Model Training History'); plt.show()