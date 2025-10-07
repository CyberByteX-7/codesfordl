import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomBrightness
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import gaussian_filter
NUM_SAMPLES, IMG_SIZE = 2000, 128
print(f"Generating {NUM_SAMPLES} synthetic X-ray images for demonstration...")
X = np.zeros((NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
y = np.zeros(NUM_SAMPLES, dtype=np.int32)

for i in range(NUM_SAMPLES):
    label = np.random.randint(0, 2)
    y[i] = label
    image = np.random.rand(IMG_SIZE, IMG_SIZE) * 0.2
    for r in range(8):
        y_pos = int(IMG_SIZE * (r * 0.1 + 0.1))
        image[y_pos-1:y_pos+1, :] += np.random.uniform(0.05, 0.1)
    if label == 1:
        for _ in range(np.random.randint(2, 5)):
            patch = np.zeros((IMG_SIZE, IMG_SIZE))
            center_x, center_y = np.random.randint(int(IMG_SIZE*0.2), int(IMG_SIZE*0.8), 2)
            size = np.random.randint(int(IMG_SIZE*0.1), int(IMG_SIZE*0.25))
            patch[center_y-size//2:center_y+size//2, center_x-size//2:center_x+size//2] = 1
            blurred_patch = gaussian_filter(patch.astype(float), sigma=np.random.uniform(5, 10))
            image += blurred_patch * np.random.uniform(0.2, 0.5)
    image = np.clip(image, 0, 1)
    X[i] = np.dstack([image, image, image])
print("Synthetic data generation complete.")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

data_augmentation = Sequential([
    RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    RandomRotation(0.1),
    RandomBrightness(0.1),
])
model = Sequential([
    data_augmentation,
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nValidation Accuracy: {val_acc*100:.2f}%')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy'); plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.suptitle('Model Training History'); plt.show()
print("\nDisplaying sample predictions from validation set...")
class_names = ['NORMAL', 'PNEUMONIA']
indices = np.random.choice(len(X_val), 10)
plt.figure(figsize=(15, 8))
for i, index in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_val[index]); plt.axis('off')
    true_name = class_names[y_val[index]]
    plt.title(f"True: {true_name}")
plt.tight_layout(); plt.show()