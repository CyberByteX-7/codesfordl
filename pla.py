import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=num_samples)
    for i in range(num_samples):
        img = np.zeros((img_size, img_size, 3)); img[:, :, :] = [0.8, 0.9, 0.8]
        leaf_color = [0.1, np.random.uniform(0.4, 0.6), 0.1]
        Y, X = np.ogrid[:img_size, :img_size]
        center_x, center_y = img_size//2, img_size//2
        axis_x, axis_y = np.random.randint(img_size//4, img_size//2), np.random.randint(img_size//3, img_size//2)
        leaf_mask = np.sqrt(((X-center_x)/axis_x)**2 + ((Y-center_y)/axis_y)**2) <= 1
        img[leaf_mask] = leaf_color + np.random.normal(0, 0.05, img[leaf_mask].shape)
        if labels[i] != 0:
            spot_color = [0.6, 0.3, 0.05] if labels[i]==1 else [0.2, 0.2, 0.2]
            spot_size_range = (1, 5) if labels[i]==1 else (2, 7)
            for _ in range(np.random.randint(10, 30)):
                spot_coords = np.argwhere(leaf_mask)
                spot_cy, spot_cx = spot_coords[np.random.randint(len(spot_coords))]
                s_size = np.random.randint(*spot_size_range)
                sY, sX = np.ogrid[:img_size, :img_size]
                spot_mask = ((sX - spot_cx)**2 + (sY - spot_cy)**2) < s_size**2
                img[spot_mask] = spot_color
        img = rotate(img, np.random.uniform(-45, 45), reshape=False, mode='nearest')
        images[i] = np.clip(img, 0, 1)
    return images, labels
def display_predictions(model, images, labels, num_samples=10):
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1); plt.imshow(images[idx]); plt.axis('off')
        pred = model.predict(np.expand_dims(images[idx], 0), verbose=0)
        pred_name = CLASS_NAMES[np.argmax(pred)]
        true_name = CLASS_NAMES[np.argmax(labels[idx])]
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=('green' if pred_name==true_name else 'red'))
    plt.tight_layout(); plt.show()
X, y = generate_synthetic_plant_data()
y_cat = to_categorical(y, num_classes=NUM_CLASSES)
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.show()
display_predictions(model, X_val, y_val)