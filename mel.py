import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.ndimage import gaussian_filter
IMG_SIZE = 96
CLASS_NAMES = ['Benign', 'Melanoma']
NUM_SAMPLES = 1000
def generate_synthetic_melanoma_data(num_samples=NUM_SAMPLES, img_size=IMG_SIZE):
    images = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    labels = np.random.randint(0, 2, size=num_samples)
    for i in range(num_samples):
        skin_color = np.array([234, 192, 134])/255. + np.random.normal(0, 0.05, 3)
        img = np.ones((img_size, img_size, 3)) * skin_color
        Y, X = np.ogrid[:img_size, :img_size]
        center_x, center_y = np.random.randint(img_size*0.3, img_size*0.7, 2)
        if labels[i] == 0:
            axis_x = np.random.uniform(0.1, 0.2) * img_size
            axis_y = axis_x * np.random.uniform(0.9, 1.1)
            mask = np.sqrt(((X - center_x)/axis_x)**2 + ((Y - center_y)/axis_y)**2) <= 1
            lesion_color = np.array([101, 67, 33])/255. + np.random.normal(0, 0.03, 3)
            img[mask] = lesion_color + np.random.normal(0, 0.02, img[mask].shape)
        else:
            mask = np.zeros((img_size, img_size), dtype=bool)
            for _ in range(np.random.randint(2, 4)):
                cx_off, cy_off = np.random.randint(-img_size*0.1, img_size*0.1, 2)
                ax, ay = np.random.uniform(0.08, 0.18, 2) * img_size
                mask |= np.sqrt(((X - (center_x+cx_off))/ax)**2 + ((Y - (center_y+cy_off))/ay)**2) <= 1
            mask = (mask.astype(float) + np.random.uniform(-0.4, 0.4, mask.shape)) > 0.5
            img[mask] = np.array([80, 40, 20])/255.
            for _ in range(np.random.randint(3, 6)):
                if np.sum(mask) > 0:
                    px, py = np.argwhere(mask)[np.random.randint(np.sum(mask))]
                    p_mask = np.sqrt((X - py)**2 + (Y - px)**2) < np.random.uniform(0.02, 0.08)*img_size
                    p_color = np.array([[10,5,5],[139,0,0]])[np.random.randint(2)]/255.
                    img[mask & p_mask] = p_color
        images[i] = np.clip(gaussian_filter(img, sigma=0.5), 0, 1)
    return images, labels
def display_predictions(model, images, labels, class_names, num_samples=10):
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx]); plt.axis('off')
        score = model.predict(np.expand_dims(images[idx], 0), verbose=0)[0][0]
        pred_label = class_names[1 if score > 0.5 else 0]
        true_label = class_names[labels[idx]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.tight_layout()
    plt.show()
X, y = generate_synthetic_melanoma_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=25,
                    batch_size=160,
                    verbose=1,
                    callbacks=[early_stop])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1);
plt.plot(history.history['accuracy'], label='Train Accuracy');
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2);
plt.plot(history.history['loss'], label='Train Loss');
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.suptitle('Model Training History'); plt.show()
class_names = ['Benign', 'Melanoma']
display_predictions(model, X_val, y_val, class_names)