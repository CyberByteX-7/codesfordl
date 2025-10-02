import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.15
SYNTHETIC_SAMPLES = 500

def generate_synthetic_data(num_samples=SYNTHETIC_SAMPLES, height=IMG_HEIGHT, width=IMG_WIDTH):
    print(f"Generating {num_samples} synthetic images for demonstration...")
    images = np.zeros((num_samples, height, width, 3), dtype=np.float32)
    masks = np.zeros((num_samples, height, width, 1), dtype=np.float32)
    for i in range(num_samples):
        background_color = np.array([np.random.uniform(0, 0.1), np.random.uniform(0, 0.2), np.random.uniform(0.3, 0.7)])
        img = np.ones((height, width, 3)) * background_color
        img += np.random.normal(0, 0.03, img.shape)
        img = np.clip(img, 0, 1)
        mask = np.zeros((height, width, 1), dtype=np.float32)
        num_ships = np.random.randint(1, 4)
        for _ in range(num_ships):
            ship_h = np.random.randint(5, 20)
            ship_w = np.random.randint(10, 40)
            if np.random.rand() > 0.5:
                ship_h, ship_w = ship_w, ship_h
            start_y = np.random.randint(0, height - ship_h)
            start_x = np.random.randint(0, width - ship_w)
            ship_color = np.random.uniform(0.4, 0.8)
            img[start_y:start_y+ship_h, start_x:start_x+ship_w, :] = ship_color
            mask[start_y:start_y+ship_h, start_x:start_x+ship_w, 0] = 1.0
        images[i] = img
        masks[i] = mask
    print("Synthetic data generation complete.")
    return images, masks

def build_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def plot_history(history, metric_name='dice_coefficient'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric_name], label='Training Dice Coeff')
    plt.plot(history.history[f'val_{metric_name}'], label='Validation Dice Coeff')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def display_prediction(model, images, masks, num_samples=3):
    indices = np.random.choice(len(images), num_samples)
    plt.figure(figsize=(10, num_samples * 4))
    for i, index in enumerate(indices):
        img = images[index]
        true_mask = masks[index]
        pred_mask = model.predict(np.expand_dims(img, axis=0))[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(np.squeeze(true_mask), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(np.squeeze(pred_mask), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

if __name__ == '__main__':
    images, masks = generate_synthetic_data()
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print("Building the U-Net model...")
    model = build_unet_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient])
    model.summary()
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    model.save('ship_segmentation_unet.keras')
    print("Model saved successfully as 'ship_segmentation_unet.keras'")
    print("Plotting training history...")
    plot_history(history)
    print("Displaying predictions on validation data...")
    display_prediction(model, X_val, y_val)
