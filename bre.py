import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("/content/breast-cancer.csv")
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
x = df.drop("diagnosis", axis=1)
y = df['diagnosis']
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
input_features = x.shape[1]

model = Sequential([
    Dense(48, activation='relu', input_dim=input_features),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training the model...")
history = model.fit(x_train, y_train, epochs=10, verbose=1)

print("\nEvaluating on Training Data:")
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print("-" * 30)

print("Evaluating on Test Data:")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("-" * 30)
