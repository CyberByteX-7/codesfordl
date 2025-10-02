import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
data = np.random.rand(200, 3) * [50, 100, 100] + [20, 0, 0]
df = pd.DataFrame(data, columns=["Age", "Income", "SpendingScore"])

scaler = StandardScaler()
X = scaler.fit_transform(df)

input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
encoded = Dense(encoding_dim, activation="relu")(encoded)
decoded = Dense(8, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
autoencoder.fit(X, X, epochs=50, batch_size=16, shuffle=True, verbose=0)

X_encoded = encoder.predict(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_encoded)
df["Cluster"] = clusters

plt.figure(figsize=(8, 6))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=clusters, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=200, marker="X")
plt.title("Customer Segmentation with Autoencoder + KMeans")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.colorbar(label="Cluster")
plt.show()

print(df.head(10))
