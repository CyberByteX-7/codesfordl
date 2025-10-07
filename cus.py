import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
df = pd.DataFrame(np.random.rand(200, 3) * [50, 100, 100] + [20, 0, 0], columns=["Age", "Income", "SpendingScore"])
X_scaled = StandardScaler().fit_transform(df)
input_layer = Input(shape=(X_scaled.shape[1],))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(2, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(X_scaled.shape[1])(decoded)
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer=Adam(0.01), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)
X_encoded = encoder.predict(X_scaled, verbose=0)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_encoded)
plt.figure(figsize=(8, 6))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=df['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X')
plt.title('Customer Segmentation with Autoencoder + KMeans')
plt.xlabel('Encoded Feature 1'); plt.ylabel('Encoded Feature 2')
plt.colorbar(label='Cluster'); plt.show()
print(df.head(10))