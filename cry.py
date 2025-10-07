import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv('/content/BTC-USD(1).csv', index_col='Date', parse_dates=True)[['Close']]
scaler = MinMaxScaler()
s_data = scaler.fit_transform(df)
lb = 60
X = np.array([s_data[i-lb:i, 0] for i in range(lb, len(s_data))])[..., np.newaxis]
y = s_data[lb:, 0]
model = Sequential([LSTM(50, input_shape=(lb, 1)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, callbacks=[EarlyStopping(patience=3)], verbose=1)
test_slice = s_data[-(lb + 30):]
X_test = np.array([test_slice[i-lb:i, 0] for i in range(lb, len(test_slice))])[..., np.newaxis]
preds = scaler.inverse_transform(model.predict(X_test))
plt.figure(figsize=(12, 5))
plt.plot(df.index[-30:], df['Close'].values[-30:], label='Actual')
plt.plot(df.index[-30:], preds, label='Predicted')
plt.title('BTC Price Prediction'); plt.legend(); plt.show()