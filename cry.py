import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('/content/BTC-USD(1).csv', index_col='Date', parse_dates=True)[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

look_back = 60
X_train = np.array([scaled_data[i-look_back:i, 0] for i in range(look_back, len(scaled_data))])
y_train = scaled_data[look_back:, 0]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[EarlyStopping(monitor='loss', patience=10)])

test_data = scaled_data[-(look_back + 30):]
X_test = np.array([test_data[i-look_back:i, 0] for i in range(look_back, len(test_data))])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_prices = scaler.inverse_transform(model.predict(X_test))
actual_prices = data['Close'].values[-len(predicted_prices):]

plt.figure(figsize=(14, 6))
plt.plot(data.index[-len(predicted_prices):], actual_prices, color='blue', label='Actual Price')
plt.plot(data.index[-len(predicted_prices):], predicted_prices, color='red', label='Predicted Price')
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
