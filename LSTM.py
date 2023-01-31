import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ticker = input("Input stock ticker: ")

today = pd.Timestamp.today().date()

stock_data = yf.download(ticker, start='2016-01-01', end=today)
close_prices = stock_data['Close']
values = close_prices.values

training_data_len = math.ceil(len(values)* 0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 1, epochs=25)

data = values[-1]
dataScaled = scaler.transform(data.reshape(-1, 1))

tomorrow_pred = model.predict(dataScaled)
tomorrow_pred = scaler.inverse_transform(tomorrow_pred)

print(f"Last closing price: ${close_prices.iloc[-1]}")
print(f"Tomorrow's closing price: ${tomorrow_pred[0][0]}")
