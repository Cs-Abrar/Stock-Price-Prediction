import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime as dt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Downloading the data
st.title('Stock Price Prediction Web App')
st.sidebar.title('Stock Price Prediction')


stock_ticker = st.sidebar.text_input('Enter Stock Name: ', 'BTC-USD')
start_date = st.sidebar.date_input('Start Date:', dt.date(2010, 1, 1))
end_date = st.sidebar.date_input('End Date:', dt.date.today())
data = yf.download(stock_ticker, start_date, end_date)

# Historical Stock Data
st.subheader(f'Historical Stock Data of {stock_ticker}')
st.write(data.head())

# Calculate Moving Averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Plotting Moving Averages
st.subheader('Moving Averages (50-day & 200-day)')
fig_ma = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA_50'], label='50-day SMA', color='orange')
plt.plot(data['SMA_200'], label='200-day SMA', color='green')
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_ma)

# Calculate Price Volatility (Standard Deviation)
data['Volatility'] = data['Close'].rolling(window=30).std()

# Plotting Volatility
st.subheader('Price Volatility')
fig_vol = plt.figure(figsize=(12, 6))
plt.plot(data['Volatility'], label='30-day Volatility', color='purple')
plt.title('Price Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
st.pyplot(fig_vol)

# Calculate Relative Strength Index (RSI)
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Plotting RSI
st.subheader('Relative Strength Index (RSI)')
fig_rsi = plt.figure(figsize=(12, 6))
plt.plot(data['RSI'], label='RSI', color='brown')
plt.axhline(70, linestyle='--', color='red', label='Overbought Threshold')
plt.axhline(30, linestyle='--', color='green', label='Oversold Threshold')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
st.pyplot(fig_rsi)

# Plotting the historical Close Price
st.subheader(f'Closing Price of {stock_ticker}')
fig = plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title(f'Close Price History of {stock_ticker}')
plt.xlabel('Date')
plt.ylabel("Close Price USD ($)")
st.pyplot(fig)

# Preparing the data for training the model
X = data.drop(['Close'], axis=1)
y = data['Close']

# Normalizing the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])  # Scale only 'Close' price

# Create Time-Series Sequences
sequence_length = 30  # Use the last 30 days to predict the next day
X, y = [], []

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])  # Last 30 days as input
    y.append(data_scaled[i])  # Next day's price as output

X, y = np.array(X), np.array(y)  # Convert to numpy arrays

# Split Data into Training and Testing Sets
train_size = int(len(X) * 0.8)  # 80% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# EarlyStopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with a progress spinner
with st.spinner('Training the LSTM model...'):
    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=1
    )

# Model Training History Plot
st.subheader("Model Training Loss")
fig_loss = plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
st.pyplot(fig_loss)

# Predicting on Test Data
predictions = model.predict(X_test)

# Inverse Scaling to Get Predictions in Original Price Range
predictions = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Performance Metrics
mse = mean_squared_error(y_test_inverse, predictions)
mae = mean_absolute_error(y_test_inverse, predictions)
r2 = r2_score(y_test_inverse, predictions)

# Displaying Metrics
st.subheader("Model Performance Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plotting the results
st.subheader(f"Predicted vs Actual Prices")
fig_pred = plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title(f'{stock_ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
st.pyplot(fig_pred)
