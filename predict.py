import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

savedModel=load_model("trained_model.h5")
BATCH_SIZE = 256

with open("valid_data.pkl", 'rb') as f:
    SPLIT_TIME, WINDOW_SIZE, series, series_valid, time_valid = pkl.load(f)

def plot_series(time, real, predicted, format="-", start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], real[start:end], label="real")
    plt.plot(time[start:end], predicted[start:end], label="predicted")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

def model_forecast(model, series, window_size):
    """Generates a forecast using your trained model"""
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def compute_metrics(true_series, forecast):
    """Computes MSE and MAE metrics for the forecast"""
    mse = tf.keras.losses.MSE(true_series, forecast)
    mae = tf.keras.losses.MAE(true_series, forecast)
    return mse, mae

rnn_forecast = model_forecast(savedModel, series[SPLIT_TIME-WINDOW_SIZE:-1], WINDOW_SIZE).squeeze()

# Plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, rnn_forecast)

mse, mae = compute_metrics(series_valid, rnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")