import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# global variables
SPLIT_RATE = 0.8
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 100

price_df = pd.read_csv("../sales_data/TimeSeriesData.csv", parse_dates=[0])
print("Dataframe size ", price_df.shape)

SPLIT_TIME = int(price_df.shape[0]*SPLIT_RATE)

plt.plot(price_df["Date"], price_df["Product"])
plt.grid()
plt.show()
# check if there are any missing values
print(price_df.isnull().values.any())

def train_val_split(time, series):
    """ Splits time series into train and validations sets"""
    time_train = time[:SPLIT_TIME]
    series_train = series[:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series_valid = series[SPLIT_TIME:]

    return time_train, series_train, time_valid, series_valid

# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(price_df.index.values, price_df["Product"].values)

def windowed_dataset(series, window_size):
    """Creates windowed dataset"""
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(BATCH_SIZE).prefetch(1)
    return dataset


# Generate the dataset windows
train_dataset = windowed_dataset(series_train, window_size=WINDOW_SIZE)
train_dataset.save("train_data")
filehandler = open(b"valid_data.pkl","wb")
pkl.dump((SPLIT_TIME, WINDOW_SIZE, price_df["Product"].values, series_valid, time_valid), filehandler)
