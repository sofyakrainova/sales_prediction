import tensorflow as tf
import keras_tuner as kt
import pickle as pkl

train_data = tf.data.Dataset.load("train_data")
with open("valid_data.pkl", 'rb') as f:
    SPLIT_TIME, WINDOW_SIZE, series, series_valid, time_valid = pkl.load(f)

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(WINDOW_SIZE,1)))

  # Tune the embedding dimension
  # Choose an optimal value between 8-32
  rrn_units = hp.Int('rrn_units', min_value=10, max_value=50, step=5)
  model.add(tf.keras.layers.SimpleRNN(rrn_units, return_sequences=True),)
  rrn_units2 = hp.Int('rrn_units2', min_value=10, max_value=50, step=5)
  model.add(tf.keras.layers.SimpleRNN(rrn_units2), )
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  const = hp.Int('const', min_value=50, max_value=150, step=10)
  model.add(tf.keras.layers.Lambda(lambda x: x * const))

  # Tune the learning rate for the optimizer
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
  optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, momentum=0.9)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.Huber(),
                metrics=["mse"])

  return model

def model_forecast(series, window_size):
    """Generates a forecast using your trained model"""
    #series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    return ds

valid_dataset = model_forecast(series[SPLIT_TIME-WINDOW_SIZE:-1], WINDOW_SIZE)

tuner = kt.Hyperband(model_builder,
                     max_epochs=50,
                     objective="mse",
                     overwrite=True,
                     directory="tuner_dir",
                     project_name="price_prediction"
                     )
tuner.search(train_data,
             epochs=20,
             #validation_data = valid_dataset,
             )

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete.\n
 rrn_units is {best_hps.get('rrn_units')}. \n 
 rrn_units2 is {best_hps.get('rrn_units2')}. \n 
 const is {best_hps.get('const')}. \n 
 The optimal learning rate is {best_hps.get('learning_rate')}. \n
 """)