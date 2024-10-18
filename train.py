import tensorflow as tf
import matplotlib.pyplot as plt

WINDOW_SIZE = 64

print("Load datasets")
train_data = tf.data.Dataset.load("train_data")


# Build the model

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(WINDOW_SIZE,1)),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 110.0)
])

"""
# Build the Model

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE,1)), 
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu"),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

"""
learning_rate = 1e-6
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mse"])

# Train the model
history = model.fit(train_data,epochs=100)

# Plot utility
def plot_graphs(history, string1, string2):
  plt.plot(history.history[string1], label=string1)
  plt.plot(history.history[string2], label=string2)
  plt.grid()
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylim(0, 20)
  plt.show()

# Plot the mae and loss
plot_graphs(history, "mse", "loss")

# Save the weights
model.save('trained_model.h5')
