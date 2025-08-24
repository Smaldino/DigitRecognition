import tensorflow as tf
from tensorflow import layers, models
import numpy as np
import matplotlib.pyplot as plt


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to fit the model (flatten 28x28 to 784)
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

print(f"Training data shape: {x_train.shape}")  # Should be (60000, 784)
print(f"Training labels shape: {y_train.shape}")  # Should be (60000,)