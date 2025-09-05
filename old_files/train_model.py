import tensorflow as tf
from tensorflow.keras import layers, models
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

model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(784,)),  # Reshape flat vector to 28x28x1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=16,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict on a few test images
number_pred = 20
predictions = model.predict(x_test[:number_pred])
predicted_labels = np.argmax(predictions, axis=1)

# Plot the first x test images and predictions
plt.figure(figsize=(10, 4))
for i in range(number_pred):
    plt.subplot(1, number_pred, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}")
    plt.axis('off')
plt.show()

model.save('./trained_model/DR3.keras')
print("Model saved!")