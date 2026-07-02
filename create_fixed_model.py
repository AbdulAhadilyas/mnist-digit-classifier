import tensorflow as tf
from tensorflow import keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model (this will take 1-2 minutes)...")
model.fit(x_train, y_train, epochs=6, batch_size=128, validation_split=0.2, verbose=1)

# Save models
model.save("mnist_model_fixed.h5")
model.save("mnist_model_fixed.keras")

print("\n✅ Models saved successfully!")
print("Files should be here:")
import os
print(os.listdir("."))
