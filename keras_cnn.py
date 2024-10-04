import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from keras.layers import Conv2D  # Convolutional layer
from keras.layers import MaxPooling2D  # Add pooling layer
from keras.layers import Flatten  # Flatten data of fully connected layers

import matplotlib.pyplot as plt

# Import MNIST dataset (handwriting)
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalise between 0-1
X_train = X_train / 255
X_test = X_test / 255

# Convert target to binary categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1] 

def build_convolutional_model():
	
	# Create
	model = Sequential()

	# Conv layer with 16 filters, then max pooling
	model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# Conv layer with 8 filters
	model.add(Conv2D(8, (2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	# Fully connected layer
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))

	# Output layer
	model.add(Dense(num_classes, activation='softmax'))

	# Compile
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model

model = build_convolutional_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

# Plot the training and validation loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot loss
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
