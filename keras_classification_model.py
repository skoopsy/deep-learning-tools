import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# Using MINST data from Keras library
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Show some info
print(X_train.shape)
plt.imshow(X_train[0])
plt.waitforbuttonpress(0)
plt.close()

# Flatten images into 1D vector
num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalise pixel values to between 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encoding of catagories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

def classification_model():
	# Create
	model = Sequential()
	model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	# Compile
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Train and test
model = classification_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# Evaluate
scores = model.evaluate(X_test, y_test, verbose=0)

figure, axis = plt.subplots(2,1)

# Plot training & validation loss values
axis[0].plot(history.history['loss'], label='Training Loss')
axis[0].plot(history.history['val_loss'], label='Validation Loss')
axis[0].set_title('Model Loss vs Epochs')
axis[0].set_ylabel('Loss')
axis[0].set_xlabel('Epoch')
axis[0].legend()

# Plot training & validation accuracy values
axis[1].plot(history.history['accuracy'], label='Training Accuracy')
axis[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axis[1].set_title('Model Accuracy vs Epochs')
axis[1].set_ylabel('Accuracy')
axis[1].set_xlabel('Epoch')
axis[1].legend()

plt.subplots_adjust(hspace=0.5) 

plt.show()

# Save trained model
print("Saving model...")
model_save_name = 'classification_model.keras'
#model.save(model_save_name)  # Legacy save and file format (.h5)
keras.saving.save_model(model, model_save_name)
print(f"Model saved as: {model_save_name}")

"""
For reloading the trained model:
	from keras.models import load_model
	pretrained_model = load_model('model-name.h5')
"""
