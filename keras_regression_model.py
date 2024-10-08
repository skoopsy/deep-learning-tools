import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')

# Data Checks
concrete_data.head()
print(concrete_data.isnull().sum())

# Split into preictor and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Normalise data
predictors_norm = (predictors - predictors.mean()) / predictors.std()

n_cols = predictors_norm.shape[1]  # Num of predictors

def regression_model():
	# Create
	model = Sequential()
	model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))

	# Compile
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model


# Build model
model = regression_model()

# Train and test using fit
history = model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Plot the loss vs epoch
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

