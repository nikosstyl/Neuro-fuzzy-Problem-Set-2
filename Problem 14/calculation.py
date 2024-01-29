import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

# Layer 1 - Convolutional
model.add(Conv2D(100, (5,5), activation='relu', input_shape=(100, 100, 1)))

# Layer 2 - Convolutional with 'same' padding
model.add(Conv2D(100, (5,5), padding='same', activation='relu'))

# Layer 3 - MaxPooling
model.add(MaxPooling2D(pool_size=(2,2))) 

# Layer 4 - Dense
model.add(Dense(100, activation='relu'))

# Layer 5 - Dense
model.add(Dense(100, activation='relu'))

# Layer 6 - Single Output Unit
model.add(Dense(1, activation='sigmoid'))

#print(model.summary())
total_weights = sum([layer.count_params() for layer in model.layers if not isinstance(layer, Dense)])
total_biases = sum([layer.units for layer in model.layers if isinstance(layer, Dense)])

print("Total number of weights (excluding biases) in the model: ", total_weights - total_biases)
# print("Total number of weights in the model: ", model.count_params())