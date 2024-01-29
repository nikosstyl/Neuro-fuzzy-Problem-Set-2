import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPool2D, Flatten

model = Sequential()

# Layer 1 - Convolutional
model.add(Conv2D(100, (5,5), activation='relu', input_shape=(100, 100, 1)))

# Layer 2 - Convolutional
model.add(Conv2D(100, (5,5), activation='relu'))

# Layer 3 - MaxPooling
model.add(MaxPool2D(pool_size=(2,2))) 
model.add(Flatten())

# Layer 4 - Dense
model.add(Dense(100, activation='relu'))

# Layer 5 - Dense
model.add(Dense(100, activation='relu'))

# Layer 6 - Single Output Unit
model.add(Dense(1, activation='sigmoid'))

print(model.summary())