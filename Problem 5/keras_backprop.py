import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout

USE_TF = True

if USE_TF:
    from tensorflow import keras
else:
    import keras

# Define the target function
def target_function(p):
    return 1 + np.sin(3 * np.pi * p / 8)

# Define the model function
def build_model(hidden_units, learning_rate):
    model = keras.Sequential()
    
    first_layer = keras.layers.Dense(hidden_units, input_dim=1, activation='sigmoid',
                                  kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5),
                                  bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5))
    middle_layer = keras.layers.Dropout(0.15)
    second_layer = keras.layers.Dense(1, activation='relu', input_dim=hidden_units,
                                  kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5),
                                  bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5))
    
    model.add(first_layer)
    model.add(middle_layer)
    model.add(second_layer)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Generate training data
p_train = np.linspace(-2, 2, 300).reshape(-1, 1)
g_train = target_function(p_train)

# Training configurations
hidden_units = 12
hidden_layer_configs = [12]
learning_rates = [0.1]

MAX_EPOCHS = 40000

# Experiment with different configurations
for hidden_units in hidden_layer_configs:
    for lr in learning_rates:
        print(f"\nTraining with {hidden_units} hidden units and learning rate {lr}\n")

        model = build_model(hidden_units, lr)

        # Train the model
        history = model.fit(p_train, g_train, epochs=MAX_EPOCHS, batch_size=hidden_units, shuffle=True)
        
        # Plot the results
        plt.figure()
        plt.plot(history.history['loss'], label=f'loss_{hidden_units}_{lr}')
        plt.legend()
        plt.title(f'Learning Curve for {hidden_units} Hidden Units with LR {lr}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        # Predict and plot the function approximation
        p_test = np.linspace(-2, 2, 100).reshape(-1, 1)
        g_pred = model.predict(p_test)
        
        plt.figure()
        plt.plot(p_test, g_pred, label=f'Predicted - {hidden_units} units, lr {lr}')
        plt.plot(p_test, target_function(p_test), label='Target Function')
        plt.legend()
        plt.title(f'Function Approximation with {hidden_units} Hidden Units and LR {lr}')
        plt.xlabel('p')
        plt.ylabel('g(p)')
        plt.grid(True)

plt.show()