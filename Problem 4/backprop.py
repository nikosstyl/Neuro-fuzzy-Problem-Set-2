import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

# Define the target function
def target_function(p):
    return 1 + np.sin(3 * np.pi * p / 8)

# Generate training data
p_train = np.linspace(-2, 2, 200).reshape(-1, 1)
g_train = target_function(p_train)

# Define the model function
def build_model(hidden_units, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, input_dim=1, activation='sigmoid',
                              kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5),
                              bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5)),
        tf.keras.layers.Dense(1, activation='relu', 
                              kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5),
                              bias_initializer=keras.initializers.Constant(1))
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Training configurations
hidden_layer_configs = [2, 8, 12]
learning_rates = [0.001, 0.01, 0.1]

# Experiment with different configurations
for hidden_units in hidden_layer_configs:
    for lr in learning_rates:
        print(f"\nTraining with {hidden_units} hidden units and learning rate {lr}\n")
        model = build_model(hidden_units, lr)

        hidden_weights = np.random.uniform(-0.5, 0.5, size=(1, hidden_units))
        hidden_bias = np.random.uniform(-0.5, 0.5, size=(1, hidden_units))

        # Train the model
        history = model.fit(p_train, g_train, epochs=500, verbose=1, batch_size=200)
        
        # Plot the results
        plt.plot(history.history['loss'], label=f'loss_{hidden_units}_{lr}')
        plt.legend()
        plt.title(f'Learning Curve for {hidden_units} Hidden Units with LR {lr}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        # Predict and plot the function approximation
        p_test = np.linspace(-2, 2, 100).reshape(-1, 1)
        g_pred = model.predict(p_test)
        
        plt.plot(p_test, g_pred, label=f'Predicted - {hidden_units} units, lr {lr}')
        plt.plot(p_test, target_function(p_test), label='Target Function')
        plt.legend()
        plt.title(f'Function Approximation with {hidden_units} Hidden Units and LR {lr}')
        plt.xlabel('p')
        plt.ylabel('g(p)')
        plt.show()

# Note: For detailed analysis of convergence properties and accuracy,
# one should collect and analyze the loss values, and predicted vs actual function values.
