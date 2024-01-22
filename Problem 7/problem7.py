#In this code, we define a simple MLP with one hidden layer and one output layer.
#Each layer uses a linear function and the ReLU activation function.
#The plot shows that the output of the MLP is a piecewise linear function.

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define a linear function
def linear(x, a, b):
    return a * x + b

# Define a function that represents a simple MLP with one hidden layer
def mlp(x, weights):
    # Compute the output of the hidden layer
    hidden_layer_output = relu(linear(x, weights['hidden'][0], weights['hidden'][1]))
    # Compute the output of the MLP
    output = linear(hidden_layer_output, weights['output'][0], weights['output'][1])
    return output

# Define the weights of the MLP
weights = {
    'hidden': (2, -1),  # Weights for the hidden layer
    'output': (1, 0),  # Weights for the output layer
}

# Generate some input values
x = np.linspace(-10, 10, 400)

# Compute the output of the MLP for the input values
y = mlp(x, weights)

# Plot the input and output
plt.figure(figsize=(3.5, 2.5))
plt.plot(x, y, label='Model Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.legend()
plt.grid()
plt.show()
