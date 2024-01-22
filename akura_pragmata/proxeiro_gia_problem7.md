## PROBLEM 7

A Multi-Layer Perceptron (MLP) using only ReLU (Rectified Linear Unit) or pReLU (Parametric Rectified Linear Unit) as activation functions constructs a continuous piecewise linear function. Here's why:

ReLU Function: The ReLU function is defined as f(x) = max(0, x). This function is linear for all positive values and zero for all negative values. Therefore, it is a piecewise linear function.

pReLU Function: The pReLU function is a variant of ReLU that allows small negative values when the input is less than zero. It is defined as f(x) = max(0, x) + a * min(0, x), where a is a small constant. Like ReLU, pReLU is also a piecewise linear function.

Composition of Linear Functions: In an MLP, the output of each layer is a composition of the activation function (ReLU or pReLU in this case) and a linear transformation (the weighted sum of inputs). The composition of linear functions is still a linear function.

Piecewise Linear Function: Since each neuron in the MLP applies a piecewise linear function (ReLU or pReLU) to a linear transformation of its inputs, the entire MLP represents a composition of piecewise linear functions. Therefore, an MLP using only ReLU or pReLU constructs a continuous piecewise linear function.

This is a high-level explanation. The detailed mathematical proof would involve showing that the composition of piecewise linear functions is still a piecewise linear function, which is beyond the scope of this assistant's capabilities.

##### 

The detailed mathematical proof involves understanding the composition of functions and the properties of linear functions. Here's a simplified version:

A linear function is defined as f(x) = ax + b, where a and b are constants. The function is continuous and its graph is a straight line.

The ReLU function is defined as f(x) = max(0, x). This function is piecewise linear, meaning it is composed of linear segments. The function is continuous for all real numbers, and its graph is a line that bends at the origin.

In an MLP, each neuron computes a weighted sum of its inputs (a linear function) and applies the activation function to the result. If the activation function is ReLU, the output is a piecewise linear function.

The output of the entire MLP is a composition of the outputs of its neurons. Since the composition of piecewise linear functions is still a piecewise linear function, the output of the MLP is a piecewise linear function.

Now, let's demonstrate this with a simple Python code:

##### CODE ########
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the output of the MLP
plt.plot(x, y)
plt.show()

