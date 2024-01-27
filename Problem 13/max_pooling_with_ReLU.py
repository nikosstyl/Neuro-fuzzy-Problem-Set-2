import numpy as np

def relu(x):
    return np.maximum(0, x)

def max_pooling_with_relu(a, b):
    return 0.5 * (a + b + relu(a - b) + relu(b - a))

# Test the max_pooling_with_relu function
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

result = max_pooling_with_relu(a, b)

print("The result of max pooling with ReLU is:\n", result)