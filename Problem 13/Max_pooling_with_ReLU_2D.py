import numpy as np

def relu(x):
    return np.maximum(0, x)

def max_pooling_with_relu(array):
    return np.max(relu(array))

def max_pooling_with_relu_2d(input_array, pool_size=(2, 2)):
    # Get the shape of the input array
    input_height, input_width = input_array.shape

    # Calculate the shape of the output array
    output_height = input_height // pool_size[0]
    output_width = input_width // pool_size[1]

    # Create the output array
    output_array = np.zeros((output_height, output_width))

    # Iterate over the input array in the shape of the pooling window
    for i in range(0, input_height, pool_size[0]):
        for j in range(0, input_width, pool_size[1]):
            # Apply max pooling to the current window
            output_array[i // pool_size[0], j // pool_size[1]] = max_pooling_with_relu(
                input_array[i:i+pool_size[0], j:j+pool_size[1]]
            )

    return output_array

# Create a 4x4 2D array
input_array = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Apply max pooling with ReLU
output_array = max_pooling_with_relu_2d(input_array)

print("The result of 2D max pooling with ReLU is:\n", output_array)