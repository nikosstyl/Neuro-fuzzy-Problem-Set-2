import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

# Define the pooling operation
def max_pooling(input_matrix, pool_size, stride):
    output_matrix = block_reduce(input_matrix, block_size=pool_size, func=np.max)
    return output_matrix

#### QUESTION A
# Define the image matrix
I = np.array([
    [20, 35, 35, 35, 35, 20],
    [29, 46, 44, 42, 42, 27],
    [16, 25, 21, 19, 19, 12],
    [66, 120, 116, 154, 114, 62],
    [74, 216, 174, 252, 172, 112],
    [70, 210, 170, 250, 170, 110]
])

# Define the stride and kernel
stride = (1,1)

kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Perform 2D convolution
result = convolve2d(I, kernel, mode='valid')
print("The result of the convolution is:\n", result)

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the feature map
plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Feature Map')
plt.axis('off')

# Display the plots
plt.show()

### QUESTION B
# Apply max pooling
pool_size = (2, 2)
stride = (2, 2)  # Note: block_reduce doesn't support strides. It will always be the same as pool_size

pooled_result = max_pooling(result, pool_size, stride)
print("\nThe max pooling result is:\n", pooled_result)