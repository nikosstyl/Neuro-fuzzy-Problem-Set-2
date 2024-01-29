import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Define the image matrix
I = np.array([
    [20, 35, 35, 35, 35, 20],
    [29, 46, 44, 42, 42, 27],
    [16, 25, 21, 19, 19, 12],
    [66, 120, 116, 154, 114, 62],
    [74, 216, 174, 252, 172, 112],
    [70, 210, 170, 250, 170, 110]
])

### KERNEL F1
# F = np.array([
    #  [-10, -10, -10],
    #  [5, 5, 5],
    #  [-10, -10, -10]
# ])
 
### KERNEL F2
F = np.array([
    [2, 2, 2],
    [2, -12, 2],
    [2, 2, 2]
])

### KERNEL F3
#F = np.array([
#    [-20, -10, 0, 5, 10],
#    [-10, 0, 5, 10, 5],
#    [0, 5, 10, 5, 0],
#    [5, 10, 5, 0, -10],
#    [10, 5, 0, -10, -20]
#])

# Apply the kernel to the image using 2D convolution
feature_map_F = convolve2d(I, F, mode='valid')
print(feature_map_F)

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
# plt.axis('off')

# Plot the feature map
plt.subplot(1, 2, 2)
plt.imshow(feature_map_F, cmap='gray')
plt.title('After Applying Kernel')
# plt.axis('off')

# Display the plots
plt.show()
