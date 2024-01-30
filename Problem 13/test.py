import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 2x2 convolutional kernel
# conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, dilation=2)
# 
# Initialize the weights of the kernel manually
# conv.weight.data = torch.tensor([[[[-1., -1.], [-1., -1.]]]])
# 
# Example usage
# input_tensor = torch.randn(1, 1, 6, 6)  # Batch size of 1, 1 channel, 6x6 input
# print("The input tensor:\n",input_tensor)
# 
# output_tensor = conv(input_tensor)
# print("The convolved tensor:\n",output_tensor)
# 
# Apply ReLU activation function
# output_tensor = F.relu(output_tensor)
# print("Output after ReLU:\n",output_tensor)
# 
# Negate the output tensor to get the max pooling result
# output_tensor = -output_tensor
# print("Output after max pooling:\n",output_tensor)


# Define a max pooling layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Example usage
input_tensor = torch.randn(1, 1, 6, 6)  # Batch size of 1, 1 channel, 6x6 input
print("The input tensor:\n",input_tensor)

# Apply max pooling to the input tensor
output_tensor = max_pool(input_tensor)
print("Output after max pooling:\n",output_tensor)

