import numpy as np
import requests
import pandas as pd
import os
import random
import torch
import torch.nn.functional as F

def relu(x):
    return np.maximum(0, x)

def max_pooling_conv(input_tensor, pool_size):
    batch_size, channels, height, width = input_tensor.size()
    kernel_size = (pool_size, pool_size)
    stride = (pool_size, pool_size)
    output_shape = F.conv2d(input_tensor, torch.ones((channels, 1, pool_size, pool_size)), stride=stride, padding=0, groups=channels).shape
    output_tensor = torch.tensor([-torch.inf]*output_shape.numel()).reshape(output_shape)
    
    # Define the average pooling kernel
    for i in range(pool_size*pool_size):
        kernel = torch.zeros(pool_size*pool_size)
        kernel[i] = 1
        kernel = kernel.reshape(channels, 1, pool_size, pool_size)
    
    # Apply the convolution operation with average pooling kernel
        temp = F.conv2d(input_tensor, kernel, stride=stride, padding=0, groups=channels)
        output_tensor = relu(output_tensor - temp) + temp
    return output_tensor

# Example usage
input_tensor = torch.randn(1, 1, 6, 6)  # Batch size of 1, 3 channels, 6x6 input
pool_size = 2
output_tensor = max_pooling_conv(input_tensor, pool_size)
print(input_tensor)
print(output_tensor)