# Layer 1
input_channels_1 = 3
kernel_size_1 = 3
output_channels_1 = 4

weights_1 = input_channels_1 * kernel_size_1 * output_channels_1
biases_1 = output_channels_1

# Layer 2
input_channels_2 = output_channels_1  # output of previous layer is input to next layer
kernel_size_2 = 5
output_channels_2 = 10

weights_2 = input_channels_2 * kernel_size_2 * output_channels_2
biases_2 = output_channels_2

print(f"Layer 1: {weights_1} weights and {biases_1} biases")
print(f"Layer 2: {weights_2} weights and {biases_2} biases")