import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt

def calculate_times(image, kernel, delta):
    # Method 1: Scan horizontally across the source, 
    #reading a k-wide strip and computing the 1-wide output strip one value at a time.
    start_time = time.time()
    for i in range(image.shape[1] - kernel.shape[1] + 1):
        strip = image[:, i:i+kernel.shape[1]]
        output = signal.convolve2d(strip, kernel, mode='valid')
    time1 = time.time() - start_time

    # Method 2: Read a k + Δ wide strip and compute a Δ-wide output strip
    start_time = time.time()
    for i in range(0, image.shape[1] - kernel.shape[1] + 1, delta):
        strip = image[:, i:i+kernel.shape[1]+delta]
        output = signal.convolve2d(strip, kernel, mode='valid')
    time2 = time.time() - start_time

    return time1, time2

def write_times_to_file(kernel, time1, time2,iteration,kernels):
    with open('samples.txt', 'a') as f:
        if kernel is kernels[0]:  # If this is the first kernel in the list
            f.write("Iteration: {}\n".format(iteration))
        f.write("Kernel: {}\n".format(kernel))
        f.write("Time in method 1: {}\n".format(time1))
        f.write("Time in method 2: {}\n".format(time2))
        f.write("Difference in time: {}\n".format(abs(time1 - time2)))
        f.write("\n")  # Add a newline for readability

def main():
    # Create a sample image of 228x228
    image = np.random.rand(228, 228)

    # Create a series of 3x3, 7x7 and 11x11 filters(kernels) with random values
    kernels = [np.random.rand(k, k) for k in [3, 7, 11]]

    # You can adjust this value
    delta = 2 

    for kernel in kernels:
        times1 = []
        times2 = [] 

        for i in range(5):
            time1, time2 = calculate_times(image, kernel, delta)
            times1.append(time1)
            times2.append(time2)
            print("Used kernel: ", kernel)
            print("Method 1 execution time for this kernel: ", time1)
            print("Method 2 execution time for this kernel: ", time2)
            print("Difference in time: ", abs(time1 - time2))
            write_times_to_file(kernel, time1, time2,i+1,kernels)  # Pass the list of kernels to the function

        # Plot the times for this kernel
        plt.figure()
        plt.plot(range(1, len(times1) + 1), times1, label='Method 1')  # Include the iteration numbers starting from 1
        plt.plot(range(1, len(times2) + 1), times2, label='Method 2')  # Include the iteration numbers starting from 1
        plt.title('Times for kernel size {}'.format(kernel.shape[0]))
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.legend()


        # Set the x-axis ticks to the values 1, 2, 3, 4, 5
        plt.xticks(range(1, 6))

        # Add a grid
        plt.grid(True)

    
    plt.show()

if __name__ == "__main__":
    main()
 
