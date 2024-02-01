import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Define the new function F(w)
def F(w):
    return 0.1*(w[0] + w[1])**2 + 2*(w[0] - w[1])**2

# Reinitialize parameters for AdaDelta (without a fixed learning rate alpha)
w = torch.tensor([2.0, 2.0], requires_grad=True)  # Initial guess
St = torch.zeros(2)    # Running average of squared gradients
Edelta = torch.zeros(2) # Running average of squared parameter updates
rho = 0.9
epsilon = 1e-6
lr = 0.4

# Store the trajectory for plotting
trajectory_adadelta = [w.detach().numpy().copy()]

# AdaDelta optimization (without fixed learning rate alpha)
for _ in range(300):
    loss = F(w)
   
    loss.backward()  # Compute the gradient
    grad = w.grad  # Get the gradient
    with torch.no_grad():  # Update weights without tracking gradients
        St = rho * St + (1 - rho) * grad**2
        update = torch.sqrt(Edelta + epsilon) / torch.sqrt(St + epsilon) * grad
        # w = w - lr*update
        Edelta = rho * Edelta + (1 - rho) * update**2
        w_new = (w - lr * update).clone().detach().requires_grad_(True)
        # w_new = w - lr * update
        w.grad.zero_()  # Reset the gradient to zero for the next iteration
    
    w = w_new
    # w.add_(-lr * update)  # Update weights in-place
    # w = w - lr*update
    trajectory_adadelta.append(w.detach().numpy().copy())
    num_iterations = len(trajectory_adadelta) - 1

    print(num_iterations)


# Create a grid of points for the contour plot
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-2.5, 2.5, 100)
X1, X2 = np.meshgrid(x, y)
F_values = 0.1*X1**2 + 2*X2**2

# Create contour plot for AdaDelta trajectory
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, F_values, levels=50, cmap='viridis')
plt.plot(np.array(trajectory_adadelta)[:, 0], np.array(trajectory_adadelta)[:, 1], 'b.-')  # Blue dots for the trajectory
plt.title('Contour Plot and Trajectory of AdaDelta on F(w)')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.show()
