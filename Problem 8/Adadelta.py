import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the function F(w)
def F(w):
    return 0.1*w[0]**2 + 2*w[1]**2

# Reinitialize parameters for AdaDelta (without a fixed learning rate alpha)
w = torch.tensor([5.0, 5.0], requires_grad=True)  # Initial guess
Eg = torch.zeros(2)    # Running average of squared gradients
Edelta = torch.zeros(2) # Running average of squared parameter updates
rho = 0.9
epsilon = 1e-6

# Store the trajectory for plotting
trajectory_adadelta = [w.detach().numpy().copy()]

# AdaDelta optimization (without fixed learning rate alpha)
for _ in range(100):
    loss = F(w)
    loss.backward()  # Compute the gradient
    grad = w.grad  # Get the gradient
    with torch.no_grad():  # Update weights without tracking gradients
        Eg = rho * Eg + (1 - rho) * grad**2
        update = -torch.sqrt(Edelta + epsilon) / torch.sqrt(Eg + epsilon) * grad
        w += update
        Edelta = rho * Edelta + (1 - rho) * update**2
        w.grad.zero_()  # Reset the gradient to zero for the next iteration

    trajectory_adadelta.append(w.detach().numpy().copy())

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