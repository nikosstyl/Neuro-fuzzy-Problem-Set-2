import numpy as np
import matplotlib.pyplot as plt
import torch

def adadelta_optimizer(w1_init, w2_init):
    w = torch.tensor([w1_init, w2_init], requires_grad=True)
    rho = 0.9
    epsilon = 1e-6
    St = torch.zeros_like(w)
    Edelta = torch.zeros_like(w)
    trajectory = [w.detach().numpy().copy()]

    for _ in range(300):
        loss = 0.1*(w[0] - w[1])**2 + 2*(w[0] + w[1])**2  # Modified objective function
        loss.backward()
        grad = w.grad
        with torch.no_grad():
            St = rho * St + (1 - rho) * grad**2
            update = torch.sqrt(Edelta + epsilon) / torch.sqrt(St + epsilon) * grad
            Edelta = rho * Edelta + (1 - rho) * update**2
            w -= update
            w.grad.zero_()
        trajectory.append(w.detach().numpy().copy())
    return trajectory
# Create a grid of points for the contour plot
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-2.5, 2.5, 100)
X1, X2 = np.meshgrid(x, y)
F_values = 0.1*X1**2 + 2*X2**2

# Run the Adadelta optimizer
w1_init = -2.5
w2_init = 2.5
trajectory = adadelta_optimizer(w1_init, w2_init)

# Plot the contour plot and optimizer trajectory
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, F_values, levels=50, cmap='viridis')
plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'b.-')  # Blue dots for the trajectory
plt.title('Contour Plot and Trajectory of AdaDelta on F(w)')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.show()