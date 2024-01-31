import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the function F(w)
def F(w):
    return 0.1*w[0]**2 + 2*w[1]**2

# Initialize the parameters w1 and w2
w = torch.tensor([2.0, 2.0], requires_grad=True)

# Implement the Adadelta optimizer
optimizer = torch.optim.Adadelta([w], lr=0.4)

# Store the values of w for plotting
ws = []
ws.append(w.detach().numpy().copy())

# Update the parameters using the optimizer in a loop until convergence
for i in range(100):
    optimizer.zero_grad()
    loss = F(w)
    loss.backward()
    optimizer.step()
    ws.append(w.detach().numpy().copy())

# Convert the list of w values to a numpy array for plotting
ws = np.array(ws)

# Create a grid of points for plotting the contour plot
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z = 0.1*X**2 + 2*Y**2

# Plot the contour plot of F(w)
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap=plt.cm.jet)
plt.plot(ws[:, 0], ws[:, 1], 'r-', linewidth=2, label='Adadelta optimizer path')
plt.legend()
plt.show()