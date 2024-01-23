import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Define the function
def F(w):
    return 0.1 * w[0]**2 + 2 * w[1]**2

# Initialize the weights
w = torch.tensor([2.0, 2.0], requires_grad=True)

# Define the optimizer
optimizer = torch.optim.Adadelta([w], lr=0.4)

# Store the weights and function values for plotting
ws = []
values = []

# Perform the optimization
for i in range(100):
    optimizer.zero_grad()
    output = F(w)
    output.backward()
    optimizer.step()

    ws.append(w.detach().numpy().copy())
    values.append(output.item())
    #print(f'Iteration {i+1}: w = {w.detach().numpy()}')

# Convert to numpy arrays for easier plotting
ws = np.array(ws)
values = np.array(values)

# Create a grid of values for the contour plot
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = 0.1 * X**2 + 2 * Y**2

# Create the contour plot
plt.figure(figsize=(5, 5))
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)

plt.plot(ws[:, 0], ws[:, 1], 'w+', linewidth=2)
plt.title('Contour plot with trajectory of Adadelta optimizer')
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()