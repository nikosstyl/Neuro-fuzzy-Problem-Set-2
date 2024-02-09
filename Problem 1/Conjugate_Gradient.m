syms x1 x2

% Define the function
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;

% Calculate the gradient
grad = gradient(f, [x1, x2]);

% Calculate the minimizing dirrection
s = -grad;

% Define initial point
x0 = [3; 3];