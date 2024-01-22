% Define the ReLU function
relu = @(x) max(0, x);

% Define a linear function
linear = @(x, a, b) a * x + b;

% Define a function that represents a simple MLP with one hidden layer
mlp = @(x, weights) linear(relu(linear(x, weights.hidden(1), weights.hidden(2))), weights.output(1), weights.output(2));

% Define the weights of the MLP
weights.hidden = [2, -1];  % Weights for the hidden layer
weights.output = [1, 0];  % Weights for the output layer

% Generate some input values
x = linspace(-10, 10, 400);

% Compute the output of the MLP for the input values
y = arrayfun(@(x) mlp(x, weights), x);

% Plot the output of the MLP
plot(x, y);