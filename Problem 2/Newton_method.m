syms x1 x2

% Define Function
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;

% Define the derivative of f with respect to x1 and x2
dfx1 = diff(f, x1);
dfx2 = diff(f, x2);

% Convert the symbolic expressions to function handles
dfx1_func = matlabFunction(dfx1);
dfx2_func = matlabFunction(dfx2);


% Insert the derivatives into a common array
%df = [dfx1; dfx2];

% Initialize the starting point
x0 = [0; 0]

% Set the fixed step size
lambda = 1;

% Set the maximum number of iterations
maxIterations = 100;

% Perform the Newton method
x = x0;
for k = 1:maxIterations
    % Update the value of x using the Newton method
    x = x - lambda * [dfx1_func(x(1), x(2)); dfx2_func(x(1), x(2))];

   
    % Check for convergence
    if norm([dfx1_func(x(1), x(2)); dfx2_func(x(1), x(2))]) < 1e-6
        break;
    end
   
end

% Convert the symbolic function f to a function handle
f_func = matlabFunction(f);

% Display the minimum value of f and the corresponding x
fprintf('Minimum value of f: %.4f\n', f_func(x(1), x(2)));
fprintf('x at minimum: %.4f, %.4f\n', x(1), x(2));


%%%%%%%%%%%%%%%%% PLOT THE FUNCTION AND VISUALIZE THE MIN %%%%%%%%%%%

% Convert the symbolic function f to a function handle
f_func = matlabFunction(f);

% Generate a grid of points in the x1-x2 plane
[x1_grid, x2_grid] = meshgrid(-10:0.1:10, -10:0.1:10);

% Compute the function values at the grid points
f_grid = f_func(x1_grid, x2_grid);

% Plot the function
surf(x1_grid, x2_grid, f_grid);
hold on;

% Plot the minimum point found using the Newton's method
plot3(x(1), x(2), f_func(x(1), x(2)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% Add labels and title
xlabel('x1');
ylabel('x2');
zlabel('f(x1, x2)');
title('Function and minimum point');

% Enable the grid
grid on;

% Release the hold on the current figure
hold off;