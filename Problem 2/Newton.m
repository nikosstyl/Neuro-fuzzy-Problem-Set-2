syms x1 x2

% Define the function
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;

% Define the gradient and the Hessian
grad_f = gradient(f, [x1, x2]);
hess_f = hessian(f, [x1, x2]);

disp('grad_f = ');
disp(grad_f);
disp('hess_f = ');
disp(hess_f);

% Define inverse Hessian
inv_hess_f = inv(hess_f);

disp('inv_hess_f = ');
disp(inv_hess_f);

% Define the minimizing direction
s = -inv_hess_f*grad_f;

disp('s = ');
disp(s);

% Define the initial point
x0 = [3; 3];

% Define the tolerance
tol = 1e-6;

% Initialize lambda
lambda = 1;

% Initialize counter
counter = 0;

% Initialize a matrix to store the points
points = x0;

% Perform the Newton's method update
while norm(subs(grad_f, [x1, x2], x0')) > tol
    % Calculate the inverse Hessian and the direction s at the current point
    inv_hess_f = inv(double(subs(hess_f, [x1, x2], x0')));
    s = -inv_hess_f * double(subs(grad_f, [x1, x2], x0'));

    %Update the point
    x0 = x0 + lambda*s;

    % Store the point
    points = [points, x0];
   
    % Increment the counter
    counter = counter + 1;
end

disp('The minimum of the function occurs at');
disp(x0);
disp('The number of iterations is');
disp(counter);

% Convert the symbolic function to a function handle
f_handle = matlabFunction(f);

% Generate a grid of points
[x1_grid, x2_grid] = meshgrid(-5:0.1:5, -5:0.1:5);

% Evaluate the function at the grid points
z = f_handle(x1_grid, x2_grid);

% Create a new figure for the contour plot
figure;
contour(x1_grid, x2_grid, z, 50);
hold on;

% Plot the path of the Newton's method on the contour plot
plot(points(1,:), points(2,:), '-ro');
hold off;
title('Contour plot of the function and path of the Newton''s method');
xlabel('x1');
ylabel('x2');

% Create a new figure for the 3D surface plot
figure;
surf(x1_grid, x2_grid, z);
hold on;
% Plot the path of the Newton's method on the 3D surface plot
plot3(points(1,:), points(2,:), f_handle(points(1,:), points(2,:)), '-ro');
hold off;

title('3D plot of the function and path of the Newton''s method');
xlabel('x1');
ylabel('x2');
zlabel('f(x1, x2)');
