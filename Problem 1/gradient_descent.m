syms x1 x2

% Define function
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;

% Define gradient
grad_f = gradient(f, [x1, x2]);
fprintf("The gradient of the function is: %s\n", grad_f);

% Initial guess
x = [3;3];

%  Step size - unit step movement
alpha = 0.01;

%  Define tolerance
tol = 1e-6;

%  Perform 10 Iterations
for i = 1:10
    %  Compute gradient
    grad = double(subs(grad_f, [x1, x2], [x(1), x(2)]));
    fprintf("Gradient on iteration %d is: %s\n", round(i), grad);

    %  Compute step
    step = alpha*grad;

    %  If step size is smaller than tolerance, break
    if norm(step) < tol
        fprintf('Step size is smaller than the tolerance. Breaking on iter:%d...\n', i);
        break;
    end

    %  Update x
    x = x - step;
    % fprintf("Updated x on iteration %d is: %s\n", i, x)

    % Print the current state
    fprintf('After iteration %d, x = [%f, %f]\n', i, x(1), x(2));
    
end

% Print the final state
fprintf('After 10 iterations, x = [%f, %f]\n', x(1), x(2));