% Define the function
f = @(x) x(1)^2 + x(2)^2 + (0.5*x(1) + x(2))^2 + (0.5*x(1) + x(2))^4;

% Define the gradient of the function
grad_f = @(x) [2*x(1) + x(2) + 2*(0.5*x(1) + x(2))^3; 2*x(2) + x(1) + 2*(0.5*x(1) + x(2))^3];

% Initial guess
w = [3; 3];

% Conjugate Gradient Fletcher-Reeves method
max_iter = 100; % maximum number of iterations
tol = 1e-6; % tolerance for the norm of the gradient

for k = 0:max_iter
    g = grad_f(w);
    if norm(g) < tol
        break;
    end
    if k == 1
        d = -g;
    else
        beta = (g' * g) / (g_prev' * g_prev);x
        d = -g + beta * d;
    end
    g_prev = g;
    
    % Line search (Backtracking)
    alpha = 1;
    while f(w + alpha * d) > f(w) + 0.5 * alpha * g' * d
        alpha = alpha / 2;
    end
    
    % Update
    w = w + alpha * d;
     
 
    
end

disp('The minimum of the function is at:')
disp(w)