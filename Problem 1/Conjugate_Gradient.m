function Conjugate_Gradient
% Define the function F(w)
F = @(w) w(1)^2 + w(2)^2 + (0.5*w(1) + w(2))^2 + (0.5*w(1) + w(2))^4;

% Define the gradient of the function F(w)
gradF = @(w) [2*w(1) + 2*0.5*(0.5*w(1) + w(2)) + 4*(0.5*w(1) + w(2))^3 * 0.5; 
              2*w(2) + 2*(0.5*w(1) + w(2)) + 4*(0.5*w(1) + w(2))^3];

% Choose a starting point for w (can be random or zero-initialized)
w = [3;3];

% Fletcher-Reeves Conjugate Gradient Algorithm with Backtracking Line Search
g = gradF(w);
s = -g;
k = 0;
maxIter = 100;
tol = 1e-4;

while norm(g) > tol && k < maxIter
    k = k + 1;
    
    % Backtracking line search parameters
    lambda = 1;
    rho = 0.5;
    c = 1e-4;
    
    % Backtracking line search
    while F(w + lambda * s) > F(w) + c * lambda * (g' * s)
        lambda = rho * lambda;
    end
    
    % Update w
    w = w + lambda * s;
    
    % Calculate the new gradient
    g_new = gradF(w);
    
    % Fletcher-Reeves update for omega
    omega = (norm(g_new)^2) / (norm(g)^2);
    
    % Update search direction
    s = -g_new + omega * s;
    
    % Update the gradient
    g = g_new;
    
    % Display current iteration stats
    % fprintf('Iteration %d: w = [%f, %f], F(w) = %f\n', k, w(1), w(2), F(w));
    % fprintf('Lambda: %f, New search direction: %f, Sk norm: %f, Omega: %f\n', lambda, s, norm(s), omega);
    % fprintf('\n');
    fprintf('Iteration: %d, W = [%f, %f], F(w) = %f\n', k, w(1), w(2), F(w));
    fprintf('Lambda: %f, New S_k: %s, ||S_k||: %f, Omega: %f\n\n', lambda, mat2str(s), norm(s), omega);
end

% Display final result
fprintf('The minimum of the function is at w = [%f, %f] with F(w) = %f\n', w(1), w(2), F(w));
end 