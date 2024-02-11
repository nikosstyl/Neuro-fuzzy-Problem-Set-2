syms x1 x2

% Define the function
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;

% Calculate the gradient
grad_f = gradient(f, [x1, x2]);

% Initial points
x = [3;3];

% Calculate the gradient at the initial points
grad_at_x = subs(grad_f, [x1, x2], x');

% syms w1 w2

% Define the function and its derivative
% f = w1^2 + w2^2 + (0.5*w1 + w2)^2 + (0.5*w1 + w2)^4;

% Calculate the gradient
% grad = matlabFunction(gradient(f, [w1, w2]));

% Initial guess
% w0 = [3; 3];



disp(grad_at_x)
% Call the function
% w = cg_fletcher_reeves_obj(f, grad, w0);

% Print the result
% disp('The minimum is at:'), disp(w)
% 
% function x = cg_fletcher_reeves_obj(f, grad, w0, tol, max_iter)
    % if nargin < 4
        % tol = 1e-10;
    % end
    % if nargin < 5
        % max_iter = 1000;
    % end
% 
    % 1. Start with an Initial Guess:
    % w = w0;
% 
    % 2. Initial Gradient and Direction:
    % r = -grad(w(1), w(2));
    % p = r;
    % rsold = r' * r;
% 
    % iter = 0;  % Initialize counter
% 
    % Loop until the norm of the gradient is less than the tolerance
    % while norm(grad(w(1), w(2))) > tol
        % Print the counter
        % disp(['Iteration: ', num2str(iter+1)])
% 
        % 3. Line Search for Step Size:
        % alpha = rsold / (p' * grad(p(1), p(2)));
% 
        % 4. Update Position:
        % w = w + alpha * p;
% 
        % 5. Iterate with Conjugate Directions:
        % r = -grad(w(1), w(2));
        % rsnew = r' * r;
% 
        % p = r + (rsnew / rsold) * p;
        % rsold = rsnew;
% 
        % iter = iter + 1;  % Increment counter
        % if iter >= max_iter
            % break;  % Break the loop if maximum number of iterations is reached
        % end
    % end
% 
    % 7. Repeat or Terminate:
    % x = w;
% end