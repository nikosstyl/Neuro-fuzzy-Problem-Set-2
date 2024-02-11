function x = cg_fletcher_reeves_obj(f, df, x0, tol, max_iter)
    if nargin < 4
        tol = 1e-10;
    end
    if nargin < 5
        max_iter = 1000;
    end

    x = x0;
    r = -df(x);
    p = r;
    rsold = r' * r;

    for i = 1:max_iter
        alpha = rsold / (p' * df(p));
        x = x + alpha * p;
        r = -df(x);
        rsnew = r' * r;
        if sqrt(rsnew) < tol
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

% Define the function and its derivative
f = @(w) w(1)^2 + w(2)^2 + (0.5*w(1) + w(2))^2 + (0.5*w(1) + w(2))^4;
df = @(w) [2*w(1) + w(2) + 2*(0.5*w(1) + w(2))^3; 2*w(2) + w(1) + 4*(0.5*w(1) + w(2))^3];

% Initial guess
x0 = [3; 3];

% Call the function
x = cg_fletcher_reeves_obj(f, df, x0);

% Print the result
disp('The minimum is at:'), disp(x)