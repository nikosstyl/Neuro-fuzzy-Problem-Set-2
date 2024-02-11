format long
% Function Definition (Enter your Function here):
syms x1 x2;
%Objective function
% f = 8*x1^2-6*x1*x2+8*x2^2 -x1 +x2;
f = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4;
% Initial Guess:
x_1(1) = 3;
x_2(1) = 3;
Epsilon = 10^(-6); % Convergence Criteria
i = 1; % Iteration Counter
% Contour
x1Label = linspace(-10,100,100);
x2Label = linspace(-10,40,100);
[x,y] = meshgrid(x1Label,x2Label);
% f2 =  8*x.^2-6*x.*y+8*y.^2-x+y;  % objective Function 
f2 = x.^2 + y.^2 + (0.5*x + y).^2 + (0.5*x + y).^4;
figure(1)
contour(x,y,f2,'Fill','On')
hold on
plot(x_1(1),x_2(1),'*-k');
text(50,0,['Initial Point (x1,x2) = (' num2str(x_1(1)),', ',num2str(x_2(1)) ')'],'Color','k')
xlabel('x1')
ylabel('x2')
title('Conjugate gredient Method')
grid on
hold on

% Gradient Computation:
df_dx1 = diff(f, x1);
df_dx2 = diff(f, x2);
J = [subs(df_dx1,[x1,x2], [x_1(1),x_2(1)]) subs(df_dx2, [x1,x2], [x_1(1),x_2(1)])]; % Gradient
S = -(J); % Search Direction

% Initialize lambda storage
lambda_values = [];
fun_value = [];

% Minimization Condition:
while norm(S) > Epsilon 
    I = [x_1(i),x_2(i)]';
    syms lambda; % Step size
    g = subs(f, [x1,x2], [x_1(i)+S(1)*lambda,x_2(i)+lambda*S(2)]);
    %Optimize the step length
    dg_dlambda = (diff(g,lambda)==0);
    lambda = double(solve(dg_dlambda, lambda)); 
    lambda =  lambda(imag(lambda)==0);
    if isempty(lambda)
        lambda = NaN;  % Or some other default value
    end
    lambda_values = [lambda_values; lambda];  % Store lambda value
    % disp(lambda);
    for k = 1:size(lambda,1)
        fun_value(k) = subs(f, [x1,x2],[(I(1)+lambda(k,1)*S(1)),(I(2)+lambda(k,1)*S(2))]);
    end
    [fun, index] = min(fun_value);
    %Optimum step length
    lambda = lambda(index);
    

    figure(1)
    %update the optimum point on plot
    plot(x_1(i),x_2(i),'*-r');
    hold on
    % Update the value
    x_1(i+1) = I(1)+lambda*S(1); % New x value
    x_2(i+1) = I(2)+lambda*S(2); % New y value
    J_old = [subs(df_dx1,[x1,x2], [x_1(i),x_2(i)]) subs(df_dx2, [x1,x2], [x_1(i),x_2(i)])];
    i = i+1;
    J_new = [subs(df_dx1,[x1,x2], [x_1(i),x_2(i)]) subs(df_dx2, [x1,x2], [x_1(i),x_2(i)])]; % Updated Gradient
    S = -(J_new)+((norm(J_new))^2/(norm(J_old))^2)*S; % New Search Direction
end

% Result Table:`
Iter = 1:i;
X_coordinate = x_1';
Y_coordinate = x_2';
% After the while loop, before creating the table
if length(lambda_values) < length(x_1)
    lambda_values = [lambda_values; NaN(length(x_1) - length(lambda_values), 1)];
end
Lambda_values = lambda_values';
Iterations = Iter;
% Check lengths of variables
fprintf('Length of Iter: %d\n', length(Iterations));
fprintf('Length of X_coordinate: %d\n', length(X_coordinate));
fprintf('Length of Y_coordinate: %d\n', length(Y_coordinate));
fprintf('Length of Lambda_values: %d\n', length(Lambda_values));

% Create the table
T = table(Iter, X_coordinate, Y_coordinate, Lambda_values);

% Now create the table
% T = table(Iter, X_coordinate, Y_coordinate, Lambda_values);
T = table(Iterations,X_coordinate,Y_coordinate, Lambda_values);
figure(1)
plot(x_1,x_2,'*-r');
hold on
plot(x_1(i),x_2(i),'*-k');
hold on
plot(x_1(1),x_2(1),'*-k');
hold on
text(0,-2,['optimum point [x1*,x2*] = ','[' num2str(x_1(i)),', ',num2str(x_2(i)),']',],'Color','k')
% Output:
fprintf('Initial Objective Function Value: %d\n\n',subs(f,[x1,x2], [x_1(1),x_2(1)]));
if (norm(S) < Epsilon)
    fprintf('Minimum succesfully obtained...\n\n');
end
fprintf('Number of Iterations for Convergence: %d\n\n', i);
fprintf('Point of Minima: [%d,%d]\n\n', x_1(i), x_2(i));
fprintf('Objective Function Minimum Value: %d\n\n', subs(f,[x1,x2], [x_1(i),x_2(i)]));
disp(T)
