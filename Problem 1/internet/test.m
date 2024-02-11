% Unconstrained optimization Problem
% Fletcher-Reeves Method for conjugate gradient method
% Direct method to solve the unconstrained optimization problem
% Coded based on algorithm is given in SS Rao Optimization book
% Algorithm is given in SS Rao book
% Coded by : Narayan Das Ahirwar
% Contact me: ndahirwar93@gmail.com
% Matlab function used; syms(for drfinr thr variables), fmincon
% Gradient(calculate the gradient of objective function with respect to defined variables),
% Subs(for substitute the new point [x,y] instead of previous point and calculate),
% Solve(for solve the equation withrespect to variables)
% VPA (Variable precision arithmetic for numerically evaluates each element
% of the double matrix and results you will get in sym type)
%%
clc
clear
format long
% Function Definition (Enter your Function here):
syms x1 x2;
%Objective function
f = 8*x1^2-6*x1*x2+8*x2^2 -x1 +x2;
% Initial Guess:
x_1(1) = 100;
x_2(1) = 0;
Epsilon = 10^(-4); % Convergence Criteria
i = 1; % Iteration Counter
% Contour
x1Label = linspace(-10,100,100);
x2Label = linspace(-10,40,100);
[x,y] = meshgrid(x1Label,x2Label);
f2 =  8*x.^2-6*x.*y+8*y.^2-x+y;  % objective Function 
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
% Minimization Condition:
while norm(S) > Epsilon 
    I = [x_1(i),x_2(i)]';
    syms lambda; % Step size
    g = subs(f, [x1,x2], [x_1(i)+S(1)*lambda,x_2(i)+lambda*S(2)]);
    %Optimize the step length
    dg_dlambda = (diff(g,lambda)==0);
    lambda = double(solve(dg_dlambda, lambda)); 
    lambda =  lambda(imag(lambda)==0);
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
Iterations = Iter';
T = table(Iterations,X_coordinate,Y_coordinate);
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
