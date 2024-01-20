function [w1, b1, w2, b2] = calc_propagation(p, t, w1, b1, w2, b2, alpha, iterations)
    m_sigmoid = @(x) 1 ./ (1+exp(x));
    % Define the Swish and its derivative
    swish = @(x) x ./ (1 + exp(-x));
    d_swish = @(x) swish(x) + m_sigmoid(x) .* (1 - swish(x));

  
    % Define the LReLU and its derivative
    lrelu = @(x) max(0.001*x, x);
    d_lrelu = @(x) (x > 0) + 0.001 * (x <= 0);

    % Perform the specified number of iterations of backpropagation
    for i = 1:iterations
        % Forward pass
        n1 = w1 * p + b1;
        a1 = swish(n1);
        n2 = w2 * a1 + b2;
        a2 = lrelu(n2);

        % Compute the error
        e = t - a2;

        % Backward pass for the second layer
        s2 = -2 * d_lrelu(n2) .* e;
        w2_grad = s2 * a1';
        b2_grad = s2;

        % Backward pass for the first layer
        s1 = (w2' * s2) .* d_swish(n1);
        w1_grad = s1 * p';
        b1_grad = s1;

        % Update weights and biases
        w2 = w2 - alpha * w2_grad;
        b2 = b2 - alpha * b2_grad;
        w1 = w1 - alpha * w1_grad;
        b1 = b1 - alpha * b1_grad;
    end
end
