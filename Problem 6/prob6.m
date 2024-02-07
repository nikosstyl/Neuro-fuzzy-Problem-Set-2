function prob6

    input = [0 1]';
    input(3) = input(2);
    input(2) = input(1)*input(3);

    weights = [1 0.5 -1];
    bias = 1;

    
    [new_weights, new_bias] = backprop(input, weights, bias, 0.75, 1);

    fprintf('Initial weights: %s\n', mat2str(weights));
    fprintf('Initial bias: %s\n', mat2str(bias));

    fprintf('\nNew weights: %s\n', mat2str(new_weights));
    fprintf('New bias: %s\n', mat2str(new_bias));

    output = fwd_pass(input, new_weights, new_bias);

    fprintf('\nOutput: %s\n', mat2str(output));
end

function result = fwd_pass (input, weights, bias)
    n1 = weights * input + bias;
    result = tanh(n1);
end

function [new_weights, new_bias, loss] = backprop(input, weights, bias, target, lr)
    % calculate the output of the network
    output = fwd_pass(input, weights, bias);
    
    loss = 1/2 * (target - output)^2;
    loss_grad = output - target;
    act_grad = 1 - tanh(output)^2;

    loss_weight_grad = loss_grad * act_grad .* input;
    loss_bias_grad = loss_grad * act_grad;

    % Update weights and bias
    new_weights = weights - lr .* loss_weight_grad';
    new_bias = bias - lr .* loss_bias_grad;
end