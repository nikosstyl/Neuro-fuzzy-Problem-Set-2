function backpropagation 

    s = [2 8 12];
    lr = [0.1 0.01 0.001];

    for i=s
        for j=lr
            calc_backpropagation(i, j);
        end
    end

end


function calc_backpropagation (S, learning_rate)

    fprintf('1-%d-1 Network, S = %d, lr = %g\n', S, S, learning_rate);

    % Initialize weights and biases
    W1 = rand(S, 1) - 0.5;
    b1 = rand(S, 1) - 0.5;
    W2 = rand(1, S) - 0.5;
    b2 = rand(1, 1) - 0.5;

    % W1_init = W1;
    % b1_init = b1;
    % W2_init = W2;
    % b2_init = b2;

    fprintf('* W1\n')
    for i = 1:length(W1)
        fprintf('\t* %g\n', W1(i));
    end
    fprintf('* b1\n');
    for i = 1:length(b1)
        fprintf('\t* %g\n', b1(i));
    end
    fprintf('* W2\n');
    for i = 1:length(W2)
        fprintf('\t* %g\n', W2(i));
    end
    fprintf('* b2\n\t* %g\n', b2);
    fprintf('---------------\n\n');

    MAX_EPOCHS = 1000;

    % Training data
    p = linspace(-2, 2, 200)';
    t = 1 + sin(3 * pi * p / 8);

    e = zeros(1, MAX_EPOCHS);

    % Training
    for epoch = 1:MAX_EPOCHS
        sum_squared_error = 0;
        for i = 1:length(p)
            % Forward pass
            a1 = logsig(W1 * p(i) + b1);
            a2 = relu(W2 * a1 + b2);

            % Backpropagation
            e(epoch) = t(i) - a2;
            sum_squared_error = sum_squared_error + e(epoch)^2; 

            % The following is the derivative of the ReLU function
            if a2 > 0
                df2 = 1;
            else
                df2 = 0;
            end
            
            s2 = -2*e(epoch)*df2;
            s1 = (W2' * s2) .* a1 .* (1 - a1);

            % Update weights and biases
            W2 = W2 - learning_rate * s2 * a1';
            b2 = b2 - learning_rate * s2;
            W1 = W1 - learning_rate * s1 * p(i)';
            b1 = b1 - learning_rate * s1;
        end
        e(epoch) = sum_squared_error / length(p);
    end

    % Plot error
    figure;
    plot(1:MAX_EPOCHS, e, 'LineWidth', 2);
    title(sprintf('Error vs. Epochs w/ a=%g', learning_rate));
    xlabel('Epochs');
    ylabel('Error');
    grid('on');
    
    % Test the network
    p_test = linspace(-2, 2, 100)';
    a1_test = logsig(W1 * p_test' + b1);
    a2_test = max(0, W2 * a1_test + b2);

    % Plot
    figure;
    plot(p_test, a2_test, '-', p_test, 1 + sin(3 * pi * p_test / 8), '--', 'LineWidth', 2);
    legend('Network Output', 'Target Function', 'Location', 'best');
    title(sprintf('1-%d-1 Network Approximation w/ a=%g', S, learning_rate));
    xlabel('p');
    ylabel('g(p)'); 
end

% My implementation of ReLU
function result = relu (x)
    result = max(0, x);
end
