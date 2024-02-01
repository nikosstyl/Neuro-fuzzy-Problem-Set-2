function backpropagation (S, learning_rate)

    % Define network structure

    % Initialize weights and biases
    W1 = rand(S, 1) - 0.5;
    b1 = rand(S, 1) - 0.5;
    % W2 = rand(1, S) - 0.5;
    W2 = rand(1, S);
    % b2 = rand(1, 1) - 0.5;
    b2 = rand(1, 1);

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
    fprintf('---------------\n');

    MAX_EPOCHS = 1000;

    % Training data
    p = linspace(-2, 2, 100)';
    t = 1 + sin(3 * pi * p / 8);

    % Training
    for epoch = 1:MAX_EPOCHS
        for i = 1:length(p)
            % Forward pass
            a1 = logsig(W1 * p(i) + b1);
            a2 = max(0, W2 * a1 + b2);

            % Backpropagation
            e = t(i) - a2;
            if a2 > 0
                df2 = 1;
            else
                df2 = 0;
            end
            
            s2 = -2*e*df2;
            s1 = (W2' * s2) .* a1 .* (1 - a1);

            % Update weights and biases
            W2 = W2 - learning_rate * s2 * a1';
            b2 = b2 - learning_rate * s2;
            W1 = W1 - learning_rate * s1 * p(i)';
            b1 = b1 - learning_rate * s1;
        end
    end

    % Test the network
    p_test = linspace(-2, 2, 100)';
    a1_test = logsig(W1 * p_test' + b1);
    a2_test = max(0, W2 * a1_test + b2);

    % Plot
    figure;
    plot(p_test, a2_test, 'r-', p_test, 1 + sin(3 * pi * p_test / 8), 'b--');
    legend('Network Output', 'Target Function', 'Location', 'best');
    title(sprintf('1-%d-1 Network Approximation w/ a=%g', S, learning_rate));
    xlabel('p');
    ylabel('g(p)'); 
end
