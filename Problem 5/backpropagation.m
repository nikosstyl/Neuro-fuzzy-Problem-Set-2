function backpropagation (export_figs)

    if nargin < 1
        export_figs = false;
    end

    s =  12;
    lr = 0.1;
    
    for i=s
        for j=lr
            failed_num = 0;
            failed = calc_backpropagation(i, j, export_figs);
            failed_num = failed_num + 1;
            while failed
                failed = calc_backpropagation(i, j, export_figs);
                failed_num = failed_num + 1;
                if failed_num > 10
                    error('Failed 10 times in a row')
                end
            end
        end
    end

end


function failed = calc_backpropagation (S, learning_rate, export_figs)

    fprintf('1-%d-1 Network, S = %d, lr = %g\n', S, S, learning_rate);

    % Initialize weights and biases
    W1 = rand(S, 1) - 0.5;
    b1 = rand(S, 1) - 0.5;
    W2 = rand(1, S) - 0.5;
    b2 = rand(1, 1) - 0.5;

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

    MAX_EPOCHS = 20000;

    % Training data
    p = linspace(-2, 2, 200)';
    g = 1 + sin(3 * pi * p / 8);

    err_max = zeros(1, MAX_EPOCHS);

    convergence_threshold = 2e-3;

    % Training
    for epoch = 1:MAX_EPOCHS
        for i = 1:length(p)
            % Forward pass
            n1 = W1 * p(i) + b1;
            a1 = logsig(n1);

            n2 = W2 * a1 + b2;
            a2 = relu(n2);

            % Backpropagation
            e = g(i) - a2;

            df2 = relu_derivative(n2);
            
            s2 = -2*e*df2;
            s1 = logsig_derivative(n1) .* W2' .* s2;

            if s2 == 0 && epoch == 1
                % error('s2 is 0')
                fprintf('s2 is 0\n\n')
                failed = true;
                return;
            end            

            % Update weights and biases
            W2 = W2 - learning_rate * s2 * a1';
            b2 = b2 - learning_rate * s2;
            W1 = W1 - learning_rate * s1 * p(i)';
            b1 = b1 - learning_rate * s1;
        end

        a1_error = logsig(W1 * p' + b1);
        a2_error = max(0, W2 * a1_error + b2);

        err = abs(a2_error' - g).^2;
        err_max(epoch) = sqrt(mean(err));

        if err_max(epoch) < convergence_threshold
            fprintf('Converged at epoch %d\n', epoch);
            break;
        end
    end

    failed = false;

    % Plot error
    fig = figure("Name", sprintf('1-%d-1 NN w/ a=%g', S, learning_rate));
    tiledlayout(fig, 2, 1);
    nexttile;
    plot(1:MAX_EPOCHS, err_max, 'LineWidth', 2);
    % semilogy(1:MAX_EPOCHS, err_max, 'LineWidth', 2); % Use this for better visualization in smaller values
    title(sprintf('Error vs. Epochs w/ a=%g', learning_rate));
    xlabel('Epochs');
    ylabel('Error');
    grid('on');
    
    % Test the network
    a1_test = logsig(W1 * p' + b1);
    a2_test = relu(W2 * a1_test + b2);

    % figure;
    nexttile;
    plot(p, a2_test, '-', p, g, '--', 'LineWidth', 2);
    legend('Network Output', 'Target Function', 'Location', 'best');
    title(sprintf('1-%d-1 Network Approximation w/ a=%g', S, learning_rate));
    xlabel('p');
    ylabel('g(p)');
    grid('on');

    if export_figs == true
        exportgraphics(fig, sprintf('nn_images/1-%d-1_NN_a=%g.pdf', S, learning_rate), "ContentType", "vector");
    end
end

% My implementation of ReLU
function result = relu (x)
    result = max(0, x);
end

function result = relu_derivative (x)
    if x > 0
        result = 1;
    else
        result = 0;
    end
end

function result = logsig_derivative (x)
    result = logsig(x) .* (1 - logsig(x));
end