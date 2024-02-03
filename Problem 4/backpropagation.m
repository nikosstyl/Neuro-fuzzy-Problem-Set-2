function backpropagation 

    % s = [2 8 12];
    s=12;
    % lr = [0.1 0.01 0.001];
    lr = 0.01;
    
    for i=s
        for j=lr
            calc_backpropagation(i, j);
        end
    end

end


function calc_backpropagation (S, learning_rate)

    fprintf('1-%d-1 Network, S = %d, lr = %g\n', S, S, learning_rate);

    % Initialize weights and biases
    W1 = rand(S, 1) ; %- 0.5;
    b1 = rand(S, 1) ; %- 0.5;
    W2 = rand(1, S) ; %- 0.5;
    b2 = rand(1, 1) ; %- 0.5;

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

    e = zeros(1, MAX_EPOCHS);

    convergence_threshold = 1e-5;

    w2_f = figure;
    w2_f.Name = sprintf('W2 vs. Epochs w/ a=%g', learning_rate);
    w2_f_ax = axes;
    w2_f_ax.Title.String = 'W2 vs. Epochs';
    hold(w2_f_ax, "on");

    % Training
    for epoch = 1:MAX_EPOCHS
        sum_squared_error = 0;
        for i = 1:length(p)
            % Forward pass
            n1 = W1 * p(i) + b1;
            a1 = logsig(n1);

            n2 = W2 * a1 + b2;
            a2 = relu(n2);

            % Backpropagation
            e(epoch) = g(i) - a2;
            sum_squared_error = sum_squared_error + e(epoch)^2; 

            df2 = relu_derivative(n2);
            
            s2 = -2*e(epoch)*df2;
            s1 = logsig_derivative(n1) .* W2' .* s2;

            if s2 == 0 && epoch == 1
                error('s2 is 0')
            end            

            % Update weights and biases
            W2 = W2 - learning_rate * s2 * a1';
            b2 = b2 - learning_rate * s2;
            W1 = W1 - learning_rate * s1 * p(i)';
            b1 = b1 - learning_rate * s1;
        end

        plot(w2_f_ax, epoch, W2(1), 'bx', epoch, W2(2), 'ro');
        e(epoch) = sum_squared_error / length(g);


        a1_error = logsig(W1 * p' + b1);
        a2_error = max(0, W2 * a1_error + b2);

        err  = abs(a2_error - g);
        if abs(max(err)) < convergence_threshold
            fprintf('Converged at epoch %d\n', epoch);
            break;
        end
        % if abs(e(epoch) - old_error) < convergence_threshold
        %     fprintf('Converged at epoch %d\n', epoch);
        %     break;
        % end
        % old_error = e(epoch);
    end

    % Plot error
    figure;
    plot(1:MAX_EPOCHS, e, 'LineWidth', 2);
    title(sprintf('Error vs. Epochs w/ a=%g', learning_rate));
    xlabel('Epochs');
    ylabel('Error');
    grid('on');
    
    % Test the network
    a1_test = logsig(W1 * p' + b1);
    a2_test = max(0, W2 * a1_test + b2);

    % Plot
    figure;
    plot(p, a2_test, '-', p, g, '--', 'LineWidth', 2);
    legend('Network Output', 'Target Function', 'Location', 'best');
    title(sprintf('1-%d-1 Network Approximation w/ a=%g', S, learning_rate));
    xlabel('p');
    ylabel('g(p)'); 
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