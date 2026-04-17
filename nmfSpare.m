function [W, H] = nmfSpare(data_matrix, num_components, sparsity_W, sparsity_H, max_iterations, init_W, init_H)
    if min(data_matrix(:)) < 0
        error('Input data_matrix contains negative values!');
    end
    data_matrix = data_matrix / max(data_matrix(:));
    [num_features, num_samples] = size(data_matrix);
    if nargin < 7 || isempty(init_W)
        W = abs(randn(num_features, num_components));
    else
        W = init_W;
    end
    if nargin < 8 || isempty(init_H)
        H = abs(randn(num_components, num_samples));
        row_norms_H = sqrt(sum(H.^2, 2));
        H = H ./ (row_norms_H * ones(1, num_samples));
    else
        H = init_H;
    end
    if ~isempty(sparsity_W)
        L1_bound_W = sqrt(num_features) - (sqrt(num_features) - 1) * sparsity_W;
        for i = 1:num_components
            W(:, i) = projfunc(W(:, i), L1_bound_W, 1, 1);
        end
    end
    if ~isempty(sparsity_H)
        L1_bound_H = sqrt(num_samples) - (sqrt(num_samples) - 1) * sparsity_H;
        for i = 1:num_components
            H(i, :) = (projfunc(H(i, :)', L1_bound_H, 1, 1))';
        end
    end
    loss_history = 0.5 * sum(sum((data_matrix - W * H).^2));    
    step_size_W = 1;
    step_size_H = 1;
    for iter = 1:max_iterations
        if ~isempty(sparsity_H)
            gradient_H = W' * (W * H - data_matrix);
            current_loss = loss_history(end);
            search_steps = 1;
            
            while true
                H_candidate = H - step_size_H * gradient_H;
                for i = 1:num_components
                    H_candidate(i, :) = (projfunc(H_candidate(i, :)', L1_bound_H, 1, 1))';
                end
                candidate_loss = 0.5 * sum(sum((data_matrix - W * H_candidate).^2));
                if candidate_loss <= current_loss || search_steps >= 10
                    break;
                end
                step_size_H = step_size_H / 2;
                search_steps = search_steps + 1;
            end
            step_size_H = step_size_H * 1.2;
            H = H_candidate;
        else
            H = H .* (W' * data_matrix) ./ (W' * W * H + 1e-9);
            row_norms = sqrt(sum(H.^2, 2));
            H = H ./ (row_norms * ones(1, num_samples));
            W = W .* (ones(num_features, 1) * row_norms');
        end
        if ~isempty(sparsity_W)
            gradient_W = (W * H - data_matrix) * H';
            current_loss = 0.5 * sum(sum((data_matrix - W * H).^2));
            search_steps = 1;
            while true
                W_candidate = W - step_size_W * gradient_W;
                col_norms = sqrt(sum(W_candidate.^2, 1));
                for i = 1:num_components
                    W_candidate(:, i) = projfunc(W_candidate(:, i), L1_bound_W * col_norms(i), (col_norms(i)^2), 1);
                end
                candidate_loss = 0.5 * sum(sum((data_matrix - W_candidate * H).^2));
                if candidate_loss <= current_loss || search_steps >= 10
                    break;
                end
                step_size_W = step_size_W / 2;
                search_steps = search_steps + 1;
            end
            step_size_W = step_size_W * 1.2;
            W = W_candidate;
        else
            W = W .* (data_matrix * H') ./ (W * H * H' + 1e-9);
        end
        new_loss = 0.5 * sum(sum((data_matrix - W * H).^2));
        loss_history = [loss_history, new_loss];
    end
end