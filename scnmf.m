function [final_labels, basis_W, consensus_matrices, dispersion_scores] = scnmf(data_matrix, candidate_k_values, sparsity_H, max_iterations, num_replicates)
    if nargin < 3 || isempty(sparsity_H)
        sparsity_H = 0.3;
    end
    if nargin < 4 || isempty(max_iterations)
        max_iterations = 100;
    end
    if nargin < 5 || isempty(num_replicates)
        num_replicates = 10;
    end
    num_k_options = length(candidate_k_values);
    num_samples = size(data_matrix, 2);
    
    if num_k_options == 1
        num_replicates = 1; 
        consensus_matrices = [];
        dispersion_scores = [];
    else
        consensus_matrices = zeros(num_samples, num_samples, num_k_options);
        dispersion_scores = zeros(1, num_k_options);
    end
    all_run_labels = zeros(num_replicates, num_samples, num_k_options);
    for idx_k = 1:num_k_options
        current_k = candidate_k_values(idx_k);
        for idx_rep = 1:num_replicates
            [current_W, current_H] = nmfSpare(data_matrix, current_k, [], sparsity_H, max_iterations, 0);
            [~, all_run_labels(idx_rep, :, idx_k)] = max(current_H, [], 1);
            if num_k_options == 1
                basis_W = current_W;
            end
        end
    end
    if num_k_options > 1
        for idx_k = 1:num_k_options
            labels_for_current_k = all_run_labels(:, :, idx_k);
            
            for sample_i = 1:num_samples
                for sample_j = 1:num_samples
                    same_cluster_count = sum(labels_for_current_k(:, sample_i) == labels_for_current_k(:, sample_j));
                    consensus_matrices(sample_i, sample_j, idx_k) = same_cluster_count / num_replicates;
                end
            end
            current_consensus = consensus_matrices(:, :, idx_k);
            dispersion_scores(idx_k) = sum(sum(4 * ((current_consensus - 0.5).^2))) / (num_samples^2);
        end
        [~, best_dispersion_idx] = max(dispersion_scores);
        optimal_k = candidate_k_values(best_dispersion_idx);
        [basis_W, optimal_H] = nmfSpare(data_matrix, optimal_k, [], sparsity_H, max_iterations, 0);
        [~, final_labels] = max(optimal_H, [], 1);
    else
        final_labels = all_run_labels(1, :, 1);
    end
end