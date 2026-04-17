function out = evaluate_separation_step(s_true, s_est, signal_names)
% Evaluate separation for BSS using bss_eval_sources (from bss_eval toolbox).
% Inputs:
%   s_true       : cell{1 x Ntrue} true sources (time-domain)
%   s_est        : cell{1 x Nest}  estimated sources (time-domain)
%   signal_names : cell{1 x Ntrue} true source names (optional)
%
% Output struct fields (key ones):
%   out.matched_est_idx_for_true    : [Ntrue x 1] est index per true (0 if unmatched)
%   out.matched_true_idx_for_est    : [Nest x 1] true index per est (0 if not selected in 1-1)
%   out.sdr_true/sir_true/sar_true  : [Ntrue x 1] group-wise metrics (NaN for unmatched true)
%   out.sdr_est/sir_est/sar_est     : [Nest x 1] mapped metrics to est (NaN for est not in 1-1)
%   out.score_matrix_sdr_pair       : [Nest x Ntrue] pairwise single-source SDR
%   out.best_true_idx_for_est       : [Nest x 1] many-to-one best true for each est (for plotting)
%   out.best_sdr_for_est            : [Nest x 1] corresponding single-source SDR (for plotting)
%   out.coverage                    : scalar, matched true count / Ntrue
    num_true = numel(s_true);
    num_est  = numel(s_est);
    if num_true == 0 || num_est == 0
        error('True sources or estimated sources cannot be empty.');
    end
    if nargin < 3 || isempty(signal_names)
        signal_names = arrayfun(@(k) sprintf('True%d', k), 1:num_true, 'UniformOutput', false);
    end
    min_len = inf;
    for j = 1:num_true, min_len = min(min_len, numel(s_true{j})); end
    for i = 1:num_est,  min_len = min(min_len, numel(s_est{i}));  end
    if ~isfinite(min_len) || min_len < 10
        error('Signals too short after alignment.');
    end
    ref_mat = zeros(num_true, min_len);
    for j = 1:num_true
        ref_mat(j,:) = real(s_true{j}(1:min_len)).';
    end
    est_all = zeros(num_est, min_len);
    for i = 1:num_est
        est_all(i,:) = real(s_est{i}(1:min_len)).';
    end

    score_matrix = zeros(num_est, num_true);
    for i = 1:num_est
        for j = 1:num_true
            sdr_ij = bss_eval_sources(ref_mat(j,:), est_all(i,:));
            if numel(sdr_ij) > 1, sdr_ij = sdr_ij(1); end
            score_matrix(i,j) = sdr_ij;
        end
    end
    best_true_idx_for_est = zeros(num_est,1);
    best_sdr_for_est = nan(num_est,1);
    for i = 1:num_est
        [best_sdr_for_est(i), best_true_idx_for_est(i)] = max(score_matrix(i,:));
    end
    matched_est_idx_for_true = best_assignment_exhaustive(score_matrix); 
    matched_true_idx_for_est = zeros(num_est, 1);
    for j = 1:num_true
        i = matched_est_idx_for_true(j);
        if i > 0
            matched_true_idx_for_est(i) = j;
        end
    end
    matched_true_mask = (matched_est_idx_for_true > 0);
    matched_true_idx  = find(matched_true_mask);
    coverage = sum(matched_true_mask) / num_true;
    sdr_true = nan(num_true,1);
    sir_true = nan(num_true,1);
    sar_true = nan(num_true,1);
    if ~isempty(matched_true_idx)
        ref_sub = ref_mat(matched_true_idx, :);
        est_sub = zeros(numel(matched_true_idx), min_len);
        for k = 1:numel(matched_true_idx)
            j = matched_true_idx(k);
            i = matched_est_idx_for_true(j);
            est_sub(k,:) = est_all(i,:);
        end
        [sdr_sub, sir_sub, sar_sub] = bss_eval_sources(ref_sub, est_sub);
        sdr_true(matched_true_idx) = sdr_sub(:);
        sir_true(matched_true_idx) = sir_sub(:);
        sar_true(matched_true_idx) = sar_sub(:);
    end    
    sdr_est = nan(num_est,1);
    sir_est = nan(num_est,1);
    sar_est = nan(num_est,1);
    matched_true_name_for_est = cell(1, num_est);
    for j = 1:num_true
        i = matched_est_idx_for_true(j);
        if i > 0
            sdr_est(i) = sdr_true(j);
            sir_est(i) = sir_true(j);
            sar_est(i) = sar_true(j);
            matched_true_name_for_est{i} = signal_names{j};
        end
    end
    out = struct();
    out.matched_est_idx_for_true = matched_est_idx_for_true(:);
    out.matched_true_idx_for_est = matched_true_idx_for_est(:);
    out.sdr_true = sdr_true(:);
    out.sir_true = sir_true(:);
    out.sar_true = sar_true(:);
    out.sdr_est = sdr_est(:);
    out.sir_est = sir_est(:);
    out.sar_est = sar_est(:);
    out.score_matrix_sdr_pair = score_matrix;
    out.best_true_idx_for_est = best_true_idx_for_est(:);
    out.best_sdr_for_est = best_sdr_for_est(:);
    out.matched_true_name_for_est = matched_true_name_for_est;
    out.coverage = coverage;
end

function matched_est_idx_for_true = best_assignment_exhaustive(score_matrix)
    [num_est, num_true] = size(score_matrix);
    matched_est_idx_for_true = zeros(num_true,1);
    if num_est < num_true
        true_sets = nchoosek(1:num_true, num_est);
        bestScore = -Inf;
        bestMatch = matched_est_idx_for_true;

        perms_est = perms(1:num_est);
        for k = 1:size(true_sets,1)
            trues = true_sets(k,:);
            for p = 1:size(perms_est,1)
                est_perm = perms_est(p,:);
                total = 0;
                tmp = zeros(num_true,1);
                for t = 1:num_est
                    j = trues(t);
                    i = est_perm(t);
                    total = total + score_matrix(i,j);
                    tmp(j) = i;
                end
                if total > bestScore
                    bestScore = total;
                    bestMatch = tmp;
                end
            end
        end
        matched_est_idx_for_true = bestMatch;
        return;
    end

    est_subsets = nchoosek(1:num_est, num_true);
    bestScore = -Inf;
    for k = 1:size(est_subsets,1)
        subset = est_subsets(k,:);
        perms_subset = perms(subset);
        for p = 1:size(perms_subset,1)
            est_perm = perms_subset(p,:);
            total = 0;
            for j = 1:num_true
                total = total + score_matrix(est_perm(j), j);
            end
            if total > bestScore
                bestScore = total;
                matched_est_idx_for_true = est_perm(:);
            end
        end
    end
end
