function [best_class_index, best_silhouette, k_optimal] = Clustering_Step(feature, k_range, sH, iter_num, use_cascade, secondary_feature)
% Stage-1: NMF sparse clustering 
% Stage-2: K-means refinement

if nargin < 3
    sH = 0.3;
    iter_num = 100;
    use_cascade = false;
    secondary_feature = [];
elseif nargin < 4
    iter_num = 100;
    use_cascade = false;
    secondary_feature = [];
elseif nargin < 5
    use_cascade = false;
    secondary_feature = [];
end
[class_index, ~, ~, ~] = scnmf(feature, k_range, sH, iter_num, 10);
class_index = reshape(class_index, 1, []);
class_index = reorganize_cluster_labels(feature, class_index);
data_for_silhouette = feature';
try
    silhouette_value = mean(silhouette(data_for_silhouette, class_index'));
catch
    silhouette_value = -1;
end
best_class_index = class_index;
best_silhouette  = silhouette_value;
k_optimal        = max(class_index);
fprintf('Clustering completed: k=%d, silhouette coefficient=%.4f\n', k_optimal, best_silhouette);
if use_cascade && ~isempty(secondary_feature) && k_optimal > 1
    best_class_index = cascade_separation(secondary_feature, best_class_index, k_optimal);
    try
        best_silhouette = mean(silhouette(data_for_silhouette, best_class_index'));
    catch
        best_silhouette = -1;
    end
    k_optimal = max(best_class_index);
end
end
%% Helper function
function class_index = reorganize_cluster_labels(feature, class_index)
    if size(feature, 1) >= 1 && max(class_index) > 1
        cluster_means = zeros(1, max(class_index));
        for i = 1:max(class_index)
            cluster_samples = find(class_index == i);
            if ~isempty(cluster_samples)
                cluster_means(i) = mean(feature(1, cluster_samples));
            else
                cluster_means(i) = Inf;
            end
        end
        [~, sort_idx] = sort(cluster_means);
        new_labels = zeros(size(class_index));
        for i = 1:length(sort_idx)
            old_label = sort_idx(i);
            new_labels(class_index == old_label) = i;
        end
        class_index = new_labels;
    end
end

function final_class_index = cascade_separation(secondary_feature, initial_class_index, k_primary)
    final_class_index = initial_class_index;
    current_max_label = max(initial_class_index);
    hc_linkage  = 'average';
    hc_dist     = 'cosine';
    sil_thresh  = 0.20;
    for cluster_id = 1:k_primary
        cluster_mask = (initial_class_index == cluster_id);
        cluster_samples = find(cluster_mask);
        num_samples = length(cluster_samples);
        if num_samples < 10
            fprintf('Cluster %d has too few samples (%d), skipping further separation\n', cluster_id, num_samples);
            continue;
        end
        X = secondary_feature(:, cluster_mask)';
        max_possible_k = min(4, floor(num_samples / 4));
        if max_possible_k < 2
            fprintf('Cluster %d has insufficient samples for further separation\n', cluster_id);
            continue;
        end
        k_candidates = 2:max_possible_k;
        [k_hc, sil_hc, labels_hc] = find_optimal_hc_for_k(X, k_candidates, hc_linkage, hc_dist);
        if k_hc <= 1 || sil_hc <= sil_thresh
            fprintf('Cluster %d kept (HC bestSil=%.3f -> choose K=1)\n', cluster_id, sil_hc);
            continue;
        end
      
        [labels_km, sil_km] = run_kmeans(X, k_hc);
        should_separate = should_separate_cluster(sil_km, k_hc, X, labels_km);
        if should_separate
            fprintf('Cluster %d further separated into %d subclusters (K from HC: %.3f, kmeans sil=%.3f)\n', ...
                cluster_id, k_hc, sil_hc, sil_km);
            for sub_idx = 1:k_hc
                sub_mask = (labels_km == sub_idx);
                final_class_index(cluster_samples(sub_mask)) = current_max_label + sub_idx;
            end
            current_max_label = current_max_label + k_hc;
        else
            fprintf('Cluster %d remains as 1 cluster (K from HC: %.3f, kmeans sil=%.3f)\n', ...
                cluster_id, sil_hc, sil_km);
        end
    end
    if max(final_class_index) > k_primary
        unique_labels = unique(final_class_index);
        label_map = containers.Map(unique_labels, 1:length(unique_labels));
        for i = 1:length(final_class_index)
            final_class_index(i) = label_map(final_class_index(i));
        end
        fprintf('Cascaded separation completed, final number of clusters: %d\n', max(final_class_index));
    else
        fprintf('Cascaded separation completed, all clusters remain unchanged\n');
    end
end

% HC: choose K by scanning candidates and maximizing mean silhouette
function [best_k, best_sil, best_labels] = find_optimal_hc_for_k(X, k_candidates, linkage_method, dist_metric)
    best_sil = -Inf;
    best_k = 1;
    best_labels = ones(size(X,1), 1);
    try
        D = pdist(X, dist_metric);
        Z = linkage(D, linkage_method);
    catch ME
        fprintf('HC failed: %s\n', ME.message);
        return;
    end
    for k = k_candidates
        try
            lab = cluster(Z, 'maxclust', k);
            s = silhouette(X, lab);
            ms = mean(s, 'omitnan');
            if ~isnan(ms) && ms > best_sil
                best_sil = ms;
                best_k = k;
                best_labels = lab;
            end
        catch
            % ignore invalid k
        end
    end
end

% k-means
function [best_labels, best_silhouette] = run_kmeans(X, k)
    best_silhouette = -Inf;
    best_labels = ones(size(X,1),1);
    max_repeats = 5;
    best_inertia = inf;
    best_k_labels = [];
    for repeat = 1:max_repeats
        try
            [labels, ~, sumd] = kmeans(X, k, 'Distance','cosine', 'Replicates', 3, 'MaxIter', 200);
            inertia = sum(sumd);
            if inertia < best_inertia
                best_inertia = inertia;
                best_k_labels = labels;
            end
        catch
        end
    end
    if isempty(best_k_labels)
        return;
    end
    try
        silh = silhouette(X, best_k_labels);
        best_silhouette = mean(silh, 'omitnan');
        best_labels = best_k_labels;
    catch
        best_silhouette = -Inf;
        best_labels = best_k_labels;
    end
end

function should_separate = should_separate_cluster(silhouette_score, k, data, labels)
    % Condition 1: Silhouette coefficient threshold
    condition1 = silhouette_score > 0;
    % Condition 2: Number of clusters greater than 1
    condition2 = k > 1;
    % Condition 3: Minimum sample size per subcluster
    min_cluster_size = inf;
    for i = 1:k
        cluster_size = sum(labels == i);
        if cluster_size < min_cluster_size
            min_cluster_size = cluster_size;
        end
    end
    condition3 = min_cluster_size >= 5;
    % Condition 4: Ratio of inter-cluster to intra-cluster distance
    condition4 = check_cluster_separation(data, labels);
    should_separate = condition1 && condition2 && condition3 && condition4;
    if ~should_separate
        fprintf('Reasons for not separating: sil=%.3f, k=%d, min_size=%d, sep_ok=%d\n', ...
            silhouette_score, k, min_cluster_size, condition4);
    end
end

function is_well_separated = check_cluster_separation(data, labels)
    k = max(labels);
    if k == 1
        is_well_separated = false;
        return;
    end
    intra_distances = zeros(1, k);
    for i = 1:k
        cluster_data = data(labels == i, :);
        if size(cluster_data, 1) > 1
            centroid = mean(cluster_data, 1);
            distances = sqrt(sum((cluster_data - centroid).^2, 2));
            intra_distances(i) = mean(distances);
        else
            intra_distances(i) = 0;
        end
    end
    mean_intra_distance = mean(intra_distances);
    centroids = zeros(k, size(data, 2));
    for i = 1:k
        centroids(i, :) = mean(data(labels == i, :), 1);
    end
    inter_distances = [];
    for i = 1:k-1
        for j = i+1:k
            distance = norm(centroids(i, :) - centroids(j, :));
            inter_distances = [inter_distances, distance]; %#ok<AGROW>
        end
    end
    mean_inter_distance = mean(inter_distances);
    separation_ratio = mean_inter_distance / (mean_intra_distance + eps);
    is_well_separated = separation_ratio > 1.2;
    fprintf('Separation ratio: %.3f (inter/intra)\n', separation_ratio);
end