comb_type = 2;
audio_root_dir = 'fish_sound_combinations';
signal_names = {'DS', 'LT', 'LDS', 'RPS', 'PS', 'LPS'};
num_true_sources = length(signal_names);
s_true = cell(1, num_true_sources);
switch comb_type
    case 1
        mix_audio_path = fullfile(audio_root_dir, 'fish_sounds_2comb_30s.wav');
        single_audio_paths = {
            fullfile(audio_root_dir, 'fish_2comb_single_DS_30s.wav');
            fullfile(audio_root_dir, 'fish_2comb_single_LT_30s.wav');
            fullfile(audio_root_dir, 'fish_2comb_single_LDS_30s.wav');
            fullfile(audio_root_dir, 'fish_2comb_single_RPS_30s.wav');
            fullfile(audio_root_dir, 'fish_2comb_single_PS_30s.wav');
            fullfile(audio_root_dir, 'fish_2comb_single_LPS_30s.wav');
        };
    case 2
        mix_audio_path = fullfile(audio_root_dir, 'fish_sounds_3comb_40s.wav');
        single_audio_paths = {
            fullfile(audio_root_dir, 'fish_3comb_single_DS_40s.wav');
            fullfile(audio_root_dir, 'fish_3comb_single_LT_40s.wav');
            fullfile(audio_root_dir, 'fish_3comb_single_LDS_40s.wav');
            fullfile(audio_root_dir, 'fish_3comb_single_RPS_40s.wav');
            fullfile(audio_root_dir, 'fish_3comb_single_PS_40s.wav');
            fullfile(audio_root_dir, 'fish_3comb_single_LPS_40s.wav');
        };
    case 3
        mix_audio_path = fullfile(audio_root_dir, 'fish_sounds_4comb_30s.wav');
        single_audio_paths = {
            fullfile(audio_root_dir, 'fish_4comb_single_DS_30s.wav');
            fullfile(audio_root_dir, 'fish_4comb_single_LT_30s.wav');
            fullfile(audio_root_dir, 'fish_4comb_single_LDS_30s.wav');
            fullfile(audio_root_dir, 'fish_4comb_single_RPS_30s.wav');
            fullfile(audio_root_dir, 'fish_4comb_single_PS_30s.wav');
            fullfile(audio_root_dir, 'fish_4comb_single_LPS_30s.wav');
        };
    case 4
        mix_audio_path = fullfile(audio_root_dir, 'fish_sounds_5comb_30s.wav');
        single_audio_paths = {
            fullfile(audio_root_dir, 'fish_5comb_single_DS_30s.wav');
            fullfile(audio_root_dir, 'fish_5comb_single_LT_30s.wav');
            fullfile(audio_root_dir, 'fish_5comb_single_LDS_30s.wav');
            fullfile(audio_root_dir, 'fish_5comb_single_RPS_30s.wav');
            fullfile(audio_root_dir, 'fish_5comb_single_PS_30s.wav');
            fullfile(audio_root_dir, 'fish_5comb_single_LPS_30s.wav');
        };
    case 5
        mix_audio_path = fullfile(audio_root_dir, 'fish_sounds_6comb_20s.wav');
        single_audio_paths = {
            fullfile(audio_root_dir, 'fish_6comb_single_DS_20s.wav');
            fullfile(audio_root_dir, 'fish_6comb_single_LT_20s.wav');
            fullfile(audio_root_dir, 'fish_6comb_single_LDS_20s.wav');
            fullfile(audio_root_dir, 'fish_6comb_single_RPS_20s.wav');
            fullfile(audio_root_dir, 'fish_6comb_single_PS_20s.wav');
            fullfile(audio_root_dir, 'fish_6comb_single_LPS_20s.wav');
        };
end
[x, fs] = audioread(mix_audio_path);
x = x(:,1);
for i = 1:num_true_sources
    if exist(single_audio_paths{i}, 'file')
        [tmp, fs2] = audioread(single_audio_paths{i});
        tmp = tmp(:,1);
        if fs2 ~= fs
            tmp = resample(tmp, fs, fs2);
        end
        L = min(length(tmp), length(x));
        s_true{i} = tmp(1:L);
    else
        s_true{i} = [];
    end
end
window_size = hamming(512);
overlap = 0.5*512;
nfft = 2048;
[S, F, T] = spectrogram(x, window_size, overlap, nfft, fs);
data = abs(S);
basis_num = 150;
iter_num  = 500;
sparseness_W = [];
sparseness_H = [];
pretrain_iter_num = 20;
rmse = zeros(1, pretrain_iter_num);
W_pretrain_all = cell(1, pretrain_iter_num);
H_pretrain_all = cell(1, pretrain_iter_num);
for m = 1:pretrain_iter_num
    [W_pretrain, H_pretrain] = nmfSpare(data, basis_num, sparseness_W, sparseness_H, pretrain_iter_num, 0);
    W_pretrain_all{m} = W_pretrain;
    H_pretrain_all{m} = H_pretrain;
    V_recon = W_pretrain * H_pretrain;
    V_recon = (V_recon - min(V_recon(:))) ./ (max(V_recon(:)) - min(V_recon(:)) + 1e-9);
    rmse(m) = sqrt(mean(abs(data - V_recon).^2, 'all'));
end
[~, best_idx] = min(rmse);
[W, H] = nmfSpare(data, basis_num, sparseness_W, sparseness_H, ...
    iter_num - pretrain_iter_num, 0, W_pretrain_all{best_idx}, H_pretrain_all{best_idx});

freq_bands = 0:20:fs/2;
fre_feature    = calc_frequency_features(W, fs, nfft, freq_bands);     
period_feature = calc_period_features(H, fs, 256);                    
wave_feature   = calc_wavelet_features(H, fs, 256);                 
cat_feature    = [fre_feature wave_feature];                   

feature_list = {
    'W'
    'H'
    'Frequency Feature'
    'Period Feature'
    'wave Feature'
    'Cat Feature'
};
results_table = table();
comb_subdir = sprintf('comb_type_%d', comb_type);
save_dir = fullfile('single_cluster', comb_subdir);
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
method_name = 'One-stage';
for idx_feat = 1:numel(feature_list)
    feat_name = feature_list{idx_feat};
    switch feat_name
        case 'W'
            feat_mat = W;           
        case 'H'
            feat_mat = H';        
        case 'Frequency Feature'
            feat_mat = fre_feature';  
        case 'Period Feature'
            feat_mat = period_feature'; 
        case 'wave Feature'
            feat_mat = wave_feature';  
        case 'Cat Feature'
            feat_mat = cat_feature';  
        otherwise
            error('Unknown feature name: %s', feat_name);
    end
    [cluster_labels_pc, ~] = scnmf(feat_mat, [2 6], 0.3, iter_num, 10);
    cluster_labels_pc = cluster_labels_pc(:)';   % 1×K
    separate_source_pc = max(cluster_labels_pc);
    reconstructed_signals_pc = perform_source_separation(W, H, cluster_labels_pc, S, window_size, overlap, nfft, fs);

    [reconstructed_signals_pc, ~] = merge_separated_sources(reconstructed_signals_pc, fs, 0.80, 0.80);
    separate_source_pc = numel(reconstructed_signals_pc);
    E_pc = evaluate_separation_step(s_true, reconstructed_signals_pc, signal_names);
    matched_true_mask_pc = (E_pc.matched_est_idx_for_true > 0);
    avg_sdr_pc = mean(E_pc.sdr_true(matched_true_mask_pc), 'omitnan');
    avg_sir_pc = mean(E_pc.sir_true(matched_true_mask_pc), 'omitnan');
    avg_sar_pc = mean(E_pc.sar_true(matched_true_mask_pc), 'omitnan');
    coverage_pc = sum(matched_true_mask_pc) / numel(E_pc.matched_est_idx_for_true);
    matched_true_idx_for_est = E_pc.matched_true_idx_for_est(:);
    sdr_est_list = E_pc.sdr_est(:).';
    sir_est_list = E_pc.sir_est(:).';
    sar_est_list = E_pc.sar_est(:).';

    matched_true_name_for_est = cell(1, numel(matched_true_idx_for_est));
    for ii = 1:numel(matched_true_idx_for_est)
        ti = matched_true_idx_for_est(ii);
        if ti > 0 && ti <= numel(signal_names)
            matched_true_name_for_est{ii} = signal_names{ti};
        else
            matched_true_name_for_est{ii} = 'None';
        end
    end

    matched_true_name_str = strjoin(matched_true_name_for_est, ',');
    matched_true_idx_str  = mat2str(matched_true_idx_for_est(:).', 4);
    sdr_str = mat2str(sdr_est_list, 4);
    sir_str = mat2str(sir_est_list, 4);
    sar_str = mat2str(sar_est_list, 4);

    result_row = table();
    result_row.Method       = {method_name};
    result_row.Comb_Type    = comb_type;
    result_row.FeatureUsed  = {feat_name};
    result_row.SepSourceNum = separate_source_pc;
    result_row.Coverage     = coverage_pc;
    result_row.MatchedTrueName_forEst = {matched_true_name_str};
    result_row.MatchedTrueIdx_forEst  = {matched_true_idx_str};
    result_row.SDR_Est_List = {sdr_str};
    result_row.SIR_Est_List = {sir_str};
    result_row.SAR_Est_List = {sar_str};
    result_row.Avg_SDR = avg_sdr_pc;
    result_row.Avg_SIR = avg_sir_pc;
    result_row.Avg_SAR = avg_sar_pc;

    if isempty(results_table)
        results_table = result_row;
    else
        results_table = [results_table; result_row];
    end
end
results_file = fullfile(save_dir, sprintf('comb%d_single_results.xlsx', comb_type));
writetable(results_table, results_file);