function fre_feature = calc_frequency_features(W, fs, nfft, freq_bands)
% Calculate frequency feature matrix from NMF's W matrix
% Inputs:
%   W           - Frequency basis matrix, dimension [number of frequency points × number of bases] (frequency×basis)
%   fs          - Sampling rate of original signal 
%   nfft        - Number of FFT points 
%   freq_bands  - Frequency band division vector
% Output:
%   fre_feature - Frequency feature matrix, dimension [number of bases × number of features]. Feature order:[frequency bandwidth, frequency variance, band 1 energy ratio, band 2 energy ratio, ..., spectral entropy, peak count, average peak spacing]
f = (0:nfft/2) * (fs / nfft); 
num_freq = length(f);
if size(W, 1) ~= num_freq
    error('Number of rows in W must equal nfft/2 + 1 (number of positive frequency points)');
end
[~, num_basis] = size(W); 
num_band = length(freq_bands) - 1; 
fre_feature = zeros(num_basis, 3 + num_band + 2); 
for j = 1:num_basis
    w_j = W(:, j);
    total_energy = sum(w_j);
    if total_energy < 1e-10 
        fre_feature(j, :) = zeros(1, size(fre_feature, 2));
        continue;
    end
    [max_amp, peak_idx] = max(w_j);
    threshold = max_amp * 0.5;
    left_idx = peak_idx;
    while left_idx > 1 && w_j(left_idx) >= threshold
        left_idx = left_idx - 1;
    end
    right_idx = peak_idx;
    while right_idx < num_freq && w_j(right_idx) >= threshold
        right_idx = right_idx + 1;
    end
    bandwidth = f(right_idx) - f(left_idx);
    mean_freq = sum(f .* w_j) / total_energy;
    freq_var = sum((f - mean_freq).^2 .* w_j') / total_energy;
    band_ratio = zeros(1, num_band);
    for b = 1:num_band
        band_mask = (f >= freq_bands(b)) & (f < freq_bands(b+1));
        band_energy = sum(w_j(band_mask));
        band_ratio(b) = band_energy / total_energy;
    end
    p = w_j / total_energy;
    p(p < 1e-10) = 1e-10;  
    spectral_entropy = -sum(p .* log2(p));
    min_peak_height = max_amp * 0.02;
    min_peak_distance = 5; 
    [~, peaks_idx] = findpeaks(w_j, 'MinPeakHeight', min_peak_height, 'MinPeakDistance', min_peak_distance);
    peak_count = length(peaks_idx);
    if peak_count >= 2
        peak_freqs = f(peaks_idx);
        peak_spacing = mean(diff(peak_freqs));
    else
        peak_spacing = 0;
    end
    fre_feature(j, :) = [bandwidth, freq_var, spectral_entropy, peak_count, peak_spacing, band_ratio];
end
end