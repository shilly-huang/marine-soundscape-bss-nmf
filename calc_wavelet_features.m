function wave_feature = calc_wavelet_features(H, fs, hop_length)
% Calculate wavelet-based multi-resolution features from NMF's H matrix.
% Inputs:
%   H           - Temporal activation matrix, [num_basis × num_frames] 
%   fs          - Sampling rate of original signal (Hz)
%   hop_length  - STFT hop length
%
% Output:
%   wave_feature - Wavelet feature matrix, [num_basis × num_features]
%                  Each row concatenates:
%                  [energy_ratio_level1..L , entropy_level1..L]
    [num_basis, num_frames] = size(H);
    wname = 'sym4';   
    Lmax  = 6;        

    wave_feature = [];

    for i = 1:num_basis
        h_i = double(H(i, :));
        h_i = h_i - mean(h_i); 

        N = numel(h_i);
        if N < 16
            if isempty(wave_feature)
                wave_feature = zeros(num_basis, 2); 
            end
            wave_feature(i, :) = 0;
            continue;
        end
        L = min(Lmax, max(1, floor(log2(N)) - 2));

        if isempty(wave_feature)
            wave_feature = zeros(num_basis, 2 * L);
        elseif size(wave_feature,2) ~= 2*L
            L = size(wave_feature,2) / 2;
        end

        if exist('modwt','file') == 2
            W = modwt(h_i, wname, L);
            detail = W(1:L, :); 
        else
            [C, Lvec] = wavedec(h_i, L, wname);
            detail = zeros(L, N);
            for lev = 1:L
                d = wrcoef('d', C, Lvec, wname, lev);
                detail(lev, :) = d(:).';
            end
        end
        E = sum(detail.^2, 2);          % [L×1]
        Etot = sum(E) + eps;
        energy_ratio = (E / Etot).';    % [1×L]
        entropy_levels = zeros(1, L);
        for lev = 1:L
            p = detail(lev,:).^2;
            p = p / (sum(p) + eps);
            entropy_levels(lev) = -sum(p .* log(p + eps));
        end
        wave_feature(i, :) = [energy_ratio, entropy_levels];
    end
end
