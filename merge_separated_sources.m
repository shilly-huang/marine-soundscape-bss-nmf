function [merged_signals, merge_info] = merge_separated_sources(reconstructed_signals, fs, tau_env, tau_spec)
% Merge over-segmented sources after separation.
% A pair is merged only if BOTH:
%   (1) envelope correlation >= tau_env
%   (2) spectral similarity (log-PSD corr) >= tau_spec
%
% Inputs:
%   reconstructed_signals : cell array {1 x K}, each is a time-domain signal
%   fs                   : sampling rate
%   tau_env              : threshold for envelope correlation
%   tau_spec             : threshold for spectral similarity
% Outputs:
%   merged_signals : cell array after iterative merging
%   merge_info     : struct recording merges
    if nargin < 3 || isempty(tau_env),  tau_env = 0.70;  end
    if nargin < 4 || isempty(tau_spec), tau_spec = 0.70; end
    % Ensure row vectors & real
    sigs = reconstructed_signals;
    for i = 1:numel(sigs)
        sigs{i} = real(sigs{i}(:)).';
    end
    merge_info = struct();
    merge_info.tau_env = tau_env;
    merge_info.tau_spec = tau_spec;
    merge_info.merges = []; 
    changed = true;
    while changed
        changed = false;
        K = numel(sigs);
        if K <= 1, break; end
        bestScore = -Inf;
        bestPair = [];
        for i = 1:K-1
            for j = i+1:K
                [envCorr, specSim] = pair_similarity(sigs{i}, sigs{j}, fs);
                if envCorr >= tau_env && specSim >= tau_spec
                    score = 0.5 * envCorr + 0.5 * specSim;
                    if score > bestScore
                        bestScore = score;
                        bestPair = [i, j, envCorr, specSim];
                    end
                end
            end
        end
      
        if ~isempty(bestPair)
            i = bestPair(1);
            j = bestPair(2);
            envCorr = bestPair(3);
            specSim = bestPair(4);

            xi = sigs{i};
            xj = sigs{j};
            L = min(numel(xi), numel(xj));
            xi = xi(1:L);
            xj = xj(1:L);
            x_merge = xi + xj;
            rms_i = rms(xi) + eps;
            rms_j = rms(xj) + eps;
            rms_target = max(rms_i, rms_j);
            x_merge = x_merge * (rms_target / (rms(x_merge) + eps));
            sigs{i} = x_merge;
            sigs(j) = [];
            merge_info.merges = [merge_info.merges; bestPair]; 
            changed = true;
        end
    end
    merged_signals = sigs;
end

function [envCorr, specSim] = pair_similarity(x1, x2, fs)
    L = min(numel(x1), numel(x2));
    x1 = x1(1:L);
    x2 = x2(1:L);
    x1 = x1 - mean(x1);
    x2 = x2 - mean(x2);
    e1 = abs(hilbert(x1));
    e2 = abs(hilbert(x2));
    envCorr = corr_safe(e1(:), e2(:));
    nfft = 2048;
    win = hamming(1024);
    nover = 512;
    [P1, ~] = pwelch(x1, win, nover, nfft, fs);
    [P2, ~] = pwelch(x2, win, nover, nfft, fs);

    lp1 = log10(P1 + eps);
    lp2 = log10(P2 + eps);
    specSim = corr_safe(lp1(:), lp2(:));
end

function r = corr_safe(a, b)
    a = a(:); b = b(:);
    if numel(a) ~= numel(b) || numel(a) < 5
        r = NaN; return;
    end
    sa = std(a); sb = std(b);
    if sa < 1e-12 || sb < 1e-12
        r = NaN; return;
    end
    r = corr(a, b, 'Type', 'Pearson');
end
