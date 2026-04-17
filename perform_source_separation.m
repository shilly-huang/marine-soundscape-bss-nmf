function reconstructed_signals = perform_source_separation(W, H, cluster_labels, S, window_size, overlap, nfft, fs)
    num_sources = max(cluster_labels);
    output = [];
    for m = 1:num_sources
        W_selected = W(:, cluster_labels == m);
        H_selected = H(cluster_labels == m, :);
        recon_cascaded = W_selected * H_selected;
        output(:, :, m) = recon_cascaded;
    end
    reconstructed_signals = cell(1, num_sources);
    for i = 1:num_sources
        separated_mag = output(:, :, i);
        original_phase = angle(S);
        separated_stft_coeff = separated_mag .* exp(1j * original_phase);
        win=hamming(512);
        reconstructed_signals{i} = real(istft(separated_stft_coeff, fs, 'Window', win,'OverlapLength', overlap, 'FFTLength', nfft));
    end
end