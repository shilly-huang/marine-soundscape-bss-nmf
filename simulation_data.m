fs = 44100;  
duration = 60; 
N = duration * fs;

%% 1. DS
sweep_duration = 1;
freq_bands = [1000, 850;  
              750,  600;  
              500,  350];
t_sweep = 0:1/fs:sweep_duration-1/fs;
ds_signal = zeros(1, length(t_sweep));
for i = 1:size(freq_bands, 1)
    f_start = freq_bands(i, 1);
    f_end = freq_bands(i, 2);
    sweep_single = chirp(t_sweep, f_start, sweep_duration, f_end, 'linear');
    env_sweep = tukeywin(length(sweep_single), 0.3)';
    sweep_single = sweep_single .* env_sweep;
    ds_signal = ds_signal + sweep_single;
end
ds_signal = ds_signal / max(abs(ds_signal)) * 0.3;
%% 2. LT
lt_duration = 1; 
t_lt = 0:1/fs:lt_duration-1/fs;
lt_freq = 75; 
lt_signal = sin(2*pi*lt_freq*t_lt);
env_lt = tukeywin(length(lt_signal), 0.1)';
lt_signal = lt_signal .* env_lt;
lt_signal = lt_signal / max(abs(lt_signal)) * 0.3;
%% 3. LDS
lds_duration = 1; 
t_lds = 0:1/fs:lds_duration-1/fs;
lds_signal = chirp(t_lds, 200, lds_duration, 50, 'linear');
env_lds = tukeywin(length(lds_signal), 0.2)';
lds_signal = lds_signal .* env_lds;
lds_signal = lds_signal / max(abs(lds_signal)) * 0.3;
%% 4. RPS
rps_pulses = 10; 
rps_interval = 0.15; 
pulse_duration = 0.01; 
rps_start_freq = 500; 
rps_end_freq = 900; 
t_pulse = 0:1/fs:pulse_duration-1/fs;
rps_pulse = chirp(t_pulse, rps_start_freq, pulse_duration, rps_end_freq, 'linear');
pulse_env = gausswin(length(rps_pulse))';
rps_pulse = rps_pulse .* pulse_env;
rps_signal = zeros(1, N);
for i = 1:rps_pulses
    start_time = (i-1) * rps_interval;
    start_idx = round(start_time * fs) + 1;
    end_idx = start_idx + length(rps_pulse) - 1;
    
    if end_idx <= N
        rps_signal(start_idx:end_idx) = rps_signal(start_idx:end_idx) + rps_pulse;
    end
end
rps_signal = rps_signal / max(abs(rps_signal)) * 0.7;
%% 5. PS
ps_pulses = 10; 
ps_center_freq = 200; 
pulse_duration = 0.05; 
t_pulse = 0:1/fs:pulse_duration-1/fs;
ps_pulse = sin(2*pi*ps_center_freq*t_pulse);
pulse_env = gausswin(length(ps_pulse))';
ps_pulse = ps_pulse .* pulse_env;
ps_signal = zeros(1, N);
pulse_times = cumsum(0.1 + 0.6*rand(1, ps_pulses));
for i = 1:ps_pulses
    if pulse_times(i) < duration - pulse_duration
        start_idx = round(pulse_times(i) * fs) + 1;
        end_idx = start_idx + length(ps_pulse) - 1;
        ps_signal(start_idx:end_idx) = ps_signal(start_idx:end_idx) + ps_pulse;
    end
end
ps_signal = ps_signal / max(abs(ps_signal)) * 0.7;
%% 6. LPS
lps_pulses = 10;
lps_center_freq = 50;
pulse_duration = 0.05;
t_pulse = 0:1/fs:pulse_duration-1/fs;
lps_pulse = sin(2*pi*lps_center_freq*t_pulse);
pulse_env = gausswin(length(lps_pulse))';
lps_pulse = lps_pulse .* pulse_env;
lps_signal = zeros(1, N);
pulse_times = cumsum(0.1 + 0.6*rand(1, lps_pulses));
for i = 1:lps_pulses
    if pulse_times(i) < duration - pulse_duration
        start_idx = round(pulse_times(i) * fs) + 1;
        end_idx = start_idx + length(lps_pulse) - 1;
        lps_signal(start_idx:end_idx) = lps_signal(start_idx:end_idx) + lps_pulse;
    end
end
lps_signal = lps_signal / max(abs(lps_signal)) * 0.7;
%% 
signals = {ds_signal, lt_signal, lds_signal, rps_signal, ps_signal, lps_signal};
signal_names = {'DS', 'LT', 'LDS', 'RPS', 'PS', 'LPS'};
for i = 1:length(signals)
    if max(abs(signals{i})) > 0
        if ismember(signal_names{i}, {'DS', 'LDS'})
            signals{i} = signals{i} / max(abs(signals{i})) * 0.3;
        else
            signals{i} = signals{i} / max(abs(signals{i})) * 0.7;
        end
    end
end
%% 
total_duration = 60; 
N_total = total_duration * fs;
composite_signal = zeros(1, N_total);
ds_solo = zeros(1, N_total);  
lt_solo = zeros(1, N_total);    
lds_solo = zeros(1, N_total); 
rps_solo = zeros(1, N_total);  
ps_solo = zeros(1, N_total);   
lps_solo = zeros(1, N_total); 
ds_segment = ds_signal;
lt_segment = lt_signal;
[lds_start, lds_end] = findNonZeroSegment(lds_signal);
lds_segment = lds_signal(lds_start:lds_end);
[rps_start, rps_end] = findNonZeroSegment(rps_signal);
rps_segment = rps_signal(rps_start:rps_end);
[ps_start, ps_end] = findNonZeroSegment(ps_signal);
ps_segment = ps_signal(ps_start:ps_end);
[lps_start, lps_end] = findNonZeroSegment(lps_signal);
lps_segment = lps_signal(lps_start:lps_end);
ds_duration = length(ds_segment) / fs;
lt_duration = length(lt_segment) / fs;
lds_duration = length(lds_segment) / fs;
rps_duration = length(rps_segment) / fs;
ps_duration = length(ps_segment) / fs;
lps_duration = length(lps_segment) / fs;
current_time = 0;
signal_order = {'DS', 'LT', 'LDS', 'RPS', 'PS', 'LPS'};
signal_index = 1;
interval = 0.5; 
event_count = 0;

event_info = [];
while current_time < total_duration
    event_count = event_count + 1;
    
    switch signal_index
        case 1 % DS
            segment = ds_segment;
            seg_duration = ds_duration;            
        case 2 % LT
            segment = lt_segment;
            seg_duration = lt_duration;
        case 3 % LDS
            segment = lds_segment;
            seg_duration = lds_duration;            
        case 4 % RPS
            segment = rps_segment;
            seg_duration = rps_duration;           
        case 5 % PS
            segment = ps_segment;
            seg_duration = ps_duration;
        case 6 % LPS
            segment = lps_segment;
            seg_duration = lps_duration;
    end
    event_info = [event_info; event_count, signal_index, current_time, current_time + seg_duration];
    start_idx = round(current_time * fs) + 1;
    end_idx = start_idx + length(segment) - 1;
    
    if end_idx <= N_total
        composite_signal(start_idx:end_idx) = composite_signal(start_idx:end_idx) + segment;
    else
        available_length = N_total - start_idx + 1;
        composite_signal(start_idx:end) = composite_signal(start_idx:end) + segment(1:available_length);
        break;
    end
    
    switch signal_index
        case 1
            if end_idx <= N_total
                ds_solo(start_idx:end_idx) = ds_solo(start_idx:end_idx) + segment;
            else
                ds_solo(start_idx:end) = ds_solo(start_idx:end) + segment(1:available_length);
            end
        case 2
            if end_idx <= N_total
                lt_solo(start_idx:end_idx) = lt_solo(start_idx:end_idx) + segment;
            else
                lt_solo(start_idx:end) = lt_solo(start_idx:end) + segment(1:available_length);
            end
        case 3
            if end_idx <= N_total
                lds_solo(start_idx:end_idx) = lds_solo(start_idx:end_idx) + segment;
            else
                lds_solo(start_idx:end) = lds_solo(start_idx:end) + segment(1:available_length);
            end
        case 4
            if end_idx <= N_total
                rps_solo(start_idx:end_idx) = rps_solo(start_idx:end_idx) + segment;
            else
                rps_solo(start_idx:end) = rps_solo(start_idx:end) + segment(1:available_length);
            end
        case 5
            if end_idx <= N_total
                ps_solo(start_idx:end_idx) = ps_solo(start_idx:end_idx) + segment;
            else
                ps_solo(start_idx:end) = ps_solo(start_idx:end) + segment(1:available_length);
            end
        case 6
            if end_idx <= N_total
                lps_solo(start_idx:end_idx) = lps_solo(start_idx:end_idx) + segment;
            else
                lps_solo(start_idx:end) = lps_solo(start_idx:end) + segment(1:available_length);
            end
    end

    current_time = current_time + seg_duration + interval;
    signal_index = mod(signal_index, length(signal_order)) + 1; 
    
    if current_time >= total_duration
        break;
    end
end

composite_signal = composite_signal / max(abs(composite_signal)) * 0.8;
ds_solo = ds_solo / max(abs(ds_solo)) * 0.8;
lt_solo = lt_solo / max(abs(lt_solo)) * 0.8;
lds_solo = lds_solo / max(abs(lds_solo)) * 0.8;
rps_solo = rps_solo / max(abs(rps_solo)) * 0.8;
ps_solo = ps_solo / max(abs(ps_solo)) * 0.8;
lps_solo = lps_solo / max(abs(lps_solo)) * 0.8;

audiowrite('fish_sounds_composite_60s.wav', composite_signal, fs);
audiowrite('fish_DS_solo_60s.wav', ds_solo, fs);
audiowrite('fish_LT_solo_60s.wav', lt_solo, fs);
audiowrite('fish_LDS_solo_60s.wav', lds_solo, fs);
audiowrite('fish_RPS_solo_60s.wav', rps_solo, fs);
audiowrite('fish_PS_solo_60s.wav', ps_solo, fs);
audiowrite('fish_LPS_solo_60s.wav', lps_solo, fs);

event_table = array2table(event_info, 'VariableNames', {'EventID', 'SignalType', 'StartTime', 'EndTime'});
signal_names_table = {'DS', 'LT', 'LDS', 'RPS', 'PS', 'LPS'};
event_table.SignalName = signal_names_table(event_table.SignalType)';
writetable(event_table, 'event_timestamps_60s.csv');
