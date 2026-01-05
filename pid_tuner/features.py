import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

def extract_features_from_window(window_df, axis_idx, dt):
    """
    Extracts advanced features including P/I/D-specific spectral analysis.
    """
    # 1. Setup Column Names
    gyro_col = f'gyroADC[{axis_idx}]'
    setpoint_col = f'setpoint[{axis_idx}]'
    p_col = f'axisP[{axis_idx}]'
    i_col = f'axisI[{axis_idx}]'
    d_col = f'axisD[{axis_idx}]'
    
    if len(window_df) < 10: return None
    
    gyro = window_df[gyro_col].values
    setpoint = window_df[setpoint_col].values
    error = gyro - setpoint
    
    feats = {}
    feats['axis_idx'] = axis_idx
    
    # --- TIME DOMAIN ---
    feats['rmse'] = np.sqrt(np.mean(error**2))
    feats['p2p_gyro'] = np.ptp(gyro)
    feats['p2p_error'] = np.ptp(error)
    feats['std_error'] = np.std(error)
    
    # --- CONTROL ACTIVITY & RATIOS ---
    # We want to know: "Who is doing the work?"
    p_std = window_df[p_col].std() if p_col in window_df else 0
    i_std = window_df[i_col].std() if i_col in window_df else 0
    d_std = window_df[d_col].std() if d_col in window_df else 0
    
    total_activity = p_std + i_std + d_std + 1e-9
    feats['p_contribution'] = p_std / total_activity
    feats['i_contribution'] = i_std / total_activity
    feats['d_contribution'] = d_std / total_activity
    
    feats['p_std'] = p_std
    feats['d_std'] = d_std

    # --- OVERSHOOT DETECTOR ---
    # Look for sharp setpoint changes (Step inputs)
    sp_diff = np.diff(setpoint)
    max_step = np.max(np.abs(sp_diff))
    
    if max_step > 5: # If pilot moved sticks quickly
        # Find max error during this window
        max_err = np.max(np.abs(error))
        # Overshoot Ratio: How big was the error compared to the move?
        feats['overshoot_severity'] = max_err / (max_step + 1e-9)
    else:
        feats['overshoot_severity'] = 0

    # --- GENERIC FFT FUNCTION ---
    def get_spectral_features(signal, prefix):
        n = len(signal)
        fs = 1.0 / dt
        if n < 10: return {}
        
        yf = fft(signal)
        xf = fftfreq(n, dt)
        mag = np.abs(yf)[:n//2]
        freqs = xf[:n//2]
        mag[0] = 0 # No DC
        
        total_e = np.sum(mag) + 1e-9
        mag_norm = mag / total_e
        
        # Peak analysis
        peak_idx = np.argmax(mag)
        peak_freq = freqs[peak_idx]
        peak_val = mag[peak_idx]
        spectral_purity = peak_val / total_e # 1.0 = Pure Sine, 0.0 = White Noise

        def band(f_min, f_max):
            mask = (freqs >= f_min) & (freqs < f_max)
            return np.sum(mag[mask]) / total_e

        return {
            f'{prefix}_peak_freq': peak_freq,
            f'{prefix}_purity': spectral_purity,
            f'{prefix}_energy_low': band(0, 20),      # Control / I-term
            f'{prefix}_energy_mid': band(20, 100),    # P-term / Propwash
            f'{prefix}_energy_high': band(100, fs/2), # D-term / Noise
            f'{prefix}_total_power': total_e
        }

    # 1. Gyro FFT
    feats.update(get_spectral_features(gyro, 'gyro'))
    
    # 2. P-Term FFT (CRITICAL for P_High)
    if p_col in window_df:
        feats.update(get_spectral_features(window_df[p_col].values, 'pterm'))
        
    # 3. D-Term FFT (CRITICAL for D_High)
    if d_col in window_df:
        feats.update(get_spectral_features(window_df[d_col].values, 'dterm'))

    # 4. I-Term FFT (For I_High wobbles)
    if i_col in window_df:
        # I-term usually oscillates slow (< 10Hz)
        i_feats = get_spectral_features(window_df[i_col].values, 'iterm')
        feats['iterm_energy_low'] = i_feats.get('iterm_energy_low', 0)
        feats['iterm_total_power'] = i_feats.get('iterm_total_power', 0)

    # 5. Lag (Response Latency)
    if np.std(setpoint) > 5: 
        n = len(gyro)
        xcorr = correlate(setpoint, gyro, mode='same')
        lag_idx = np.argmax(xcorr)
        feats['lag_score'] = (lag_idx - n//2)
    else:
        feats['lag_score'] = 0

    return feats

def segment_and_extract(df, window_size_sec=0.8, overlap=0.75):
    """
    Slices log into windows. 
    overlap=0.75 -> More samples from the same data (Data Augmentation)
    """
    time_us = df['time'].values
    diffs = np.diff(time_us)
    valid_diffs = diffs[diffs > 0]
    dt = np.median(valid_diffs) * 1e-6 if len(valid_diffs) > 0 else 0.002
    
    window_len = int(window_size_sec / dt)
    step_len = int(window_len * (1 - overlap)) # Smaller step = More windows
    
    samples = []
    
    for start in range(0, len(df) - window_len, step_len):
        end = start + window_len
        window_df = df.iloc[start:end]
        
        # --- FILTER: ACTIVE FLIGHT ---
        # Keep if: Motors ON AND (Sticks Moving OR Gyro Shaking)
        if 'motor[0]' in df.columns and window_df['motor[0]'].mean() < 100:
            continue

        is_active = False
        
        # 1. Stick Movement?
        sp_activity = 0
        for i in range(3):
            if f'setpoint[{i}]' in window_df:
                sp_activity += window_df[f'setpoint[{i}]'].std()
        if sp_activity > 5: is_active = True
            
        # 2. Shaking? (Even if hovering)
        if not is_active:
            for i in range(3):
                if window_df[f'gyroADC[{i}]'].std() > 20: 
                    is_active = True
                    break
        
        if not is_active: continue
        
        for axis_idx, axis_name in [(0, 'Roll'), (1, 'Pitch'), (2, 'Yaw')]:
            feats = extract_features_from_window(window_df, axis_idx, dt)
            if feats:
                feats['axis_name'] = axis_name
                samples.append(feats)
                
    return samples