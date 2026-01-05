import joblib
import pandas as pd
import numpy as np
from collections import Counter
from .features import segment_and_extract
from .utils import load_log_csv
from .rules import get_dynamic_recommendation  # Changed import

def diagnose_log(file_path, model_path):
    # ... (Load resources and CSV as before) ...
    try:
        clf = joblib.load(model_path)
    except:
        return [{"axis": "System", "diagnosis": "Error", "advice": "Model not found."}]

    try:
        df = load_log_csv(file_path)
    except Exception as e:
        return [{"axis": "System", "diagnosis": "Error", "advice": f"CSV Error: {e}"}]

    # Slice windows
    windows = segment_and_extract(df, window_size_sec=0.8, overlap=0.75)
    
    if not windows:
        return [{"axis": "System", "diagnosis": "Insufficient Data", "advice": "Log too short or inactive."}]

    axis_data = {'Roll': [], 'Pitch': [], 'Yaw': []}
    for w in windows:
        name = w['axis_name']
        if name in axis_data:
            w_feat = w.copy()
            del w_feat['axis_name']
            axis_data[name].append(w_feat)

    results = []

    for axis_name, feats_list in axis_data.items():
        if not feats_list: 
            results.append({"axis": axis_name, "diagnosis": "N/A", "advice": "No active data."})
            continue
        
        feat_df = pd.DataFrame(feats_list)
        
        # --- Handle Missing Columns (Safety Fix) ---
        if hasattr(clf, "feature_names_in_"):
            expected_cols = clf.feature_names_in_
            missing_cols = set(expected_cols) - set(feat_df.columns)
            for c in missing_cols: feat_df[c] = 0.0
            feat_df = feat_df[expected_cols]

        # Predict
        preds = clf.predict(feat_df)
        
        # Vote Logic
        vote_counts = Counter(preds)
        most_common_label, count = vote_counts.most_common(1)[0]
        confidence = count / len(preds)
        
        # Transient P_High check
        if "P_High" in preds:
            p_high_count = list(preds).count("P_High")
            if p_high_count > len(preds) * 0.15: 
                most_common_label = "P_High"
                confidence = p_high_count / len(preds)

        # --- NEW: SEVERITY CALCULATOR ---
        # We calculate how "intense" the fault is based on feature magnitude
        severity_score = 0.0
        
        if most_common_label == "P_High":
            # How pure is the oscillation? (0.0 to 1.0)
            if 'pterm_purity' in feat_df:
                severity_score = feat_df[preds == "P_High"]['pterm_purity'].mean()
        
        elif most_common_label in ["P_Low", "I_Low"]:
            # How bad is the overshoot/lag?
            if 'overshoot_severity' in feat_df:
                 # filter for windows where overshoot actually happened
                bad_windows = feat_df[feat_df['overshoot_severity'] > 0]
                if len(bad_windows) > 0:
                    severity_score = bad_windows['overshoot_severity'].mean()
        
        elif most_common_label == "D_High":
            # How noisy is the D-term relative to Gyro?
            if 'd_gyro_ratio' in feat_df:
                severity_score = feat_df['d_gyro_ratio'].mean()

        # Get Smart Advice
        advice = get_dynamic_recommendation(most_common_label, axis_name, severity_score)
        
        # Oscillation Flag
        is_oscillating = False
        if 'gyro_energy_high' in feat_df.columns:
            if any(feat_df['gyro_energy_high'] > 0.35): is_oscillating = True
        
        results.append({
            'axis': axis_name,
            'diagnosis': most_common_label,
            'confidence': f"{confidence:.2f}",
            'oscillation': is_oscillating,
            'advice': advice
        })
        
    return results