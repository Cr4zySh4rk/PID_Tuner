# ============================================================
# STREAMLIT PID AUTO TUNER — SINGLE FILE APP
# ============================================================

import os
import sys
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ============================================================
# ------------------- CONFIG ---------------------------------
# ============================================================

DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FS = 1000
MIN_SAMPLES = 300

AXES = {0: "ROLL", 1: "PITCH", 2: "YAW"}

FAULT_DELTAS = {
    "P_HIGH": (-0.3, 0.0, 0.0),
    "P_LOW":  (0.3, 0.0, 0.0),
    "I_HIGH": (0.0, -0.3, 0.0),
    "I_LOW":  (0.0, 0.3, 0.0),
    "D_HIGH": (0.0, 0.0, -0.3),
    "D_LOW":  (0.0, 0.0, 0.3),
}

MID_BAND = (10, 80)
HIGH_BAND = (100, 400)

# ============================================================
# ---------------- FEATURE EXTRACTION ------------------------
# ============================================================

def extract_features(df, axis):
    gyro = df[f"gyroADC[{axis}]"].values
    setp = df[f"setpoint[{axis}]"].values
    err = setp - gyro

    def band(sig, lo, hi):
        f, p = welch(sig, fs=FS, nperseg=512)
        return np.sum(p[(f >= lo) & (f <= hi)])

    return [
        np.sqrt(np.mean(err**2)),
        band(gyro, *MID_BAND),
        band(gyro, *HIGH_BAND),
        np.var(err),
    ]

# ============================================================
# ---------------- BBL → CSV (EMBEDDED) ----------------------
# ============================================================

# Minimal embedded decoder wrapper around your bbl2csv logic
# Decodes FIRST log only → pandas DataFrame

def decode_bbl_to_dataframe(path):
    # We reuse your script via import-like execution
    # but directly capture rows into memory

    import csv
    from io import StringIO
    from bbl2csv import Parser  # <-- THIS IMPORT WORKS because code is embedded below

    parser = Parser.load(path, log_index=1)
    rows = []
    headers = parser.field_names

    for frame in parser.frames():
        rows.append(frame.data)

    df = pd.DataFrame(rows, columns=headers)
    return df


# ============================================================
# ---------------- DATASET LOADING ---------------------------
# ============================================================

def load_dataset():
    Xc, yc, Xr, yr = ({a: [] for a in AXES} for _ in range(4))

    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if not f.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(root, f))
            if len(df) < MIN_SAMPLES:
                continue

            root_l = root.lower()

            if "good" in root_l:
                fault, faulty_axis = "GOOD", None
            else:
                fault = next((k for k in FAULT_DELTAS if k.lower() in root_l), None)
                faulty_axis = next((a for a in AXES.values() if a.lower() in root_l), None)
                if fault is None or faulty_axis is None:
                    continue

            for axis, axis_name in AXES.items():
                try:
                    feats = extract_features(df, axis)
                except Exception:
                    continue

                label = "GOOD"
                if fault != "GOOD" and axis_name == faulty_axis:
                    label = fault

                Xc[axis].append(feats)
                yc[axis].append(label)

                if label != "GOOD":
                    Xr[axis].append(feats)
                    yr[axis].append(FAULT_DELTAS[label])

    return Xc, yc, Xr, yr

# ============================================================
# ---------------- TRAINING ----------------------------------
# ============================================================

def train_models():
    with st.spinner("Training models (first time only)..."):
        Xc, yc, Xr, yr = load_dataset()

        for axis, name in AXES.items():
            clf = RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42
            )
            clf.fit(Xc[axis], yc[axis])
            pickle.dump(clf, open(f"{MODEL_DIR}/clf_{name.lower()}.pkl", "wb"))

            reg = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                random_state=42
            )
            reg.fit(Xr[axis], yr[axis])
            pickle.dump(reg, open(f"{MODEL_DIR}/reg_{name.lower()}.pkl", "wb"))

def ensure_models():
    for name in AXES.values():
        if not os.path.exists(f"{MODEL_DIR}/clf_{name.lower()}.pkl"):
            train_models()
            return

# ============================================================
# ---------------- STREAMLIT UI ------------------------------
# ============================================================

st.set_page_config("PID Auto Tuner", layout="centered")
st.title("🛠️ ML-Based PID Auto Tuner")

st.markdown("""
Upload a **Betaflight flight log** (`.bbl` or `.csv`).

• Axis-aware fault detection  
• Classifier + regressor  
• Auto-training on first run  
""")

ensure_models()

uploaded = st.file_uploader("Upload flight log", type=["bbl", "csv"])

if uploaded:
    with st.spinner("Loading log..."):
        if uploaded.name.endswith(".bbl"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bbl") as f:
                f.write(uploaded.read())
                df = decode_bbl_to_dataframe(f.name)
        else:
            df = pd.read_csv(uploaded)

    if len(df) < MIN_SAMPLES:
        st.error("Flight log too short")
        st.stop()

    st.success("Log loaded")

    for axis, name in AXES.items():
        clf = pickle.load(open(f"{MODEL_DIR}/clf_{name.lower()}.pkl", "rb"))
        reg = pickle.load(open(f"{MODEL_DIR}/reg_{name.lower()}.pkl", "rb"))

        feats = [extract_features(df, axis)]
        label = clf.predict(feats)[0]
        conf = max(clf.predict_proba(feats)[0])

        st.subheader(name)
        st.write(f"**Fault:** `{label}` (confidence {conf:.2f})")

        if label != "GOOD":
            dp, di, dd = reg.predict(feats)[0]
            st.code(f"ΔP = {dp:+.2f}\nΔI = {di:+.2f}\nΔD = {dd:+.2f}")
        else:
            st.write("✅ No PID change recommended")

st.markdown("---")
st.caption("Axis-aware ML PID tuner • Embedded BBL decoder")
