import pandas as pd

def load_log_csv(filepath):
    """
    Robust CSV loader for Blackbox logs.
    """
    try:
        # Try reading normally
        df = pd.read_csv(filepath)
        
        # Check if header is correct, sometimes header is on row 1
        required = ['time']
        if not any(col in df.columns for col in required):
            df = pd.read_csv(filepath, skiprows=1)
            
        return df
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")