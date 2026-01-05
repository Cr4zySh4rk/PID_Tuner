import os
import pandas as pd
from .features import segment_and_extract
from .utils import load_log_csv

def build_dataset(data_dir):
    all_features = []
    print(f"Scanning dataset at: {data_dir}")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.csv'): continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, data_dir)
            parts = rel_path.split(os.sep)
            
            fault_type = parts[0] 
            target_axis = parts[1] if len(parts) > 1 else None
            
            try:
                df = load_log_csv(file_path)
                
                # Using Higher Overlap (0.75) for Data Augmentation
                window_samples = segment_and_extract(df, window_size_sec=0.8, overlap=0.75)
                
                for sample in window_samples:
                    axis_name = sample['axis_name']
                    
                    add_sample = False
                    label = "Unknown"

                    if fault_type.lower() == 'good':
                        add_sample = True
                        label = "Good"
                    elif target_axis and axis_name == target_axis:
                        add_sample = True
                        label = fault_type
                    
                    if add_sample:
                        s = sample.copy()
                        del s['axis_name']
                        s['label'] = label
                        all_features.append(s)
                        
            except Exception as e:
                pass

    return pd.DataFrame(all_features)