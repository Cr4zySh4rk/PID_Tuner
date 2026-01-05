import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from .preprocess import build_dataset

def train_model(data_dir, model_save_path):
    print("Building dataset (Enhanced Features + Augmentation)...")
    df = build_dataset(data_dir)
    
    if df.empty:
        print("No valid data found.")
        return

    X = df.drop(columns=['label'])
    y = df['label']
    
    print(f"Total Samples: {len(X)}")
    print(f"Classes:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [20, None],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    # Using 'f1_macro' to balance performance across all fault types
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_clf = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}")
    
    print("\n--- Final Test Set Evaluation ---")
    preds = best_clf.predict(X_test)
    print(classification_report(y_test, preds))
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_clf, model_save_path)
    print(f"Model saved to {model_save_path}")