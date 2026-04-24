import os
import joblib
import zipfile
import tempfile
import streamlit as st

MODEL_DIR = "models"

def save_model_pipeline(model, scaler, feature_cols, target_col, model_name):
    """
    Save model, scaler, and metadata to disk.
    Returns: path to saved file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    
    pipeline = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'model_name': model_name
    }
    
    joblib.dump(pipeline, path)
    return path

def load_model_pipeline(path):
    """
    Load saved pipeline from disk.
    Returns: dict with model, scaler, feature_cols, target_col, model_name
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    pipeline = joblib.load(path)
    return pipeline

def save_model_zip(models_dict, scaler, feature_cols, target_col):
    """
    Save all models as a ZIP file for download.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save each model
        for name, model in models_dict.items():
            joblib.dump(model, os.path.join(tmpdir, f"{name}_model.pkl"))
        
        # Save scaler and metadata
        joblib.dump(scaler, os.path.join(tmpdir, "scaler.pkl"))
        joblib.dump({
            'feature_cols': feature_cols,
            'target_col': target_col
        }, os.path.join(tmpdir, "metadata.pkl"))
        
        # Create ZIP
        zip_path = os.path.join(tmpdir, "models_package.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for f in os.listdir(tmpdir):
                if f != "models_package.zip":
                    zf.write(os.path.join(tmpdir, f), arcname=f)
        
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
    
    return zip_data

def load_model_from_zip(zip_file):
    """
    Load models from uploaded ZIP file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)
        
        # Load metadata
        metadata = joblib.load(os.path.join(tmpdir, "metadata.pkl"))
        
        # Load scaler
        scaler = joblib.load(os.path.join(tmpdir, "scaler.pkl"))
        
        # Load all models
        models = {}
        for f in os.listdir(tmpdir):
            if f.endswith("_model.pkl"):
                model_name = f.replace("_model.pkl", "")
                models[model_name] = joblib.load(os.path.join(tmpdir, f))
        
        return models, scaler, metadata['feature_cols'], metadata['target_col']