import pandas as pd
import streamlit as st

def load_uploaded_file(uploaded_file):
    """
    Load uploaded file (Excel or CSV) into DataFrame.
    Returns (DataFrame, filename) or (None, None) on error.
    """
    if uploaded_file is None:
        return None, None
    
    filename = uploaded_file.name
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"Unsupported file format: {filename}")
            return None, None
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df, filename
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def detect_column_types(df):
    """
    Classify columns as numeric, categorical, or datetime.
    """
    col_info = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info[col] = 'datetime'
        else:
            col_info[col] = 'categorical'
    return col_info

def get_preview_data(df, n_rows=100):
    """
    Return first n_rows of dataframe for preview.
    """
    return df.head(n_rows)