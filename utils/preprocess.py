import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def apply_range_imputation(df, col, lower=None, upper=None, strategy='mean'):
    """Impute values outside specified range."""
    if col not in df.columns:
        return df
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        return df
    
    mask = pd.Series(False, index=series.index)
    if lower is not None:
        mask |= (series < lower)
    if upper is not None:
        mask |= (series > upper)
    
    if mask.any():
        if strategy == 'mean':
            fill_val = series[~mask].mean()
        elif strategy == 'median':
            fill_val = series[~mask].median()
        elif strategy == 'mode':
            valid_mode = series[~mask].mode()
            fill_val = valid_mode[0] if len(valid_mode) > 0 else np.nan
        else:
            fill_val = np.nan
        df.loc[mask, col] = fill_val
    
    return df

def engineer_ad_features(df):
    """
    Create Anaerobic Digestion specific features.
    """
    df_new = df.copy()
    
    # VS/TS ratio (biodegradability indicator)
    if 'VS' in df_new.columns and 'TS' in df_new.columns:
        df_new['VS_TS_Ratio'] = df_new['VS'] / (df_new['TS'] + 1e-6)
    
    # C/N ratio (optimal around 20-30:1)
    if 'Carbon' in df_new.columns and 'Nitrogen' in df_new.columns:
        df_new['CN_Ratio'] = df_new['Carbon'] / (df_new['Nitrogen'] + 1e-6)
    
    # OLR * HRT interaction
    if 'OLR' in df_new.columns and 'HRT' in df_new.columns:
        df_new['OLR_HRT_Product'] = df_new['OLR'] * df_new['HRT']
    
    # Biochar dose relative to VS
    if 'Biochar_Dose' in df_new.columns and 'VS' in df_new.columns:
        df_new['Biochar_VS_Ratio'] = df_new['Biochar_Dose'] / (df_new['VS'] + 1e-6)
    
    # VFA/Alkalinity ratio (system stability)
    if 'VFA' in df_new.columns and 'Alkalinity' in df_new.columns:
        df_new['VFA_Alk_Ratio'] = df_new['VFA'] / (df_new['Alkalinity'] + 1e-6)
    
    # Temperature-pH interaction
    if 'Temperature' in df_new.columns and 'pH' in df_new.columns:
        df_new['Temp_pH_Interaction'] = df_new['Temperature'] * df_new['pH']
    
    # Log transformations for highly skewed variables
    for col in ['Biochar_Dose', 'OLR', 'VFA']:
        if col in df_new.columns:
            df_new[f'Log_{col}'] = np.log1p(df_new[col])
    
    # Square root transformations
    for col in ['Surface_Area', 'Pore_Volume']:
        if col in df_new.columns:
            df_new[f'Sqrt_{col}'] = np.sqrt(df_new[col] + 1e-6)
    
    return df_new

def engineer_biochar_features(df):
    """
    Create biochar-specific features.
    """
    df_new = df.copy()
    
    # Fixed carbon calculation
    if all(col in df_new.columns for col in ['Ash_Content', 'Volatile_Matter']):
        df_new['Fixed_Carbon'] = 100 - df_new['Ash_Content'] - df_new['Volatile_Matter']
        df_new['Biochar_Stability'] = df_new['Fixed_Carbon'] / (df_new['Volatile_Matter'] + 1e-6)
    
    # Surface area-pore volume interaction
    if 'Surface_Area' in df_new.columns and 'Pore_Volume' in df_new.columns:
        df_new['SA_PV_Product'] = df_new['Surface_Area'] * df_new['Pore_Volume']
    
    # pH buffering capacity proxy
    if 'pH_H2O' in df_new.columns and 'CEC' in df_new.columns:
        df_new['pH_CEC_Index'] = df_new['pH_H2O'] * df_new['CEC']
    
    return df_new

def clean_data(df, target_col, null_strategy='mean', outlier_cap=True, 
               encode_cat=True, add_ad_features=True, add_biochar_features=True):
    """
    Complete data cleaning and feature engineering pipeline.
    
    Returns: (cleaned_df, feature_columns_list)
    """
    df_clean = df.copy()
    
    # 1. Handle missing values in features
    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    if null_strategy == 'drop':
        df_clean = df_clean.dropna()
    else:
        for col in feature_cols:
            if df_clean[col].isnull().sum() == 0:
                continue
            
            if null_strategy == 'mode':
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
            elif null_strategy in ['mean', 'median']:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_val = df_clean[col].mean() if null_strategy == 'mean' else df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                else:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
            elif null_strategy == 'ffill':
                df_clean[col] = df_clean[col].fillna(method='ffill')
            elif null_strategy == 'bfill':
                df_clean[col] = df_clean[col].fillna(method='bfill')
            elif null_strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].interpolate()
    
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_col])
    
    # 2. Outlier capping (IQR method)
    if outlier_cap:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)
    
    # 3. Encode categorical variables
    if encode_cat:
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    # 4. AD-specific feature engineering
    if add_ad_features:
        df_clean = engineer_ad_features(df_clean)
    
    # 5. Biochar-specific feature engineering
    if add_biochar_features:
        df_clean = engineer_biochar_features(df_clean)
    
    # 6. Final cleanup - drop any remaining NaN in features
    feature_cols = [c for c in df_clean.columns if c != target_col]
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=feature_cols)
    after = len(df_clean)
    if before > after:
        print(f"Dropped {before - after} rows with remaining NaN values.")
    
    return df_clean, feature_cols