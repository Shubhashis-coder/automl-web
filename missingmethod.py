"""
Multi‑Method Missing Data Imputation for AD‑Biochar Dataset
- Loads raw data (with missing values and messy strings)
- Cleans to numeric (ranges → mid, <, > removed)
- Applies 6 imputation methods:
    1. Linear Interpolation (simple, per column)
    2. Linear Interpolation (sorted by BC production temperature)
    3. MICE (Random Forest)
    4. kNN (k=5)
    5. Median
    6. Predictive Mean Matching (PMM) – NEW
- Clips to user ranges
- Saves each imputed dataset as separate Excel file
- Compares results (distributions, correlations)
- Recommends best method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. Load raw data (with missing values)
# ============================================
file_path = "raw_data_468rows.xlsx"   # <-- CHANGE to your raw (un‑imputed) file
df_raw = pd.read_excel(file_path)

# Columns to impute (from your list)
impute_cols = [
    'AD - Temperature [°C]',
    'ISR_VS_std',
    'AD substrate - Volatile Solids [%TS] - mean',
    'AD substrate - pH [-] - mean',
    'BC - Dose [g /L]',
    'BC - particle size [mm] - mean',
    'BC - BET surface area [m^2/g] - mean',
    'BC - Total pore volume [cm^3/g] - mean',
    'BC - Average pore diameter [nm] - mean',
    'BC - pH [-] - mean',
    'BC - Electrical conductivity [μS/cm] - mean',
    'BC - Total Carbon [% wt] - mean',
    'BC - H content [% wt] - mean',
    'BC - N content [% wt] - mean',
    'BC - Electron donating capacity [m-mol eq/g] - mean',
    'BC - Electron accepting capacity [m-mol eq/g] - mean'
]

# Keep existing columns
impute_cols = [c for c in impute_cols if c in df_raw.columns]
print(f"Columns to impute: {len(impute_cols)}")

# ============================================
# 2. Clean messy strings to numeric (no imputation yet)
# ============================================
def clean_numeric_column(series):
    def clean_single(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # Handle ranges: "3.0-3.1" -> mid
        if '-' in s and not s.startswith('-'):
            parts = s.split('-')
            if len(parts) == 2:
                try:
                    low = float(parts[0])
                    high = float(parts[1])
                    return (low + high) / 2.0
                except:
                    pass
        # Remove leading < or >
        if s.startswith('<') or s.startswith('>'):
            s = s[1:]
        # Extract first number
        match = re.search(r'(\d+\.?\d*)', s)
        if match:
            return float(match.group(1))
        return np.nan
    return series.apply(clean_single)

df_clean = df_raw.copy()
for col in impute_cols:
    df_clean[col] = clean_numeric_column(df_clean[col])
print("Data cleaning completed (messy strings → numeric, NaN kept).")

# ============================================
# 3. Define allowed ranges (clipping after imputation)
# ============================================
ranges = {
    'AD - Temperature [°C]': (20, 55),
    'ISR_VS_std': (0.06, 4.2),
    'AD substrate - Volatile Solids [%TS] - mean': (49, 99),
    'AD substrate - pH [-] - mean': (4, 9),
    'BC - Dose [g /L]': (1, 50),
    'BC - particle size [mm] - mean': (0.005, 5),
    'BC - BET surface area [m^2/g] - mean': (0.49, 800),
    'BC - Total pore volume [cm^3/g] - mean': (0.01, 0.4),
    'BC - Average pore diameter [nm] - mean': (0.9, 80),
    'BC - pH [-] - mean': (5, 12),
    'BC - Electrical conductivity [μS/cm] - mean': (0.1, 30000),
    'BC - Total Carbon [% wt] - mean': (11, 85),
    'BC - H content [% wt] - mean': (0.2, 8),
    'BC - N content [% wt] - mean': (0.01, 7),
    'BC - Electron donating capacity [m-mol eq/g] - mean': (0.01, 0.6),
    'BC - Electron accepting capacity [m-mol eq/g] - mean': (0.2, 1)
}
ranges = {k: v for k, v in ranges.items() if k in impute_cols}

# ============================================
# 4. Helper: clip to ranges
# ============================================
def clip_to_ranges(df, ranges):
    df_clipped = df.copy()
    for col, (low, high) in ranges.items():
        if col in df_clipped.columns:
            df_clipped[col] = pd.to_numeric(df_clipped[col], errors='coerce')
            df_clipped[col] = df_clipped[col].clip(low, high)
    return df_clipped

# ============================================
# 5. Imputation functions
# ============================================

# 5a. Simple linear interpolation (per column, no sorting)
def impute_linear_simple(df, cols):
    df_imp = df.copy()
    for col in cols:
        if df_imp[col].isnull().any():
            df_imp[col] = df_imp[col].interpolate(method='linear', limit_direction='both')
            df_imp[col] = df_imp[col].fillna(method='ffill').fillna(method='bfill')
    return df_imp

# 5b. Linear interpolation after sorting by BC production temperature
def impute_linear_sorted(df, cols):
    df_sorted = df.copy()
    if 'BC production - Temperature [° C]' in df.columns:
        df_sorted = df_sorted.sort_values('BC production - Temperature [° C]')
    for col in cols:
        if df_sorted[col].isnull().any():
            df_sorted[col] = df_sorted[col].interpolate(method='linear', limit_direction='both')
            df_sorted[col] = df_sorted[col].fillna(method='ffill').fillna(method='bfill')
    return df_sorted

# 5c. MICE with Random Forest
def impute_mice(df, cols, n_estimators=50, max_iter=10):
    df_imp = df.copy()
    X = df_imp[cols]
    # Fill fully missing columns with median to avoid crash
    for col in cols:
        if X[col].isnull().all():
            X[col] = X[col].fillna(X[col].median())
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        max_iter=max_iter,
        random_state=42
    )
    X_imputed = imputer.fit_transform(X)
    df_imp[cols] = X_imputed
    return df_imp

# 5d. kNN imputation
def impute_knn(df, cols, n_neighbors=5):
    df_imp = df.copy()
    X = df_imp[cols]
    for col in cols:
        if X[col].isnull().all():
            X[col] = X[col].fillna(X[col].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed_scaled = imputer.fit_transform(X_scaled)
    X_imputed = scaler.inverse_transform(X_imputed_scaled)
    df_imp[cols] = X_imputed
    return df_imp

# 5e. Median imputation (baseline)
def impute_median(df, cols):
    df_imp = df.copy()
    for col in cols:
        if df_imp[col].isnull().any():
            median_val = df_imp[col].median()
            df_imp[col].fillna(median_val, inplace=True)
    return df_imp

# 5f. Predictive Mean Matching (PMM)
def impute_pmm(df, cols, n_neighbors=5, n_estimators=50):
    """
    Predictive Mean Matching using Random Forest.
    For each missing value, train a model on complete cases, predict for missing,
    find k observed donors with closest predictions, randomly select one donor's actual value.
    """
    df_imp = df.copy()
    # Iterate over columns (like MICE but simpler: one pass)
    for col in cols:
        missing_mask = df_imp[col].isna()
        if missing_mask.sum() == 0:
            continue
        # Separate observed and missing rows
        observed = ~missing_mask
        X_train = df_imp.loc[observed].drop(columns=[col])
        y_train = df_imp.loc[observed, col]
        X_missing = df_imp.loc[missing_mask].drop(columns=[col])
        
        if len(X_train) < n_neighbors:
            # Not enough donors, fallback to median
            median_val = df_imp[col].median()
            df_imp.loc[missing_mask, col] = median_val
            continue
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predict for all rows (to get distances in prediction space)
        X_all = df_imp.drop(columns=[col])
        pred_all = rf.predict(X_all)
        pred_observed = pred_all[observed]
        pred_missing = pred_all[missing_mask]
        
        # For each missing, find k nearest observed donors by prediction
        for i, idx in enumerate(df_imp.index[missing_mask]):
            dist = np.abs(pred_observed - pred_missing[i])
            k = min(n_neighbors, len(dist))
            nearest_indices = np.argsort(dist)[:k]
            donor_values = y_train.iloc[nearest_indices].values
            chosen = np.random.choice(donor_values)
            df_imp.loc[idx, col] = chosen
    return df_imp

# ============================================
# 6. Show missing values before imputation
# ============================================
print("\nMissing values BEFORE imputation:")
missing_before = df_clean[impute_cols].isnull().sum()
missing_before = missing_before[missing_before > 0]
if len(missing_before) > 0:
    print(missing_before)
else:
    print("  No missing values found. All methods will return original data.")

# ============================================
# 7. Apply all methods and save each output
# ============================================
methods = {
    'Linear_Simple': impute_linear_simple,
    'Linear_Sorted': impute_linear_sorted,
    'MICE_RF': impute_mice,
    'kNN_k5': impute_knn,
    'Median': impute_median,
    'PMM': impute_pmm      # NEW
}

imputed_dfs = {}
print("\nApplying imputation methods and saving Excel files...")
for name, func in methods.items():
    print(f"  {name}...")
    df_imp = func(df_clean, impute_cols)
    df_imp = clip_to_ranges(df_imp, ranges)
    imputed_dfs[name] = df_imp
    # Save individual Excel file
    output_file = f"imputed_{name}.xlsx"
    df_imp.to_excel(output_file, index=False)
    print(f"    Saved: {output_file}")

# ============================================
# 8. Compare distributions (histograms)
# ============================================
example_cols = ['BC - BET surface area [m^2/g] - mean', 'BC - Total Carbon [% wt] - mean', 
                'BC - H content [% wt] - mean', 'AD - Temperature [°C]']
example_cols = [c for c in example_cols if c in impute_cols]

fig, axes = plt.subplots(len(example_cols), len(methods)+1, figsize=(20, len(example_cols)*4))
for i, col in enumerate(example_cols):
    orig_vals = df_clean[col].dropna()
    axes[i, 0].hist(orig_vals, bins=20, alpha=0.7, color='gray')
    axes[i, 0].set_title(f'{col} (original, non-missing)')
    axes[i, 0].set_ylabel('Frequency')
    for j, (name, df_imp) in enumerate(imputed_dfs.items()):
        axes[i, j+1].hist(df_imp[col], bins=20, alpha=0.7)
        axes[i, j+1].set_title(f'{name}')
plt.tight_layout()
plt.savefig('imputation_histograms_comparison.png', dpi=150)
plt.close()
print("✓ Saved: imputation_histograms_comparison.png")

# ============================================
# 9. Boxplot comparison
# ============================================
col_box = 'BC - BET surface area [m^2/g] - mean' if 'BC - BET surface area [m^2/g] - mean' in impute_cols else impute_cols[0]
data_to_plot = [df_clean[col_box].dropna()] + [df[col_box] for df in imputed_dfs.values()]
labels = ['Original (non-missing)'] + list(imputed_dfs.keys())
plt.figure(figsize=(12,6))
plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title(f'Boxplot comparison: {col_box}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('imputation_boxplots.png', dpi=150)
plt.close()
print("✓ Saved: imputation_boxplots.png")

# ============================================
# 10. Correlation matrix differences
# ============================================
corr_orig = df_clean[impute_cols].corr()
corr_diffs = {}
for name, df_imp in imputed_dfs.items():
    corr_imp = df_imp[impute_cols].corr()
    diff = (corr_imp - corr_orig).abs().max().max()
    corr_diffs[name] = diff
    print(f"  {name}: max abs correlation difference = {diff:.4f}")

best_method = min(corr_diffs, key=corr_diffs.get)
fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.heatmap(corr_orig, ax=axes[0], cmap='coolwarm', center=0, annot=False)
axes[0].set_title('Original correlation (non-missing data)')
sns.heatmap(imputed_dfs[best_method][impute_cols].corr(), ax=axes[1], cmap='coolwarm', center=0, annot=False)
axes[1].set_title(f'{best_method} imputed correlation')
plt.tight_layout()
plt.savefig('imputation_best_correlation.png', dpi=150)
plt.close()
print(f"✓ Best method by correlation preservation: {best_method}")

# ============================================
# 11. Generate comparison report
# ============================================
with open('imputation_comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("IMPUTATION METHOD COMPARISON FOR AD-BIOCHAR DATASET\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total rows: {len(df_clean)}\n")
    f.write(f"Columns imputed: {len(impute_cols)}\n\n")
    f.write("Missing values before imputation:\n")
    if len(missing_before) > 0:
        f.write(missing_before.to_string())
    else:
        f.write("  None\n")
    f.write("\n\nCorrelation preservation (max absolute difference, lower is better):\n")
    for name, diff in sorted(corr_diffs.items(), key=lambda x: x[1]):
        f.write(f"  {name:20} : {diff:.4f}\n")
    f.write(f"\nRecommended method: {best_method}\n")
    f.write("Reason: Lowest distortion of correlation structure while respecting all ranges.\n")
    f.write("Note: PMM (Predictive Mean Matching) selects real observed donor values, making it highly realistic.\n")

print("✓ Saved: imputation_comparison_report.txt")
print("\nAll done. Individual Excel files saved for each method.")
