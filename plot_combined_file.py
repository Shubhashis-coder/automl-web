import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

# =========================
# SETTINGS
# =========================
file_path = "combine.xlsx"
output_dir = "analysis_outputs"

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# =========================
# SAFE NAME FUNCTION
# =========================
def safe_name(name):
    return re.sub(r"[^\w\-_. ]", "_", str(name))

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(file_path)

# Clean column names (IMPORTANT)
df.columns = df.columns.str.strip()

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# =========================
# SUMMARY
# =========================
print("\n===== SUMMARY STATISTICS =====")
summary = df.describe()
print(summary)

print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
print(missing)

# =========================
# UNIQUE VALUES
# =========================
print("\n===== UNIQUE VALUE COUNTS =====")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# =========================
# CORRELATION
# =========================
print("\n===== CORRELATION MATRIX =====")
corr = df.corr(numeric_only=True)
print(corr)

corr.to_excel(os.path.join(output_dir, "correlation_matrix.xlsx"))

# =========================
# TARGET COLUMN
# =========================
target = "CH4_BC_std"

# =========================
# 1. TARGET DISTRIBUTION
# =========================
plt.figure()
df[target].hist(bins=40)
plt.title("Methane Yield Distribution")
plt.xlabel(target)
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "1_target_distribution.png"))
plt.close()

# =========================
# 2. FEATURE DISTRIBUTIONS
# =========================
for col in df.select_dtypes(include=np.number).columns:
    if col == target:
        continue
    
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Distribution: {col}")
    
    filename = f"dist_{safe_name(col)}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# =========================
# 3. CORRELATION HEATMAP
# =========================
plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2_correlation_heatmap.png"))
plt.close()

# =========================
# 4. TARGET VS TOP FEATURES
# =========================
top_features = corr[target].abs().sort_values(ascending=False).index[1:6]

for col in top_features:
    plt.figure()
    plt.scatter(df[col], df[target])
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target}")
    
    filename = f"scatter_{safe_name(col)}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# =========================
# 5. PAIRPLOT (LIMITED)
# =========================
sample_cols = list(top_features[:4]) + [target]

pd.plotting.scatter_matrix(df[sample_cols], figsize=(8,8))
plt.savefig(os.path.join(output_dir, "3_pairplot.png"))
plt.close()

# =========================
# 6. DATASET COMPOSITION
# =========================
if "source" in df.columns:
    print("\n===== SOURCE DISTRIBUTION =====")
    print(df["source"].value_counts())

    plt.figure()
    df["source"].value_counts().plot(kind="bar")
    plt.title("Dataset Composition")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "4_dataset_composition.png"))
    plt.close()

# =========================
# SAVE STRUCTURE JSON
# =========================
structure = {
    "columns": df.columns.tolist(),
    "shape": df.shape,
    "summary": summary.to_dict(),
    "missing_values": missing.to_dict(),
    "correlation_with_target": corr[target].to_dict()
}

with open(os.path.join(output_dir, "dataset_structure.json"), "w") as f:
    json.dump(structure, f, indent=4)

print("\n✅ ALL ANALYSIS COMPLETE")
print(f"📁 Outputs saved in folder: {output_dir}")
