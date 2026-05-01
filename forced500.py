import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("imputed_MICE_RF.xlsx")

# 🔥 CRITICAL FIX (same as script 1)
df.columns = (
    df.columns
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(" ", "_")
)

target = "CH4_BC_std"

X = df.drop(columns=[target])
y = df[target]

# =========================
# OVERFITTED MODEL
# =========================
model_perfect = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=10,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0,
    reg_alpha=0,
    random_state=42
)

model_perfect.fit(X, y)

# =========================
# EVALUATION
# =========================
y_pred = model_perfect.predict(X)
r2 = r2_score(y, y_pred)

print("\n===== FORCED MODEL =====")
print("R2 (same data):", r2)

# =========================
# SYNTHETIC DATA
# =========================
X_new = X.sample(500, replace=True).reset_index(drop=True)

y_new = model_perfect.predict(X_new)

# tiny noise
y_new = y_new + np.random.normal(0, 2, size=500)

y_new = np.clip(y_new, 0, 850)

synthetic_perfect = X_new.copy()
synthetic_perfect[target] = y_new

synthetic_perfect.to_csv("synthetic_forced_500.csv", index=False)

print("🔥 Done!")
