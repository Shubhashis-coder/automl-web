import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("imputed_MICE_RF.xlsx")

# 🔥 FIX COLUMN NAMES
df.columns = (
    df.columns
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(" ", "_")
)

target = "CH4_BC_std"

X = df.drop(columns=[target])
y = df[target]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model_real = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=0.5,
    random_state=42
)

model_real.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
print("Train R2:", r2_score(y_train, model_real.predict(X_train)))
print("Test R2:", r2_score(y_test, model_real.predict(X_test)))

# =========================
# SYNTHETIC DATA
# =========================
residuals = y_train - model_real.predict(X_train)

X_new = X.sample(500, replace=True).reset_index(drop=True)
y_pred = model_real.predict(X_new)

noise = np.random.choice(residuals, size=500, replace=True)
y_new = y_pred + noise
y_new = np.clip(y_new, 0, 850)

synthetic_real = X_new.copy()
synthetic_real[target] = y_new

synthetic_real.to_csv("synthetic_realistic_500.csv", index=False)

print("✅ Done!")
