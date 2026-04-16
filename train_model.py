"""
train_model.py
--------------
Trains multiple regression models on the rent dataset,
picks the best one by R² score, and saves:
  - models/best_model.pkl
  - models/preprocessor.pkl
  - models/model_metadata.json
  - models/label_encoders.pkl
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────── paths ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────── load ─────────────────────────────────
print("=" * 60)
print("  RENT PREDICTION MODEL TRAINER")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✔ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────── feature engineering ──────────────────────────
# Drop rows with nulls (there are none, but safety first)
df.dropna(inplace=True)

# Encode categoricals
label_encoders = {}
for col in ["city", "locality", "furnishing"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].str.strip())
    label_encoders[col] = le

# Save unique values for validation at inference time
meta_values = {
    "cities":      sorted(df["city"].unique().tolist()),
    "localities":  sorted(df["locality"].unique().tolist()),
    "furnishings": sorted(df["furnishing"].unique().tolist()),
    "beds_range":  [int(df["beds"].min()), int(df["beds"].max())],
    "area_range":  [float(df["area"].min()), float(df["area"].max())],
    "rent_range":  [float(df["rent"].min()), float(df["rent"].max())],
}

FEATURE_COLS = [
    "area", "beds", "bathrooms", "balconies",
    "area_rate",
    "city_enc", "locality_enc", "furnishing_enc"
]
TARGET = "rent"

X = df[FEATURE_COLS].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✔ Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─────────────────────── model competition ────────────────────────────
candidates = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Ridge": Ridge(alpha=10.0),
    "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=42),
}

print("\n── Model Comparison ─────────────────────────────────────────────")
print(f"{'Model':<22} {'CV R²':>8} {'Test R²':>8} {'MAE':>12} {'RMSE':>12}")
print("-" * 65)

best_name, best_score, best_model = None, -np.inf, None

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

results = {}
for name, model in candidates.items():
    cv_scores = cross_val_score(
        model, X_train_sc, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_r2 = cv_scores.mean()

    results[name] = {"cv_r2": round(cv_r2, 4), "test_r2": round(r2, 4),
                     "mae": round(mae, 2), "rmse": round(rmse, 2)}
    print(f"{name:<22} {cv_r2:>8.4f} {r2:>8.4f} {mae:>12,.0f} {rmse:>12,.0f}")

    if r2 > best_score:
        best_score = r2
        best_name  = name
        best_model = model

print("-" * 65)
print(f"\n✔ Best model: {best_name}  (Test R² = {best_score:.4f})")

# ─────────────────────────── save artefacts ───────────────────────────
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

metadata = {
    "best_model":    best_name,
    "feature_cols":  FEATURE_COLS,
    "target":        TARGET,
    "model_results": results,
    **meta_values,
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✔ Artefacts saved to: {MODEL_DIR}/")
print("   ├── best_model.pkl")
print("   ├── preprocessor.pkl")
print("   ├── label_encoders.pkl")
print("   └── model_metadata.json")
print("\n✅ Training complete!\n")