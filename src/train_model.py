"""Module 5 — train a daily patronage prediction model.

Predicts state-wide daily trips for one mode from calendar features
(day of week, month, year), then saves the trained model for the app.

Run from the project root:
    python src/train_model.py
"""

import os
import sqlite3

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DB_PATH = "data/opal.db"
MODEL_PATH = "app/patronage_model.pkl"
MODE = "Train"

# --- 1. Load the clean, state-wide daily series ---
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    """
    SELECT trip_origin_date, mode_name, trips
    FROM patronage
    WHERE data_quality_flag = 0 AND is_nsw_total = 1
    """,
    conn,
)
conn.close()
df["trip_origin_date"] = pd.to_datetime(df["trip_origin_date"])

mode_df = df[df["mode_name"] == MODE].copy()
print(f"Loaded {len(mode_df)} clean daily rows for mode '{MODE}'")

# --- 2. Features (X) and target (y) ---
mode_df["dow"] = mode_df["trip_origin_date"].dt.dayofweek
mode_df["month"] = mode_df["trip_origin_date"].dt.month
mode_df["year"] = mode_df["trip_origin_date"].dt.year

X = mode_df[["dow", "month", "year"]]
y = mode_df["trips"]

# --- 3. Hold back 20% of days as an unseen test set ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train ---
m = RandomForestRegressor(n_estimators=100, random_state=42)
m.fit(Xtr, ytr)

# --- 5. Evaluate on the unseen days ---
p = m.predict(Xte)
mae = mean_absolute_error(yte, p)
r2 = r2_score(yte, p)

print(f"MAE: {mae:,.0f} trips  (average size of the prediction error)")
print(f"R2:  {r2:.3f}  (share of day-to-day variation explained)")

print("\nFeature importances (which inputs the forest relies on most):")
for name, imp in zip(X.columns, m.feature_importances_):
    print(f"  {name:6s} {imp:.2f}")

# --- 6. Save the trained model for the Streamlit app (Module 6) ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(m, MODEL_PATH)
print(f"\nSaved trained model to {MODEL_PATH}")
