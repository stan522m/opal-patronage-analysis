"""Module 6 — interactive patronage explorer.

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "opal.db"
MODEL_PATH = APP_DIR / "patronage_model.pkl"

st.set_page_config(page_title="Sydney Opal Patronage", layout="wide")
st.title("🚆 Sydney Opal Patronage Explorer")
st.caption(
    "Data: Transport for NSW Open Data (CC BY). "
    "Built with SQL, Python, scikit-learn & Streamlit."
)


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM patronage WHERE data_quality_flag = 0", conn)
    df["trip_origin_date"] = pd.to_datetime(df["trip_origin_date"])
    conn.close()
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()

# --- Filters ---
col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Transport mode", sorted(df["mode_name"].unique()))
with col2:
    regions = sorted(df["ti_region"].unique())
    # Default to the state-wide series so visitors land on the full story.
    # (Filtering to ONE region at a time also avoids the All - NSW double-count trap.)
    default_ix = regions.index("All - NSW") if "All - NSW" in regions else 0
    region = st.selectbox("Region", regions, index=default_ix)

date_range = st.slider(
    "Date range",
    min_value=df["trip_origin_date"].min().to_pydatetime(),
    max_value=df["trip_origin_date"].max().to_pydatetime(),
    value=(
        df["trip_origin_date"].min().to_pydatetime(),
        df["trip_origin_date"].max().to_pydatetime(),
    ),
)

# --- Filter & plot ---
mask = (
    (df["mode_name"] == mode)
    & (df["ti_region"] == region)
    & (df["trip_origin_date"].between(date_range[0], date_range[1]))
)
filtered = df[mask].groupby("trip_origin_date", as_index=False)["trips"].sum()

if filtered.empty:
    st.warning("No data for this mode/region/date combination.")
    st.stop()

fig = px.line(
    filtered, x="trip_origin_date", y="trips", title=f"{mode} patronage — {region}"
)
st.plotly_chart(fig, width="stretch")

# --- Quick stats ---
c1, c2, c3 = st.columns(3)
c1.metric("Total trips", f"{int(filtered['trips'].sum()):,}")
c2.metric("Daily average", f"{int(filtered['trips'].mean()):,}")
c3.metric("Peak day", f"{int(filtered['trips'].max()):,}")

# --- Prediction (Module 5 model) ---
st.divider()
st.subheader("🔮 Predict daily Train patronage")
st.caption(
    "A RandomForest trained on calendar features only (day of week, month, year) "
    "for the state-wide Train series. It knows calendar patterns — not holidays, "
    "weather, events, or disruptions — so treat it as a 'typical day' estimate "
    "(MAE ≈ 86k trips, R² ≈ 0.81 on held-out days)."
)

pick = st.date_input(
    "Pick a date",
    value=df["trip_origin_date"].max().date(),
    min_value=pd.Timestamp("2020-01-01").date(),
    max_value=pd.Timestamp("2027-12-31").date(),
)

model = load_model()
features = pd.DataFrame(
    [[pick.weekday(), pick.month, pick.year]], columns=["dow", "month", "year"]
)
predicted = model.predict(features)[0]

st.metric(
    f"Predicted Train trips — {pick.strftime('%A %d %b %Y')}",
    f"{predicted:,.0f}",
)

actual = df[
    (df["mode_name"] == "Train")
    & (df["is_nsw_total"] == 1)
    & (df["trip_origin_date"] == pd.Timestamp(pick))
]
if not actual.empty:
    real = int(actual["trips"].iloc[0])
    err = predicted - real
    st.write(
        f"Actual on that day: **{real:,}** trips "
        f"(model was off by {abs(err):,.0f}, {abs(err) / real:.1%})."
    )
