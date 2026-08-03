"""Clean raw Opal patronage data and load it into SQLite.

Usage: python src/clean_data.py
"""
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repo root, works wherever you run it from
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "opal.db"


def load_raw() -> pd.DataFrame:
    # The Opal files are pipe-delimited (|). Load the count columns as strings
    # deliberately — they contain "<50" suppression markers, so they are NOT numeric yet.
    files = sorted(RAW_DIR.rglob("*.txt"))   # walks subfolders too
    df = pd.concat(
        (pd.read_csv(f, sep="|", dtype={"Tap_Ons": str, "Tap_Offs": str}) for f in files),
        ignore_index=True,
    )

    # Inspect
    print(f"Loaded {len(files)} files, {df.shape[0]:,} rows")
    print(df.head())
    print(df.dtypes)          # Tap_Ons / Tap_Offs will show as object (string) — expected
    print(df['Tap_Ons'].str.contains('<').mean())  # what share of rows is suppressed?
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Handle the suppression markers -----------------------------------
    # Small counts are privacy-suppressed, and the marker CHANGED over time:
    # "<50" up to Jun 2024, "<100" from Jul 2024. Policy decision (state it in
    # your README): treat each suppressed cell as the midpoint of its hidden
    # range (25 / 50). Alternatives: 0 (undercounts) or NaN (drops them).
    # What matters is that you CHOSE and DOCUMENTED it, not which one you picked.
    sup = {'<50': '25', '<100': '50'}
    df['suppressed'] = df['Tap_Ons'].str.strip().isin(sup)

    for col in ['Tap_Ons', 'Tap_Offs']:
        df[col] = (df[col].str.strip()
                          .replace(sup)
                          .astype(int))

    # --- 2. Parse dates, standardise text, drop unattributed taps ------------
    df['trip_origin_date'] = pd.to_datetime(df['trip_origin_date'])
    df['mode_name'] = df['mode_name'].str.strip()      # Bus, Train, Ferry, Light rail, UNKNOWN
    df['ti_region'] = df['ti_region'].str.strip()
    df = df[df['mode_name'] != 'UNKNOWN']              # unattributed taps, ~0.1% of trips
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    # --- 3. Aggregate hourly rows to a daily table ---------------------------
    # Use Tap_Ons as the "trips" measure (a tap-on ≈ a journey start).
    # Keep the All - NSW aggregate SEPARATE from the per-region rows so no
    # query ever double-counts.
    daily = (df.groupby(['trip_origin_date', 'mode_name', 'ti_region'], as_index=False)
               .agg(trips=('Tap_Ons', 'sum'),
                    tap_offs=('Tap_Offs', 'sum'),
                    suppressed_hours=('suppressed', 'sum')))

    daily['is_nsw_total'] = daily['ti_region'].eq('All - NSW')

    # --- 4. Flag anomalous / low-quality dates -------------------------------
    # Don't hardcode a guessed list — FIND bad dates two ways:
    #   (a) the dataset's official data-quality notes on the TfNSW page;
    #   (b) detection from the data itself, e.g. days far below the local median.
    chk = daily[daily['is_nsw_total']].copy()          # state-wide series per mode
    chk['med'] = (chk.groupby('mode_name')['trips']
                     .transform(lambda s: s.rolling(7, center=True, min_periods=1).median()))
    chk['suspect'] = chk['trips'] < 0.4 * chk['med']   # <40% of local median

    # Inspect what got flagged BEFORE excluding anything — never trust the rule blindly.
    # Note: genuine events (lockdowns, strikes) will flag too; that's a judgement call
    # for the README, not an automatic delete.
    print(chk[chk['suspect']][['mode_name', 'trip_origin_date', 'trips', 'med']])

    suspect_dates = set(chk.loc[chk['suspect'], 'trip_origin_date'])
    daily['data_quality_flag'] = daily['trip_origin_date'].isin(suspect_dates).astype(int)

    # --- 5. Add calendar columns for later analysis --------------------------
    daily['day_of_week'] = daily['trip_origin_date'].dt.day_name()
    daily['is_weekend']  = (daily['trip_origin_date'].dt.dayofweek >= 5).astype(int)
    daily['year']        = daily['trip_origin_date'].dt.year
    daily['month']       = daily['trip_origin_date'].dt.month
    return daily


def write_db(daily: pd.DataFrame, df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    daily.to_sql("patronage", conn, if_exists="replace", index=False)      # daily table (main)
    df.to_sql("patronage_hourly", conn, if_exists="replace", index=False)  # optional
    conn.close()


def main():
    df = clean(load_raw())
    daily = aggregate_daily(df)
    write_db(daily, df)
    print(f"Done — wrote {DB_PATH}")


if __name__ == "__main__":
    main()
