"""
features.py

Builds a daily, department-level demand time series from the DuckDB
warehouse, then engineers lag and rolling-window features for the
Day 2 forecasting models.

Why this file exists (read before you run it):
Instacart's public dataset does NOT include real order dates -- it was
anonymized to only release order_dow, order_hour_of_day, and
days_since_prior_order (the gap, in days, since that user's previous
order). There is no way to recover true calendar dates from it.

To get a daily series to forecast against, every user is assigned a
random "anchor" date drawn from a fixed window, and their orders are
then dated by walking forward through their *real* gap-day values
(days_since_prior_order). This keeps genuine purchase-interval and
weekly-seasonality behavior -- which is real, measured signal -- while
avoiding the obvious failure mode of anchoring every user's first
order to the same calendar day, which would fake a massive demand
spike on day one of the simulation.

This is a documented modeling assumption, not a reconstruction of
ground truth. Say exactly this if it comes up in an interview -- see
README.md -> Limitations for the paragraph version.

Run from the project root, after ingest.py:
    python features.py
"""

import os

import duckdb
import numpy as np
import pandas as pd

DB_PATH = os.path.join("db", "demandiq.duckdb")
SIM_START = "2024-01-01"
ANCHOR_WINDOW_DAYS = 90
RANDOM_SEED = 42


def build_order_dates(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Assigns every order_id an approximate calendar date."""
    orders = conn.execute("""
        SELECT order_id, user_id, order_number,
               COALESCE(days_since_prior, 0) AS gap_days
        FROM fact_orders
        ORDER BY user_id, order_number
    """).df()

    rng = np.random.default_rng(RANDOM_SEED)
    user_ids = orders["user_id"].unique()
    anchor_offset = rng.integers(0, ANCHOR_WINDOW_DAYS, size=len(user_ids))
    anchor_map = pd.Series(anchor_offset, index=user_ids)

    orders["anchor_offset"] = orders["user_id"].map(anchor_map)

    # Cumulative gap-days from this user's first order up to and including
    # the current one. days_since_prior on row N is the gap that *led to*
    # order N, so a plain cumsum (no subtracting the current row) lines up
    # correctly: order 1 gets offset 0, order 2 gets its own gap, order 3
    # gets order-2's gap plus its own, and so on.
    orders["cumulative_gap"] = orders.groupby("user_id")["gap_days"].cumsum()

    start = pd.Timestamp(SIM_START)
    orders["approx_date"] = (
        start
        + pd.to_timedelta(orders["anchor_offset"], unit="D")
        + pd.to_timedelta(orders["cumulative_gap"], unit="D")
    )
    return orders[["order_id", "approx_date"]]


def build_daily_department_demand(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    order_dates = build_order_dates(conn)
    conn.register("order_dates", order_dates)

    daily = conn.execute("""
        SELECT
            od.approx_date       AS ds,
            dep.department_name  AS department,
            COUNT(foi.product_id) AS y
        FROM fact_order_items foi
        JOIN order_dates od     ON foi.order_id = od.order_id
        JOIN dim_products p     ON foi.product_id = p.product_id
        JOIN dim_departments dep ON p.department_id = dep.department_id
        GROUP BY od.approx_date, dep.department_name
        ORDER BY od.approx_date
    """).df()

    print(f"Date range: {daily['ds'].min().date()} to {daily['ds'].max().date()}")
    print(f"Departments: {daily['department'].nunique()}")
    print(f"Rows: {len(daily):,}")
    print(
        "Sanity check: if the date range looks absurd (e.g. spans many "
        "years) or any department has near-zero rows, stop and inspect "
        "before moving on to Day 2."
    )
    return daily


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag, rolling-window, and calendar features per department."""
    df = df.sort_values(["department", "ds"]).reset_index(drop=True)

    df["lag_1"] = df.groupby("department")["y"].shift(1)
    df["lag_7"] = df.groupby("department")["y"].shift(7)
    df["lag_14"] = df.groupby("department")["y"].shift(14)

    df["rolling_mean_7"] = df.groupby("department")["y"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    df["rolling_std_7"] = df.groupby("department")["y"].transform(
        lambda s: s.shift(1).rolling(7).std()
    )
    df["rolling_mean_14"] = df.groupby("department")["y"].transform(
        lambda s: s.shift(1).rolling(14).mean()
    )

    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["ds"].dt.month
    df["day_of_month"] = df["ds"].dt.day

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with incomplete lag/rolling windows (expected -- "
          f"the first 14 days per department can't have a 14-day lag)")
    return df


def save_features(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    conn.register("features_view", df)
    conn.execute("""
        CREATE OR REPLACE TABLE daily_department_features AS
        SELECT * FROM features_view
    """)

    out_path = os.path.join("data", "daily_department_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows -> DuckDB table 'daily_department_features' and {out_path}")


if __name__ == "__main__":
    conn = duckdb.connect(DB_PATH)
    raw_daily = build_daily_department_demand(conn)
    features = engineer_features(raw_daily)
    save_features(conn, features)
    conn.close()
