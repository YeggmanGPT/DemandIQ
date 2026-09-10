"""
anomaly.py

Compares actual vs. forecast demand (from forecaster.py's hold-out
evaluation) per department and flags deviations using the same
thresholds documented in BUILD_LOG.md:
    < 15%   -> LOW      (normal operational variance)
    15-25%  -> MEDIUM   (moderate deviation)
    > 25%   -> HIGH     (significant anomaly -> triggers agent investigation)

Run from the project root, after forecaster.py:
    python anomaly.py
"""

import os

import duckdb
import pandas as pd

DB_PATH = os.path.join("db", "demandiq.duckdb")
HIGH_THRESHOLD = 25.0
MEDIUM_THRESHOLD = 15.0


def compute_deviation(actual: float, forecast: float) -> float:
    if forecast == 0:
        return 0.0 if actual == 0 else 100.0
    return abs(actual - forecast) / forecast * 100


def classify(deviation_pct: float) -> str:
    if deviation_pct > HIGH_THRESHOLD:
        return "HIGH"
    if deviation_pct > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def run_anomaly_scan(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("""
        SELECT department, ds, actual, forecast, model
        FROM department_forecast_evaluation
        ORDER BY department, ds
    """).df()

    df["deviation_pct"] = df.apply(lambda r: compute_deviation(r["actual"], r["forecast"]), axis=1)
    df["severity"] = df["deviation_pct"].apply(classify)

    conn.register("anomaly_view", df)
    conn.execute("CREATE OR REPLACE TABLE anomaly_scan AS SELECT * FROM anomaly_view")
    return df


def latest_status_per_department(df: pd.DataFrame) -> pd.DataFrame:
    """The most recent evaluated day per department -- this is what a
    scheduled run would actually alert on."""
    latest = df.sort_values("ds").groupby("department").tail(1).reset_index(drop=True)
    return latest.sort_values("deviation_pct", ascending=False)


if __name__ == "__main__":
    conn = duckdb.connect(DB_PATH)
    scan = run_anomaly_scan(conn)

    print(f"Scanned {len(scan):,} evaluated department-days.")
    print("\nSeverity breakdown:")
    print(scan["severity"].value_counts().to_string())

    latest = latest_status_per_department(scan)
    display_cols = ["department", "ds", "actual", "forecast", "deviation_pct", "severity"]
    formatted = latest[display_cols].copy()
    formatted[["actual", "forecast", "deviation_pct"]] = formatted[
        ["actual", "forecast", "deviation_pct"]
    ].round(1)
    print("\nMost recent status per department:")
    print(formatted.to_string(index=False))

    high = latest[latest["severity"] == "HIGH"]
    if len(high):
        print(f"\n{len(high)} department(s) at HIGH severity -- would trigger agent investigation:")
        print(high["department"].tolist())
    else:
        print("\nNo HIGH-severity departments right now -- nothing would page the agent on this run.")

    conn.close()
