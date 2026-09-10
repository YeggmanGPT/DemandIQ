"""
forecaster.py

Trains a per-department demand forecasting model tournament (Prophet vs.
XGBoost) on the daily_department_features table produced by features.py,
evaluates both on a chronological hold-out split, and saves:
  - the champion model per department -> models/<department>_<champion>.pkl
  - a leaderboard of both models' scores -> models/model_leaderboard.csv
  - the champion's hold-out predictions -> DuckDB table
    'department_forecast_evaluation' (actual vs. forecast per day), which
    anomaly.py reads to flag deviations.

Run from the project root, after features.py:
    python forecaster.py
"""

import os
import pickle
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DB_PATH = os.path.join("db", "demandiq.duckdb")
MODELS_DIR = "models"
TEST_FRACTION = 0.2
MIN_TEST_ROWS = 5

FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
    "day_of_week", "is_weekend", "month", "day_of_month",
]


def load_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("SELECT * FROM daily_department_features ORDER BY department, ds").df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Time-ordered split, not random -- a random split would leak future
    rolling-window information into the training set."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1, y_true)  # avoid divide-by-zero on zero-demand days
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def score(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def fit_prophet(train: pd.DataFrame, test: pd.DataFrame):
    from prophet import Prophet
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.fit(train[["ds", "y"]])
    forecast = m.predict(test[["ds"]])
    preds = forecast["yhat"].clip(lower=0).to_numpy()
    return m, preds


def fit_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"])
    preds = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
    return model, preds


def run_tournament(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(MODELS_DIR, exist_ok=True)
    eval_rows = []
    leaderboard = []

    for department, dept_df in df.groupby("department"):
        dept_df = dept_df.sort_values("ds").reset_index(drop=True)
        train, test = chronological_split(dept_df)
        if len(test) < MIN_TEST_ROWS:
            print(f"  [skip] {department}: only {len(dept_df)} rows -- not enough for a hold-out split")
            continue

        prophet_model, prophet_preds = fit_prophet(train, test)
        prophet_scores = score(test["y"], prophet_preds)

        xgb_model, xgb_preds = fit_xgboost(train, test)
        xgb_scores = score(test["y"], xgb_preds)

        champion = "prophet" if prophet_scores["rmse"] <= xgb_scores["rmse"] else "xgboost"
        champion_preds = prophet_preds if champion == "prophet" else xgb_preds
        champion_model = prophet_model if champion == "prophet" else xgb_model

        safe_name = "".join(c if c.isalnum() else "_" for c in department)
        model_path = os.path.join(MODELS_DIR, f"{safe_name}_{champion}.pkl")
        try:
            with open(model_path, "wb") as f:
                pickle.dump(champion_model, f)
        except Exception as exc:  # noqa: BLE001 -- don't let a pickling quirk kill the whole run
            print(f"  [warn] could not pickle {champion} model for {department}: {exc}")

        leaderboard.append({
            "department": department,
            "prophet_rmse": round(prophet_scores["rmse"], 2),
            "prophet_mape": round(prophet_scores["mape"], 1),
            "xgboost_rmse": round(xgb_scores["rmse"], 2),
            "xgboost_mape": round(xgb_scores["mape"], 1),
            "champion": champion,
            "champion_rmse": round(min(prophet_scores["rmse"], xgb_scores["rmse"]), 2),
        })

        for ds, actual, pred in zip(test["ds"], test["y"], champion_preds):
            eval_rows.append({
                "department": department,
                "ds": ds,
                "actual": float(actual),
                "forecast": float(pred),
                "model": champion,
            })

    leaderboard_df = pd.DataFrame(leaderboard)
    eval_df = pd.DataFrame(eval_rows)

    conn.register("eval_view", eval_df)
    conn.execute("CREATE OR REPLACE TABLE department_forecast_evaluation AS SELECT * FROM eval_view")

    leaderboard_df.to_csv(os.path.join(MODELS_DIR, "model_leaderboard.csv"), index=False)
    return leaderboard_df


if __name__ == "__main__":
    conn = duckdb.connect(DB_PATH)
    df = load_features(conn)
    print(f"Loaded {len(df):,} rows across {df['department'].nunique()} departments")

    print("\nRunning Prophet vs. XGBoost tournament per department...")
    leaderboard = run_tournament(conn, df)

    print("\nModel leaderboard (lower RMSE/MAPE is better):")
    print(leaderboard.to_string(index=False))

    print(f"\nChampions: {leaderboard['champion'].value_counts().to_dict()}")
    print(f"Saved model artifacts -> {MODELS_DIR}/")
    print("Saved hold-out predictions -> DuckDB table 'department_forecast_evaluation'")

    conn.close()
