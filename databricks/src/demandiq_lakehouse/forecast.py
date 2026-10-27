"""Forecasting + anomaly detection on the gold daily demand, tracked in MLflow.

What it does, per run:
  1. Reads gold.department_daily_demand (21 departments x ~400 days -- small,
     so it is pulled into pandas on the driver).
  2. Fills missing days with 0 and trims the warm-up and tail-off periods
     that the date reconstruction creates (see ``steady_state_window``).
  3. Trains one global XGBoost model across all departments on lag,
     rolling and calendar features, and scores it against a seasonal-naive
     baseline (same weekday last week) on the last ``holdout_days`` days.
  4. Logs params, metrics, per-department errors and the model to MLflow.
     If XGBoost beats the baseline, the model is registered in Unity
     Catalog and gets the alias ``champion``.
  5. Writes gold.forecast_eval, gold.forecast_metrics and
     gold.demand_anomalies (15% / 25% thresholds, as in anomaly.py).

The change from the DuckDB version (per-department Prophet vs XGBoost) is
deliberate: one global model is a single artefact to track and register,
and the naive baseline answers "is the model worth having at all?".
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from .config import HIGH_THRESHOLD_PCT, MEDIUM_THRESHOLD_PCT, LakehouseConfig
from .storage import write_table

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
    "day_of_week", "is_weekend", "month", "day_of_month", "dept_code",
]
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=42)


# --------------------------------------------------------------------------
# pure pandas steps (unit-tested)
# --------------------------------------------------------------------------

def complete_daily_series(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per department per day over the full range, 0 where no sales.

    Without this, lag_1 would silently mean "the previous day that had
    sales" for small departments.
    """
    daily = daily.copy()
    daily["ds"] = pd.to_datetime(daily["ds"])
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    departments = sorted(daily["department"].unique())
    index = pd.MultiIndex.from_product([departments, full_range], names=["department", "ds"])
    return (
        daily.groupby(["department", "ds"])["items"].sum()
        .reindex(index, fill_value=0)
        .rename("y").reset_index()
    )


def steady_state_window(series: pd.DataFrame, min_share: float = 0.5) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Date range where total demand is at least ``min_share`` of its peak.

    Reconstructed dates ramp up over the anchor window (users "arrive")
    and fade out at the end (users' order histories run out). Neither is
    real demand behaviour, so training and scoring on them would reward
    a model for learning the reconstruction instead of the data.
    """
    total = series.groupby("ds")["y"].sum().sort_index()
    smooth = total.rolling(7, center=True, min_periods=1).mean()
    keep = smooth[smooth >= min_share * smooth.max()]
    return keep.index.min(), keep.index.max()


def engineer_features(series: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling / calendar features, as in features.py, per department."""
    df = series.sort_values(["department", "ds"]).reset_index(drop=True)
    by_dept = df.groupby("department")["y"]
    df["lag_1"] = by_dept.shift(1)
    df["lag_7"] = by_dept.shift(7)
    df["lag_14"] = by_dept.shift(14)
    df["rolling_mean_7"] = by_dept.transform(lambda s: s.shift(1).rolling(7).mean())
    df["rolling_std_7"] = by_dept.transform(lambda s: s.shift(1).rolling(7).std())
    df["rolling_mean_14"] = by_dept.transform(lambda s: s.shift(1).rolling(14).mean())
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["ds"].dt.month
    df["day_of_month"] = df["ds"].dt.day
    codes = {d: i for i, d in enumerate(sorted(df["department"].unique()))}
    df["dept_code"] = df["department"].map(codes)
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def error_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    nonzero = actual != 0  # MAPE is undefined where the actual is 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
    }


def deviation_pct(actual: float, forecast: float) -> float:
    if forecast == 0:
        return 0.0 if actual == 0 else 100.0
    return abs(actual - forecast) / forecast * 100


def severity(dev_pct: float) -> str:
    if dev_pct > HIGH_THRESHOLD_PCT:
        return "HIGH"
    if dev_pct > MEDIUM_THRESHOLD_PCT:
        return "MEDIUM"
    return "LOW"


@dataclass
class ForecastResult:
    evaluation: pd.DataFrame        # ds, department, actual, xgboost, seasonal_naive
    metrics: pd.DataFrame           # department, model, mae, rmse, mape
    feature_importance: pd.DataFrame
    model: object
    champion: str
    window: dict[str, str]
    train_frame: pd.DataFrame


def train_and_evaluate(daily: pd.DataFrame, holdout_days: int = 21, min_share: float = 0.5) -> ForecastResult:
    from xgboost import XGBRegressor

    series = complete_daily_series(daily)
    start, end = steady_state_window(series, min_share=min_share)
    series = series[(series["ds"] >= start) & (series["ds"] <= end)]
    features = engineer_features(series)

    cutoff = features["ds"].max() - pd.Timedelta(days=holdout_days - 1)
    train, test = features[features["ds"] < cutoff], features[features["ds"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError(
            f"Not enough history to train: {features['ds'].nunique()} usable days after trimming, "
            f"holdout needs {holdout_days}.")

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLS], train["y"])

    evaluation = test[["ds", "department", "y"]].rename(columns={"y": "actual"}).copy()
    evaluation["xgboost"] = np.clip(model.predict(test[FEATURE_COLS]), 0, None)
    evaluation["seasonal_naive"] = test["lag_7"].to_numpy()

    rows = []
    for model_name in ("xgboost", "seasonal_naive"):
        rows.append({"department": "ALL", "model": model_name,
                     **error_metrics(evaluation["actual"], evaluation[model_name])})
        for dept, part in evaluation.groupby("department"):
            rows.append({"department": dept, "model": model_name,
                         **error_metrics(part["actual"], part[model_name])})
    metrics = pd.DataFrame(rows)

    overall = metrics[metrics["department"] == "ALL"].set_index("model")["rmse"]
    champion = "xgboost" if overall["xgboost"] <= overall["seasonal_naive"] else "seasonal_naive"

    importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False)
    window = {
        "window_start": str(start.date()), "window_end": str(end.date()),
        "train_start": str(train["ds"].min().date()), "train_end": str(train["ds"].max().date()),
        "test_start": str(test["ds"].min().date()), "test_end": str(test["ds"].max().date()),
    }
    return ForecastResult(evaluation.reset_index(drop=True), metrics, importance, model, champion,
                          window, train)


def detect_anomalies(evaluation: pd.DataFrame, champion: str) -> pd.DataFrame:
    df = evaluation[["ds", "department", "actual"]].copy()
    df["forecast"] = evaluation[champion].astype(float)
    df["model"] = champion
    df["deviation_pct"] = [deviation_pct(a, f) for a, f in zip(df["actual"], df["forecast"], strict=True)]
    df["severity"] = df["deviation_pct"].map(severity)
    latest = df.groupby("department")["ds"].transform("max")
    df["is_latest"] = df["ds"] == latest
    return df.sort_values(["department", "ds"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------

def _metric_key(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()


def _log_xgb_model(model, **kwargs):
    import mlflow.xgboost

    params = inspect.signature(mlflow.xgboost.log_model).parameters
    name_arg = "name" if "name" in params else "artifact_path"  # MLflow 3 vs 2
    return mlflow.xgboost.log_model(model, **{name_arg: "model"}, **kwargs)


def log_to_mlflow(result: ForecastResult, cfg: LakehouseConfig, experiment_path: str,
                  register_model: bool, tracking_uri: str | None, holdout_days: int) -> dict:
    import mlflow
    from mlflow.models import infer_signature

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if register_model:
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_path)

    model_name = f"{cfg.catalog}.{cfg.schema('gold')}.demand_forecaster"
    info = {"mlflow_run_id": None, "registered_model": None, "model_version": None}
    with mlflow.start_run(run_name=f"demand_forecaster_{cfg.run_id}") as run:
        info["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags({"project": "demandiq", "job_run_id": cfg.run_id, "champion": result.champion})
        mlflow.log_params({**XGB_PARAMS, "holdout_days": holdout_days, "features": ",".join(FEATURE_COLS),
                           "train_rows": len(result.train_frame), **result.window})

        overall = result.metrics[result.metrics["department"] == "ALL"]
        for _, row in overall.iterrows():
            for m in ("mae", "rmse", "mape"):
                mlflow.log_metric(f"{row['model']}_{m}", row[m])
        per_dept = result.metrics[(result.metrics["department"] != "ALL") & (result.metrics["model"] == "xgboost")]
        for _, row in per_dept.iterrows():
            if not np.isnan(row["mape"]):
                mlflow.log_metric(f"mape_{_metric_key(row['department'])}", row["mape"])

        mlflow.log_text(result.metrics.to_csv(index=False), "metrics_by_department.csv")
        mlflow.log_text(result.feature_importance.to_csv(index=False), "feature_importance.csv")
        mlflow.log_text(json.dumps(result.window, indent=2), "data_window.json")

        # float64 so the logged signature accepts NaN-safe numeric input
        sample = result.train_frame[FEATURE_COLS].head(50).astype("float64")
        signature = infer_signature(sample, result.model.predict(sample))
        should_register = register_model and result.champion == "xgboost"
        _log_xgb_model(result.model, signature=signature, input_example=sample.head(5),
                       registered_model_name=model_name if should_register else None)

    if should_register:
        from mlflow import MlflowClient

        client = MlflowClient()
        versions = [v for v in client.search_model_versions(f"name='{model_name}'")
                    if v.run_id == info["mlflow_run_id"]]
        if versions:
            version = max(int(v.version) for v in versions)
            client.set_registered_model_alias(model_name, "champion", version)
            info.update(registered_model=model_name, model_version=str(version))
    return info


# --------------------------------------------------------------------------
# job task
# --------------------------------------------------------------------------

def run(spark: SparkSession, cfg: LakehouseConfig, experiment_path: str, register_model: bool = True,
        tracking_uri: str | None = None, holdout_days: int = 21) -> ForecastResult:
    daily = spark.table(cfg.gold("department_daily_demand")).select("ds", "department", "items").toPandas()
    result = train_and_evaluate(daily, holdout_days=holdout_days)
    print(f"[forecast] window {result.window}")
    print(result.metrics[result.metrics["department"] == "ALL"].to_string(index=False))
    print(f"[forecast] champion: {result.champion}")

    info = log_to_mlflow(result, cfg, experiment_path, register_model, tracking_uri, holdout_days)
    if info["registered_model"]:
        print(f"[forecast] MLflow run {info['mlflow_run_id']}; registered {info['registered_model']} "
              f"version {info['model_version']} as @champion")
    else:
        print(f"[forecast] MLflow run {info['mlflow_run_id']}; model logged, not registered "
              f"(register_model={register_model}, champion={result.champion})")

    trained_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

    evaluation = result.evaluation.copy()
    evaluation["ds"] = evaluation["ds"].dt.date
    write_table(spark.createDataFrame(evaluation), cfg.gold("forecast_eval"), cfg,
                comment="Hold-out days: actual items vs XGBoost and seasonal-naive forecasts.")

    metrics = result.metrics.copy()
    metrics["is_champion"] = metrics["model"] == result.champion
    metrics["mlflow_run_id"] = info["mlflow_run_id"]
    metrics["job_run_id"] = cfg.run_id
    metrics["test_start"] = result.window["test_start"]
    metrics["test_end"] = result.window["test_end"]
    metrics["trained_at"] = trained_at
    write_table(spark.createDataFrame(metrics), cfg.gold("forecast_metrics"), cfg,
                comment="Hold-out MAE/RMSE/MAPE per department and overall ('ALL'), per model.")

    anomalies = detect_anomalies(result.evaluation, result.champion)
    anomalies["ds"] = anomalies["ds"].dt.date
    write_table(spark.createDataFrame(anomalies), cfg.gold("demand_anomalies"), cfg,
                comment="Forecast-vs-actual deviation per department-day, LOW/MEDIUM/HIGH.")
    high = anomalies[anomalies["is_latest"] & (anomalies["severity"] == "HIGH")]
    print(f"[forecast] {len(high)} department(s) at HIGH severity on the latest day: "
          f"{sorted(high['department'])}")
    return result
