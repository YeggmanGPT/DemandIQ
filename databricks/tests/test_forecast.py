"""Forecasting helpers (pure pandas) and the outputs of the forecast task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandiq_lakehouse import forecast


def _daily(days: int, departments=("a", "b"), level=100.0, ramp=0):
    rows = []
    start = pd.Timestamp("2024-01-01")
    for d in departments:
        for i in range(days):
            value = level * min(1.0, (i + 1) / ramp) if ramp else level
            rows.append({"ds": start + pd.Timedelta(days=i), "department": d, "items": value})
    return pd.DataFrame(rows)


def test_missing_days_become_zero():
    daily = _daily(10).drop(index=[3, 4])  # department "a" loses two days
    series = forecast.complete_daily_series(daily)
    assert len(series) == 20
    assert series.loc[series["department"] == "a", "y"].tolist().count(0) == 2


def test_steady_state_window_trims_the_ramp():
    series = forecast.complete_daily_series(_daily(200, ramp=60))
    start, end = forecast.steady_state_window(series, min_share=0.5)
    assert pd.Timestamp("2024-01-25") <= start <= pd.Timestamp("2024-02-05")
    assert end == pd.Timestamp("2024-07-18")


def test_features_only_look_backwards():
    series = forecast.complete_daily_series(_daily(40))
    series["y"] = np.arange(len(series), dtype=float)
    feats = forecast.engineer_features(series)
    a = feats[feats["department"] == "a"].reset_index(drop=True)
    assert (a["lag_1"] == a["y"] - 1).all()
    assert (a["lag_7"] == a["y"] - 7).all()
    assert (a["rolling_mean_7"] == a["y"] - 4).all()  # mean of the 7 previous values


def test_thresholds_match_the_duckdb_anomaly_check():
    assert forecast.severity(forecast.deviation_pct(100, 100)) == "LOW"
    assert forecast.severity(forecast.deviation_pct(115, 100)) == "LOW"
    assert forecast.severity(forecast.deviation_pct(120, 100)) == "MEDIUM"
    assert forecast.severity(forecast.deviation_pct(126, 100)) == "HIGH"
    assert forecast.deviation_pct(0, 0) == 0.0
    assert forecast.deviation_pct(5, 0) == 100.0


def test_mape_ignores_zero_actuals():
    m = forecast.error_metrics([0, 10], [5, 12])
    assert m["mape"] == pytest.approx(20.0)
    assert m["mae"] == pytest.approx(3.5)


def test_too_little_history_raises_a_clear_error():
    with pytest.raises(ValueError, match="Not enough history"):
        forecast.train_and_evaluate(_daily(20), holdout_days=21)


def test_forecast_task_outputs(spark, built):
    cfg = built["cfg"]
    metrics = spark.table(cfg.gold("forecast_metrics")).toPandas()
    overall = metrics[metrics["department"] == "ALL"]
    assert set(overall["model"]) == {"xgboost", "seasonal_naive"}
    assert overall["is_champion"].sum() == 1
    assert metrics["mlflow_run_id"].nunique() == 1

    evaluation = spark.table(cfg.gold("forecast_eval")).toPandas()
    assert evaluation["ds"].nunique() == 21
    assert (evaluation["xgboost"] >= 0).all()

    anomalies = spark.table(cfg.gold("demand_anomalies")).toPandas()
    latest = anomalies[anomalies["is_latest"]]
    assert latest["department"].is_unique
    assert set(anomalies["severity"]) <= {"LOW", "MEDIUM", "HIGH"}


def test_mlflow_run_was_logged(built):
    import mlflow

    mlflow.set_tracking_uri(built["mlruns"])
    runs = mlflow.search_runs(experiment_names=["demandiq_tests"])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["metrics.xgboost_rmse"] > 0
    assert run["tags.job_run_id"] == "test-run"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run["run_id"])}
    assert {"model", "metrics_by_department.csv", "feature_importance.csv"} <= artifacts
