"""Entry point for every task of the DemandIQ Databricks job.

The job (resources/demandiq_job.yml) calls this file once per task:

    bronze    -> CSVs in the landing volume to bronze tables
    silver    -> typed, de-duplicated, conformed tables + quarantine
    quality   -> table-level checks; fails the run before gold on errors
    gold      -> business tables + gold checks
    forecast  -> MLflow-tracked forecasting + anomaly table
    all       -> everything above in order (handy for a local run)

Run locally (after `pip install -r requirements-dev.txt`) against a folder
with the six CSVs:
    python src/run_task.py --task all --catalog spark_catalog \
        --landing-path ./data --table-format parquet --register-model false \
        --mlflow-tracking-uri file:./mlruns --min-orders 1
"""

from __future__ import annotations

import argparse
import os
import sys


def _put_package_on_path() -> None:
    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv and sys.argv[0]:
        candidates.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    candidates.append(os.getcwd())
    for folder in candidates:
        if os.path.isdir(os.path.join(folder, "demandiq_lakehouse")):
            if folder not in sys.path:
                sys.path.insert(0, folder)
            return
    raise RuntimeError(f"Could not find the demandiq_lakehouse package next to run_task.py (looked in {candidates})")


def _bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["bronze", "silver", "quality", "gold", "forecast", "all"])
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema-prefix", default="demandiq_")
    p.add_argument("--landing-path", default="/Volumes/workspace/demandiq_landing/raw/instacart")
    p.add_argument("--table-format", default="delta", choices=["delta", "parquet"])
    p.add_argument("--run-id", default="manual")
    p.add_argument("--min-orders", type=int, default=3_000_000,
                   help="quality gate: fewer silver orders than this fails the run")
    p.add_argument("--holdout-days", type=int, default=21)
    p.add_argument("--experiment-path", default="/Shared/demandiq_forecasting")
    p.add_argument("--register-model", type=_bool, default=True)
    p.add_argument("--mlflow-tracking-uri", default=None,
                   help="leave unset on Databricks; e.g. file:./mlruns locally")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _put_package_on_path()

    from pyspark.sql import SparkSession

    from demandiq_lakehouse import bronze, forecast, gold, quality, silver
    from demandiq_lakehouse.config import LakehouseConfig

    spark = SparkSession.builder.getOrCreate()
    cfg = LakehouseConfig(
        catalog=args.catalog, schema_prefix=args.schema_prefix, landing_path=args.landing_path,
        table_format=args.table_format, run_id=args.run_id,
    )
    steps = {
        "bronze": lambda: bronze.run(spark, cfg),
        "silver": lambda: silver.run(spark, cfg),
        "quality": lambda: quality.run(spark, cfg, min_orders=args.min_orders),
        "gold": lambda: gold.run(spark, cfg),
        "forecast": lambda: forecast.run(
            spark, cfg, experiment_path=args.experiment_path, register_model=args.register_model,
            tracking_uri=args.mlflow_tracking_uri, holdout_days=args.holdout_days),
    }
    for name in (list(steps) if args.task == "all" else [args.task]):
        print(f"=== {name} (run {cfg.run_id}, {cfg.catalog}.{cfg.schema_prefix}*) ===")
        steps[name]()


if __name__ == "__main__":
    main()
