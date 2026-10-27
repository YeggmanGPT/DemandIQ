"""Shared fixtures: a local Spark session and a lakehouse built from synthetic CSVs.

By default the tables are Parquet so the suite runs anywhere Java is
installed. Set DEMANDIQ_TEST_DELTA=1 (CI does) to run the same suite on
real Delta tables, which also exercises the CHECK constraints.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import synthetic  # noqa: E402

USE_DELTA = os.environ.get("DEMANDIQ_TEST_DELTA") == "1"
# Databricks serverless runs PySpark over Spark Connect; DEMANDIQ_TEST_CONNECT=1
# runs the suite the same way against a local Connect server.
USE_CONNECT = os.environ.get("DEMANDIQ_TEST_CONNECT") == "1"
TABLE_FORMAT = "delta" if USE_DELTA else "parquet"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    builder = (
        (SparkSession.builder.remote("local[2]") if USE_CONNECT else SparkSession.builder.master("local[2]"))
        .appName("demandiq-tests")
        .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("warehouse")))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )
    if USE_DELTA:
        from delta import configure_spark_with_delta_pip

        builder = configure_spark_with_delta_pip(
            builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    session = builder.getOrCreate()
    if not USE_CONNECT:
        session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def clean_landing(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("landing_clean"))
    counts = synthetic.generate(path)
    return path, counts


def make_config(prefix: str, landing: str, run_id: str = "test-run"):
    from demandiq_lakehouse.config import LakehouseConfig

    return LakehouseConfig(catalog="spark_catalog", schema_prefix=prefix, landing_path=landing,
                           table_format=TABLE_FORMAT, run_id=run_id)


@pytest.fixture(scope="session")
def built(spark, clean_landing, tmp_path_factory):
    """Run the whole job once through the real CLI entry point."""
    import run_task

    landing, counts = clean_landing
    mlruns = tmp_path_factory.mktemp("mlruns")
    run_task.main([
        "--task", "all", "--catalog", "spark_catalog", "--schema-prefix", "t_",
        "--landing-path", landing, "--table-format", TABLE_FORMAT, "--run-id", "test-run",
        "--min-orders", "1000", "--register-model", "false",
        "--mlflow-tracking-uri", mlruns.as_uri(), "--experiment-path", "demandiq_tests",
    ])
    return {"cfg": make_config("t_", landing), "csv_counts": counts, "mlruns": mlruns.as_uri()}
