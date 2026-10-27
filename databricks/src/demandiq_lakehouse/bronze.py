"""Bronze layer: land the six Instacart CSVs as-is, one table per file.

Every column is read as STRING on purpose. Bronze is the faithful copy
of what arrived; typing and cleaning happen in silver, where a value
that does not parse can be quarantined with a reason instead of being
silently turned into NULL at read time.
"""

from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from .config import RAW_SOURCES, LakehouseConfig
from .storage import ensure_schemas, write_table

KAGGLE_URL = "https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis"


def read_raw_csv(spark: SparkSession, path: str, columns: list[str], run_id: str) -> DataFrame:
    schema = StructType([StructField(c, StringType(), True) for c in columns])
    return (
        spark.read.format("csv")
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')            # RFC 4180: "" inside quotes is a literal quote
        .option("enforceSchema", False)  # header must match `columns`, else fail
        .option("mode", "PERMISSIVE")    # short rows load with NULLs; silver quarantines them
        .schema(schema)
        .load(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_run_id", F.lit(run_id))
    )


def _path_exists(spark: SparkSession, path: str) -> bool:
    if os.path.exists(path):
        return True
    try:  # fall back to Spark, for paths the local filesystem cannot see
        spark.read.format("text").load(path).limit(0).collect()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not readable"
        return False


def ensure_landing_volume(spark: SparkSession, landing_path: str) -> None:
    """Create the Unity Catalog schema + volume behind a /Volumes path.

    /Volumes/<catalog>/<schema>/<volume>/... -> CREATE ... IF NOT EXISTS,
    so a first run on a fresh workspace leaves an empty volume ready for
    the upload instead of an unexplained 'path not found'.
    """
    parts = landing_path.strip("/").split("/")
    if len(parts) < 4 or parts[0] != "Volumes":
        return
    catalog, schema, volume = parts[1], parts[2], parts[3]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    spark.sql(
        f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}` "
        "COMMENT 'Landing zone for the raw Instacart CSVs'"
    )


def missing_landing_files(spark: SparkSession, landing_path: str) -> list[str]:
    return [
        filename
        for filename, _ in RAW_SOURCES.values()
        if not _path_exists(spark, os.path.join(landing_path, filename))
    ]


def run(spark: SparkSession, cfg: LakehouseConfig) -> dict[str, int]:
    ensure_schemas(spark, cfg)
    ensure_landing_volume(spark, cfg.landing_path)

    missing = missing_landing_files(spark, cfg.landing_path)
    if missing:
        raise FileNotFoundError(
            f"Missing raw files in {cfg.landing_path}: {missing}. "
            f"Download the Instacart dataset ({KAGGLE_URL}) and upload the six CSVs "
            "there (Catalog Explorer -> the volume -> Upload, or "
            "`databricks fs cp <local folder> dbfs:<landing path> --recursive`)."
        )

    counts = {}
    for table_name, (filename, columns) in RAW_SOURCES.items():
        df = read_raw_csv(spark, os.path.join(cfg.landing_path, filename), columns, cfg.run_id)
        counts[table_name] = write_table(
            df,
            cfg.bronze(table_name),
            cfg,
            comment=f"Raw copy of {filename} (all columns STRING) plus load metadata.",
        )
        print(f"[bronze] {table_name:<26} {counts[table_name]:>12,} rows")
    return counts
