"""Small helpers for creating schemas and writing tables.

On Databricks every table is a Unity Catalog managed Delta table. The
local test suite can run the exact same code against Parquet tables
(``table_format="parquet"``) because a plain laptop or CI runner may not
have the Delta jars; CI runs it against real Delta as well.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from .config import LAYERS, LakehouseConfig


def ensure_schemas(spark: SparkSession, cfg: LakehouseConfig) -> None:
    for layer in LAYERS:
        spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS {cfg.schema_fqn(layer)} "
            f"COMMENT 'DemandIQ {layer} layer'"
        )


def write_table(
    df: DataFrame,
    table: str,
    cfg: LakehouseConfig,
    comment: str | None = None,
    mode: str = "overwrite",
) -> int:
    """Write ``df`` to ``table`` and return the row count that landed.

    Full refresh (overwrite) is deliberate: the source is a static
    snapshot, so every run rebuilds each layer from the one below it and
    a re-run is always safe. Delta keeps the previous versions, so an
    overwrite can still be inspected or rolled back with time travel.
    """
    writer = df.write.format(cfg.table_format).mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table)

    spark = df.sparkSession
    if comment:
        spark.sql(f"COMMENT ON TABLE {table} IS '{_sql_string(comment)}'")
    return spark.table(table).count()


def add_check_constraints(
    spark: SparkSession, table: str, cfg: LakehouseConfig, constraints: dict[str, str]
) -> list[str]:
    """Add Delta CHECK constraints that are not on the table yet.

    Delta enforces these on every later write, so they are a second line
    of defence behind the quarantine logic: if a code change ever lets a
    bad row through, the write fails instead of publishing it.
    Returns the names that were added. No-op for non-Delta tables.
    """
    if not cfg.is_delta:
        return []
    existing = {
        row["key"].lower()
        for row in spark.sql(f"SHOW TBLPROPERTIES {table}").collect()
    }
    added = []
    for name, expression in constraints.items():
        if f"delta.constraints.{name.lower()}" in existing:
            continue
        spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({expression})")
        added.append(name)
    return added


def _sql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
