"""Gold layer: business-level tables for the dashboard and the forecaster.

These are the eight analytics queries from ``sql/`` in the DuckDB
version, rebuilt as tables, plus the daily department demand series the
forecaster trains on and a one-row overview for the KPI counters. Gold
reads only from silver, so it only ever sees rows that passed the gate.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from . import quality
from .config import DAY_NAMES, LakehouseConfig
from .storage import write_table


def _day_name(col: str = "order_dow"):
    mapping = F.create_map(*[x for k, v in DAY_NAMES.items() for x in (F.lit(k), F.lit(v))])
    return mapping[F.col(col)]


def department_daily_demand(lines: DataFrame) -> DataFrame:
    """Items per department per (estimated) day -- the forecasting input."""
    return lines.groupBy(
        F.col("est_order_date").alias("ds"), F.col("department_name").alias("department")
    ).agg(
        F.count("*").alias("items"),
        F.countDistinct("order_id").alias("orders"),
        F.sum("reordered").alias("reordered_items"),
    )


def department_kpis(lines: DataFrame) -> DataFrame:
    """Volume share, reorder rate and reorder interval per department.

    Replaces department_volume_share.sql, reorder_rate_by_department.sql
    and reorder_interval_by_department.sql (same definitions: the
    interval is averaged over order lines, as in the SQL version).
    """
    total = F.sum("item_volume").over(Window.partitionBy())
    return (
        lines.groupBy(F.col("department_name").alias("department"))
        .agg(
            F.count("*").alias("item_volume"),
            F.countDistinct("order_id").alias("orders"),
            F.countDistinct("product_id").alias("products_sold"),
            F.sum("reordered").alias("reordered_items"),
            F.avg("days_since_prior").alias("_avg_gap"),
            F.median("days_since_prior").alias("median_days_between_orders"),
        )
        .withColumn("volume_share_pct", F.round(F.col("item_volume") * 100.0 / total, 2))
        .withColumn("reorder_rate_pct", F.round(F.col("reordered_items") * 100.0 / F.col("item_volume"), 2))
        .withColumn("avg_days_between_orders", F.round("_avg_gap", 2))
        .drop("_avg_gap")
    )


def peak_order_hours(orders: DataFrame) -> DataFrame:
    """peak_order_hours.sql -- all orders, including the item-less test split."""
    total = F.sum("total_orders").over(Window.partitionBy())
    return (
        orders.groupBy("order_hour").agg(F.count("*").alias("total_orders"))
        .withColumn("pct_of_total", F.round(F.col("total_orders") * 100.0 / total, 2))
    )


def basket_size_by_dow(lines: DataFrame) -> DataFrame:
    """basket_size_by_dow.sql. 0=Sunday is a labelled guess, as in the SQL version."""
    return (
        lines.groupBy("order_dow")
        .agg(F.countDistinct("order_id").alias("total_orders"), F.count("*").alias("total_items"))
        .withColumn("day_name", _day_name())
        .withColumn("avg_basket_size", F.round(F.col("total_items") / F.col("total_orders"), 2))
    )


def department_demand_by_hour(lines: DataFrame) -> DataFrame:
    return lines.groupBy(F.col("department_name").alias("department"), "order_hour").agg(
        F.count("*").alias("items"))


def department_demand_by_dow(lines: DataFrame) -> DataFrame:
    """dow_demand_by_department.sql, kept long (one row per department-day)
    rather than pivoted, which is the shape dashboards want."""
    return (
        lines.groupBy(F.col("department_name").alias("department"), "order_dow")
        .agg(F.count("*").alias("items"))
        .withColumn("day_name", _day_name())
    )


def top_products_by_department(lines: DataFrame, top_n: int = 3) -> DataFrame:
    """top_products_by_department.sql -- DENSE_RANK so ties share a rank."""
    counts = lines.groupBy(F.col("department_name").alias("department"), "product_name").agg(
        F.count("*").alias("order_count"))
    rank = F.dense_rank().over(Window.partitionBy("department").orderBy(F.col("order_count").desc()))
    return counts.withColumn("rank", rank).filter(F.col("rank") <= top_n)


def overview_kpis(orders: DataFrame, lines: DataFrame, cfg: LakehouseConfig) -> DataFrame:
    order_stats = orders.agg(
        F.count("*").alias("total_orders"),
        F.countDistinct("user_id").alias("total_customers"),
    )
    line_stats = lines.agg(
        F.count("*").alias("total_order_lines"),
        F.round(F.sum("reordered") * 100.0 / F.count("*"), 2).alias("reorder_rate_pct"),
        F.countDistinct("department_name").alias("departments"),
        F.countDistinct("product_id").alias("products_sold"),
        F.min("est_order_date").alias("first_est_date"),
        F.max("est_order_date").alias("last_est_date"),
    )
    return (
        order_stats.crossJoin(line_stats)
        .withColumn("avg_basket_size", F.round(F.col("total_order_lines") / F.col("total_orders"), 2))
        .withColumn("run_id", F.lit(cfg.run_id))
        .withColumn("refreshed_at", F.current_timestamp())
    )


GOLD_TABLES = {
    "department_daily_demand": "Items, orders and reorders per department per estimated day.",
    "department_kpis": "Volume share, reorder rate and reorder interval per department.",
    "peak_order_hours": "Orders by hour of day.",
    "basket_size_by_dow": "Average items per order by day of week (0=Sunday assumed).",
    "department_demand_by_hour": "Items per department per hour of day.",
    "department_demand_by_dow": "Items per department per day of week.",
    "top_products_by_department": "Top 3 products per department (dense rank).",
    "overview_kpis": "One-row headline numbers for the dashboard.",
}


def run(spark: SparkSession, cfg: LakehouseConfig) -> dict[str, int]:
    lines = spark.table(cfg.silver("order_lines"))
    orders = spark.table(cfg.silver("orders"))
    builders = {
        "department_daily_demand": lambda: department_daily_demand(lines),
        "department_kpis": lambda: department_kpis(lines),
        "peak_order_hours": lambda: peak_order_hours(orders),
        "basket_size_by_dow": lambda: basket_size_by_dow(lines),
        "department_demand_by_hour": lambda: department_demand_by_hour(lines),
        "department_demand_by_dow": lambda: department_demand_by_dow(lines),
        "top_products_by_department": lambda: top_products_by_department(lines),
        "overview_kpis": lambda: overview_kpis(orders, lines, cfg),
    }
    counts = {}
    for name, build in builders.items():
        counts[name] = write_table(build(), cfg.gold(name), cfg, comment=GOLD_TABLES[name])
        print(f"[gold] {name:<28} {counts[name]:>10,} rows")

    results = quality.gold_checks(spark, cfg)
    print("[gold] checks:")
    quality.print_results(results)
    quality.record_results(spark, cfg, results)
    quality.enforce(results)
    return counts
