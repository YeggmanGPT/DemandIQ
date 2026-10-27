"""Data-quality checks and the gate between silver and gold.

Two kinds of protection run in this pipeline:

1. Row level (silver.py): each bad record is quarantined with a reason.
2. Table level (this file): after silver, and again after gold, a set of
   checks looks at whole tables -- reconciliation of row counts between
   layers, key uniqueness, volume, referential integrity, quarantine rate.

Every check result is appended to ``gold.dq_check_results`` (one row per
check per run), which the dashboard's Data quality page reads. A failed
``error`` check raises ``DataQualityError``; the job task fails and gold
is not rebuilt from bad silver data. ``warn`` checks are recorded only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType

from .config import LakehouseConfig
from .storage import write_table

RESULTS_TABLE = "dq_check_results"
QUARANTINE_SUMMARY_TABLE = "dq_quarantine_summary"

RESULT_SCHEMA = StructType([
    StructField("stage", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("observed", DoubleType(), True),
    StructField("expected", StringType(), False),
    StructField("passed", BooleanType(), False),
    StructField("details", StringType(), True),
])


class DataQualityError(RuntimeError):
    pass


@dataclass
class CheckResult:
    stage: str
    check_name: str
    table_name: str
    severity: str  # "error" fails the run, "warn" is recorded only
    observed: float
    expected: str
    passed: bool
    details: str = ""


def _scalar(spark: SparkSession, sql: str) -> float:
    value = spark.sql(sql).collect()[0][0]
    return float(value) if value is not None else 0.0


def _check(stage, name, table, severity, observed, expected_text, passed, details=""):
    return CheckResult(stage, name, table, severity, float(observed), expected_text, bool(passed), details)


# --------------------------------------------------------------------------
# silver
# --------------------------------------------------------------------------

def silver_checks(
    spark: SparkSession, cfg: LakehouseConfig, min_orders: int, max_quarantine_pct: float = 0.1
) -> list[CheckResult]:
    s, b, stage = cfg.silver, cfg.bronze, "silver"
    out: list[CheckResult] = []

    quarantined = {
        row["entity"]: row["n"]
        for row in spark.table(s("quarantine")).groupBy("entity").agg(F.count("*").alias("n")).collect()
    }

    # 1. Reconciliation: every bronze row is either in silver or quarantined.
    entities = {
        "departments": ([b("departments_raw")], s("departments")),
        "aisles": ([b("aisles_raw")], s("aisles")),
        "products": ([b("products_raw")], s("products")),
        "orders": ([b("orders_raw")], s("orders")),
        "order_lines": ([b("order_products_prior_raw"), b("order_products_train_raw")], s("order_lines")),
    }
    bronze_counts, silver_counts = {}, {}
    for entity, (bronze_tables, silver_table) in entities.items():
        bronze_counts[entity] = sum(spark.table(t).count() for t in bronze_tables)
        silver_counts[entity] = spark.table(silver_table).count()
        accounted = silver_counts[entity] + quarantined.get(entity, 0)
        out.append(_check(
            stage, f"{entity}_reconciled_bronze_to_silver", silver_table, "error",
            accounted, f"= {bronze_counts[entity]:,} bronze rows",
            accounted == bronze_counts[entity],
            f"silver={silver_counts[entity]:,}, quarantined={quarantined.get(entity, 0):,}",
        ))

    # 2. Quarantine rate: a handful of bad rows is normal, a lot means a
    #    systemic problem (wrong file, changed export format).
    for entity in ("orders", "order_lines", "products"):
        rate = 100.0 * quarantined.get(entity, 0) / max(bronze_counts[entity], 1)
        out.append(_check(
            stage, f"{entity}_quarantine_rate_pct", s(entity), "error", rate,
            f"<= {max_quarantine_pct}%", rate <= max_quarantine_pct,
        ))

    # 3. Volume: catches a truncated or partial upload.
    out.append(_check(
        stage, "orders_min_volume", s("orders"), "error", silver_counts["orders"],
        f">= {min_orders:,}", silver_counts["orders"] >= min_orders,
    ))

    # 4. Keys.
    dup_orders = _scalar(spark, f"SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM {s('orders')}")
    out.append(_check(stage, "orders_order_id_unique", s("orders"), "error", dup_orders, "= 0",
                      dup_orders == 0))
    dup_lines = _scalar(
        spark, f"SELECT COUNT(*) - COUNT(DISTINCT order_id, product_id) FROM {s('order_lines')}")
    out.append(_check(stage, "order_lines_key_unique", s("order_lines"), "error", dup_lines, "= 0",
                      dup_lines == 0, "key = (order_id, product_id)"))

    # 5. Enrichment and referential integrity in the table gold reads.
    unenriched = _scalar(spark, f"""
        SELECT COUNT(*) FROM {s('order_lines')}
        WHERE department_name IS NULL OR est_order_date IS NULL OR order_dow IS NULL""")
    out.append(_check(stage, "order_lines_fully_enriched", s("order_lines"), "error", unenriched,
                      "= 0", unenriched == 0, "department, date and order attributes present"))

    # Lines from order_products__prior.csv must belong to 'prior' orders,
    # lines from __train.csv to 'train' orders. A mismatch means files were
    # swapped or mislabelled at upload.
    split_mismatch = _scalar(
        spark, f"SELECT COUNT(*) FROM {s('order_lines')} WHERE source_split <> eval_set")
    out.append(_check(stage, "order_lines_split_matches_order", s("order_lines"), "error",
                      split_mismatch, "= 0", split_mismatch == 0))

    # 6. Known properties of the dataset (warn only).
    for name, table, expected in (("departments_count", "departments", 21),
                                  ("aisles_count", "aisles", 134)):
        n = silver_counts[table]
        out.append(_check(stage, name, s(table), "warn", n, f"= {expected}", n == expected))

    first_with_gap = _scalar(spark, f"""
        SELECT COUNT(*) FROM {s('orders')} WHERE order_number = 1 AND days_since_prior IS NOT NULL""")
    out.append(_check(stage, "first_orders_have_no_gap", s("orders"), "warn", first_with_gap, "= 0",
                      first_with_gap == 0))
    repeat_without_gap = _scalar(spark, f"""
        SELECT COUNT(*) FROM {s('orders')} WHERE order_number > 1 AND days_since_prior IS NULL""")
    out.append(_check(stage, "repeat_orders_have_gap", s("orders"), "warn", repeat_without_gap, "= 0",
                      repeat_without_gap == 0))
    # How well the reconstructed calendar agrees with the recorded weekday.
    # First orders match by construction; later ones only match if the
    # gap-days are exact (Instacart caps gaps at 30, which breaks chains).
    dow_match = _scalar(spark, f"""
        SELECT 100.0 * AVG(CASE WHEN dayofweek(est_order_date) - 1 = order_dow THEN 1 ELSE 0 END)
        FROM {s('orders')}""")
    out.append(_check(stage, "est_date_matches_order_dow_pct", s("orders"), "warn", dow_match, ">= 50%",
                      dow_match >= 50.0, "share of orders whose estimated date falls on order_dow"))

    test_lines = _scalar(spark, f"SELECT COUNT(*) FROM {s('order_lines')} WHERE eval_set = 'test'")
    out.append(_check(stage, "test_orders_have_no_lines", s("order_lines"), "warn", test_lines, "= 0",
                      test_lines == 0, "Kaggle never released items for the 'test' split"))
    return out


# --------------------------------------------------------------------------
# gold
# --------------------------------------------------------------------------

def gold_checks(spark: SparkSession, cfg: LakehouseConfig) -> list[CheckResult]:
    g, s, stage = cfg.gold, cfg.silver, "gold"
    out: list[CheckResult] = []

    silver_lines = spark.table(s("order_lines")).count()
    gold_items = _scalar(spark, f"SELECT SUM(items) FROM {g('department_daily_demand')}")
    out.append(_check(stage, "daily_demand_reconciled_to_silver", g("department_daily_demand"), "error",
                      gold_items, f"= {silver_lines:,} silver order lines", gold_items == silver_lines))

    negative = _scalar(spark, f"SELECT COUNT(*) FROM {g('department_daily_demand')} WHERE items < 0")
    out.append(_check(stage, "daily_demand_non_negative", g("department_daily_demand"), "error",
                      negative, "= 0", negative == 0))

    share = _scalar(spark, f"SELECT SUM(volume_share_pct) FROM {g('department_kpis')}")
    out.append(_check(stage, "volume_share_sums_to_100", g("department_kpis"), "error", share,
                      "100 +/- 0.5", abs(share - 100.0) <= 0.5))

    expected_departments = _scalar(
        spark, f"SELECT COUNT(DISTINCT department_name) FROM {s('order_lines')}")
    kpi_departments = spark.table(g("department_kpis")).count()
    out.append(_check(stage, "kpis_cover_all_departments", g("department_kpis"), "error",
                      kpi_departments, f"= {int(expected_departments)}",
                      kpi_departments == expected_departments))

    max_rank = _scalar(spark, f"SELECT MAX(`rank`) FROM {g('top_products_by_department')}")
    out.append(_check(stage, "top_products_rank_le_3", g("top_products_by_department"), "error",
                      max_rank, "<= 3", max_rank <= 3))
    return out


# --------------------------------------------------------------------------
# recording + gate
# --------------------------------------------------------------------------

def record_results(spark: SparkSession, cfg: LakehouseConfig, results: list[CheckResult]) -> None:
    # plain tuples in schema order work the same on classic Spark and Spark Connect (serverless)
    rows = [tuple(asdict(r)[f.name] for f in RESULT_SCHEMA.fields) for r in results]
    df = (
        spark.createDataFrame(rows, schema=RESULT_SCHEMA)
        .withColumn("run_id", F.lit(cfg.run_id))
        .withColumn("checked_at", F.current_timestamp())
    )
    write_table(df, cfg.gold(RESULTS_TABLE), cfg, mode="append",
                comment="One row per data-quality check per pipeline run.")


def write_quarantine_summary(spark: SparkSession, cfg: LakehouseConfig) -> None:
    summary = (
        spark.table(cfg.silver("quarantine"))
        .groupBy("entity", "reason").agg(F.count("*").alias("rows"))
        .withColumn("run_id", F.lit(cfg.run_id))
        .withColumn("checked_at", F.current_timestamp())
    )
    write_table(summary, cfg.gold(QUARANTINE_SUMMARY_TABLE), cfg,
                comment="Quarantined rows by entity and reason, latest run.")


def print_results(results: list[CheckResult]) -> None:
    for r in results:
        status = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "WARN")
        print(f"  [{status}] {r.stage}.{r.check_name}: observed={r.observed:,.2f} expected {r.expected}"
              + (f" ({r.details})" if r.details else ""))


def enforce(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.severity == "error" and not r.passed]
    if failed:
        names = ", ".join(f"{r.stage}.{r.check_name}" for r in failed)
        raise DataQualityError(f"{len(failed)} blocking data-quality check(s) failed: {names}")


def run(spark: SparkSession, cfg: LakehouseConfig, min_orders: int) -> list[CheckResult]:
    """Job task: the gate between silver and gold."""
    results = silver_checks(spark, cfg, min_orders=min_orders)
    print("[quality] silver checks:")
    print_results(results)
    record_results(spark, cfg, results)
    write_quarantine_summary(spark, cfg)
    enforce(results)
    return results
