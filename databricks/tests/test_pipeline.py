"""End-to-end behaviour of bronze -> silver -> quality gate -> gold on clean data."""

from __future__ import annotations

import pytest

from conftest import USE_DELTA


def test_bronze_keeps_every_csv_row(spark, built):
    cfg, csv = built["cfg"], built["csv_counts"]
    assert spark.table(cfg.bronze("orders_raw")).count() == csv["orders.csv"]
    assert spark.table(cfg.bronze("order_products_prior_raw")).count() == csv["order_products__prior.csv"]
    assert spark.table(cfg.bronze("order_products_train_raw")).count() == csv["order_products__train.csv"]
    assert spark.table(cfg.bronze("products_raw")).count() == csv["products.csv"]
    # bronze is raw text plus load metadata
    fields = {f.name: f.dataType.simpleString() for f in spark.table(cfg.bronze("orders_raw")).schema}
    assert fields["order_id"] == "string"
    assert {"_source_file", "_ingested_at", "_run_id"} <= set(fields)


def test_clean_data_is_not_quarantined(spark, built):
    cfg, csv = built["cfg"], built["csv_counts"]
    assert spark.table(cfg.silver("quarantine")).count() == 0
    assert spark.table(cfg.silver("orders")).count() == csv["orders.csv"]
    lines = csv["order_products__prior.csv"] + csv["order_products__train.csv"]
    assert spark.table(cfg.silver("order_lines")).count() == lines


def test_silver_types_and_enrichment(spark, built):
    cfg = built["cfg"]
    types = dict((f.name, f.dataType.simpleString()) for f in spark.table(cfg.silver("orders")).schema)
    assert types["order_id"] == "bigint"
    assert types["days_since_prior"] == "double"   # "8.0" text parsed
    assert types["est_order_date"] == "date"
    row = spark.table(cfg.silver("order_lines")).filter("department_name IS NULL").count()
    assert row == 0


def test_quoted_csv_values_survive_ingestion(spark, built):
    cfg = built["cfg"]
    names = {r["product_name"] for r in spark.table(cfg.silver("products")).collect()}
    assert 'Product 7, size 2 "Family Pack"' in names
    assert spark.table(cfg.silver("quarantine")).filter("entity = 'products'").count() == 0


def test_estimated_dates_are_deterministic_and_on_the_recorded_weekday(spark, built):
    from demandiq_lakehouse.silver import add_estimated_order_date

    cfg = built["cfg"]
    orders = spark.table(cfg.silver("orders"))
    again = add_estimated_order_date(orders.drop("est_order_date"))
    diff = orders.select("order_id", "est_order_date").exceptAll(again.select("order_id", "est_order_date"))
    assert diff.count() == 0
    # synthetic gaps are exact, so every order should land on its order_dow
    mismatched = orders.filter("dayofweek(est_order_date) - 1 <> order_dow").count()
    assert mismatched == 0


def test_all_quality_checks_recorded_and_blocking_ones_pass(spark, built):
    cfg = built["cfg"]
    results = spark.table(cfg.gold("dq_check_results")).filter("run_id = 'test-run'")
    stages = {r["stage"] for r in results.select("stage").distinct().collect()}
    assert stages == {"silver", "gold"}
    failed_errors = results.filter("severity = 'error' AND NOT passed").collect()
    assert failed_errors == []
    assert results.count() >= 20


def test_gold_tables_exist_and_reconcile(spark, built):
    cfg = built["cfg"]
    for name in ("department_daily_demand", "department_kpis", "peak_order_hours", "basket_size_by_dow",
                 "department_demand_by_hour", "department_demand_by_dow", "top_products_by_department",
                 "overview_kpis", "forecast_eval", "forecast_metrics", "demand_anomalies",
                 "dq_quarantine_summary"):
        spark.table(cfg.gold(name))  # raises if missing
    kpis = spark.table(cfg.gold("overview_kpis")).collect()
    assert len(kpis) == 1
    lines = spark.table(cfg.silver("order_lines")).count()
    assert kpis[0]["total_order_lines"] == lines
    share = spark.sql(f"SELECT SUM(volume_share_pct) FROM {cfg.gold('department_kpis')}").collect()[0][0]
    assert abs(share - 100) <= 0.5


@pytest.mark.skipif(not USE_DELTA, reason="CHECK constraints need Delta (set DEMANDIQ_TEST_DELTA=1)")
def test_delta_check_constraints_block_bad_writes(spark, built):
    cfg = built["cfg"]
    props = {r["key"] for r in spark.sql(f"SHOW TBLPROPERTIES {cfg.silver('orders')}").collect()}
    assert "delta.constraints.valid_order_hour" in props
    with pytest.raises(Exception, match="(?i)check constraint|DELTA_VIOLATE_CONSTRAINT|CHECK_CONSTRAINT"):
        spark.sql(f"""INSERT INTO {cfg.silver('orders')}
                      VALUES (999999999, 1, 'prior', 1, 2, 31, NULL, DATE'2024-01-01')""")


def test_rerunning_the_job_is_idempotent(spark, built):
    """The schedule re-runs every layer over existing tables (and, on Delta,
    over tables that already carry CHECK constraints). Counts must not drift."""
    from dataclasses import replace

    from demandiq_lakehouse import bronze, gold, quality, silver

    cfg = replace(built["cfg"], run_id="rerun")
    before = {name: spark.table(cfg.silver(name)).count() for name in ("orders", "order_lines", "products")}
    bronze.run(spark, cfg)
    silver.run(spark, cfg)
    quality.run(spark, cfg, min_orders=1000)
    gold.run(spark, cfg)
    after = {name: spark.table(cfg.silver(name)).count() for name in before}
    assert after == before
    reruns = spark.table(cfg.gold("dq_check_results")).filter("run_id = 'rerun'")
    assert reruns.filter("severity = 'error' AND NOT passed").count() == 0
    if USE_DELTA:
        props = {r["key"] for r in spark.sql(f"SHOW TBLPROPERTIES {cfg.silver('orders')}").collect()}
        assert "delta.constraints.valid_order_hour" in props
