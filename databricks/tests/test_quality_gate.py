"""Bad input must be quarantined with a reason, and the gate must stop gold."""

from __future__ import annotations

import os
import shutil

import pytest

import synthetic
from conftest import make_config
from demandiq_lakehouse import bronze, quality, silver
from demandiq_lakehouse.quality import DataQualityError


@pytest.fixture(scope="module")
def dirty(spark, clean_landing, tmp_path_factory):
    """Clean synthetic data plus a few known-bad rows of each kind."""
    src, _ = clean_landing
    path = str(tmp_path_factory.mktemp("landing_dirty"))
    for name in os.listdir(src):
        shutil.copy(os.path.join(src, name), path)
    synthetic.append_rows(path, "orders.csv", [
        {"order_id": 9000001, "user_id": 1, "eval_set": "prior", "order_number": 1, "order_dow": 3,
         "order_hour_of_day": 25, "days_since_prior_order": ""},                       # hour out of range
        {"order_id": "abc", "user_id": 1, "eval_set": "prior", "order_number": 2, "order_dow": 3,
         "order_hour_of_day": 10, "days_since_prior_order": "5.0"},                    # unparseable id
        {"order_id": 9000002, "user_id": 2, "eval_set": "holdout", "order_number": 1, "order_dow": 0,
         "order_hour_of_day": 9, "days_since_prior_order": ""},                        # unknown eval_set
        {"order_id": 1, "user_id": 1, "eval_set": "prior", "order_number": 1, "order_dow": 0,
         "order_hour_of_day": 9, "days_since_prior_order": ""},                        # duplicate order_id 1
    ])
    synthetic.append_rows(path, "order_products__prior.csv", [
        {"order_id": 987654321, "product_id": 1, "add_to_cart_order": 1, "reordered": 0},   # orphan order
        {"order_id": 1, "product_id": 99999, "add_to_cart_order": 9, "reordered": 0},       # orphan product
        {"order_id": 2, "product_id": 5, "add_to_cart_order": 3, "reordered": 7},           # bad flag
    ])
    cfg = make_config("dirty_", path, run_id="dirty-run")
    bronze.run(spark, cfg)
    silver.run(spark, cfg)
    return cfg


def test_each_bad_row_is_quarantined_with_its_reason(spark, dirty):
    reasons = {(r["entity"], r["reason"])
               for r in spark.table(dirty.silver("quarantine")).collect()}
    assert ("orders", "invalid_order_hour") in reasons
    assert ("orders", "invalid_type:order_id") in reasons
    assert ("orders", "invalid_eval_set") in reasons
    assert ("orders", "duplicate:order_id") in reasons
    assert ("order_lines", "orphan:order_id") in reasons
    assert ("order_lines", "orphan:product_id") in reasons
    assert ("order_lines", "invalid_reordered") in reasons
    assert spark.table(dirty.silver("quarantine")).count() == 7


def test_quarantine_keeps_the_raw_values(spark, dirty):
    row = spark.table(dirty.silver("quarantine")).filter("reason = 'invalid_type:order_id'").collect()[0]
    assert '"order_id":"abc"' in row["raw_record"]


def test_no_bad_value_reaches_silver(spark, dirty):
    orders = spark.table(dirty.silver("orders"))
    assert orders.filter("order_hour > 23 OR eval_set NOT IN ('prior','train','test')").count() == 0
    assert orders.filter("order_id = 1").count() == 1


def test_gate_passes_small_quarantine_and_reconciles(spark, dirty):
    results = quality.silver_checks(spark, dirty, min_orders=1000)
    by_name = {r.check_name: r for r in results}
    assert by_name["orders_reconciled_bronze_to_silver"].passed
    assert by_name["order_lines_reconciled_bronze_to_silver"].passed
    quality.enforce(results)  # a handful of bad rows is not a systemic failure


def test_gate_blocks_when_quarantine_rate_is_too_high(spark, dirty):
    results = quality.silver_checks(spark, dirty, min_orders=1000, max_quarantine_pct=0.001)
    with pytest.raises(DataQualityError, match="quarantine_rate"):
        quality.enforce(results)


def test_gate_blocks_a_truncated_upload(spark, dirty):
    results = quality.silver_checks(spark, dirty, min_orders=10_000_000)
    with pytest.raises(DataQualityError, match="orders_min_volume"):
        quality.enforce(results)


def test_quality_task_records_results_before_failing(spark, dirty):
    with pytest.raises(DataQualityError):
        quality.run(spark, dirty, min_orders=10_000_000)
    recorded = spark.table(dirty.gold("dq_check_results")).filter(
        "run_id = 'dirty-run' AND check_name = 'orders_min_volume'").collect()
    assert recorded and not recorded[-1]["passed"]


def test_missing_files_fail_with_instructions(spark, tmp_path):
    cfg = make_config("missing_", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="orders.csv.*Upload|Upload.*orders.csv|orders.csv"):
        bronze.run(spark, cfg)


def test_wrong_header_fails_at_ingestion(spark, clean_landing, tmp_path):
    src, _ = clean_landing
    for name in os.listdir(src):
        shutil.copy(os.path.join(src, name), tmp_path)
    text = (tmp_path / "orders.csv").read_text().replace("order_hour_of_day", "hour", 1)
    (tmp_path / "orders.csv").write_text(text)
    cfg = make_config("badheader_", str(tmp_path))
    with pytest.raises(Exception, match="(?i)header|schema|hour"):
        bronze.run(spark, cfg)
