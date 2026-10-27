"""Gold must give the same answers as the original DuckDB queries in ../sql.

The eight .sql files are run unchanged through Spark SQL against views
named like the DuckDB star schema (fact_orders, fact_order_items,
dim_products, dim_departments), built straight from the CSVs. Then each
result is compared with the gold table that replaces it.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SQL_DIR = os.path.join(REPO_ROOT, "sql")

pytestmark = pytest.mark.skipif(not os.path.isdir(SQL_DIR), reason="original sql/ folder not found")


@pytest.fixture(scope="module")
def reference(spark, built):
    landing = built["cfg"].landing_path

    def csv(name):
        # DuckDB's read_csv_auto treats "" inside quotes as a quote (RFC 4180); match it
        return (spark.read.option("header", True).option("inferSchema", True).option("escape", '"')
                .csv(os.path.join(landing, name)))

    csv("orders.csv").selectExpr(
        "order_id", "user_id", "eval_set", "order_number", "order_dow",
        "order_hour_of_day AS order_hour", "CAST(days_since_prior_order AS DOUBLE) AS days_since_prior",
    ).createOrReplaceTempView("fact_orders")
    csv("order_products__prior.csv").unionByName(csv("order_products__train.csv")).selectExpr(
        "order_id", "product_id", "add_to_cart_order", "reordered AS is_reordered",
    ).createOrReplaceTempView("fact_order_items")
    csv("products.csv").createOrReplaceTempView("dim_products")
    csv("departments.csv").selectExpr("department_id", "department AS department_name") \
        .createOrReplaceTempView("dim_departments")

    def run(name):
        with open(os.path.join(SQL_DIR, name)) as f:
            return spark.sql(f.read().strip().rstrip(";")).toPandas()

    return run


def _gold(spark, built, name):
    return spark.table(built["cfg"].gold(name)).toPandas()


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    # Spark SQL returns DECIMAL for ROUND(x * 100.0 / y); compare as floats
    return df.apply(lambda c: c.map(lambda v: float(v) if isinstance(v, Decimal) else v))


def _same(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]):
    a = _plain(a).sort_values(keys).reset_index(drop=True)
    b = _plain(b).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=0.011)


def test_reorder_rate_by_department(spark, built, reference):
    ref = reference("reorder_rate_by_department.sql")
    gold = _gold(spark, built, "department_kpis").rename(columns={"department": "department_name"})
    _same(ref[["department_name", "total_item_orders", "reorder_rate_pct"]],
          gold.rename(columns={"item_volume": "total_item_orders"})[
              ["department_name", "total_item_orders", "reorder_rate_pct"]], ["department_name"])


def test_department_volume_share(spark, built, reference):
    ref = reference("department_volume_share.sql")
    gold = _gold(spark, built, "department_kpis").rename(columns={"department": "department_name"})
    _same(ref, gold[["department_name", "item_volume", "volume_share_pct"]], ["department_name"])


def test_reorder_interval_by_department(spark, built, reference):
    ref = reference("reorder_interval_by_department.sql")
    gold = _gold(spark, built, "department_kpis").rename(columns={"department": "department_name"})
    _same(ref, gold[["department_name", "avg_days_between_orders", "median_days_between_orders"]],
          ["department_name"])


def test_peak_order_hours(spark, built, reference):
    _same(reference("peak_order_hours.sql"),
          _gold(spark, built, "peak_order_hours")[["order_hour", "total_orders", "pct_of_total"]],
          ["order_hour"])


def test_basket_size_by_dow(spark, built, reference):
    _same(reference("basket_size_by_dow.sql"),
          _gold(spark, built, "basket_size_by_dow")[
              ["order_dow", "day_name", "total_orders", "total_items", "avg_basket_size"]],
          ["order_dow"])


def test_department_demand_by_hour(spark, built, reference):
    gold = _gold(spark, built, "department_demand_by_hour").rename(
        columns={"department": "department_name", "items": "hourly_orders"})
    _same(reference("department_demand_by_hour.sql"), gold[["department_name", "order_hour", "hourly_orders"]],
          ["department_name", "order_hour"])


def test_dow_demand_by_department(spark, built, reference):
    ref = reference("dow_demand_by_department.sql")
    long = _gold(spark, built, "department_demand_by_dow")
    wide = long.pivot(index="department", columns="order_dow", values="items").fillna(0)
    wide.columns = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
    wide["total_demand"] = wide.sum(axis=1)
    wide = wide.reset_index().rename(columns={"department": "department_name"})
    _same(ref, wide[ref.columns.tolist()], ["department_name"])


def test_top_products_by_department(spark, built, reference):
    ref = reference("top_products_by_department.sql")
    gold = _gold(spark, built, "top_products_by_department").rename(columns={"department": "department_name"})
    _same(ref, gold[["department_name", "product_name", "order_count"]], ["department_name", "product_name"])
