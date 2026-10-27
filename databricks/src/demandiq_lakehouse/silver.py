"""Silver layer: typed, de-duplicated, conformed tables plus a quarantine.

Every bronze row ends up in exactly one place: a silver table, or
``silver.quarantine`` with the first rule it broke. That makes the
bronze -> silver step auditable (the quality gate reconciles the row
counts) and means one bad record never blocks the whole load.

Tables written:
  departments, aisles     lookups
  products                products + aisle and department names
  orders                  one row per order, plus an estimated order date
  order_lines             one row per product in an order, joined to its
                          order and product (the table gold reads from)
  quarantine              rejected rows: entity, reason, raw record as JSON
"""

from __future__ import annotations

from datetime import date
from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import ANCHOR_SEED, ANCHOR_WINDOW_DAYS, SIM_START, LakehouseConfig
from .storage import add_check_constraints, write_table

REASON = "_dq_reason"
META_COLS = ["_source_file", "_run_id"]

ORDER_CONSTRAINTS = {
    "valid_order_dow": "order_dow BETWEEN 0 AND 6",
    "valid_order_hour": "order_hour BETWEEN 0 AND 23",
    "valid_days_since_prior": "days_since_prior IS NULL OR days_since_prior BETWEEN 0 AND 30",
    "positive_order_number": "order_number >= 1",
}
ORDER_LINE_CONSTRAINTS = {
    "valid_reordered": "reordered IN (0, 1)",
    "positive_cart_position": "add_to_cart_order >= 1",
}


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def _raw(c: str) -> str:
    return f"_raw_{c}"


def _keep_raw(df: DataFrame, cols: list[str]) -> DataFrame:
    for c in cols:
        df = df.withColumnRenamed(c, _raw(c))
    return df


def _try_cast(c: str, sql_type: str) -> Column:
    # Numbers go through DOUBLE first so "8.0" (how pandas exports some
    # integer columns) still parses; with ANSI mode on, a direct
    # try_cast('8.0' AS INT) would give NULL.
    if sql_type in ("INT", "BIGINT"):
        return F.expr(f"try_cast(try_cast(`{_raw(c)}` AS DOUBLE) AS {sql_type})")
    return F.expr(f"try_cast(`{_raw(c)}` AS {sql_type})")


def _is_blank(c: str) -> Column:
    return F.col(_raw(c)).isNull() | (F.trim(F.col(_raw(c))) == "")


def _required(c: str) -> tuple[str, Column]:
    return f"missing:{c}", _is_blank(c)


def _parse_failed(c: str) -> tuple[str, Column]:
    return f"invalid_type:{c}", ~_is_blank(c) & F.col(c).isNull()


def _first_failure(rules: list[tuple[str, Column]]) -> Column:
    expr = None
    for reason, condition in rules:
        expr = F.when(condition, F.lit(reason)) if expr is None else expr.when(condition, F.lit(reason))
    return expr if expr is not None else F.lit(None).cast("string")


def _flag_duplicates(df: DataFrame, keys: list[str], reason: str) -> DataFrame:
    """Keep the first still-valid row per key and mark the other valid ones.

    Rows already rejected for another reason are left alone, so a broken
    row never "uses up" the slot a good duplicate should get.
    """
    still_valid = F.col(REASON).isNull()
    rank = F.row_number().over(
        Window.partitionBy(*keys, still_valid).orderBy(F.col("_source_file"), F.col("_ingested_at"))
    )
    return df.withColumn(
        REASON, F.when(still_valid & (rank > 1), F.lit(reason)).otherwise(F.col(REASON))
    )


def _apply_rules(df: DataFrame, rules: list[tuple[str, Column]]) -> DataFrame:
    """Set REASON to the first failing rule, keeping any earlier reason."""
    new_reason = _first_failure(rules)
    if REASON in df.columns:
        return df.withColumn(REASON, F.coalesce(F.col(REASON), new_reason))
    return df.withColumn(REASON, new_reason)


def _valid(df: DataFrame, cols: list[str]) -> DataFrame:
    return df.filter(F.col(REASON).isNull()).select(*cols)


def _quarantined(df: DataFrame, entity: str, raw_cols: list[str]) -> DataFrame:
    return df.filter(F.col(REASON).isNotNull()).select(
        F.lit(entity).alias("entity"),
        F.col(REASON).alias("reason"),
        F.to_json(F.struct(*[F.col(_raw(c)).alias(c) for c in raw_cols])).alias("raw_record"),
        F.col("_source_file").alias("source_file"),
        F.col("_run_id").alias("run_id"),
        F.current_timestamp().alias("quarantined_at"),
    )


# --------------------------------------------------------------------------
# entity rules
# --------------------------------------------------------------------------

def prepare_lookup(raw: DataFrame, id_col: str, name_col: str, out_name: str) -> DataFrame:
    """departments / aisles: integer id + non-empty name, unique id."""
    df = _keep_raw(raw, [id_col, name_col])
    df = df.withColumn(id_col, _try_cast(id_col, "INT")).withColumn(out_name, F.trim(F.col(_raw(name_col))))
    df = _apply_rules(df, [_required(id_col), _parse_failed(id_col), _required(name_col)])
    return _flag_duplicates(df, [id_col], f"duplicate:{id_col}")


def prepare_products(raw: DataFrame, departments: DataFrame, aisles: DataFrame) -> DataFrame:
    cols = ["product_id", "product_name", "aisle_id", "department_id"]
    df = _keep_raw(raw, cols)
    df = (
        df.withColumn("product_id", _try_cast("product_id", "INT"))
        .withColumn("product_name", F.trim(F.col(_raw("product_name"))))
        .withColumn("aisle_id", _try_cast("aisle_id", "INT"))
        .withColumn("department_id", _try_cast("department_id", "INT"))
    )
    df = _apply_rules(df, [
        _required("product_id"), _parse_failed("product_id"), _required("product_name"),
        _required("aisle_id"), _parse_failed("aisle_id"),
        _required("department_id"), _parse_failed("department_id"),
    ])
    df = (
        df.join(F.broadcast(departments.select("department_id", "department_name")), "department_id", "left")
        .join(F.broadcast(aisles.select("aisle_id", "aisle_name")), "aisle_id", "left")
    )
    df = _apply_rules(df, [
        ("unknown_department", F.col("department_name").isNull()),
        ("unknown_aisle", F.col("aisle_name").isNull()),
    ])
    return _flag_duplicates(df, ["product_id"], "duplicate:product_id")


def prepare_orders(raw: DataFrame) -> DataFrame:
    cols = ["order_id", "user_id", "eval_set", "order_number", "order_dow",
            "order_hour_of_day", "days_since_prior_order"]
    df = _keep_raw(raw, cols)
    df = (
        df.withColumn("order_id", _try_cast("order_id", "BIGINT"))
        .withColumn("user_id", _try_cast("user_id", "BIGINT"))
        .withColumn("eval_set", F.lower(F.trim(F.col(_raw("eval_set")))))
        .withColumn("order_number", _try_cast("order_number", "INT"))
        .withColumn("order_dow", _try_cast("order_dow", "INT"))
        .withColumn("order_hour_of_day", _try_cast("order_hour_of_day", "INT"))
        .withColumn("days_since_prior_order", _try_cast("days_since_prior_order", "DOUBLE"))
    )
    df = _apply_rules(df, [
        _required("order_id"), _parse_failed("order_id"),
        _required("user_id"), _parse_failed("user_id"),
        _required("order_number"), _parse_failed("order_number"),
        _required("order_dow"), _parse_failed("order_dow"),
        _required("order_hour_of_day"), _parse_failed("order_hour_of_day"),
        _parse_failed("days_since_prior_order"),
        ("invalid_eval_set", ~F.col("eval_set").isin("prior", "train", "test")),
        ("invalid_order_number", F.col("order_number") < 1),
        ("invalid_order_dow", ~F.col("order_dow").between(0, 6)),
        ("invalid_order_hour", ~F.col("order_hour_of_day").between(0, 23)),
        ("invalid_days_since_prior",
         F.col("days_since_prior_order").isNotNull() & ~F.col("days_since_prior_order").between(0, 30)),
    ])
    return _flag_duplicates(df, ["order_id"], "duplicate:order_id")


def add_estimated_order_date(orders: DataFrame) -> DataFrame:
    """Give every order an approximate calendar date.

    Instacart's public data has no real dates, only day-of-week, hour and
    days since the user's previous order. Each user gets a deterministic
    pseudo-random anchor day in a 90-day window (hash of user_id, so the
    result is identical on every run and every cluster), nudged forward
    by up to 6 days so the first order falls on its recorded order_dow
    (0 = Sunday, the same labelled guess as the SQL layer). The user's
    real gap-days are then summed forward. Gaps and weekdays are measured;
    the absolute calendar date is a modelling assumption.

    The DuckDB version used a random anchor without the weekday nudge, so
    the weekdays in its daily series did not follow the recorded order_dow
    (the real weekly mix was spread evenly across the week). The quality gate
    reports how often a later order's estimated date still lands on its
    recorded weekday (``est_date_matches_order_dow_pct``).
    """
    start_dow = date.fromisoformat(SIM_START).isoweekday() % 7  # Sunday = 0
    raw_anchor = F.pmod(F.xxhash64(F.col("user_id"), F.lit(ANCHOR_SEED)), F.lit(ANCHOR_WINDOW_DAYS))
    first_dow = F.first(
        F.when(F.col("order_number") == 1, F.col("order_dow")), ignorenulls=True
    ).over(Window.partitionBy("user_id"))
    shift = F.coalesce(F.pmod(first_dow - F.pmod(raw_anchor + F.lit(start_dow), F.lit(7)), F.lit(7)),
                       F.lit(0))
    anchor = raw_anchor + shift
    by_user = (
        Window.partitionBy("user_id").orderBy("order_number")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    cumulative_gap = F.sum(F.coalesce(F.col("days_since_prior"), F.lit(0.0))).over(by_user)
    return orders.withColumn(
        "est_order_date",
        F.date_add(F.to_date(F.lit(SIM_START)), (anchor + cumulative_gap).cast("int")),
    )


def prepare_order_lines(
    prior_raw: DataFrame, train_raw: DataFrame, orders: DataFrame, products: DataFrame
) -> DataFrame:
    cols = ["order_id", "product_id", "add_to_cart_order", "reordered"]
    lines = prior_raw.withColumn("_split", F.lit("prior")).unionByName(
        train_raw.withColumn("_split", F.lit("train"))
    )
    df = _keep_raw(lines, cols)
    df = (
        df.withColumn("order_id", _try_cast("order_id", "BIGINT"))
        .withColumn("product_id", _try_cast("product_id", "INT"))
        .withColumn("add_to_cart_order", _try_cast("add_to_cart_order", "INT"))
        .withColumn("reordered", _try_cast("reordered", "INT"))
    )
    df = _apply_rules(df, [
        _required("order_id"), _parse_failed("order_id"),
        _required("product_id"), _parse_failed("product_id"),
        _required("add_to_cart_order"), _parse_failed("add_to_cart_order"),
        _required("reordered"), _parse_failed("reordered"),
        ("invalid_reordered", ~F.col("reordered").isin(0, 1)),
        ("invalid_add_to_cart_order", F.col("add_to_cart_order") < 1),
    ])
    df = _flag_duplicates(df, ["order_id", "product_id"], "duplicate:order_line")

    # One join per parent table does double duty: the referential check
    # (no match -> quarantine) and the enrichment gold needs.
    order_attrs = orders.select(
        "order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour",
        "days_since_prior", "est_order_date",
    ).withColumn("_has_order", F.lit(True))
    product_attrs = products.select(
        "product_id", "product_name", "aisle_id", "aisle_name", "department_id", "department_name",
    ).withColumn("_has_product", F.lit(True))
    df = df.join(order_attrs, "order_id", "left").join(F.broadcast(product_attrs), "product_id", "left")
    return _apply_rules(df, [
        ("orphan:order_id", F.col("_has_order").isNull()),
        ("orphan:product_id", F.col("_has_product").isNull()),
    ])


# --------------------------------------------------------------------------
# job task
# --------------------------------------------------------------------------

ORDER_COLS = ["order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour",
              "days_since_prior", "est_order_date"]
ORDER_LINE_COLS = ["order_id", "product_id", "add_to_cart_order", "reordered", "source_split",
                   "user_id", "eval_set", "order_number", "order_dow", "order_hour",
                   "days_since_prior", "est_order_date", "product_name", "aisle_id", "aisle_name",
                   "department_id", "department_name"]


def run(spark: SparkSession, cfg: LakehouseConfig) -> dict[str, int]:
    b = cfg.bronze
    counts: dict[str, int] = {}
    quarantine: list[DataFrame] = []

    departments = prepare_lookup(spark.table(b("departments_raw")), "department_id", "department",
                                 "department_name")
    aisles = prepare_lookup(spark.table(b("aisles_raw")), "aisle_id", "aisle", "aisle_name")
    quarantine += [_quarantined(departments, "departments", ["department_id", "department"]),
                   _quarantined(aisles, "aisles", ["aisle_id", "aisle"])]
    counts["departments"] = write_table(
        _valid(departments, ["department_id", "department_name"]), cfg.silver("departments"), cfg,
        "Instacart departments (21 expected).")
    counts["aisles"] = write_table(
        _valid(aisles, ["aisle_id", "aisle_name"]), cfg.silver("aisles"), cfg,
        "Instacart aisles (134 expected).")

    products = prepare_products(spark.table(b("products_raw")), spark.table(cfg.silver("departments")),
                                spark.table(cfg.silver("aisles")))
    quarantine.append(_quarantined(products, "products",
                                   ["product_id", "product_name", "aisle_id", "department_id"]))
    counts["products"] = write_table(
        _valid(products, ["product_id", "product_name", "aisle_id", "aisle_name", "department_id",
                          "department_name"]),
        cfg.silver("products"), cfg, "Products with aisle and department names.")

    orders = prepare_orders(spark.table(b("orders_raw")))
    quarantine.append(_quarantined(orders, "orders", [
        "order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour_of_day",
        "days_since_prior_order"]))
    valid_orders = add_estimated_order_date(
        _valid(orders, ["order_id", "user_id", "eval_set", "order_number", "order_dow",
                        "order_hour_of_day", "days_since_prior_order"])
        .withColumnRenamed("order_hour_of_day", "order_hour")
        .withColumnRenamed("days_since_prior_order", "days_since_prior")
    )
    counts["orders"] = write_table(
        valid_orders.select(*ORDER_COLS), cfg.silver("orders"), cfg,
        "One row per order. est_order_date is reconstructed (see README).")
    add_check_constraints(spark, cfg.silver("orders"), cfg, ORDER_CONSTRAINTS)

    lines = prepare_order_lines(
        spark.table(b("order_products_prior_raw")), spark.table(b("order_products_train_raw")),
        spark.table(cfg.silver("orders")), spark.table(cfg.silver("products")),
    ).withColumnRenamed("_split", "source_split")
    quarantine.append(_quarantined(lines, "order_lines",
                                   ["order_id", "product_id", "add_to_cart_order", "reordered"]))
    counts["order_lines"] = write_table(
        _valid(lines, ORDER_LINE_COLS), cfg.silver("order_lines"), cfg,
        "One row per product in an order, joined to order and product attributes.")
    add_check_constraints(spark, cfg.silver("order_lines"), cfg, ORDER_LINE_CONSTRAINTS)

    counts["quarantine"] = write_table(
        reduce(lambda a, b: a.unionByName(b), quarantine), cfg.silver("quarantine"), cfg,
        "Rows rejected on the way from bronze to silver, with the first rule they broke.")

    for name, n in counts.items():
        print(f"[silver] {name:<14} {n:>12,} rows")
    return counts
