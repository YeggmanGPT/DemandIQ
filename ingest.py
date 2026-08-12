"""
ingest.py

Loads the raw Instacart CSV exports into a local DuckDB warehouse:
three dimension tables (products, aisles, departments) and two fact
tables (orders, order line items).

Run from the project root, after placing the six Instacart CSVs in data/:
    python ingest.py
"""

import os
import sys
import time

import duckdb

DB_PATH = os.path.join("db", "demandiq.duckdb")
DATA_DIR = "data"

REQUIRED_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]


def check_data_files() -> None:
    """Confirm the raw Kaggle CSVs are present before touching DuckDB."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Missing files in '{DATA_DIR}/': {missing}")
        print("Download the dataset from:")
        print("  https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis")
        print(f"and extract the CSVs into {DATA_DIR}/")
        sys.exit(1)
    print("All required raw CSV files found.")


def build_warehouse(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the dimension and fact tables from the raw CSVs."""
    print("Building warehouse...")

    conn.execute("""
        CREATE OR REPLACE TABLE dim_departments AS
        SELECT
            CAST(department_id AS INTEGER) AS department_id,
            CAST(department AS VARCHAR)    AS department_name
        FROM read_csv_auto('data/departments.csv');
    """)

    conn.execute("""
        CREATE OR REPLACE TABLE dim_aisles AS
        SELECT
            CAST(aisle_id AS INTEGER) AS aisle_id,
            CAST(aisle AS VARCHAR)    AS aisle_name
        FROM read_csv_auto('data/aisles.csv');
    """)

    conn.execute("""
        CREATE OR REPLACE TABLE dim_products AS
        SELECT
            CAST(product_id AS INTEGER)   AS product_id,
            CAST(product_name AS VARCHAR) AS product_name,
            CAST(aisle_id AS INTEGER)     AS aisle_id,
            CAST(department_id AS INTEGER) AS department_id
        FROM read_csv_auto('data/products.csv');
    """)

    conn.execute("""
        CREATE OR REPLACE TABLE fact_orders AS
        SELECT
            CAST(order_id AS BIGINT)               AS order_id,
            CAST(user_id AS BIGINT)                AS user_id,
            CAST(eval_set AS VARCHAR)              AS eval_set,
            CAST(order_number AS INTEGER)          AS order_number,
            CAST(order_dow AS INTEGER)             AS order_dow,
            CAST(order_hour_of_day AS INTEGER)     AS order_hour,
            CAST(days_since_prior_order AS DOUBLE) AS days_since_prior
        FROM read_csv_auto('data/orders.csv');
    """)

    # 'prior' and 'train' cover disjoint order_ids -- the Kaggle 'test'
    # split orders (eval_set = 'test') never had their item data released,
    # so a plain UNION ALL here is safe and there's nothing to de-dupe.
    # Those ~75k test orders still live in fact_orders (useful for
    # order-level queries) but won't join to any items -- expected, not a bug.
    conn.execute("""
        CREATE OR REPLACE TABLE fact_order_items AS
        SELECT
            CAST(order_id AS BIGINT)           AS order_id,
            CAST(product_id AS INTEGER)        AS product_id,
            CAST(add_to_cart_order AS INTEGER) AS add_to_cart_order,
            CAST(reordered AS INTEGER)         AS is_reordered
        FROM read_csv_auto('data/order_products__prior.csv')
        UNION ALL
        SELECT
            CAST(order_id AS BIGINT)           AS order_id,
            CAST(product_id AS INTEGER)        AS product_id,
            CAST(add_to_cart_order AS INTEGER) AS add_to_cart_order,
            CAST(reordered AS INTEGER)         AS is_reordered
        FROM read_csv_auto('data/order_products__train.csv');
    """)

    print("Warehouse tables created.")


def print_row_counts(conn: duckdb.DuckDBPyConnection) -> None:
    tables = ["dim_departments", "dim_aisles", "dim_products", "fact_orders", "fact_order_items"]
    print("\nRow counts:")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<20} {count:,}")


def validate_warehouse(conn: duckdb.DuckDBPyConnection) -> bool:
    """A handful of sanity checks. Returns False if anything looks wrong."""
    checks = {
        "products with null name": """
            SELECT COUNT(*) FROM dim_products WHERE product_name IS NULL
        """,
        "orders with null day-of-week": """
            SELECT COUNT(*) FROM fact_orders WHERE order_dow IS NULL
        """,
        "order_items with no matching order": """
            SELECT COUNT(*) FROM fact_order_items foi
            LEFT JOIN fact_orders fo ON foi.order_id = fo.order_id
            WHERE fo.order_id IS NULL
        """,
        "products with no matching department": """
            SELECT COUNT(*) FROM dim_products p
            LEFT JOIN dim_departments d ON p.department_id = d.department_id
            WHERE d.department_id IS NULL
        """,
    }
    print("\nData quality checks:")
    all_passed = True
    for name, query in checks.items():
        result = conn.execute(query).fetchone()[0]
        ok = result == 0
        all_passed = all_passed and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {result}")
    return all_passed


def run_sql_file(conn: duckdb.DuckDBPyConnection, filename: str, preview_rows: int = 10):
    """Run one of the sql/*.sql files against the warehouse and preview it.
    Handy for smoke-testing a query from a REPL:
        >>> import duckdb, ingest
        >>> conn = duckdb.connect(ingest.DB_PATH)
        >>> ingest.run_sql_file(conn, "reorder_rate_by_department.sql")
    """
    path = os.path.join("sql", filename)
    with open(path) as f:
        query = f.read()
    result = conn.execute(query).df()
    print(f"\n--- {filename} ({len(result)} rows) ---")
    print(result.head(preview_rows).to_string(index=False))
    return result


if __name__ == "__main__":
    check_data_files()
    os.makedirs("db", exist_ok=True)

    start = time.time()
    conn = duckdb.connect(DB_PATH)
    build_warehouse(conn)
    print(f"Ingested in {time.time() - start:.1f}s")

    print_row_counts(conn)
    passed = validate_warehouse(conn)
    if not passed:
        print("\nOne or more data quality checks failed -- inspect before continuing to Day 2.")

    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\nWarehouse file: {DB_PATH} ({db_size_mb:.1f} MB)")

    conn.close()
