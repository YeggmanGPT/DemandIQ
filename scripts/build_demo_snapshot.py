"""
scripts/build_demo_snapshot.py

Builds db/demandiq_demo.duckdb: a small copy of the warehouse that holds only
*derived aggregates*, so it is safe to commit to git. The scheduled GitHub
Actions job (.github/workflows/anomaly-check.yml) runs against it.

Why it exists
-------------
The full warehouse, db/demandiq.duckdb, is gitignored: it is built from the
Instacart CSVs (third-party licensed data, and far too big to commit), so
a fresh CI checkout has no warehouse. Before this snapshot existed the monitor
crashed there with "Table with name department_forecast_evaluation does not
exist".

What goes in
------------
    department_forecast_evaluation  hold-out actual vs. forecast per department-day
                                    (REQUIRED -- scheduler/monitor.py scans this)
    daily_department_features       daily demand per department + lag/rolling features
    dim_departments                 the department names
    anomaly_scan                    the last anomaly scan, if there is one
    sql_<name>                      the result of each sql/<name>.sql query
                                    (sql_top_products_by_department holds three
                                    product names per department -- the only
                                    product-level data in the file)
    snapshot_meta                   when it was built, and with which DuckDB version

What stays out: order-level rows (fact_orders, fact_order_items) and the product
and aisle catalogues, i.e. nothing that amounts to redistributing the dataset.

Usage (project root, after ingest -> features -> forecaster -> anomaly):
    python scripts/build_demo_snapshot.py
    git add db/demandiq_demo.duckdb
    git commit -m "Refresh demo snapshot" && git push
"""

import argparse
import datetime as dt
import os
import sys

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(ROOT, "sql")
# Deliberately NOT read from DEMANDIQ_DB_PATH: in CI that variable points at the
# snapshot itself, and a builder that read from it would copy the snapshot onto itself.
DEFAULT_SOURCE = os.path.join(ROOT, "db", "demandiq.duckdb")
DEFAULT_OUT = os.path.join(ROOT, "db", "demandiq_demo.duckdb")
WARN_SIZE_MB = 25  # GitHub warns about files over 50 MB and rejects them over 100 MB

REQUIRED_TABLE = "department_forecast_evaluation"
# Copied as they are; anything that isn't in the source is skipped with a note.
COPIED_TABLES = [REQUIRED_TABLE, "daily_department_features", "dim_departments", "anomaly_scan"]


def _literal(path: str) -> str:
    """A path as the body of a single-quoted SQL string."""
    return path.replace("'", "''")


def build(source: str, out: str) -> None:
    if not os.path.exists(source):
        sys.exit(
            f"No warehouse at {source}. Build it first:\n"
            "    python ingest.py && python features.py && python forecaster.py && python anomaly.py"
        )
    if os.path.abspath(source) == os.path.abspath(out):
        sys.exit("--source and --out are the same file; refusing to overwrite the warehouse.")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Build beside the target and swap it in at the end, so a failed run never
    # leaves a half-written (or deleted) snapshot behind.
    building = os.path.splitext(out)[0] + ".building.duckdb"
    for leftover in (building, building + ".wal"):
        if os.path.exists(leftover):
            os.remove(leftover)

    conn = duckdb.connect(":memory:")
    conn.execute(f"ATTACH '{_literal(source)}' AS src (READ_ONLY)")
    conn.execute(f"ATTACH '{_literal(building)}' AS demo")
    try:
        present = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_catalog = 'src' AND table_schema = 'main'"
            ).fetchall()
        }
        if REQUIRED_TABLE not in present:
            sys.exit(f"{source} has no {REQUIRED_TABLE} table. Run `python forecaster.py` first.")
        if conn.execute(f"SELECT COUNT(*) FROM src.main.{REQUIRED_TABLE}").fetchone()[0] == 0:
            sys.exit(f"{REQUIRED_TABLE} in {source} is empty. Re-run `python forecaster.py`.")

        notes = []
        for name in COPIED_TABLES:
            if name in present:
                conn.execute(f'CREATE TABLE demo.main."{name}" AS SELECT * FROM src.main."{name}"')
            else:
                notes.append(f"skipped {name} (not in the source warehouse)")

        # The saved queries use unqualified table names, so point the session at
        # the source warehouse while they run; each result lands in demo.
        conn.execute("USE src")
        if not os.path.isdir(SQL_DIR):
            notes.append(f"WARNING: {SQL_DIR} not found, so no sql_* tables were added; run this script from inside the repo")
        sql_files = sorted(f for f in os.listdir(SQL_DIR) if f.endswith(".sql")) if os.path.isdir(SQL_DIR) else []
        for fname in sql_files:
            stem = os.path.splitext(fname)[0]
            with open(os.path.join(SQL_DIR, fname), encoding="utf-8") as f:
                query = f.read().strip().rstrip(";")
            try:
                # The newline before ")" keeps a trailing "-- comment" from swallowing it.
                conn.execute(f'CREATE TABLE demo.main."sql_{stem}" AS (\n{query}\n)')
            except duckdb.Error as exc:
                notes.append(f"skipped sql_{stem}: {str(exc).splitlines()[0]}")

        conn.execute("CREATE TABLE demo.main.snapshot_meta (built_at TIMESTAMP, duckdb_version VARCHAR, note VARCHAR)")
        conn.execute(
            "INSERT INTO demo.main.snapshot_meta VALUES (?, ?, ?)",
            [
                dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                duckdb.__version__,
                "Derived aggregates only (no order-level rows). Built by scripts/build_demo_snapshot.py.",
            ],
        )
        conn.execute("USE memory")
        conn.execute("DETACH demo")
    except BaseException:
        conn.close()
        for leftover in (building, building + ".wal"):
            if os.path.exists(leftover):
                os.remove(leftover)
        raise
    conn.close()

    os.replace(building, out)
    report(out, notes)


def report(out: str, notes: list) -> None:
    check = duckdb.connect(out, read_only=True)  # also proves the finished file opens
    try:
        tables = [r[0] for r in check.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()]
        counts = [(t, check.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) for t in tables]
    finally:
        check.close()

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"Built {os.path.relpath(out)} ({size_mb:.1f} MB, DuckDB {duckdb.__version__})")
    for table, n in counts:
        print(f"  {table:<40} {n:>9,} rows")
    for note in notes:
        print(f"  note: {note}")
    if size_mb > WARN_SIZE_MB:
        print(f"WARNING: {size_mb:.0f} MB is large for a file that lives in git; check what went in.")
    print(
        "\nNext: commit it so the scheduled workflow can use it:\n"
        f"    git add {os.path.relpath(out)}\n"
        '    git commit -m "Refresh demo snapshot" && git push'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the small, commit-safe demo snapshot used by the scheduled CI job.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="full warehouse to read (default: db/demandiq.duckdb)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="snapshot to write (default: db/demandiq_demo.duckdb)")
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
