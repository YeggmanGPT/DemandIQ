"""
scheduler/monitor.py

The script the GitHub Actions cron workflow actually runs. Re-scans for
anomalies and, if anything is at HIGH severity, asks the agent to
investigate and (optionally) posts the resulting report to Slack.

Deliberately a plain script invoked by cron (GitHub Actions), not an
in-process scheduler like APScheduler -- an in-process scheduler dies the
moment the host app sleeps or restarts, which defeats the point of
"runs on a schedule." See BUILD_LOG.md, Module 5.

Which warehouse it scans: anomaly.DB_PATH, i.e. db/demandiq.duckdb unless the
DEMANDIQ_DB_PATH environment variable says otherwise (set it in the shell or the
workflow -- not in .env, which is only read after the path is fixed). CI points it
at the committed snapshot, db/demandiq_demo.duckdb (see
scripts/build_demo_snapshot.py), because the full warehouse is gitignored and a
fresh checkout doesn't have it.

Run manually with:
    python -m scheduler.monitor
"""
import os
import sys

import duckdb
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly import run_anomaly_scan, latest_status_per_department, DB_PATH
from agent.agent import ask

REQUIRED_TABLE = "department_forecast_evaluation"


def _fail(message: str) -> None:
    """Stop with a one-line, actionable reason (exit code 1).

    Inside GitHub Actions the ::error:: command also puts the message in the
    run's annotations -- the first thing you see -- instead of a bare
    "Process completed with exit code 1"."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error title=DemandIQ monitor::{message}")
    sys.exit(f"[monitor] {message}")


def preflight(db_path: str) -> None:
    """Check the warehouse is there and has something to scan *before* scanning it.

    duckdb.connect() quietly creates an empty database file when the path doesn't
    exist, so without this check a missing warehouse shows up as a confusing
    "Table with name department_forecast_evaluation does not exist" from deep
    inside the scan."""
    if not os.path.exists(db_path):
        _fail(
            f"No warehouse at '{db_path}'. Locally: build it with ingest.py, features.py and forecaster.py "
            "(README, steps 1-3). In CI: the full warehouse is gitignored, so commit the snapshot "
            "(`python scripts/build_demo_snapshot.py`, then `git add db/demandiq_demo.duckdb`) "
            "and set DEMANDIQ_DB_PATH=db/demandiq_demo.duckdb on the job."
        )
    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # noqa: BLE001 -- DuckDB's own message says what's wrong (version, corruption, ...)
        _fail(f"Could not open '{db_path}' with DuckDB {duckdb.__version__}: {exc}")
    try:
        tables = {row[0] for row in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if REQUIRED_TABLE not in tables:
            _fail(
                f"'{db_path}' has no {REQUIRED_TABLE} table (it has: {', '.join(sorted(tables)) or 'no tables'}). "
                "Run forecaster.py on the full warehouse, then rebuild the snapshot with scripts/build_demo_snapshot.py."
            )
        n_rows, last_day = conn.execute(f"SELECT COUNT(*), MAX(ds) FROM {REQUIRED_TABLE}").fetchone()
        if n_rows == 0:
            _fail(f"{REQUIRED_TABLE} in '{db_path}' is empty. Re-run forecaster.py and rebuild the snapshot.")
        built = ""
        if "snapshot_meta" in tables:
            row = conn.execute("SELECT built_at FROM snapshot_meta ORDER BY built_at DESC LIMIT 1").fetchone()
            if row:
                built = f", snapshot built {str(row[0])[:16]} UTC"
        print(
            f"[monitor] Warehouse OK: {db_path} (DuckDB {duckdb.__version__}) -- "
            f"{n_rows:,} evaluated department-days, latest {str(last_day)[:10]}{built}."
        )
    finally:
        conn.close()


def post_to_slack(text: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("[monitor] SLACK_WEBHOOK_URL not set -- skipping Slack post, printing instead:\n")
        print(text)
        return
    try:
        resp = requests.post(webhook, json={"text": text}, timeout=10)
        if not resp.ok:
            # Slack answers a bad webhook with a short reason ("no_service", "invalid_payload", ...).
            print(f"[monitor] Slack rejected the post (HTTP {resp.status_code}: {resp.text[:200]}); report was:\n{text}")
    except Exception as exc:  # noqa: BLE001 -- don't let a network hiccup crash the whole run
        # Only the exception type: requests' messages can include the webhook URL, which is a secret.
        print(f"[monitor] Slack post failed ({type(exc).__name__}); report was:\n{text}")


def main() -> None:
    preflight(DB_PATH)
    conn = duckdb.connect(DB_PATH)
    scan = run_anomaly_scan(conn)
    latest = latest_status_per_department(scan)
    conn.close()

    high = latest[latest["severity"] == "HIGH"]
    if high.empty:
        print("[monitor] No HIGH-severity departments this run. Nothing to investigate.")
        return

    for _, row in high.iterrows():
        dept = row["department"]
        print(f"[monitor] HIGH severity in {dept} ({row['deviation_pct']:.1f}% deviation) -- asking the agent to investigate...")
        try:
            report = ask(f"Investigate the demand anomaly in {dept} and write a report.")
        except Exception as exc:  # noqa: BLE001 -- one bad LLM call shouldn't sink the whole scan
            print(f"[monitor] Agent investigation failed for {dept}: {exc}")
            post_to_slack(
                f":warning: DemandIQ alert -- {dept} is at HIGH severity "
                f"({row['deviation_pct']:.1f}% deviation) but the agent investigation "
                f"failed ({exc}). Check it manually."
            )
            continue
        post_to_slack(f":rotating_light: DemandIQ alert -- {dept}\n\n{report}")


if __name__ == "__main__":
    main()
