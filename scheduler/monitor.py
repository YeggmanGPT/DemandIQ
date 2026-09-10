"""
scheduler/monitor.py

The script the GitHub Actions cron workflow actually runs. Re-scans for
anomalies and, if anything is at HIGH severity, asks the agent to
investigate and (optionally) posts the resulting report to Slack.

Deliberately a plain script invoked by cron (GitHub Actions), not an
in-process scheduler like APScheduler -- an in-process scheduler dies the
moment the host app sleeps or restarts, which defeats the point of
"runs on a schedule." See BUILD_LOG.md, Module 5.

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


def post_to_slack(text: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("[monitor] SLACK_WEBHOOK_URL not set -- skipping Slack post, printing instead:\n")
        print(text)
        return
    try:
        requests.post(webhook, json={"text": text}, timeout=10)
    except Exception as exc:  # noqa: BLE001 -- don't let a network hiccup crash the whole run
        print(f"[monitor] Slack post failed ({exc}); report was:\n{text}")


def main() -> None:
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
