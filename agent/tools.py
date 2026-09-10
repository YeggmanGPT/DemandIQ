"""
agent/tools.py

The three tools the ReAct agent uses to investigate a demand anomaly:
    run_sql_query             - read-only SELECT access to the DuckDB warehouse
    get_forecast_delta        - the latest actual-vs-forecast deviation for a department
    generate_business_report  - formats findings into a 4-part executive summary

Kept deliberately narrow (see BUILD_LOG.md, Module 4): each tool does one
thing, run_sql_query is hard-blocked from anything but SELECT, so the
agent can't accidentally mutate the warehouse.
"""

import os
import re

import duckdb
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "demandiq.duckdb")

# Matches a run of leading whitespace, "-- line comments", and /* block
# comments */ -- anything that can legally precede real SQL. Every file in
# sql/ opens with a "-- ..." header, so a naive `.startswith("select")`
# check (an earlier version of this guard) rejects all of them; this
# strips that noise first, then checks what's actually left.
_LEADING_SQL_NOISE = re.compile(r"^(?:\s+|--[^\n]*\n?|/\*.*?\*/)*", re.DOTALL)


def is_select_only(query: str) -> bool:
    """True if `query`, once leading whitespace/comments are stripped, is a
    read-only SELECT (optionally preceded by a WITH clause). Shared by
    run_sql_query below and the dashboard's SQL Explorer tab, so the one
    safety property -- no writes reach the warehouse -- holds in one place."""
    core = _LEADING_SQL_NOISE.sub("", query or "", count=1)
    return bool(re.match(r"(select|with)\b", core, re.IGNORECASE))


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


@tool
def run_sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query against the DemandIQ DuckDB warehouse
    and return the result as a small text table. Use this to inspect order
    volumes, reorder rates, department/aisle breakdowns, or anything else in
    the warehouse. Only SELECT statements are allowed -- no INSERT, UPDATE,
    DELETE, or DDL. Keep results small (the tool truncates to 25 rows)."""
    if not is_select_only(query):
        return "Rejected: only read-only SELECT (or WITH ... SELECT) queries are allowed."
    try:
        conn = _connect()
        result = conn.execute(query).df()
        conn.close()
    except Exception as exc:  # noqa: BLE001 -- surfaced to the agent, not the user
        return f"Query failed: {exc}"
    if result.empty:
        return "Query returned no rows."
    return result.head(25).to_string(index=False)


@tool
def get_forecast_delta(department: str) -> str:
    """Look up the most recent actual-vs-forecast deviation for a given
    department name (e.g. 'produce'), including the severity classification
    (LOW / MEDIUM / HIGH) computed by anomaly.py. Use this before digging
    into raw SQL, to confirm whether -- and how badly -- a department is
    actually deviating from its forecast."""
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT department, ds, actual, forecast, deviation_pct, severity
            FROM anomaly_scan
            WHERE department = ?
            ORDER BY ds DESC
            LIMIT 1
            """,
            [department],
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return f"No forecast evaluation found for department '{department}'. Run forecaster.py and anomaly.py first, or check the department name."
    dep, ds, actual, forecast, deviation_pct, severity = row
    return (
        f"{dep} on {ds}: actual={actual:.0f}, forecast={forecast:.0f}, "
        f"deviation={deviation_pct:.1f}%, severity={severity}."
    )


@tool
def generate_business_report(headline: str, findings: str, data_evidence: str, recommendation: str) -> str:
    """Assemble a four-part executive report (Headline, Findings, Data
    Evidence, Actionable Recommendation) from the investigation results.
    Call this last, once run_sql_query and get_forecast_delta have given
    you enough to write each section. Kept as a plain template rather than
    a second LLM call, so a live demo doesn't depend on two model calls
    succeeding back to back."""
    return (
        f"HEADLINE\n{headline}\n\n"
        f"FINDINGS\n{findings}\n\n"
        f"DATA EVIDENCE\n{data_evidence}\n\n"
        f"RECOMMENDATION\n{recommendation}"
    )


TOOLS = [run_sql_query, get_forecast_delta, generate_business_report]
