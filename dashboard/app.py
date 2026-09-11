"""
dashboard/app.py

Single-file Streamlit dashboard for DemandIQ. Four tabs:
    Overview       - warehouse-level KPIs and department volume share
    SQL Explorer   - run any of the sql/*.sql files (or your own read-only
                     query) straight against the DuckDB warehouse
    Forecasts      - actual vs. forecast per department, plus the
                     Prophet vs. XGBoost leaderboard from forecaster.py
    Agent Chat     - talk to the LangGraph/Groq agent in agent/agent.py

Run from the project root, after ingest -> features -> forecaster -> anomaly:
    streamlit run dashboard/app.py
"""

import os
import sys

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()  # must run before any os.environ.get("GROQ_API_KEY") check below

from agent.tools import is_select_only  # noqa: E402

DB_PATH = os.path.join("db", "demandiq.duckdb")
SQL_DIR = "sql"
MODELS_DIR = "models"

st.set_page_config(page_title="DemandIQ", layout="wide")


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


def run_query(conn: duckdb.DuckDBPyConnection, query: str, params: list | None = None) -> pd.DataFrame:
    return conn.execute(query, params or []).df()


st.title("DemandIQ")
st.caption("Agentic demand forecasting & anomaly monitoring for a grocery e-commerce warehouse")

if not os.path.exists(DB_PATH):
    st.error(
        f"No warehouse found at `{DB_PATH}`. Run `python ingest.py` first "
        "(after placing the Instacart CSVs in `data/`)."
    )
    st.stop()

conn = get_connection()

tab_overview, tab_sql, tab_forecast, tab_agent = st.tabs(
    ["Overview", "SQL Explorer", "Forecasts", "Agent Chat"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    total_orders = run_query(conn, "SELECT COUNT(*) AS n FROM fact_orders")["n"][0]
    total_items = run_query(conn, "SELECT COUNT(*) AS n FROM fact_order_items")["n"][0]
    n_departments = run_query(conn, "SELECT COUNT(*) AS n FROM dim_departments")["n"][0]
    avg_basket = run_query(
        conn,
        """
        SELECT ROUND(COUNT(foi.product_id) * 1.0 / COUNT(DISTINCT foi.order_id), 2) AS n
        FROM fact_order_items foi
        """,
    )["n"][0]

    col1.metric("Total orders", f"{total_orders:,}")
    col2.metric("Total line items", f"{total_items:,}")
    col3.metric("Departments", n_departments)
    col4.metric("Avg basket size", avg_basket)

    st.subheader("Department volume share")
    volume_share = run_query(
        conn,
        """
        SELECT
            d.department_name,
            COUNT(foi.order_id) AS item_volume,
            ROUND(COUNT(foi.order_id) * 100.0 / (SELECT COUNT(*) FROM fact_order_items), 2) AS volume_share_pct
        FROM fact_order_items foi
        JOIN dim_products p    ON foi.product_id = p.product_id
        JOIN dim_departments d ON p.department_id = d.department_id
        GROUP BY d.department_name
        ORDER BY item_volume DESC
        """,
    )
    fig = px.bar(volume_share, x="department_name", y="item_volume", text="volume_share_pct")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Item volume")
    st.plotly_chart(fig, width='stretch')

    if table_exists(conn, "anomaly_scan"):
        st.subheader("Current anomaly status (latest evaluated day per department)")
        latest = run_query(
            conn,
            """
            SELECT department, ds, actual, forecast, ROUND(deviation_pct, 1) AS deviation_pct, severity
            FROM anomaly_scan
            QUALIFY ROW_NUMBER() OVER (PARTITION BY department ORDER BY ds DESC) = 1
            ORDER BY deviation_pct DESC
            """,
        )

        def _highlight(row):
            color = {"HIGH": "#a8000f", "MEDIUM": "#fff3cd", "LOW": "#d4edda"}.get(row["severity"], "")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(latest.style.apply(_highlight, axis=1), width='stretch')
    else:
        st.info("Run `python forecaster.py` and `python anomaly.py` to populate anomaly status here.")

# ---------------------------------------------------------------- SQL Explorer
with tab_sql:
    st.subheader("Run a saved query")
    sql_files = sorted(f for f in os.listdir(SQL_DIR) if f.endswith(".sql")) if os.path.isdir(SQL_DIR) else []
    choice = st.selectbox("Saved queries (sql/*.sql)", ["(none)"] + sql_files)

    default_query = "SELECT * FROM dim_departments LIMIT 10"
    if choice != "(none)":
        with open(os.path.join(SQL_DIR, choice)) as f:
            default_query = f.read()

    query_text = st.text_area("SQL (read-only SELECT)", value=default_query, height=220)
    if st.button("Run query"):
        if not is_select_only(query_text):
            st.error("Only SELECT (or WITH ... SELECT) queries are allowed here.")
        else:
            try:
                result = run_query(conn, query_text)
                st.success(f"{len(result):,} rows")
                st.dataframe(result, width='stretch')
            except Exception as exc:  # noqa: BLE001 -- shown to the user, not swallowed
                st.error(f"Query failed: {exc}")

# ---------------------------------------------------------------- Forecasts
with tab_forecast:
    if not table_exists(conn, "department_forecast_evaluation"):
        st.info("Run `python forecaster.py` first to populate forecast evaluations.")
    else:
        leaderboard_path = os.path.join(MODELS_DIR, "model_leaderboard.csv")
        if os.path.exists(leaderboard_path):
            st.subheader("Model leaderboard (Prophet vs. XGBoost)")
            st.dataframe(pd.read_csv(leaderboard_path), width='stretch')

        departments = run_query(
            conn, "SELECT DISTINCT department FROM department_forecast_evaluation ORDER BY department"
        )["department"].tolist()
        selected = st.selectbox("Department", departments)

        eval_df = run_query(
            conn,
            "SELECT ds, actual, forecast FROM department_forecast_evaluation WHERE department = ? ORDER BY ds",
            [selected],
        )

        melted = eval_df.melt(id_vars="ds", value_vars=["actual", "forecast"], var_name="series", value_name="units")
        fig = px.line(melted, x="ds", y="units", color="series", markers=True,
                      title=f"Actual vs. forecast -- {selected} (hold-out period)")
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- Agent Chat
def _trace_steps(messages) -> list[tuple[str, str]]:
    """Turn a LangGraph message list into (label, body) steps for display --
    just the tool calls and their results, not the human question or the
    final answer (both already shown as the chat bubble itself)."""
    steps = []
    for m in messages:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                steps.append((f"Called `{tc['name']}`", str(tc.get("args", {}))))
        elif m.type == "tool":
            steps.append((f"`{getattr(m, 'name', 'tool')}` returned", str(m.content)))
    return steps


def _render_trace(trace: list[tuple[str, str]]) -> None:
    if not trace:
        return
    with st.expander("Reasoning trace (tool calls)"):
        for label, body in trace:
            st.markdown(f"**{label}**")
            st.code(body)


with tab_agent:
    st.subheader("Ask the DemandIQ agent")
    st.caption("Runs the LangGraph ReAct agent (agent/agent.py) against Groq. Needs GROQ_API_KEY set.")

    if not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "GROQ_API_KEY is not set in this environment, so the agent can't run yet. "
            "Copy `.env.example` to `.env`, add your key from https://console.groq.com/keys, "
            "and restart the app."
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {role, content, trace}

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            _render_trace(turn.get("trace"))

    prompt = st.chat_input("e.g. Which department has the worst forecast deviation right now, and why?")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "trace": None})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            trace: list[tuple[str, str]] = []
            if not os.environ.get("GROQ_API_KEY"):
                answer = "GROQ_API_KEY isn't set, so I can't reach the agent right now -- see the note above."
                st.markdown(answer)
            else:
                with st.spinner("Investigating..."):
                    try:
                        from agent.agent import ask_with_trace
                        answer, agent_messages = ask_with_trace(prompt)
                        trace = _trace_steps(agent_messages)
                    except Exception as exc:  # noqa: BLE001 -- surfaced in the chat, not a crash
                        answer = f"Agent call failed: {exc}"
                st.markdown(answer)
                _render_trace(trace)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "trace": trace})
