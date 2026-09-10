"""
agent/agent.py

Wires the three tools in agent/tools.py to a LangGraph ReAct agent running
on Groq. This is the "LangGraph ReAct agent" referenced on the resume --
LangGraph's create_react_agent is the standard, maintained
reason -> call-a-tool -> observe loop, rather than a hand-rolled one.

Model choice: Groq deprecates and retires models on a rolling schedule
(see https://console.groq.com/docs/deprecations). As of Sept 2026 the
default here, openai/gpt-oss-20b, is Groq's current fast/cheap
recommended replacement for the older Llama 3.1 8B Instant, which was
shut down 2026-08-16. If this model 404s for you, check
https://console.groq.com/docs/models for the current roster and update
DEFAULT_MODEL below -- it's a one-line change.

Usage (from the project root, after ingest/features/forecaster/anomaly):
    python -m agent.agent "Why did produce spike yesterday?"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agent.tools import TOOLS

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are DemandIQ's operations analyst agent. You investigate demand "
    "anomalies for an e-commerce grocery warehouse. When asked about a "
    "department, first call get_forecast_delta to see how severe the "
    "deviation actually is. If it's worth digging into, use run_sql_query "
    "to look at the underlying order data for a root cause (e.g. reorder "
    "rate, basket size, peak hours -- see the sql/ folder for the kind of "
    "questions worth asking). Finish by calling generate_business_report "
    "with a headline, findings, the data evidence you found, and one "
    "concrete recommendation. Keep SQL read-only and keep results small."
)


def build_agent(model: str = DEFAULT_MODEL):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
            "(https://console.groq.com/keys)."
        )
    llm = ChatGroq(model=model, api_key=api_key, temperature=0)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def ask_with_trace(question: str, model: str = DEFAULT_MODEL):
    """Like ask(), but also returns the full list of intermediate messages
    (tool calls + tool outputs, not just the final answer) so a caller --
    e.g. the dashboard's Agent Chat tab -- can show the agent's reasoning
    trace rather than just the last line."""
    agent = build_agent(model=model)
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]
    return messages[-1].content, messages


def ask(question: str, model: str = DEFAULT_MODEL) -> str:
    answer, _messages = ask_with_trace(question, model=model)
    return answer


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Which department has the worst forecast deviation right now, and why?"
    print(f"Q: {query}\n")
    print(ask(query))
