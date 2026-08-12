# DemandIQ — Engineering Build Log & Technical Decisions

This document details the architectural rationale, design trade-offs, data modeling decisions, and debugging iterations made during the development of DemandIQ.

---

## Module 1: Data Warehouse Architecture & Schema Validation

### Objective

Ingest 3.4M+ anonymized Instacart e-commerce orders into an embedded analytical storage layer, enforce relational schema integrity, and execute pre-feature validation checks.

### Architectural Decisions & Rationale

1. **Embedded OLAP (DuckDB) vs. Client-Server (PostgreSQL) vs. OLTP (SQLite)**
* *Trade-off*: Standing up a PostgreSQL instance adds operational hosting overhead and network serialization latency for single-node analytical workloads. SQLite is row-oriented (OLTP), making large column aggregations slow.
* *Decision*: Selected **DuckDB**. Its vectorized execution engine and columnar storage format allow scanning and joining 3.4M rows in sub-second times directly inside the Python runtime environment.


2. **Unified Order Fact Table (`fact_order_items`)**
* *Trade-off*: Instacart splits basket items across `order_products__prior.csv` (32M rows) and `order_products__train.csv` (1.3M rows).
* *Decision*: Executed an in-memory `UNION ALL` during table creation in DuckDB to consolidate all order line items into a single fact table, ensuring consistent query indexing across downstream feature pipelines.



### Data Quality Assertions & Defensive Engineering

Implemented `validate_warehouse()` inside `ingest.py` to catch data quality bugs before ML training:

* **Orphan Key Check**: Verified zero orphaned records in `fact_order_items` lacking matching parent `order_id` entries in `fact_orders`.
* **Null Attributes**: Enforced non-null constraints across critical lookup fields (`product_name`, `department_id`).
* **Sequence Validation**: Validated that `order_number` values are strictly positive integers.

---

## Module 2: Time-Series Temporal Reconstruction & Feature Engineering

### Objective

Convert static order logs into a continuous daily time-series matrix per department, complete with lag variables and rolling window statistics.

### The Temporal Reconstruction Challenge

* **Problem**: Instacart anonymizes order timestamps, providing only relative fields: `order_dow` (0–6), `order_hour_of_day` (0–23), and `days_since_prior_order` (0–30). Standard naive approaches either inject artificial random noise or anchor every user to the exact same start date (which creates a massive synthetic spike on day one).
* **Solution**: Developed a cumulative user-chain algorithm in DuckDB (`features.py`). Grouped orders by `user_id`, ordered by `order_number`, and calculated cumulative window sums of `days_since_prior_order`. Each user chain was anchored relative to a baseline calendar start point (`2024-01-01`), accurately preserving true repurchase cadences and weekly consumer behavior patterns.

### Feature Selection Rationale

* **Lag Variables (`lag_1`, `lag_7`, `lag_14`)**: Captures day-over-day momentum and weekly recurring seasonality (e.g., weekend grocery restocking spikes).
* **Rolling Window Statistics (`rolling_mean_7`, `rolling_std_7`)**: Establishes moving baseline volume and variance bounds. These features allow the downstream anomaly detection engine to compute deviation standard scores ($Z$-scores).

---

## Module 3: Predictive ML Tournament & Anomaly Detection

### Objective

Train and evaluate predictive demand models across 21 product departments and implement an automated anomaly trigger engine.

### Model Tournament Design (Prophet vs. XGBoost)

To ensure high forecast accuracy across diverse category trends (e.g., stable high-volume *Produce* vs. volatile *Bulk Goods*), two distinct modeling approaches were evaluated per department:

1. **Facebook Prophet**: Modeled macro trends, day-of-week additive seasonality, and holiday effects cleanly with minimal parameter tuning.
2. **XGBoost Regressor**: Gradient boosted decision trees trained directly on the engineered lag and rolling statistics matrix.

### Evaluation Metrics

* Models were split on chronological time boundaries (80% train, 20% test split) to avoid temporal data leakage.
* Evaluated champion models using **RMSE** (penalizes large outlier forecast errors) and **MAPE** (provides relative percentage accuracy for business stakeholders).

### Anomaly Engine Logic

* **Deviation Percentage**: $\text{Delta} = \frac{\vert{}\text{Actual} - \text{Forecast}\vert{}}{\text{Forecast}} \times 100$
* **Severity Matrix**:
* `< 15%`: Normal operational variance (`LOW`).
* `15% – 25%`: Moderate deviation (`MEDIUM`).
* `> 25%`: Significant operational anomaly (`HIGH`) $\rightarrow$ Triggers automated AI agent investigation and Slack dispatch.



---

## Module 4: Agentic AI Core & Tool Call Architecture

### Objective

Deploy an autonomous reasoning agent capable of investigating forecast anomalies, querying the warehouse for root causes, and generating executive reports.

### LLM Inference & Framework Selection

* **Framework**: LangChain configured with the **ReAct (Reason + Act)** loop paradigm.
* **LLM Engine**: **Groq LLaMA 3**. Selected Groq's LPU hardware acceleration to keep tool invocation latency sub-second (~500 tokens/sec), ensuring smooth interaction during live chat sessions.

### Custom Tool Boundaries

Designed three single-responsibility tools (`agent/tools.py`) with strict prompt constraints to prevent hallucinated inputs or unbounded SQL queries:

1. `run_sql_query`: Accepts read-only `SELECT` queries against DuckDB to inspect basket sizes, reorder rates, or peak purchasing hours.
2. `get_forecast_delta`: Fetches actual demand vs. predicted values and severity ratings for a target department.
3. `generate_business_report`: Converts raw technical findings into a structured four-part executive summary (*Headline, Findings, Data Evidence, Actionable Recommendation*).

---

## Module 5: Automation, Monitoring & Production Deployment

### Objective

Deploy the system to a cloud environment with background anomaly monitoring and real-time operational notifications.

### Architecture Trade-offs: In-App Scheduler vs. External CI/CD Trigger

* *Initial Consideration*: Running `APScheduler` inside the Streamlit application process.
* *Limitation*: Streamlit Cloud puts idle apps to sleep, which would pause background scheduler threads when no user actively holds the webpage open.
* *Production Solution*: Offloaded scheduled anomaly checks to a **GitHub Actions Cron Workflow**. The workflow runs independently on a defined schedule, executes `anomaly.py`, triggers the agent on `HIGH` severity anomalies, and posts alerts to a designated **Slack Webhook**.

### Live Cloud Hosting

* Deployed the dashboard to **Streamlit Community Cloud** synced directly to the primary GitHub branch.
* Environment secrets (`GROQ_API_KEY`, `SLACK_WEBHOOK_URL`) were securely injected via Streamlit Cloud Secrets Management.