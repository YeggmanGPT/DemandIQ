# DemandIQ — Autonomous Demand Intelligence System

> **An end-to-end grocery demand analytics platform featuring an embedded SQL warehouse, machine learning demand forecasting, autonomous anomaly detection, and an AI agent that investigates operational spikes—built on 3.4M+ real e-commerce transactions.**

---

## 📋 Executive Overview

**DemandIQ** bridges the gap between traditional data engineering, predictive machine learning, and modern agentic AI.

In grocery e-commerce (companies like Picnic, Zalando, and Delivery Hero), supply chain teams spend hundreds of hours manually checking dashboards to spot stockouts or demand surges. DemandIQ automates this entire analytical lifecycle:

1. **Warehouse & Data Engineering**: Ingests and transforms 3.4M+ raw order records into an in-process, high-performance columnar DuckDB star schema.
2. **Predictive Analytics**: Reconstructs realistic consumer purchase timelines and trains department-level forecasting models (Prophet vs. XGBoost) to predict daily demand.
3. **Agentic Intelligence**: Deploys a **LangChain ReAct Agent** powered by **Groq LLaMA 3** that autonomously monitors forecast deviations, queries the SQL warehouse using custom tools to diagnose root causes, and drafts executive business reports.
4. **Automated Alerting**: Continuously runs background monitoring loops to fire real-time push notifications to operational Slack channels without human intervention.

---

## 🏗️ Architecture Overview

The system operates as an integrated pipeline across four major layers: Data Warehouse, Predictive ML, Agentic Reasoning, and Web Presentation.

```mermaid
flowchart TD
    subgraph Data Layer ["1. Warehouse & Ingestion"]
        A[Instacart Dataset\n3.4M+ CSV Orders] -->|ingest.py| B[(DuckDB Warehouse\nStar Schema)]
        B -->|validate_warehouse| QC[Data Quality Validation]
    end

    subgraph Feature & ML Layer ["2. Feature Pipeline & ML Engine"]
        B -->|features.py| C[Reconstructed Daily Features\nLags & Rolling Window Stats]
        C -->|forecaster.py| D[Model Tournament\nProphet vs. XGBoost]
        D -->|Best Artifacts| E[Forecast Engine]
        C --> E
    end

    subgraph Autonomous Layer ["3. Agentic & Automation Core"]
        E -->|anomaly.py| F[Anomaly Detector\nZ-Score & % Thresholds]
        F -->|Trigger on Anomaly| G[LangChain ReAct Agent\nGroq LLaMA 3]
        
        subgraph Tool Matrix ["Custom LangChain Tools"]
            T1[run_sql_query]
            T2[get_forecast_delta]
            T3[generate_business_report]
        end
        
        G <--> Tool Matrix
        Tool Matrix <--> B
        G -->|Slack Webhook| H[Slack Operational Alerts]
        Scheduler[GitHub Actions / Scheduler] -->|Trigger| F
    end

    subgraph UI Layer ["4. Presentation Layer"]
        B --> I[Streamlit BI Dashboard]
        E --> I
        G -->|Live Agent Chat| I
    end

    classDef data fill:#e1f5fe,stroke:#0288d1,color:#01579b;
    classDef ml fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;
    classDef agent fill:#e8f5e9,stroke:#388e3c,color:#1b5e20;
    classDef ui fill:#fff3e0,stroke:#f57c00,color:#e65100;

    class A,B,QC data;
    class C,D,E ml;
    class F,G,T1,T2,T3,H,Scheduler agent;
    class I ui;

```

---

## 📊 End-to-End Pipeline Execution Flow

### 1. Data Ingestion & Warehouse Modeling (`ingest.py`)

* Ingests 3.4M+ records across 6 source files into an embedded **DuckDB** OLAP database.
* Structures raw data into a normalized **Star Schema**:
* **Fact Tables**: `fact_orders` (order timestamps, user IDs, sequence numbers), `fact_order_items` (reorder flags, cart positions).
* **Dimension Tables**: `dim_products`, `dim_departments` (21 product categories), `dim_aisles` (134 aisles).



### 2. Automated Data Quality Gate (`validate_warehouse`)

Before downstream processing, the pipeline runs automated schema and data integrity assertions:

* Checks primary/foreign key relationships (flags orphaned order items).
* Verifies zero-null constraints on essential foreign keys (`product_id`, `department_id`).
* Asserts non-negative bounds on item order quantities and hour ranges.

### 3. Business SQL Layer (`sql/*.sql`)

Contains 8 optimized SQL analytical scripts used for business intelligence and agent tool execution:

* **Window Functions**: `DENSE_RANK() OVER (PARTITION BY department_id ORDER BY COUNT(*) DESC)` to extract top-selling products per department.
* **Repurchase Mechanics**: Reorder ratios, basket size distributions across days of the week, and inter-order interval medians.

### 4. Reconstructed Time-Series Engine (`features.py`)

* **Temporal Reconstruction Algorithm**: Maps anonymized relative user order gaps (`days_since_prior_order`) to calendar timelines using cumulative user order chains.
* **Feature Engineering Matrix**:
* **Lags**: 1-day, 7-day, and 14-day lagged order volumes to capture daily and weekly seasonality.
* **Rolling Window Statistics**: 7-day moving averages (`rolling_mean_7`) and standard deviations (`rolling_std_7`) to establish baseline variance.
* **Calendar Features**: Day-of-week indicators, weekend flags, and day-of-month indices.



### 5. Demand Forecasting Engine (`forecaster.py`)

Evaluates multiple machine learning strategies per department:

* **Prophet**: Captures macro seasonal trends and day-of-week additive effects.
* **XGBoost**: Gradient boosted decision trees trained on engineered lag and rolling window features.
* **Automated Tournament**: Evaluates both models using **RMSE**, **MAE**, and **MAPE** on a held-out test split, saving the champion model artifact per department.

### 6. Autonomous Anomaly Detection (`anomaly.py`)

* Compares actual demand vs. model forecasts in real time.
* Flags deviations exceeding standard statistical thresholds (e.g., >20% forecast deviation or Z-score > 2.0).
* Assigns risk severity levels (`LOW`, `MEDIUM`, `HIGH`) to prioritize agent investigation.

### 7. Agentic AI & Custom Tool Use (`agent/`)

Architected using **LangChain** and **Groq LLaMA 3** using the **ReAct (Reason + Act)** framework. The agent is provided 3 custom tools:

1. `run_sql_query`: Executes direct SQL against DuckDB to inspect underlying order logs, user reorder behavior, and product trends.
2. `get_forecast_delta`: Queries the forecasting engine for specific category deviation percentages and historical baselines.
3. `generate_business_report`: Synthesizes findings into structured executive summaries containing key metrics, root causes, and operational recommendations.

### 8. Background Automation & Alerting (`scheduler/` & GitHub Actions)

* Runs an automated background loop (via GitHub Actions Cron or APScheduler).
* When a `HIGH` severity anomaly is detected, the pipeline automatically triggers the LangChain agent to investigate and dispatches a formatted alert report directly to a **Slack Webhook**.

### 9. Multi-Page BI Dashboard (`dashboard/app.py`)

An interactive **Streamlit** multi-page application:

* **Page 1: Executive KPI Overview**: Top metrics, active alerts, and overall warehouse demand volume.
* **Page 2: Interactive SQL Explorer**: Executes pre-built analytics queries with auto-rendering data charts and custom SQL box.
* **Page 3: Demand Forecast Viewer**: Displays actuals vs. predictions, model evaluation metrics, and anomaly flags.
* **Page 4: Conversational AI Agent**: Natural language chat interface with live expander views showing the agent's internal tool calls and reasoning trace.

---

## 📁 Project Structure

```
DemandIQ/
├── data/                       # Raw Instacart CSV files (gitignored)
├── db/                         # DuckDB analytical warehouse (demandiq.duckdb)
├── models/                     # Trained Prophet & XGBoost model artifacts
├── agent/                      # LangChain ReAct agent & tool definitions
│   ├── tools.py                # Custom LangChain tools (SQL, Forecast, Report)
│   └── agent.py                # ReAct agent loop configuration
├── dashboard/                  # Multi-page Streamlit web application
│   └── app.py                  # Main dashboard launcher
├── scheduler/                  # Background automation tasks
│   └── monitor.py              # Anomaly detection & Slack alert runner
├── sql/                        # Warehouse SQL analytics queries
│   ├── reorder_rate_by_department.sql
│   ├── peak_order_hours.sql
│   ├── basket_size_by_dow.sql
│   ├── top_products_by_department.sql
│   ├── dow_demand_by_department.sql
│   ├── department_volume_share.sql
│   ├── reorder_interval_by_department.sql
│   └── department_demand_by_hour.sql
├── ingest.py                   # Data ingestion & schema builder
├── features.py                 # Time-series reconstruction & feature pipeline
├── forecaster.py               # Model training tournament & evaluator
├── anomaly.py                  # Anomaly detection engine
├── requirements.txt            # Project dependencies
├── .env.example                # Environment variable template
├── BUILD_LOG.md                # Engineering build log & technical decisions
└── README.md                   # System documentation

```

---

## 📋 Data Warehouse Schema

```
                      +-------------------+
                      |  dim_departments  |
                      +-------------------+
                      | department_id (PK)|
                      | department_name   |
                      +---------+---------+
                                |
                                | 1:N
                                v
+------------------+  1:N +-------------------+
|    dim_aisles    |<-----+   dim_products    |
+------------------+      +-------------------+
| aisle_id (PK)    |      | product_id (PK)   |
| aisle_name       |      | product_name      |
+------------------+      | aisle_id (FK)     |
                          | department_id (FK)|
                          +---------+---------+
                                    |
                                    | 1:N
                                    v
+------------------+  1:N +-------------------+
|   fact_orders    |<-----+ fact_order_items  |
+------------------+      +-------------------+
| order_id (PK)    |      | order_id (FK)     |
| user_id          |      | product_id (FK)   |
| order_number     |      | add_to_cart_order |
| order_dow        |      | is_reordered      |
| order_hour       |      +-------------------+
| days_since_prior |
+------------------+

```

---

## 📝 Key Architectural Trade-offs & Engineering Decisions

### 1. DuckDB In-Process Columnar Storage vs. Client-Server Postgres

* **Decision**: Selected **DuckDB** over standing up a PostgreSQL instance.
* **Rationale**: For single-node analytical processing on 3.4M+ records, DuckDB's vectorized execution engine provides sub-second aggregation speeds directly within the Python runtime, eliminating networking overhead, server administration costs, and complex container orchestration.

### 2. Reconstructed Temporal Timelines vs. Synthetic Date Injection

* **Decision**: Reconstructed purchase timelines using user order chains rather than injecting random dates.
* **Rationale**: Instacart's public dataset anonymizes calendar dates to `days_since_prior_order`. By building a cumulative window sum of order gaps grouped per user, the pipeline preserves natural customer repurchase cycles, producing authentic temporal seasonality without artificial synthetic noise.

### 3. Department Granularity vs. Per-SKU Forecasting

* **Decision**: Forecasted demand at the **Department level** (21 categories) rather than individual SKU levels (49,688 SKUs).
* **Rationale**: Department-level forecasting balances signal-to-noise ratios and provides high business value for high-level inventory planning, avoiding severe sparsity issues present at the individual product level.

---

## ⚙️ Setup & Installation

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/YeggmanGPT/DemandIQ.git
cd DemandIQ

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

> **Note on Prophet Installation**:
> If building `prophet` encounters system compilation errors:
> * **macOS / Linux**: Run `pip install pystan==2.19.1.1` prior to installing requirements.
> * **Windows**: Install Prophet via Conda (`conda install -c conda-forge prophet`).
> 
> 

### 2. Dataset Download & Workspace Configuration

1. Download the [Instacart Market Basket Analysis Dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) from Kaggle.
2. Extract the six CSV files into the `data/` folder:
`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`.
3. Configure environment variables:
```bash
cp .env.example .env

```


Edit `.env` and insert your `GROQ_API_KEY` and optional `SLACK_WEBHOOK_URL`.

### 3. Pipeline Execution Steps

```bash
# Step 1: Ingest CSVs into DuckDB & run Data Quality Validation
python ingest.py

# Step 2: Reconstruct time-series & generate feature matrix
python features.py

# Step 3: Train forecasting models (Prophet vs XGBoost)
python forecaster.py

# Step 4: Run anomaly detection engine
python anomaly.py

# Step 5: Launch Streamlit Dashboard
streamlit run dashboard/app.py

```

---

## 🧪 Testing & Data Quality Strategy

* **Warehouse Assertions**: `ingest.py` runs validation functions testing primary key uniqueness, foreign key orphan checks, and non-null constraints.
* **Feature Validation**: `features.py` enforces temporal continuity and verifies that rolling window aggregations contain no dangling NaN values.
* **Agent Trace Auditing**: Streamlit's interface exposes full ReAct thought-action-observation cycles, allowing verification of tool selection logic.

---

## 🛠️ Tech Stack Matrix

| Domain | Tool / Framework |
| --- | --- |
| **Data Warehouse** | DuckDB (Embedded Columnar OLAP) |
| **Data Processing & SQL** | Python, Pandas, NumPy, SQL |
| **Machine Learning** | Prophet, XGBoost, Scikit-Learn |
| **Agentic AI** | LangChain, Groq (LLaMA 3 LPU Hardware Acceleration) |
| **Automation & Alerts** | GitHub Actions, APScheduler, Slack Webhooks |
| **UI & Visualization** | Streamlit, Plotly |

---

## 👤 Author

**Abdul Rahman Zuhaib**

*M.Sc. Student in Global Software Development, Hochschule Fulda*

[LinkedIn](https://www.linkedin.com/in/abdul-rahman-zuhaib-b1b7b5177/)  
[GitHub](https://github.com/YeggmanGPT)