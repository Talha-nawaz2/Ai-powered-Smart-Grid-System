# ⚡ Smart Grid Intelligence & Energy Optimization Platform

An enterprise-style AI platform for demand forecasting, predictive maintenance,
energy theft detection, renewable intelligence, a knowledge graph, and an AI
grid copilot — built lightweight enough to run on a laptop with <2GB used.

## Status: ALL 10 SESSIONS COMPLETE ✅
Full platform: Grid Data API, Prophet forecasting, XGBoost maintenance,
Isolation Forest theft detection, Renewable Intelligence, Knowledge Graph,
AI Grid Copilot, a live Dashboard, and a rule-based Alert Center + Simulator.

## Architecture

```
smart-grid/
├── backend/
│   ├── api/              # FastAPI routers (Session 2+)
│   ├── models/            # SQLAlchemy ORM models / Pydantic schemas
│   ├── forecasting/       # Prophet demand forecasting (Session 3)
│   ├── maintenance/       # XGBoost transformer failure prediction (Session 4)
│   ├── theft_detection/   # Isolation Forest anomaly detection (Session 5)
│   ├── renewables/        # Solar/wind/battery intelligence (Session 6)
│   ├── knowledge_graph/   # NetworkX grid relationship graph (Session 7)
│   ├── copilot/           # AI Grid Copilot (Session 8)
│   ├── dashboard/         # Jinja2 + Bootstrap + Plotly UI (Session 9)
│   ├── alerts/            # Alert engine (Session 10)
│   ├── data/
│   │   ├── generate_data.py     # synthetic dataset generator
│   │   ├── load_to_sqlite.py    # CSV -> SQLite ETL
│   │   ├── sample_data/         # generated CSVs (~1.1 MB)
│   │   └── smart_grid.db        # SQLite database (~1.3 MB)
│   └── utils/
│       ├── config.py      # centralized settings (pydantic-settings)
│       ├── logger.py      # loguru structured logging
│       └── db.py          # SQLAlchemy engine/session
├── requirements.txt
├── .env.example
└── README.md
```

## Design principles used (and why)

| Pattern | Why it matters |
|---|---|
| Centralized `config.py` | One place to change environments (dev/prod) instead of hunting hardcoded values |
| Separated generate → load scripts | Single Responsibility Principle — each script does ONE job |
| `db.py` as sole DB gateway | Swap SQLite → Postgres later by changing one line |
| Structured logging (loguru) | Real systems never use `print()`; logs are filterable & persisted |
| Domain-driven synthetic data | Failures/theft aren't random — they follow real signal patterns (aging → lower health score → failure), so ML models trained on this data actually learn something meaningful |

## Datasets generated (`backend/data/sample_data/`)

| File | Rows | Purpose |
|---|---|---|
| substations.csv | 50 | Grid topology |
| power_plants.csv | 30 | Generation sources |
| solar_farms.csv / wind_farms.csv | 25 / 15 | Renewable assets |
| battery_storage.csv | 40 | Storage state |
| transformers.csv | 500 | **Predictive maintenance** features + failure label |
| consumers.csv / smart_meters.csv | 3000 / 3000 | Customer & meter registry |
| meter_consumption_profile.csv | 3000 | **Theft detection** features (115 synthetic theft cases) |
| weather.csv | 4998 | Drives solar/wind generation |
| grid_demand_hourly.csv | 2160 | **Demand forecasting** time series (90 days) |
| solar_generation_log.csv / wind_generation_log.csv | 500 / 300 | Renewable output tied to weather |
| transmission_lines.csv | 194 | Network edges |

**Total footprint: ~2.5 MB** (data + DB). Well within your 10GB budget.

## Setup

```bash
cd smart-grid
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Generate data + load into SQLite (already done once, re-run anytime to reset)
python -m backend.data.generate_data
python -m backend.data.load_to_sqlite
```

## Running the API

```bash
source venv/bin/activate
uvicorn backend.api.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs, or try:

| Endpoint | What it does |
|---|---|
| `GET /` | Health check |
| `GET /api/grid/overview` | Aggregated KPIs (used later by the dashboard) |
| `GET /api/grid/substations` | List substations |
| `GET /api/grid/transformers?region=Lahore` | List/filter transformers |
| `GET /api/grid/transformers/risky?threshold=40` | Business-rule-flagged risky transformers |
| `GET /api/grid/transformers/{id}` | Single transformer lookup (404 if missing) |
| `GET /api/grid/consumers?region=Karachi` | List/filter consumers |
| `GET /api/grid/demand?limit=200` | Recent hourly demand time series |
| `GET /api/grid/weather?region=Multan` | Weather history |
| `GET /api/grid/renewables` | Solar + wind farm summary |

## Demand Forecasting (Prophet)

```bash
curl -X POST http://127.0.0.1:8000/api/forecast/train
curl "http://127.0.0.1:8000/api/forecast/hourly?hours=24"
curl "http://127.0.0.1:8000/api/forecast/daily?days=7"
curl "http://127.0.0.1:8000/api/forecast/weekly?weeks=4"
```

Model trains on all 2,160 hourly points in `grid_demand_hourly`, captures daily +
weekly seasonality (no yearly — only 90 days of history), and persists to
`backend/data/models/demand_forecast_model.pkl` so it survives server restarts.

## Predictive Maintenance (XGBoost)

```bash
curl -X POST http://127.0.0.1:8000/api/maintenance/train
curl "http://127.0.0.1:8000/api/maintenance/predict/TRF-00000"
curl "http://127.0.0.1:8000/api/maintenance/at-risk?threshold=0.8"
```

**Important design note:** `health_score` is deliberately excluded from the
model's features. It was used to derive the `failed_last_year` label itself
in Session 1's data generator, so including it would be data leakage — the
model would just echo the label instead of learning from real sensor signals
(age, temperature, vibration, oil quality, past failures). Test metrics:
accuracy 0.99, ROC-AUC 1.0, recall 0.83 (misses ~1 in 6 real failures — a
realistic tradeoff to discuss, not a flaw to hide).

## Energy Theft Detection (Isolation Forest — unsupervised)

```bash
curl -X POST "http://127.0.0.1:8000/api/theft/train?contamination=0.04"
curl "http://127.0.0.1:8000/api/theft/anomalies?limit=20"
curl "http://127.0.0.1:8000/api/theft/predict/MTR-002919"
```

**Important design note:** the model is trained WITHOUT theft labels — it
only learns what "normal" consumption behavior looks like (usage_ratio,
night_usage_ratio, voltage_variance, billing_disputes) and flags outliers.
The `is_theft_flag_ground_truth` column is used only to *score* the model
afterward, never during `.fit()`. This mirrors real deployments, where you
rarely have confirmed theft labels ahead of time. Result against the 115
synthetic theft cases: precision 0.942, recall 0.983, F1 0.962.

## Renewable Intelligence (RandomForest + rule-based battery engine)

```bash
curl -X POST http://127.0.0.1:8000/api/renewables/train
curl "http://127.0.0.1:8000/api/renewables/predict/solar?solar_irradiance_wm2=900&temperature_c=32&capacity_mw=50"
curl "http://127.0.0.1:8000/api/renewables/predict/wind?wind_speed_kmh=25&capacity_mw=80"
curl "http://127.0.0.1:8000/api/renewables/battery/recommendation/BAT-0000?current_generation_mw=120&current_demand_mw=80"
```

Solar model R²=0.982, wind model R²=0.993 — both regressors recovered the
physical generation curves used to synthesize the data in Session 1.
Battery dispatch is **deliberately rule-based, not ML** — charge/discharge
thresholds are a known physical rule, not a pattern worth learning from
noisy data. Recommendations respect real capacity limits (can't charge
past 100% or discharge below 0%).

## Knowledge Graph (NetworkX)

```bash
curl -X POST http://127.0.0.1:8000/api/graph/build
curl "http://127.0.0.1:8000/api/graph/trace/CUS-000000"
curl "http://127.0.0.1:8000/api/graph/critical-substations?top_n=10"
curl "http://127.0.0.1:8000/api/graph/neighbors/SUB-0025"
curl "http://127.0.0.1:8000/api/graph/path?source=CUS-000000&target=PLT-0007"
```

Models the grid hierarchy as a directed graph: Consumer → Meter → Transformer
→ Substation → Power Plant, plus bidirectional Substation↔Substation
transmission lines. **Simplification note:** substations connect to power
plants by shared region (no explicit substation→plant table exists in the
source data) — a reasonable approximation, but worth stating explicitly
rather than hiding. This graph is what Session 8's AI Copilot will query
to answer "what powers this consumer?" and "which substation is critical?"

## AI Grid Copilot

```bash
curl -X POST http://127.0.0.1:8000/api/copilot/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Which transformer is riskiest right now?"}'
```

Try: *"What is tomorrow's demand forecast?"*, *"Show me theft regions"*,
*"What powers CUS-000000?"*, *"Which substations are critical?"*

**Two modes, same tools underneath:**
- **LLM mode** — set `OPENAI_API_KEY` in `.env`; the Copilot uses real
  OpenAI tool-calling (`gpt-4o-mini` by default) to decide which of 6
  tools to call and with what arguments.
- **Rule-based fallback** (default, no key needed) — keyword intent
  routing calls the exact same tool functions. This keeps the platform
  fully demoable without any API cost, and is a good illustration of
  what "agentic AI" actually is under the hood: something deciding which
  function to call, whether that's an LLM or a simple router.

All 6 tools (`get_grid_overview`, `get_risky_transformers`,
`forecast_demand`, `get_theft_anomalies`, `trace_consumer_supply`,
`get_critical_substations`) call directly into the services already built
in Sessions 2-7 — the Copilot adds a natural-language layer on top,
it doesn't reimplement any logic.

## Dashboard (your website)

```bash
uvicorn backend.api.main:app --reload
```
Open **http://127.0.0.1:8000/dashboard** in your browser.

Click **"Train All Models"** once (trains forecast + maintenance + theft +
builds the graph, in sequence) — then every panel populates: live KPI
cards, a demand forecast chart, a renewable mix donut, a risky-transformer
table with color-coded risk badges, a theft anomaly table, critical
substations, an Alert Center, and a working AI Copilot terminal.

**Design note:** built as a control-room / SCADA-style interface (dark
panels, amber/cyan/green status coding, monospace data readouts, grid-paper
background) rather than generic Bootstrap cards — deliberately grounded in
how real grid operations software actually looks, since that's the subject.
All data is fetched client-side from the same REST APIs built in Sessions
2-8; the FastAPI route only serves the HTML shell once.

## Alert Center & Simulator

```bash
curl http://127.0.0.1:8000/api/alerts
curl -X POST http://127.0.0.1:8000/api/alerts/simulate-tick
```

**Alert Center is deliberately rule-based, not ML** — it applies simple,
tunable thresholds (transformer failure ≥ 80%, demand ≥ configured MW,
any flagged theft meter, battery charge < 15%) on top of outputs already
produced by Sessions 3-6's models. Alert logic in real operations tooling
needs to be instantly explainable, not another black box to debug at 2am.

**Simulator** writes small physically-plausible perturbations directly into
SQLite each time you click "⚡ Simulate Tick" on the dashboard: transformer
sensor drift (recomputed with the *same* health_score formula from Session
1, so data stays internally consistent), a new live demand reading, and
battery charge drift. This is what makes repeated dashboard visits feel
like a live feed instead of one static snapshot forever.

## Full Roadmap (all complete)
- [x] Session 1: Skeleton, config, synthetic data, SQLite
- [x] Session 2: FastAPI + repository/service layers + Grid Data API
- [x] Session 3: Demand Forecasting (Prophet)
- [x] Session 4: Predictive Maintenance (XGBoost)
- [x] Session 5: Theft Detection (Isolation Forest)
- [x] Session 6: Renewable Intelligence
- [x] Session 7: Knowledge Graph (NetworkX)
- [x] Session 8: AI Grid Copilot
- [x] Session 9: Dashboard (control-room styled web UI)
- [x] Session 10: Alert Center + Simulator + final polish

## First-time setup (from scratch)

```bash
git clone <your-repo>   # or unzip the delivered project
cd smart-grid
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m backend.data.generate_data
python -m backend.data.load_to_sqlite

# Prophet needs its CmdStan backend built once:
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

uvicorn backend.api.main:app --reload
```
Open **http://127.0.0.1:8000/dashboard**, click "Train All Models" once,
then explore. API docs: **http://127.0.0.1:8000/docs**.

## What this project demonstrates (for your portfolio / internship pitch)

- **Layered architecture**: repository → service → route, consistently
  applied across 6 different domains (grid data, forecasting, maintenance,
  theft, renewables, graph)
- **5 real ML/stats techniques** used appropriately for their problem:
  Prophet (time series), XGBoost (supervised classification, with a
  deliberate data-leakage avoidance), Isolation Forest (unsupervised
  anomaly detection), RandomForest regression (physical curve learning),
  and NetworkX graph traversal
- **Judgment about when NOT to use ML**: battery dispatch and alerting are
  rule-based on purpose, not because ML wasn't tried
- **A working agentic AI layer**: LLM tool-calling with a rule-based
  fallback, sharing one tool registry
- **A real, live, browsable dashboard** — not just notebook output
- **Honest engineering**: documented simplifications (substation→plant
  region matching), real bugs found and fixed with explanations, not
  hidden
