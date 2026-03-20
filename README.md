# 🏨 Hotel Revenue Management — Price Elasticity & Dynamic Pricing

> **Hotel pricing is solved differently from e-commerce — you have perishable inventory.
> An unsold room tonight has zero value tomorrow.**

A complete price-elasticity estimation and yield-management project for hotel
revenue optimisation. Built for mid-level data analysts (2–3 years experience)
working in Travel / Hospitality (think Marriott, IHG, or a boutique hotel chain
with dynamic pricing).

---

## Table of Contents

- [Project Goal](#project-goal)
- [Why RevPAR, Not ADR or Occupancy Alone](#why-revpar-not-adr-or-occupancy-alone)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Detail](#pipeline-detail)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Yield Management Glossary](#yield-management-glossary)

---

## Project Goal

Estimate **demand elasticity** for hotel room rates by property, season, and
booking window — then build a **dynamic pricing recommendation engine** that
maximises **RevPAR** (Revenue Per Available Room) without sacrificing occupancy.

The dataset covers **15 properties** across 4 tiers (budget → luxury), 2 years
of weekly data, 7 days of the week, and 4 booking-window bands.

---

## Why RevPAR, Not ADR or Occupancy Alone

| Metric | Problem if optimised alone |
|--------|--------------------------|
| **ADR** (Average Daily Rate) | Raising price fills fewer rooms — revenue can drop |
| **Occupancy** | Filling rooms at low rates leaves money on the table |
| **RevPAR** = ADR × Occupancy | Balances both — the industry standard metric |
| **GOPPAR** | Goes further by factoring in operating costs |

RevPAR captures the **trade-off** between price and volume.  A 10 % rate hike
that only drops occupancy by 5 % is a net win.  This project finds the rate
that maximises that trade-off for each property.

---

## Key Findings

### Elasticity Insight Table by Property Tier

| Tier | Own-Price Elasticity (β) | Interpretation | Impact of +10 % Rate Hike |
|------|--------------------------|----------------|--------------------------|
| **Budget** | −1.8 | Highly elastic | Occupancy drops ~18 % — net RevPAR loss |
| **Midscale** | −1.2 | Moderately elastic | Occupancy drops ~12 % — marginal |
| **Upscale** | −0.9 | Mildly inelastic | Occupancy drops ~9 % — net RevPAR gain |
| **Luxury** | −0.7 | Nearly inelastic | Occupancy drops only ~7 % — clear gain |

> **Luxury properties are nearly inelastic (β = −0.7) — a 10 % rate increase
> only drops occupancy by 7 %. Budget properties lose 18 % occupancy for the
> same rate hike.**

### Booking Window Elasticity

| Window | Elasticity | Insight |
|--------|-----------|---------|
| 0–7 days (last-minute) | −2.5 | Extreme price sensitivity — distress pricing territory |
| 8–30 days | −1.5 | Moderate sensitivity |
| 31–90 days | −1.0 | Committed travellers |
| 90+ days | −0.9 | Early bookers — destination-committed, low sensitivity |

### Last-Minute Strategy

- **Trigger:** Occupancy < 60 % at T−5 days
- **Action:** Distress rate = max(brand floor, competitor × 0.85)
- **Channel:** OTA-only (protects direct channel rate integrity)
- **Outcome:** Recovers revenue on rooms that would otherwise expire worthless

### Yield Management Scenarios (90-day forward simulation)

| Strategy | Description | Expected Effect |
|----------|-------------|-----------------|
| **Revenue Max** | Optimal rate per booking window | Highest RevPAR |
| **Market Parity** | Match competitor rate exactly | Competitive positioning |
| **Occupancy Fill** | 10 % below competitor | Fills rooms, lower ADR |
| **Event Premium** | +25 % surcharge on event weeks | Captures event demand |

---

## Project Structure

```
hotel-revenue-management/
├── data/                         # Generated dataset (gitignored)
├── models/                       # Model artefacts & results (gitignored)
├── src/
│   ├── generate_data.py          # Synthetic hotel data generator
│   ├── log_log_model.py          # OLS log-log elasticity model
│   ├── ml_model.py               # LightGBM demand model + SHAP
│   ├── optimizer.py              # RevPAR rate optimizer
│   ├── scenarios.py              # 90-day yield management scenarios
│   └── last_minute.py            # Last-minute distress pricing engine
├── dashboard/
│   └── app.py                    # Streamlit pricing simulator (5 tabs)
├── tests/
│   └── test_hotel.py             # pytest suite
├── notebooks/                    # Exploratory analysis (optional)
├── Makefile                      # Pipeline commands
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/hotel-revenue-management.git
cd hotel-revenue-management

# Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
make data        # Generate synthetic hotel dataset
make train       # Fit log-log + LightGBM models
make optimize    # Rate optimization + scenarios + last-minute engine
make test        # Run pytest suite

# Launch the dashboard
make dashboard   # Opens Streamlit at localhost:8501
```

---

## Pipeline Detail

### Step 1 — Synthetic Data (`make data`)

Generates **15 properties × 104 weeks × 7 days ≈ 10,920 rows** with realistic
demand-elasticity relationships embedded per tier, segment, and booking window.

### Step 2 — Log-Log Elasticity Model (`make train`)

OLS regression: `ln(occupancy) = α + β·ln(ADR) + γ·ln(comp_rate) + controls`

- Fitted **per property** and **per tier**
- Segmented by booking window, day-of-week, traveller segment
- Outputs: elasticity, 95 % CI, R², p-values

### Step 3 — ML Model — LightGBM (`make train`)

- 20+ features including lagged occupancy, weather, channel mix
- Time-series cross-validation (4 folds)
- SHAP explanations — top drivers: lead-time, competitor rate, events
- Implied elasticities compared with log-log estimates

### Step 4 — RevPAR Optimizer (`make optimize`)

Finds `argmax RevPAR(p)` subject to:
- **Rate parity:** ±5 % of OTA rates
- **Brand floor:** budget ≥ £50, luxury ≥ £250
- **Competitor proximity:** within ±20 % of comp set

### Step 5 — Yield Scenarios (`make optimize`)

90-day forward simulation under 4 strategies per property.

### Step 6 — Last-Minute Pricing (`make optimize`)

Distress pricing engine with OTA-only channel strategy.

---

## Streamlit Dashboard

```bash
make dashboard
```

| Tab | Content |
|-----|---------|
| **Property Overview** | RevPAR heatmap (property × week), elasticity table |
| **Rate Optimizer** | Interactive demand curve with rate slider — live KPIs |
| **Scenario Planner** | Side-by-side 90-day strategy comparison |
| **Last-Minute Engine** | Unsold inventory alerts & distress rate recommendations |
| **Competitive Positioning** | Rate index, RGI, our ADR vs comp set |

---

## Yield Management Glossary

| Term | Definition |
|------|-----------|
| **ADR** | Average Daily Rate — average price charged per occupied room |
| **RevPAR** | Revenue Per Available Room = ADR × Occupancy Rate |
| **GOPPAR** | Gross Operating Profit Per Available Room |
| **RGI** | Revenue Generation Index = Your RevPAR ÷ Comp Set RevPAR |
| **Comp Set** | Competitive set — the hotels you benchmark against |
| **Rate Parity** | Maintaining consistent rates across direct and OTA channels |
| **BAR** | Best Available Rate — the lowest unrestricted rate offered |
| **GDS** | Global Distribution System (Amadeus, Sabre, Travelport) |
| **OTA** | Online Travel Agency (Booking.com, Expedia) |
| **Booking Window** | Lead time between reservation date and stay date |
| **Distress Pricing** | Discounting unsold inventory close to stay date |
| **Yield Management** | Adjusting price and inventory to maximise revenue |
| **Perishable Inventory** | Hotel rooms expire daily — unsold = zero revenue |

---

## Tech Stack

- **Python 3.11**
- pandas, NumPy, SciPy, statsmodels — data & econometrics
- scikit-learn, LightGBM — ML demand modelling
- SHAP — model explainability
- Plotly, Matplotlib — visualisation
- Streamlit — interactive pricing simulator
- pytest — testing

---

## License

MIT
