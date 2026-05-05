# ATM Cash Replenishment — Monte Carlo Simulation
**Authors:** Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/13737cf7-f1c3-4bf2-9492-69312f7eb807" />

## Overview
This project simulates a single ATM's cash lifecycle over one year to study how replenishment policies and demand patterns affect stockout risk. Using 10,000 Monte Carlo runs per experiment, we test three hypotheses grounded in a real-world Danish ATM transaction dataset and Federal Reserve payment statistics.

# Project Structure

```
.
├── src/
│   ├── config.py              # Central configuration — all parameters here
│   ├── simulation.py          # Core simulation engine (shared by all phases)
│   ├── fit_distributions.py   # Phase 1: fit distributions from Danish ATM data
│   ├── phase2_control.py      # Phase 2: control run validation
│   ├── h1.py                  # Phase 3: Hypothesis 1 (weekend stockouts)
│   ├── h2.py                  # Phase 3: Hypothesis 2 (policy comparison)
│   ├── h3.py                  # Phase 3: Hypothesis 3 (peak-hour stockouts)
│   └── analysis.py            # Plot generation and JSON export
├── tests/
│   ├── test_simulation.py
│   ├── test_hypothesis2.py
│   └── test_hypothesis3.py
├── data/
│   └── danish_atm.csv         # Raw dataset (not included — download separately)
├── outputs/                   # All plots and results saved here
│   ├── phase2_control.png
│   ├── h1_dow_stockouts.png
│   ├── h2_policy_comparison.png
│   ├── h3_hour_stockouts.png
│   └── results.json
└── run_simulation.py                    # Entry point — runs all phases in order
```

---

## Data Sources

| Source | Usage |
|---|---|
| [Danish ATM Transactions (Kaggle)](https://www.kaggle.com/datasets/sparnord/danish-atm-transactions) | Hourly Poisson arrival rates, day-of-week multipliers |
| Federal Reserve Payments Study 2022 | Lognormal mean withdrawal ($198.13) |
| Ekinci et al. 2015 (logistics literature) | Lognormal sigma shape parameter (0.82) |
| Weibull literature (logistics) | Truck delay distribution (shape=1.5, scale=0.5) |

---

## Setup

### Requirements

- Python 3.11+
- `numpy`, `scipy`, `pandas`, `matplotlib`

```bash
pip install numpy scipy pandas matplotlib
```

### Optional (for Phase 1 re-fitting)

Download `danish_atm.csv` from Kaggle and place it in `data/`. Then:

```bash
python -m src.fit_distributions
```

Copy the printed parameter values into `src/config.py` (already done — fitted values are present in `config.py`).

## Running the Simulation

```bash
python run_simulation.py
```

This runs all four phases in order:

1. **Phase 2 control** — validates the model against a real-world benchmark
2. **H1** — weekend stockout disproportionality
3. **H2** — fixed schedule vs demand-triggered policy
4. **H3** — peak-hour stockout clustering

Outputs are saved to `outputs/`.

### Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Simulation Design

### Core Engine (`simulation.py`)

Each simulated year runs 365 days. For each day:

1. **Check truck arrival** — if a pending dispatch is due, cash is replenished
2. **Hourly demand loop** — for each of 24 hours:
   - Draw Poisson arrivals (λ varies by hour and weekend/weekday)
   - Draw lognormal withdrawal amounts per customer
   - Deplete cash; flag a stockout if cash is exhausted
3. **Dispatch decision** — based on active policy:
   - `fixed`: refill every 3 days regardless of cash level
   - `demand`: refill when cash drops below $500,000
4. **Truck delay** — drawn from Weibull(shape=1.5, scale=0.5), minimum 1 day

### Random Variables

| Variable | Distribution | Parameters |
|---|---|---|
| Withdrawal amount | Lognormal | μ=4.95, σ=0.82 |
| Hourly customer arrivals | Poisson | λ varies by hour and day type (fitted from data) |
| Day-of-week demand scaling | Fitted multipliers | Mon=1.0 baseline |
| Truck delivery delay | Weibull | shape=1.5, scale=0.5 |

---

## Results Summary

### Phase 2 — Control Run Validation

| Metric | Value |
|---|---|
| Mean annual stockout rate | 14.83% |
| Std deviation | 0.009 |
| Benchmark range | 5%–20% |
| **Verdict** | **PASS** |

The model produces a plausible real-world stockout rate before any experimental manipulation.

---

### H1 — Weekend Stockout Disproportionality

**Hypothesis:** Saturday and Sunday cause a disproportionate share of stockouts relative to their 28.6% share of operating days.

| Metric | Value |
|---|---|
| Weekend stockout share | 38.8% |
| Expected (uniform) | 28.6% |
| Disproportionality ratio | 1.36× |
| **Finding** | **H1 SUPPORTED** |

Weekends account for nearly 39% of all stockouts despite being only 2 out of 7 days, driven by elevated transaction volume on Saturday especially.

---

### H2 — Fixed Schedule vs Demand-Triggered Policy

**Hypothesis:** A demand-triggered refill policy reduces stockout rate but increases dispatch frequency.

| Metric | Fixed (3-day) | Demand-triggered | Change |
|---|---|---|---|
| Mean stockout rate | 14.84% | 1.04% | **−93.0%** |
| Mean dispatches/year | 121 | 128 | +5.6% |
| **Finding** | — | — | **H2 SUPPORTED** |

The demand-triggered policy nearly eliminates stockouts at the cost of only 7 additional truck dispatches per year — a highly favorable trade-off.

---

### H3 — Peak-Hour Stockout Clustering

**Hypothesis:** The 8 busiest hours by transaction volume account for more than their uniform share (33.3%) of stockout events.

| Metric | Value |
|---|---|
| Peak hours (by fitted λ) | 09:00–16:00 |
| Peak hour stockout share | 63.1% |
| Expected (uniform) | 33.3% |
| Disproportionality ratio | 1.89× |
| **Finding** | **H3 SUPPORTED** |

Stockouts are nearly twice as likely during peak hours, with strong clustering around 11:00, 16:00, and 17:00–19:00.

---

## Configuration

All parameters are centralized in `src/config.py`. Key values:

```python
"initial_cash":      1_200_000   # Starting cash ($)
"refill_amount":     1_200_000   # Cash added per dispatch ($)
"fixed_refill_days": 3           # Fixed policy interval (days)
"demand_threshold":  500_000     # Demand policy trigger ($)
"sim_days":          365         # Days per run
"num_runs":          10_000      # Monte Carlo iterations
```

If `lognormal_mu`, `poisson_lambda_weekday`, etc. are set to `None`, the simulation automatically falls back to literature-based estimates. The `get_param()` helper in `config.py` manages this transparently.

---

## Reproducibility

Every experiment uses a deterministic seeding scheme: run `i` under experiment `X` uses seed `base_seed_X + i`. Base seeds are:

| Experiment | Base seed |
|---|---|
| Phase 2 control | 0 |
| H1 | 10,000 |
| H2 fixed | 20,000 |
| H2 demand | 30,000 |
| H3 | 40,000 |

---

