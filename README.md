# 🏧 ATM Cash Replenishment — Monte Carlo Simulation

**Authors:** Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/13737cf7-f1c3-4bf2-9492-69312f7eb807" />

## Overview

This project simulates a single Danish ATM's cash lifecycle over one year to study how replenishment policies and demand patterns affect stockout risk. Using 1,000 Monte Carlo runs per experiment, we test three hypotheses grounded in a real-world Danish ATM transaction dataset and ECB payment statistics.

**Bottom line:** Weekends cause a disproportionate share of stockouts relative to their 28.6% share of days (H1 supported). A demand-triggered refill policy cuts the stockout rate from ~15% to ~3% with only 7 additional truck dispatches per year — a strong operational trade-off (H2 supported). Peak hours (09:00–16:00) account for over 60% of stockouts despite being only 33% of the day (H3 supported).

---

## Important Note

Please refer to the project documentation and README for a better understanding of the simulation design, parameter calibration, and hypothesis methodology. We have included detailed explanations to make the project easier to follow and reproduce.

---

## 1. Problem Statement

How do ATM replenishment policies and demand timing patterns affect cash stockout risk for a single Danish bank-branch ATM over a one-year horizon?

This project evaluates two replenishment strategies — fixed-schedule and demand-triggered — and tests whether stockouts cluster predictably by day of week and hour of day, using a real transaction dataset from Spare Nord Bank.

---

## 2. Research Questions

Does stockout risk for a single ATM vary systematically by:

- Day of week, with weekends showing disproportionate stockout share?
- Replenishment policy, with demand-triggered refills outperforming fixed schedules?
- Hour of day, with peak transaction hours accounting for more than their uniform share of stockouts?

---

## 3. Project Type Justification (Type II)

This is a **Type II project** because it builds a Monte Carlo simulation grounded in multiple real-world data sources:

| Data Source | Description |
|---|---|
| Danish ATM Transactions (Spare Nord) | Hourly transaction counts across 38 ATMs — used to fit Poisson arrival rates and day-of-week multipliers |
| ECB Payment Statistics 2022 (EU-27) | Total ATM withdrawal value and count — used to anchor lognormal mean withdrawal at €150 ≈ DKK 1,100 |
| ATMIA European ATM Standards | Defines realistic ATM cash capacity range (DKK 400,000–1,200,000) |
| Logistics literature (Geismar et al. 2017; Ekinci et al. 2015) | Weibull truck delay parameters and lognormal sigma shape |

**Key contribution:** The project scales fleet-level transaction data (38 ATMs, ~1,952 txns/day) down to a single-ATM model (~250 txns/day) using a calibrated scale factor of 0.1280, enabling realistic single-ATM simulation while preserving the empirical hourly demand shape from the dataset.

---

## 4. Data Sources

| Source | URL |
|---|---|
| Danish ATM Transactions (Kaggle / Spare Nord) | https://www.kaggle.com/datasets/sparnord/danish-atm-transactions |
| ECB Payment Statistics 2022 | https://data.ecb.europa.eu/data/datasets/PTT |
| Danmarks Nationalbank Payments | https://www.nationalbanken.dk/en/news-and-knowledge/data-and-statistics/payments |

---

## 5. Hypotheses and Results

### H1 — Weekend Stockout Disproportionality

**Hypothesis:** Saturday and Sunday account for more than their uniform share (28.6%) of all stockout events.

**Result:** Weekend stockout share = 42.9%, expected = 28.6%, disproportionality ratio = 1.50×. **H1 Supported.** Weekends see elevated stockouts primarily because the fixed 3-day refill schedule is agnostic to weekend demand spikes, and Saturday in particular carries higher transaction volume (1.18× Monday baseline).

---

### H2 — Fixed Schedule vs. Demand-Triggered Policy

**Hypothesis:** A demand-triggered refill policy reduces the stockout rate compared to a fixed 3-day schedule, at the cost of more truck dispatches per year.

**Result:** Fixed schedule = 8.10% stockout rate, 121 dispatches/year. Demand-triggered = 3.05% stockout rate, 123 dispatches/year. Stockout rate change = -62.4%, dispatch change = +1.8%. **H2 Supported.** The demand-triggered policy nearly eliminates stockouts at the cost of roughly 2 additional truck trips per year.

---

### H3 — Peak-Hour Stockout Clustering

**Hypothesis:** The 8 peak transaction hours (09:00–16:00) account for more than their uniform share (33.3%) of stockout events.

**Result:** Peak-hour stockout share = 46.2%, expected = 33.3%, disproportionality ratio = 1.39×. **H3 Supported.** Stockouts cluster strongly in the afternoon peak, with 15:00 and 16:00 showing the highest individual shares.

---

## 6. Methodology Summary

1. **Distribution fitting (Phase 1):** Lognormal withdrawal amounts are anchored to the ECB 2022 EU-27 average (€150 ≈ DKK 1,100). Poisson hourly arrival rates are fitted from the Danish dataset and scaled by 0.1280 to represent a single ATM. Day-of-week multipliers are fitted from the dataset directly.
2. **Control validation (Phase 2):** A fixed 3-day schedule is run for 1,000 iterations and checked against a real-world benchmark stockout range of 5–20%.
3. **Hypothesis experiments (Phase 3):** H1, H2, and H3 each run 1,000 simulations with deterministic seeding for reproducibility.
4. **Truck delay:** Each dispatch draws a Weibull(shape=1.5, scale=0.5) delay, minimum 1 day.
5. **Stockout definition:** A stockout day is any day where cash is exhausted before a transaction can complete. Only the first stockout per day is recorded.

---

## 7. Limitations

- Single-ATM model only.
- Weibull truck delay parameters are drawn from general logistics literature, not Danish CIT operator data.
- The fixed 3-day refill schedule does not account for weekday vs. weekend timing, a real-world operator would likely adjust this.

---

## 8. Ethical Considerations

- All data sources are publicly available (Kaggle, ECB, Danmarks Nationalbank).
- No individual transaction records or customer data are used; the dataset contains only anonymized aggregate counts.
- Parameter choices are documented with sources to allow independent verification.

---

## 9. Reproducibility

### Setup

```bash
pip install numpy scipy pandas matplotlib numba
```

### Run the simulation

```bash
python main.py
```

This runs all four phases in order:

1. **Phase 1** — fits distributions from `data/danish_atm.csv` if present, otherwise uses fallback parameters from `config.py`
2. **Phase 2** — control run validation against 5–20% stockout benchmark
3. **Phase 3** — H1, H2, and H3 experiments
4. **Analysis** — saves plots and `results.json` to `outputs/`

### Dataset

Download `danish_atm.csv` from [Kaggle](https://www.kaggle.com/datasets/sparnord/danish-atm-transactions) and place it at `data/danish_atm.csv`. The simulation runs on fallback parameters if the file is absent.

### Run tests

```bash
python -m pytest
```

---

## 10. Repository Structure

```
.
├── README.md
├── main.py          # Entry point — runs all phases in order
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
│   ├── test_hypothesis1.py
│   ├── test_hypothesis2.py
│   └── test_hypothesis3.py
├── data/
│   └── danish_atm.csv         # Raw dataset (not included — download separately)
└── outputs/                   # After running the code, All plots and results saved here
    ├── phase2_control.png
    ├── h1_dow_stockouts.png
    ├── h2_policy_comparison.png
    ├── h3_hour_stockouts.png
    └── results.json
```

---

## 11. Results Summary Table

| Hypothesis | Test | Key Statistic | Benchmark | Verdict |
|---|---|---|---|---|
| Phase 2 Control | Mean stockout rate | 8.17% | 5–20% | **PASS** |
| H1: Weekend disproportionality | Weekend share vs. uniform | 42.7% vs. 28.6% (1.49×) | > 28.6% | **Supported** |
| H2: Policy comparison | Stockout rate change | 8.16% → 3.04% (−62.7%) | Rate decreases | **Supported** |
| H3: Peak-hour clustering | Peak share vs. uniform | 46.1% vs. 33.3% (1.38×) | > 33.3% | **Supported** |
---

## 12. Configuration

All parameters are in `src/config.py`. Key values:

```python
"initial_cash":      880_000    # DKK 
"refill_amount":     880_000    # DKK
"fixed_refill_days": 3          # Fixed policy interval (days)
"demand_threshold":  300_000    # DKK 
"sim_days":          365        # Days per run
"num_runs":          10_000      # Monte Carlo iterations
```

If `lognormal_mu`, `poisson_lambda_weekday`, etc. are set to `None`, the simulation falls back to literature-based estimates automatically via `get_param()` in `config.py`.

---

## 13. Reproducibility — Seeding Scheme

Every experiment uses deterministic seeding: run `i` under experiment `X` uses seed `base_seed_X + i`.

| Experiment | Base seed |
|---|---|
| Phase 2 control | 0 |
| H1 | 10,000 |
| H2 fixed | 20,000 |
| H2 demand | 30,000 |
| H3 | 40,000 |

---

## 14. AI Disclosure

Claude assisted in brainstorming ideas, choosing random variables and slide content. The simulation logic, dataset exploration, and results interpretation are team work.

---

## 15. Team

- Parul Mudaliar
- Nandhini Ramesh
- Suriya Gopal
