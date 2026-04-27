"""
h2.py
Phase 3 — Hypothesis 2: A demand-triggered refill policy reduces
stockout rate compared to a fixed 3-day schedule, but increases
dispatch frequency and variability.

Data grounding: Real hourly and daily demand patterns from Danish ATM
dataset calibrate the simulation. Fed Reserve 2022 anchors mean
withdrawal at $198.13.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import numpy as np
from src.config import CONFIG
from src.simulation import run_many


def run_h2(config: dict = CONFIG) -> dict:
    """Run H2 experiment: fixed schedule vs demand-triggered policy.

    Parameters
    ----------
    config : dict
        Simulation configuration from config.py.

    Returns
    -------
    dict with keys:
        fixed               : dict of stats for fixed policy
        demand              : dict of stats for demand policy
        stockout_reduction  : float
        dispatch_change     : float
        supported           : bool
        finding             : str

    Examples
    --------
    >>> from src.config import CONFIG
    >>> result = run_h2(CONFIG)
    >>> 'fixed' in result and 'demand' in result
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — H2: Fixed schedule vs demand-triggered policy")
    print("=" * 55)
    print("Prediction: demand-triggered policy reduces stockout rate")
    print("but increases dispatch frequency.")

def _summarize(results: list[dict]) -> dict:
    """Compute summary statistics from a list of simulation results.

    Parameters
    ----------
    results : list[dict]
        Output from run_many().

    Returns
    -------
    dict with keys:
        mean_stockout_rate  : float
        std_stockout_rate   : float
        p5_stockout_rate    : float
        p95_stockout_rate   : float
        mean_dispatch_count : float
        std_dispatch_count  : float
        all_rates           : list[float]

    Examples
    --------
    >>> results = [{'stockout_rate': 0.1, 'dispatch_count': 100,
    ...             'stockout_by_dow': [0]*7,
    ...             'stockout_by_hour': [0]*24,
    ...             'stockout_days': 10,
    ...             'cash_history': []}]
    >>> s = _summarize(results)
    >>> s['mean_stockout_rate'] == 0.1
    True
    """
    rates = [r["stockout_rate"] for r in results]
    dispatches = [r["dispatch_count"] for r in results]

    return {
        "mean_stockout_rate": round(float(np.mean(rates)), 6),
        "std_stockout_rate": round(float(np.std(rates)), 6),
        "p5_stockout_rate": round(float(np.percentile(rates, 5)), 6),
        "p95_stockout_rate": round(float(np.percentile(rates, 95)), 6),
        "mean_dispatch_count": round(float(np.mean(dispatches)), 2),
        "std_dispatch_count": round(float(np.std(dispatches)), 2),
        "all_rates": rates,
    }