"""
h1.py
Phase 3 — Hypothesis 1: Weekend days cause a disproportionate share
of ATM stockouts relative to their 28.6% share of operating days.

Data grounding: Danish ATM dataset weekend_flag and weekday columns
show elevated transaction volume on Saturday and Sunday.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
from typing import Any
from src.config import CONFIG
from src.simulation import run_many

DAYS: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKEND_DAYS: set[str] = {"Sat", "Sun"}
WEEKEND_SHARE_UNIFORM: float = 2 / 7  # 28.6%


def run_h1(config: dict[str, Any] = CONFIG) -> dict[str, Any]:
    """Run H1 experiment: weekend stockout disproportionality.

    Runs num_runs simulations under fixed schedule policy and measures
    what fraction of stockouts fall on Saturday and Sunday compared to
    the uniform expectation of 2/7 = 28.6%.

    Parameters
    ----------
    config : dict[str, Any]
        Simulation configuration from config.py.

    Returns
    -------
    dict[str, Any]
        Keys: weekend_stockout_fraction, weekday_stockout_fraction,
        expected_weekend_fraction, disproportionality_ratio,
        stockouts_by_dow, stockouts_by_dow_pct, total_stockouts,
        supported, finding.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 30, 'num_runs': 5,
    ...        'benchmark_min': 0.0, 'benchmark_max': 1.0}
    >>> result = run_h1(cfg)
    >>> 0.0 <= result['weekend_stockout_fraction'] <= 1.0
    True
    >>> result['expected_weekend_fraction'] == 2 / 7
    True
    >>> isinstance(result['supported'], bool)
    True
    >>> len(result['stockouts_by_dow']) == 7
    True
    >>> len(result['stockouts_by_dow_pct']) == 7
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — H1: Weekend stockout disproportionality")
    print("=" * 55)
    print("Prediction: Sat + Sun will account for more than 28.6%")
    print("of all stockout events despite being 28.6% of days.")
    
    results: list[dict[str, Any]] = run_many(
        config, policy="fixed", label="H1", base_seed=10_000,
    )
    # Aggregate stockout_by_dow across all runs
    total_by_dow: list[int] = [0] * 7
    for r in results:
        for i in range(7):
            total_by_dow[i] += r["stockout_by_dow"][i]

    total_stockouts: int = sum(total_by_dow)

    if total_stockouts == 0:
        return _zero_stockout_result()

    weekend_stockouts: int = total_by_dow[5] + total_by_dow[6]
    weekend_fraction: float = weekend_stockouts / total_stockouts
    weekday_fraction: float = 1.0 - weekend_fraction
    ratio: float = weekend_fraction / WEEKEND_SHARE_UNIFORM

    pct_by_dow: list[float] = [
        round(total_by_dow[i] / total_stockouts * 100, 2)
        for i in range(7)
    ]
    supported: bool = ratio > 1.2

    finding: str = (
        f"{weekend_fraction:.1%} of stockouts occur on weekends "
        f"vs expected {WEEKEND_SHARE_UNIFORM:.1%} "
        f"(ratio: {ratio:.2f}x). "
        f"H1 {'SUPPORTED' if supported else 'NOT SUPPORTED'}."
    )
    _print_results(total_by_dow, pct_by_dow, weekend_fraction,
                   ratio, finding)

    return {
        "weekend_stockout_fraction": round(weekend_fraction, 6),
        "weekday_stockout_fraction": round(weekday_fraction, 6),
        "expected_weekend_fraction": WEEKEND_SHARE_UNIFORM,
        "disproportionality_ratio": round(ratio, 4),
        "stockouts_by_dow": total_by_dow,
        "stockouts_by_dow_pct": pct_by_dow,
        "total_stockouts": total_stockouts,
        "supported": supported,
        "finding": finding,
    }


def _zero_stockout_result() -> dict[str, Any]:
    """Return a safe result dict when no stockouts occurred.

    Returns
    -------
    dict[str, Any]
        Result dict with zero values and unsupported finding.

    Examples
    --------
    >>> result = _zero_stockout_result()
    >>> result['total_stockouts'] == 0
    True
    >>> result['supported'] == False
    True
    >>> len(result['stockouts_by_dow']) == 7
    True
    """
    return {
        "weekend_stockout_fraction": 0.0,
        "weekday_stockout_fraction": 0.0,
        "expected_weekend_fraction": WEEKEND_SHARE_UNIFORM,
        "disproportionality_ratio": 0.0,
        "stockouts_by_dow": [0] * 7,
        "stockouts_by_dow_pct": [0.0] * 7,
        "total_stockouts": 0,
        "supported": False,
        "finding": "No stockouts observed. Check distribution parameters.",
    }


def _print_results(
    total_by_dow: list[int],
    pct_by_dow: list[float],
    weekend_fraction: float,
    ratio: float,
    finding: str,
) -> None:
    """Print formatted H1 results to terminal.

    Parameters
    ----------
    total_by_dow : list[int]
        Total stockout count per day of week.
    pct_by_dow : list[float]
        Percentage of total stockouts per day of week.
    weekend_fraction : float
        Fraction of stockouts on weekends.
    ratio : float
        Disproportionality ratio vs uniform expectation.
    finding : str
        Plain-English result statement.

    Examples
    --------
    >>> _print_results([10]*7, [14.3]*7, 0.286, 1.0, 'Test finding')
    <BLANKLINE>
      Stockouts by day of week:
        Mon:  14.3%  ...
        ...
    """
    print(f"\n  Stockouts by day of week:")
    for i, day in enumerate(DAYS):
        bar: str = "█" * int(pct_by_dow[i] / 2)
        tag: str = " <- WEEKEND" if day in WEEKEND_DAYS else ""
        print(f"    {day}: {pct_by_dow[i]:5.1f}%  {bar}{tag}")
    print(f"\n  Weekend stockout share:    {weekend_fraction:.2%}")
    print(f"  Expected (uniform):        {WEEKEND_SHARE_UNIFORM:.2%}")
    print(f"  Disproportionality ratio:  {ratio:.2f}x")
    print(f"\n  Finding: {finding}")


if __name__ == "__main__":
    run_h1()
