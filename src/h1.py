"""
Phase 3— Hypothesis 1: Weekend days cause a disproportionate share
of ATM stockouts relative to their 28.6% share of operating days.

Data grounding: Danish ATM dataset weekend_flag and weekday columns show elevated transaction volume on Saturday and Sunday.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""
import numpy as np
from src.config import CONFIG
from src.simulation import run_many

DAYS = ["Mon","Tue", "Wed", "Thu", "Fri","Sat","Sun"]
WEEKEND_DAYS = {"Sat", "Sun"}
WEEKEND_SHARE_UNIFORM= 2 / 7  # 28.6% expected if stockouts were random

def run_h1(config: dict = CONFIG) -> dict:
    """Run H1 experiment: weekend stockout disproportionality.
    Runs num_runs simulations under fixed schedule policy.
    Measures what fraction of stockouts fall on Saturday and Sunday
    compared to the uniform expectation of 2/7 = 28.6%.
    
    Parameters
    ----------
    config : dict
        Simulation configuration from config.py.

    Returns
    -------
    dict with keys:
        weekend_stockout_fraction  : float
        weekday_stockout_fraction  : float
        expected_weekend_fraction  : float  (always 2/7)
        disproportionality_ratio   : float
        stockouts_by_dow           : list[int]
        stockouts_by_dow_pct       : list[float]
        total_stockouts            : int
        supported                  : bool
        finding                    : str

    Examples
    --------
    >>> from src.config import CONFIG
    >>> result = run_h1(CONFIG)
    >>> 0.0 <= result['weekend_stockout_fraction'] <= 1.0
    True
    >>> result['expected_weekend_fraction'] == 2 / 7
    True
    >>> isinstance(result['supported'], bool)
    True
    >>> len(result['stockouts_by_dow']) == 7
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — H1: Weekend stockout disproportionality")
    print("=" * 55)
    print("Prediction: Sat + Sun will account for more than 28.6%")
    print("of all stockout events despite being 28.6% of days.")

    results = run_many(
        config,
        policy="fixed",
        label="H1",
        base_seed=10_000,
    )
    # Aggregate stockout_by_dow across all runs
    total_by_dow = [0] * 7
    for r in results:
        for i in range(7):
            total_by_dow[i] += r["stockout_by_dow"][i]

    total_stockouts = sum(total_by_dow)

    if total_stockouts == 0:
        return _zero_stockout_result()

    weekend_stockouts =total_by_dow[5] + total_by_dow[6]
    weekday_stockouts = sum(total_by_dow[:5])

    weekend_fraction= weekend_stockouts / total_stockouts
    weekday_fraction = weekday_stockouts / total_stockouts
    ratio = weekend_fraction/ WEEKEND_SHARE_UNIFORM

    pct_by_dow =[
        round(total_by_dow[i] / total_stockouts * 100, 2)
        for i in range(7)]
    supported = ratio > 1.2

    finding = (
        f"{weekend_fraction:.1%} of stockouts occur on weekends "
        f"vs expected {WEEKEND_SHARE_UNIFORM:.1%} "
        f"(ratio: {ratio:.2f}x). "
        f"H1 {'SUPPORTED' if supported else 'NOT SUPPORTED'}."
    )
    _print_results(total_by_dow, pct_by_dow, weekend_fraction,
                   ratio, finding)

    return {
        "weekend_stockout_fraction": round(weekend_fraction, 6),
        "weekday_stockout_fraction":round(weekday_fraction, 6),
        "expected_weekend_fraction": WEEKEND_SHARE_UNIFORM,
        "disproportionality_ratio": round(ratio, 4),
        "stockouts_by_dow": total_by_dow,
        "stockouts_by_dow_pct":pct_by_dow,
        "total_stockouts": total_stockouts,
        "supported":supported,
        "finding": finding,
    }

def _zero_stockout_result() -> dict:
    """Return a safe result dict when no stockouts occurred.
    Returns
    -------
    dict
        Result dict with zero values and unsupported finding.
    Examples
    --------
    >>> result = _zero_stockout_result()
    >>> result['total_stockouts'] == 0
    True
    >>> result['supported'] == False
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
    """
    print(f"\n  Stockouts by day of week:")
    for i, day in enumerate(DAYS):
        bar = "█" * int(pct_by_dow[i] / 2)
        tag = " <- WEEKEND" if day in WEEKEND_DAYS else ""
        print(f"    {day}: {pct_by_dow[i]:5.1f}%  {bar}{tag}")

    print(f"\n  Weekend stockout share:    {weekend_fraction:.2%}")
    print(f"  Expected (uniform):        {WEEKEND_SHARE_UNIFORM:.2%}")
    print(f"  Disproportionality ratio:  {ratio:.2f}x")
    print(f"\n  Finding: {finding}")

if __name__ == "__main__":
    run_h1()