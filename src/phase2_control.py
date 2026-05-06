"""
phase2_control.py
Phase 2 — Control run validation.

Run the simulation under the fixed 3-day schedule to confirm the model
produces a plausible stockout rate before we start experimenting.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

import numpy as np
from typing import Any
from src.config import CONFIG
from src.simulation import run_many


def run_control(config: dict[str, Any] = CONFIG) -> dict[str, Any]:
    """Validate the control simulation against a real-world benchmark.

    Runs the fixed 3-day replenishment policy num_runs times and checks
    whether the mean stockout rate lands in the expected range.

    Parameters
    ----------
    config : dict[str, Any]
        Simulation config from config.py.

    Returns
    -------
    dict[str, Any]
        Keys: mean_stockout_rate, std_stockout_rate, min_stockout_rate,
        max_stockout_rate, mean_dispatch_count, benchmark_pass, verdict,
        all_rates.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 30, 'num_runs': 5,
    ...        'benchmark_min': 0.0, 'benchmark_max': 1.0}
    >>> result = run_control(cfg)
    >>> 0.0 <= result['mean_stockout_rate'] <= 1.0
    True
    >>> isinstance(result['benchmark_pass'], bool)
    True
    >>> result['benchmark_pass']
    True
    >>> isinstance(result['verdict'], str)
    True
    >>> 'PASS' in result['verdict']
    True
    >>> len(result['all_rates']) == 5
    True
    """
    print("\n" + "=" * 55)
    print("PHASE 2 — Control run (fixed 3-day schedule)")
    print("=" * 55)

    results: list[dict[str, Any]] = run_many(
        config,
        policy="fixed",
        label="Control",
        base_seed=0,
    )

    rates: list[float] = [r["stockout_rate"] for r in results]
    dispatches: list[int] = [r["dispatch_count"] for r in results]

    mean_rate: float = float(np.mean(rates))
    std_rate: float = float(np.std(rates))
    min_rate: float = float(np.min(rates))
    max_rate: float = float(np.max(rates))
    mean_dispatches: float = float(np.mean(dispatches))

    benchmark_pass: bool = (
        config["benchmark_min"] <= mean_rate <= config["benchmark_max"]
    )

    if benchmark_pass:
        verdict: str = (
            f"PASS — mean stockout rate {mean_rate:.2%} is within "
            f"benchmark range "
            f"({config['benchmark_min']:.0%}–{config['benchmark_max']:.0%}). "
            f"Proceeding to Phase 3."
        )
    else:
        verdict = (
            f"WARNING — mean stockout rate {mean_rate:.2%} is outside "
            f"benchmark range "
            f"({config['benchmark_min']:.0%}–{config['benchmark_max']:.0%}). "
            f"Re-examine distribution parameters before Phase 3."
        )

    _print_results(mean_rate, std_rate, min_rate, max_rate,
                   mean_dispatches, verdict)

    return {
        "mean_stockout_rate": mean_rate,
        "std_stockout_rate": std_rate,
        "min_stockout_rate": min_rate,
        "max_stockout_rate": max_rate,
        "mean_dispatch_count": mean_dispatches,
        "benchmark_pass": benchmark_pass,
        "verdict": verdict,
        "all_rates": rates,
    }


def _print_results(
    mean_rate: float,
    std_rate: float,
    min_rate: float,
    max_rate: float,
    mean_dispatches: float,
    verdict: str,
) -> None:
    """Print the Phase 2 summary to the terminal.

    Parameters
    ----------
    mean_rate : float
        Mean stockout rate across all runs.
    std_rate : float
        Standard deviation of stockout rates.
    min_rate : float
        Minimum stockout rate observed.
    max_rate : float
        Maximum stockout rate observed.
    mean_dispatches : float
        Mean number of truck dispatches per year.
    verdict : str
        Pass or warning message.

    Examples
    --------
    >>> _print_results(0.10, 0.02, 0.05, 0.18, 122.0, 'PASS')
    <BLANKLINE>
      Mean stockout rate:   10.00%
      Std stockout rate:    0.0200
      Range:                5.00% — 18.00%
      Mean dispatches/year: 122.0
    <BLANKLINE>
      PASS
    """
    print(f"\n  Mean stockout rate:   {mean_rate:.2%}")
    print(f"  Std stockout rate:    {std_rate:.4f}")
    print(f"  Range:                {min_rate:.2%} — {max_rate:.2%}")
    print(f"  Mean dispatches/year: {mean_dispatches:.1f}")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    run_control()
