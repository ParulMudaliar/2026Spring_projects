"""
phase2_control.py
Phase 2 — Control run validation.

Run the simulation under the fixed 3-day schedule to confirm the model
produces a plausible stockout rate before we start experimenting.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
AI Disclosure: Claude  assisted in drafting this code.
"""
import numpy as np
from src.config import CONFIG
from src.simulation import run_many

def run_control(config: dict = CONFIG) -> dict:
    def run_control(config: dict = CONFIG) -> dict:
        """Validate the control simulation against a real-world benchmark.

        Runs the fixed 3-day replenishment policy num_runs times and checks
        whether the mean stockout rate lands in the expected range. If it
        doesn't, the distribution parameters need another look before we move
        on to Phase 3.

        Parameters
        ----------
        config : dict
            Simulation config from config.py.

        Returns
        -------
        dict with keys:
            mean_stockout_rate  : float
            std_stockout_rate   : float
            min_stockout_rate   : float
            max_stockout_rate   : float
            mean_dispatch_count : float
            benchmark_pass      : bool
            verdict             : str
            all_rates           : list[float]

        Examples
        --------
        >>> from src.config import CONFIG
        >>> result = run_control(CONFIG)
        >>> 0.0 <= result['mean_stockout_rate'] <= 1.0
        True
        >>> isinstance(result['benchmark_pass'], bool)
        True
        >>> isinstance(result['verdict'], str)
        True
        """

    print("\n" + "=" * 55)
    print("PHASE 2 — Control run (fixed 3-day schedule)")
    print("=" * 55)

    results = run_many(
        config,
        policy="fixed",
        label="Control",
        base_seed=0,
    )

    rates = [r["stockout_rate"] for r in results]
    dispatches = [r["dispatch_count"] for r in results]

    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    min_rate = float(np.min(rates))
    max_rate = float(np.max(rates))
    mean_dispatches = float(np.mean(dispatches))

    benchmark_pass = (
        config["benchmark_min"] <= mean_rate <= config["benchmark_max"]
    )

    if benchmark_pass:
        verdict = (
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


def _print_results(mean_rate, std_rate, min_rate, max_rate, mean_dispatches, verdict) -> None:
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
        """
    print(f"\n  Mean stockout rate:   {mean_rate:.2%}")
    print(f"  Std stockout rate:    {std_rate:.4f}")
    print(f"  Range:                {min_rate:.2%} — {max_rate:.2%}")
    print(f"  Mean dispatches/year: {mean_dispatches:.1f}")
    print(f"\n  {verdict}")