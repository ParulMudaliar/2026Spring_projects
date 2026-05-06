"""
main.py
Entry point for the ATM Cash Replenishment Monte Carlo Simulation.

Runs all phases in sequence:
    Phase 2  — Control run validation (fixed 3-day schedule)
    Phase 3  — H1: Weekend stockout disproportionality
    Phase 3  — H2: Fixed schedule vs demand-triggered policy
    Phase 3  — H3: Peak hour stockout clustering

All outputs (plots + results.json) are written to the outputs/ folder.

Usage:
    python main.py                  # full run (10,000 simulations)
    python main.py --fast           # quick run (100 simulations, for testing)

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import argparse
import sys
from typing import Any

from src.config import CONFIG
from src.phase2_control import run_control
from src.h1 import run_h1
from src.h2 import run_h2
from src.h3 import run_h3
from src.analysis import (
    plot_control_run,
    plot_h1,
    plot_h2,
    plot_h3,
    save_results,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a 'fast' boolean flag.

    Examples
    --------
    >>> args = parse_args.__wrapped__(['--fast']) if hasattr(parse_args, '__wrapped__') else None
    >>> isinstance(parse_args(), argparse.Namespace)
    True
    """
    parser = argparse.ArgumentParser(
        description="ATM Cash Replenishment Monte Carlo Simulation"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with 100 simulations instead of 10,000 (for quick testing)",
    )
    return parser.parse_args()


def build_config(fast: bool = False) -> dict[str, Any]:
    """Return CONFIG, optionally overriding num_runs for a fast test run.

    Parameters
    ----------
    fast : bool
        If True, sets num_runs to 100 for a quick smoke test.

    Returns
    -------
    dict[str, Any]
        Simulation configuration dictionary.

    Examples
    --------
    >>> cfg = build_config(fast=True)
    >>> cfg['num_runs']
    100

    >>> cfg_full = build_config(fast=False)
    >>> cfg_full['num_runs'] == CONFIG['num_runs']
    True
    """
    if fast:
        return {**CONFIG, "num_runs": 100}
    return CONFIG


def run_all(config: dict[str, Any]) -> None:
    """Run all simulation phases in sequence and save all outputs.

    Executes Phase 2 (control validation) then all three Phase 3
    hypothesis experiments. Saves plots and a results.json summary
    to the outputs/ directory.

    Parameters
    ----------
    config : dict[str, Any]
        Simulation configuration from config.py or build_config().

    Examples
    --------
    >>> from src.config import CONFIG
    >>> cfg = {**CONFIG, 'sim_days': 7, 'num_runs': 2,
    ...        'benchmark_min': 0.0, 'benchmark_max': 1.0}
    >>> run_all(cfg)  # doctest: +ELLIPSIS
    ...
    """
    print("\n" + "=" * 55)
    print("ATM CASH REPLENISHMENT — MONTE CARLO SIMULATION")
    print(f"Runs per experiment: {config['num_runs']:,}")
    print(f"Days per run:        {config['sim_days']:,}")
    print("=" * 55)

    # Phase 2 — Control run
    phase2 = run_control(config)

    if not phase2["benchmark_pass"]:
        print(
            "\n  WARNING: Control run failed benchmark. "
            "Proceeding to Phase 3 anyway — review distribution parameters."
        )

    # Phase 3 — All three hypotheses
    h1 = run_h1(config)
    h2 = run_h2(config)
    h3 = run_h3(config)

    # Save plots and results
    print("\n" + "=" * 55)
    print("SAVING OUTPUTS")
    print("=" * 55)
    plot_control_run(phase2)
    plot_h1(h1)
    plot_h2(h2)
    plot_h3(h3)
    save_results(phase2, h1, h2, h3)

    # Final summary
    print_summary(phase2, h1, h2, h3)


def main() -> None:
    """Parse arguments and run the full simulation pipeline.

    Examples
    --------
    >>> import sys
    >>> sys.argv = ['main.py']
    """
    args = parse_args()
    config = build_config(fast=args.fast)
    run_all(config)


if __name__ == "__main__":
    main()
