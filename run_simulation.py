"""
run_simulation.py
Main entry point. Runs all phases in order:

    Phase 1 — Fit distributions from Danish ATM dataset
    Phase 2 — Control run validation
    Phase 3 — H1, H2, H3 experiments
    Analysis — Plots and results JSON

Usage:
    python run_simulation.py

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

from pathlib import Path
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

DATA_PATH = Path("data/danish_atm.csv")


def main() -> None:
    """Run all simulation phases end to end.

    Examples
    --------
    >>> import os
    >>> os.path.exists('run_simulation.py')
    True
    """
    print("ATM Cash Replenishment — Monte Carlo Simulation")
    print("Team: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal")
    print("=" * 55)

    # --------------------------------------------------
    # Phase 1 — Fit distributions
    # --------------------------------------------------
    print("\nPHASE 1 — Fitting distributions")

    if DATA_PATH.exists():
        print(f"  Dataset found at {DATA_PATH}. Fitting from real data.")
        from src.fit_distributions import (
            load_data,
            fit_lognormal,
            fit_poisson,
            fit_dow_multipliers,
        )
        df = load_data(DATA_PATH)
        mu, sigma = fit_lognormal()
        poisson = fit_poisson(df)
        dow = fit_dow_multipliers(df)

        CONFIG["lognormal_mu"] = mu
        CONFIG["lognormal_sigma"] = sigma
        CONFIG["poisson_lambda_weekday"] = poisson["weekday"]
        CONFIG["poisson_lambda_weekend"] = poisson["weekend"]
        CONFIG["dow_multipliers"] = dow

        print("\n  Phase 1 complete. Real parameters loaded.")
    else:
        print(f"  Dataset not found at {DATA_PATH}.")
        print("  Using fallback parameters.")
        print("  Download from: kaggle.com/datasets/sparnord/danish-atm-transactions")
        print("  Save as data/danish_atm.csv and re-run for real parameters.")

    # --------------------------------------------------
    # Phase 2 — Control run
    # --------------------------------------------------
    phase2 = run_control(CONFIG)

    if not phase2["benchmark_pass"]:
        print("\n  Benchmark failed. Continuing anyway for demonstration.")

    # --------------------------------------------------
    # Phase 3 — Hypothesis experiments
    # --------------------------------------------------
    h1 = run_h1(CONFIG)
    h2 = run_h2(CONFIG)
    h3 = run_h3(CONFIG)

    # --------------------------------------------------
    # Analysis — Plots and results
    # --------------------------------------------------
    print("\n" + "=" * 55)
    print("Generating plots and saving results...")

    plot_control_run(phase2)
    plot_h1(h1)
    plot_h2(h2)
    plot_h3(h3)
    save_results(phase2, h1, h2, h3)
    print_summary(phase2, h1, h2, h3)


if __name__ == "__main__":
    main()