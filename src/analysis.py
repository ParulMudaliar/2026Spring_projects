"""
Generates plots and saves results for all phases.

Saves all plots to outputs/ folder.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_control_run(phase2: dict) -> None:
    """Plot histogram of stockout rates from Phase 2 control run.

    Parameters
    ----------
    phase2 : dict
        Output from phase2_control.run_control().

    Examples
    --------
    >>> plot_control_run({'all_rates': [0.1, 0.12, 0.09],
    ...                   'mean_stockout_rate': 0.103})
    """
    rates = phase2["all_rates"]
    mean = phase2["mean_stockout_rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rates, bins=50, color="#534AB7", alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(mean, color="#E8593C", linewidth=2,
               label=f"Mean = {mean:.2%}")
    ax.axvline(0.05, color="gray", linestyle="--",
               linewidth=1, label="Benchmark lower (5%)")
    ax.axvline(0.20, color="gray", linestyle=":",
               linewidth=1, label="Benchmark upper (20%)")
    ax.set_xlabel("Annual stockout rate")
    ax.set_ylabel("Number of runs")
    ax.set_title("Phase 2 — Control run: stockout rate distribution\n"
                 "(fixed 3-day schedule, 10,000 runs)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.tight_layout()
    fig.savefig(OUTPUTS / "phase2_control.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: outputs/phase2_control.png")