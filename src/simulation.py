"""
simulation.py
Phase 1 — Single ATM simulation core loop.

This is the shared engine used by Phase 2 control run and all
three hypothesis experiments. Every other file imports this.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""
import numpy as np
from src.config import CONFIG, get_param


def simulate_one_year(config: dict, policy: str = "fixed", rng: np.random.Generator | None = None,) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    cash = float(config["initial_cash"])
    sim_days = config["sim_days"]

    # Tracking
    stockout_days = 0

    for day in range(sim_days):
        dow = day % 7  # 0=Mon, 6=Sun

        for hour in range(24):
            pass

    return {
        "stockout_days": stockout_days,
    }