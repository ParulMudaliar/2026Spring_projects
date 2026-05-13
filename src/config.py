"""
config.py
Central configuration for ATM Cash Replenishment Monte Carlo Simulation.
All parameters live here. No hardcoded values anywhere else.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

CONFIG: dict = {

    # --- Cash parameters (currency: DKK — Danish Krone) ---
    # Source: ATMIA European ATM industry standards.
    # A Danish bank-branch ATM typically holds DKK 400,000–1,200,000 loaded.
    # DKK 800,000 ≈ €107,000 is a realistic mid-range operational load.
    "initial_cash": 880_000,    # DKK
    "refill_amount": 880_000,   # DKK

    # --- Fixed schedule policy (Phase 2 control + H1 + H3) ---
    "fixed_refill_days": 3,

    # Demand-triggered policy (H2)
    "demand_threshold": 300_000,    # DKK

    # Weibull truck delay
    "weibull_shape": 1.5,
    "weibull_scale": 0.5,

    # Simulation settings
    "sim_days": 365,
    "num_runs": 10_000,

    # Phase 2 benchmark

    "benchmark_min": 0.05,
    "benchmark_max": 0.20,

    # Distribution parameters (fitted from Danish dataset)
    "lognormal_mu": 6.6669,     # DKK
    "lognormal_sigma": 0.82,
    "poisson_lambda_weekday": [0.5849, 0.3639, 0.2843, 0.3939, 0.5240, 1.4146,
                                3.4595, 5.8521, 9.3241, 14.5355, 23.7136, 23.5349,
                                21.7183, 22.1515, 22.9706, 23.1748, 20.9901, 17.0418,
                                13.1326, 9.7712, 6.7912, 4.3014, 2.4366, 1.4852],
    "poisson_lambda_weekend": [1.5827, 1.1571, 0.8629, 0.8809, 0.4655, 0.6610,
                                1.6000, 4.1909, 9.1170, 16.5205, 22.0843, 23.2960,
                                20.8778, 17.9406, 15.8711, 13.6899, 12.3997, 11.4782,
                                9.2959, 7.6134, 5.5715, 3.6347, 2.2715, 1.5080],
    "dow_multipliers": [1.0, 1.1101, 1.1612, 1.2128, 1.5156, 1.1772, 0.7906],

    # Fallback parameters
    # Used automatically if distribution parameters above are None.
    # These are reasonable estimates based on Fed Reserve data and
    # logistics literature. Replace with fitted values after running
    # fit_distributions.py.
    "fallback_lognormal_mu": 6.57,   # DKK;
    "fallback_lognormal_sigma": 0.82,
    "fallback_poisson_weekday": [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.5, 1.2, 2.1, 2.8, 3.0, 2.9,
        2.7, 2.5, 2.6, 2.8, 3.1, 3.0,
        2.4, 1.8, 1.2, 0.8, 0.4, 0.2,
    ],
    "fallback_poisson_weekend": [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.4, 1.0, 2.0, 3.2, 3.8, 3.9,
        3.7, 3.5, 3.3, 3.4, 3.6, 3.4,
        2.8, 2.1, 1.4, 0.9, 0.5, 0.2,
    ],
    "fallback_dow_multipliers": [
        1.0, 1.0, 1.02, 1.05, 1.18, 1.55, 1.42
    ],
}


def get_param(config: dict, key: str) -> object:
    """Return fitted parameter if available, otherwise return fallback.

    Parameters
    ----------
    config : dict
        The CONFIG dictionary.
    key : str
        Parameter key e.g. 'lognormal_mu'.

    Returns
    -------
    object
        Fitted value if not None, else corresponding fallback value.

    Examples
    --------
    >>> cfg = {**CONFIG, 'lognormal_mu': None}
    >>> get_param(cfg, 'lognormal_mu') == cfg['fallback_lognormal_mu']
    True
    >>> cfg2 = {**CONFIG, 'lognormal_mu': 4.5}
    >>> get_param(cfg2, 'lognormal_mu')
    4.5
    """
    value = config.get(key)
    if value is not None:
        return value
    fallback_key = f"fallback_{key.replace('poisson_lambda_weekday', 'poisson_weekday').replace('poisson_lambda_weekend', 'poisson_weekend').replace('dow_multipliers', 'dow_multipliers')}"
    fallback_map = {
        "fallback_lognormal_mu":           config["fallback_lognormal_mu"],
        "fallback_lognormal_sigma":        config["fallback_lognormal_sigma"],
        "fallback_poisson_lambda_weekday": config["fallback_poisson_weekday"],
        "fallback_poisson_lambda_weekend": config["fallback_poisson_weekend"],
        "fallback_dow_multipliers":        config["fallback_dow_multipliers"],
    }
    return fallback_map.get(f"fallback_{key}", None)