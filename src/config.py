"""
config.py
Central configuration for ATM Cash Replenishment Monte Carlo Simulation.
All parameters live here. No hardcoded values anywhere else.

AI Disclosure: Claude (Anthropic) assisted in drafting this codebase.
All modeling decisions reviewed by the team.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
"""

CONFIG: dict = {

    # --- Cash parameters ---
    "initial_cash": 1_200_000,
    "refill_amount": 1_200_000,

    # --- Fixed schedule policy (Phase 2 control + H1 + H3) ---
    "fixed_refill_days": 3,

    # --- Demand-triggered policy (H2) ---
    "demand_threshold": 500_000,

    # --- Weibull truck delay ---
    # Right-skewed: most trucks arrive within 1 day, occasional delays
    # Based on logistics literature (Geismar et al. 2017)
    "weibull_shape": 1.5,
    "weibull_scale": 0.5,

    # --- Simulation settings ---
    "sim_days": 365,
    "num_runs": 10_000,

    # --- Phase 2 benchmark ---
    # Plausible real-world ATM stockout range from literature
    "benchmark_min": 0.05,
    "benchmark_max": 0.20,

    # --- Distribution parameters ---
    # Set to None until fit_distributions.py is run on the Danish dataset.
    # After running, paste printed values here to replace None.
"lognormal_mu": 4.952723380906168,
"lognormal_sigma": 0.82,
"poisson_lambda_weekday": [4.5693, 2.8429, 2.2211, 3.0776, 4.0936, 11.0512, 27.0274, 45.7192, 72.8442, 113.5583, 185.2621, 183.8668, 169.674, 173.0584, 179.4576, 181.0533, 163.9854, 133.1388, 102.5984, 76.3378, 53.0565, 33.6049, 19.036, 11.6036],
"poisson_lambda_weekend": [12.365, 9.04, 6.7414, 6.8818, 3.6364, 5.1639, 12.5, 32.7415, 71.2267, 129.0667, 172.5333, 182.0, 163.1074, 140.1611, 123.9933, 106.9524, 96.8725, 89.6733, 72.6242, 59.4797, 43.5274, 28.3958, 17.7465, 11.781],
"dow_multipliers": [1.0, 1.1101, 1.1612, 1.2128, 1.5156, 1.1772, 0.7906],

    # --- Fallback parameters ---
    # Used automatically if distribution parameters above are None.
    # These are reasonable estimates based on Fed Reserve data and
    # logistics literature. Replace with fitted values after running
    # fit_distributions.py.
    "fallback_lognormal_mu": 4.61,
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