"""
config.py
Central configuration for ATM Cash Replenishment Monte Carlo Simulation.
All parameters live here. No hardcoded values anywhere else.

Authors: Parul Mudaliar, Nandhini Ramesh, Suriya Gopal
Course: IS 597PR, UIUC Spring 2026
"""

from typing import Any

CONFIG: dict[str, Any] = {

    # Cash parameters
    "initial_cash": 1_200_000,
    "refill_amount": 1_200_000,

    # Fixed schedule policy
    "fixed_refill_days": 3,

    # Demand-triggered policy (H2)
    "demand_threshold": 500_000,

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
    "lognormal_mu": 4.952723380906168,
    "lognormal_sigma": 0.82,
    "poisson_lambda_weekday": [4.5693, 2.8429, 2.2211, 3.0776, 4.0936, 11.0512,
                               27.0274, 45.7192, 72.8442, 113.5583, 185.2621,
                               183.8668, 169.674, 173.0584, 179.4576, 181.0533,
                               163.9854, 133.1388, 102.5984, 76.3378, 53.0565,
                               33.6049, 19.036, 11.6036],
    "poisson_lambda_weekend": [12.365, 9.04, 6.7414, 6.8818, 3.6364, 5.1639,
                               12.5, 32.7415, 71.2267, 129.0667, 172.5333,
                               182.0, 163.1074, 140.1611, 123.9933, 106.9524,
                               96.8725, 89.6733, 72.6242, 59.4797, 43.5274,
                               28.3958, 17.7465, 11.781],
    "dow_multipliers": [1.0, 1.1101, 1.1612, 1.2128, 1.5156, 1.1772, 0.7906],

    # Fallback parameters (used if fitted values are None)
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


def get_param(config: dict[str, Any], key: str) -> Any:
    """Return fitted parameter if available, otherwise return fallback.

    Parameters
    ----------
    config : dict[str, Any]
        The CONFIG dictionary.
    key : str
        Parameter key e.g. 'lognormal_mu'.

    Returns
    -------
    Any
        Fitted value if not None, else corresponding fallback value.

    Examples
    --------
    >>> cfg = {**CONFIG, 'lognormal_mu': None}
    >>> get_param(cfg, 'lognormal_mu') == cfg['fallback_lognormal_mu']
    True

    >>> cfg2 = {**CONFIG, 'lognormal_mu': 4.5}
    >>> get_param(cfg2, 'lognormal_mu')
    4.5

    >>> get_param(CONFIG, 'lognormal_sigma')
    0.82

    >>> cfg3 = {**CONFIG, 'dow_multipliers': None}
    >>> get_param(cfg3, 'dow_multipliers') == cfg3['fallback_dow_multipliers']
    True

    >>> cfg4 = {**CONFIG, 'poisson_lambda_weekday': None}
    >>> get_param(cfg4, 'poisson_lambda_weekday') == cfg4['fallback_poisson_weekday']
    True
    """
    value = config.get(key)
    if value is not None:
        return value
    fallback_key = f"fallback_{key}"
    # Map fitted key names to fallback key names
    key_map: dict[str, str] = {
        "fallback_poisson_lambda_weekday": "fallback_poisson_weekday",
        "fallback_poisson_lambda_weekend": "fallback_poisson_weekend",
    }
    fallback_key = key_map.get(fallback_key, fallback_key)
    return config.get(fallback_key)
