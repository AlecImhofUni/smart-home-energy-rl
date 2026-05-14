"""
Configuration objects and default project parameters.

This module centralizes the static settings used throughout the smart-home
energy scheduling project. It defines:

- the appliance configuration dataclass,
- the global experiment configuration dataclass,
- the default set of flexible household appliances.

Keeping these values in one file makes the project easier to tune and avoids
hard-coding parameters across the environment, agents, and experiments.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApplianceConfig:
    """
    Static description of one flexible appliance.

    duration:
        Number of consecutive hours needed to complete the appliance task.

    power:
        Energy consumption per hour in kWh.

    earliest_request / latest_request:
        Typical time window during which the appliance request may appear.

    max_delay:
        Maximum number of hours allowed between request time and deadline.

    request_probability_weekday / request_probability_weekend:
        Probability that this appliance is requested on a given weekday/weekend.
        This avoids generating completely random demand from day to day.
    """

    name: str
    duration: int
    power: float
    earliest_request: int
    latest_request: int
    max_delay: int
    request_probability_weekday: float
    request_probability_weekend: float


@dataclass(frozen=True)
class Config:
    """
    Global configuration for the smart home scheduling project.
    """

    # General simulation setup
    horizon: int = 24
    seed: int = 42

    # Data generation parameters
    price_base: float = 0.25
    price_volatility: float = 0.08
    solar_peak: float = 3.0
    solar_noise: float = 0.15

    # Solar forecast uncertainty parameters
    solar_forecast_relative_error: float = 0.25
    solar_forecast_absolute_error: float = 0.10

    # Reward parameters
    cost_weight: float = 1.0
    delay_weight: float = 0.02
    missed_deadline_penalty: float = 5.0
    solar_bonus_weight: float = 0.10
    invalid_action_penalty: float = 2.0

    # Q-learning parameters
    episodes: int = 6_000
    ablation_episodes: int = 3_000
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999
    q_initial_value: float = -5.0

    # Evaluation parameters
    eval_days: int = 200

    # Paths
    results_dir: Path = Path("results")


DEFAULT_APPLIANCES = [
    ApplianceConfig(
        name="washing_machine",
        duration=2,
        power=1.0,
        earliest_request=7,
        latest_request=18,
        max_delay=8,
        request_probability_weekday=0.65,
        request_probability_weekend=0.85,
    ),
    ApplianceConfig(
        name="dishwasher",
        duration=2,
        power=1.2,
        earliest_request=18,
        latest_request=22,
        max_delay=6,
        request_probability_weekday=0.90,
        request_probability_weekend=0.90,
    ),
    ApplianceConfig(
        name="ev_charger",
        duration=4,
        power=2.0,
        earliest_request=17,
        latest_request=20,
        max_delay=7,
        request_probability_weekday=0.75,
        request_probability_weekend=0.60,
    ),
]