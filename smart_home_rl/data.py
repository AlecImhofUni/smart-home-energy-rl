"""
Synthetic data generation for the smart-home scheduling environment.

This module creates structured daily input data for the project:

- electricity price profiles,
- solar production profiles,
- appliance request times and deadlines,
- summary tables for generated days and requests.

The generated data is stochastic but not purely random. Prices follow daily
time-of-use patterns, solar production follows weather-dependent daylight
curves, and appliance requests follow appliance-specific usage windows. This
makes the environment suitable for evaluating scheduling policies under
controlled uncertainty.
"""
import numpy as np
import pandas as pd

from smart_home_rl.config import ApplianceConfig, Config, DEFAULT_APPLIANCES


def get_day_type(day_index: int) -> str:
    """
    Return whether a simulated day is a weekday or weekend.

    We assume day_index = 0 is Monday.
    """

    day_of_week = day_index % 7

    if day_of_week in (5, 6):
        return "weekend"

    return "weekday"


def sample_weather_type(rng: np.random.Generator) -> str:
    """
    Sample a simple weather category for solar generation.

    The goal is not to model meteorology precisely, but to create structured
    variation between sunny, mixed and cloudy days.
    """

    return str(rng.choice(["sunny", "mixed", "cloudy"], p=[0.35, 0.45, 0.20]))


def generate_price_profile(
    config: Config,
    rng: np.random.Generator,
    day_type: str = "weekday",
) -> np.ndarray:
    """
    Generate a structured daily electricity price profile.

    The profile is not purely random:
    - prices are lower during the night,
    - prices increase during morning and evening peaks,
    - weekend peaks are slightly smaller,
    - daily noise creates variation across simulated days.
    """

    hours = np.arange(config.horizon)

    night_discount = -0.06 * ((hours <= 6) | (hours >= 23))

    morning_peak = 0.08 * np.exp(-0.5 * ((hours - 8) / 2.0) ** 2)
    evening_peak = 0.12 * np.exp(-0.5 * ((hours - 19) / 3.0) ** 2)

    if day_type == "weekend":
        peak_factor = 0.75
    else:
        peak_factor = 1.0

    daily_shift = rng.normal(0.0, config.price_volatility * 0.4)
    hourly_noise = rng.normal(0.0, config.price_volatility, size=config.horizon)

    prices = (
        config.price_base
        + daily_shift
        + night_discount
        + peak_factor * (morning_peak + evening_peak)
        + hourly_noise
    )

    prices = np.clip(prices, 0.05, None)

    return prices


def generate_solar_profile(
    config: Config,
    rng: np.random.Generator,
    weather_type: str = "mixed",
) -> np.ndarray:
    """
    Generate a structured daily solar production profile.

    Solar production follows a bell-shaped curve:
    - zero at night,
    - peak around midday,
    - scaled by a sampled weather type.
    """

    hours = np.arange(config.horizon)

    clear_sky_curve = config.solar_peak * np.exp(-0.5 * ((hours - 13) / 3.2) ** 2)
    clear_sky_curve[(hours < 6) | (hours > 21)] = 0.0

    if weather_type == "sunny":
        weather_factor = rng.uniform(0.85, 1.15)
    elif weather_type == "mixed":
        weather_factor = rng.uniform(0.45, 0.85)
    elif weather_type == "cloudy":
        weather_factor = rng.uniform(0.15, 0.45)
    else:
        raise ValueError(f"Unknown weather type: {weather_type}")

    noise = rng.normal(0.0, config.solar_noise, size=config.horizon)

    solar = weather_factor * clear_sky_curve + noise
    solar = np.clip(solar, 0.0, None)

    return solar

def generate_solar_forecast(
    config: Config,
    rng: np.random.Generator,
    solar_actual: np.ndarray,
) -> np.ndarray:
    """
    Generate a noisy solar production forecast.

    The actual solar profile represents what really happens during the day.
    The forecast represents what the controller expects before making decisions.

    This models exogenous uncertainty such as unexpected cloud cover.
    """

    relative_error = rng.normal(
        0.0,
        config.solar_forecast_relative_error,
        size=config.horizon,
    )

    absolute_error = rng.normal(
        0.0,
        config.solar_forecast_absolute_error,
        size=config.horizon,
    )

    solar_forecast = solar_actual * (1.0 + relative_error) + absolute_error
    solar_forecast = np.clip(solar_forecast, 0.0, None)

    return solar_forecast

def _weighted_hour_sample(
    rng: np.random.Generator,
    possible_hours: np.ndarray,
    centers: list[float],
    center_weights: list[float],
    spread: float,
) -> int:
    """
    Sample an hour using a smooth preference over typical request times.
    """

    raw_weights = np.zeros_like(possible_hours, dtype=float)

    for center, weight in zip(centers, center_weights):
        raw_weights += weight * np.exp(-0.5 * ((possible_hours - center) / spread) ** 2)

    probabilities = raw_weights / raw_weights.sum()

    return int(rng.choice(possible_hours, p=probabilities))


def _sample_request_time(
    appliance: ApplianceConfig,
    config: Config,
    rng: np.random.Generator,
    day_type: str,
) -> int:
    """
    Sample a realistic request time depending on the appliance and day type.
    """

    latest_feasible_request = min(
        appliance.latest_request,
        config.horizon - appliance.duration,
    )

    if latest_feasible_request < appliance.earliest_request:
        raise ValueError(
            f"Appliance {appliance.name} has no feasible request window."
        )

    possible_hours = np.arange(appliance.earliest_request, latest_feasible_request + 1)

    if appliance.name == "washing_machine":
        if day_type == "weekend":
            centers = [11.0, 15.0]
            weights = [0.65, 0.35]
        else:
            centers = [8.0, 18.0]
            weights = [0.45, 0.55]

        return _weighted_hour_sample(rng, possible_hours, centers, weights, spread=2.0)

    if appliance.name == "dishwasher":
        centers = [20.5]
        weights = [1.0]

        return _weighted_hour_sample(rng, possible_hours, centers, weights, spread=1.5)

    if appliance.name == "ev_charger":
        centers = [18.5]
        weights = [1.0]

        return _weighted_hour_sample(rng, possible_hours, centers, weights, spread=1.5)

    return int(rng.choice(possible_hours))


def generate_appliance_requests(
    config: Config,
    rng: np.random.Generator,
    appliances: list[ApplianceConfig] | None = None,
    day_type: str = "weekday",
) -> list[dict]:
    """
    Generate appliance requests for one simulated day.

    Demand is structured:
    - each appliance has a weekday/weekend request probability,
    - each appliance has a realistic request window,
    - request times are biased toward plausible hours,
    - deadlines are derived from request times and max_delay.
    """

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    requests = []

    for appliance_id, appliance in enumerate(appliances):
        if day_type == "weekend":
            request_probability = appliance.request_probability_weekend
        else:
            request_probability = appliance.request_probability_weekday

        if rng.random() > request_probability:
            continue

        request_time = _sample_request_time(appliance, config, rng, day_type)

        deadline = min(config.horizon, request_time + appliance.max_delay)
        latest_start = deadline - appliance.duration

        if latest_start < request_time:
            raise ValueError(
                f"Infeasible request generated for {appliance.name}: "
                f"request={request_time}, deadline={deadline}, "
                f"duration={appliance.duration}"
            )

        requests.append(
            {
                "appliance_id": appliance_id,
                "name": appliance.name,
                "request_time": request_time,
                "deadline": deadline,
                "latest_start": latest_start,
                "duration": appliance.duration,
                "power": appliance.power,
            }
        )

    return requests


def generate_day(
    config: Config,
    rng: np.random.Generator,
    appliances: list[ApplianceConfig] | None = None,
    day_index: int = 0,
) -> dict:
    """
    Generate all exogenous data for one simulated day.

    The returned dictionary contains both:
    - solar_actual: the realized solar production used for real cost computation,
    - solar_forecast: the noisy forecast used by agents for planning.

    The key 'solar' is kept as an alias for solar_actual for backward compatibility.
    """

    day_type = get_day_type(day_index)
    weather_type = sample_weather_type(rng)

    prices = generate_price_profile(config, rng, day_type)
    solar_actual = generate_solar_profile(config, rng, weather_type)
    solar_forecast = generate_solar_forecast(config, rng, solar_actual)
    requests = generate_appliance_requests(config, rng, appliances, day_type)

    return {
        "day_index": day_index,
        "day_type": day_type,
        "weather_type": weather_type,
        "prices": prices,
        "solar": solar_actual,
        "solar_actual": solar_actual,
        "solar_forecast": solar_forecast,
        "requests": requests,
    }


def generate_days(
    config: Config,
    num_days: int,
    seed: int | None = None,
    appliances: list[ApplianceConfig] | None = None,
) -> list[dict]:
    """
    Generate several simulated days using one random generator.
    """

    if seed is None:
        seed = config.seed

    rng = np.random.default_rng(seed)

    return [
        generate_day(config, rng, appliances, day_index=day_index)
        for day_index in range(num_days)
    ]


def build_day_summary(days: list[dict]) -> pd.DataFrame:
    """
    Build a compact summary table of generated daily profiles.
    """

    records = []

    for day in days:
        requested_energy = sum(
            request["duration"] * request["power"]
            for request in day["requests"]
        )

        solar_actual = day.get("solar_actual", day["solar"])
        solar_forecast = day.get("solar_forecast", solar_actual)
        mean_abs_solar_forecast_error = float(
            np.mean(np.abs(solar_forecast - solar_actual))
        )

        records.append(
            {
                "day_index": day["day_index"],
                "day_type": day["day_type"],
                "weather_type": day["weather_type"],
                "mean_price": float(np.mean(day["prices"])),
                "max_price": float(np.max(day["prices"])),
                "total_solar": float(np.sum(day["solar"])),
                "max_solar": float(np.max(day["solar"])),
                "num_requests": len(day["requests"]),
                "requested_energy_kwh": requested_energy,
                "mean_abs_solar_forecast_error": mean_abs_solar_forecast_error,
            }
        )

    return pd.DataFrame.from_records(records)


def build_request_summary(days: list[dict]) -> pd.DataFrame:
    """
    Build a table containing all appliance requests across generated days.
    """

    records = []

    for day in days:
        for request in day["requests"]:
            records.append(
                {
                    "day_index": day["day_index"],
                    "day_type": day["day_type"],
                    "weather_type": day["weather_type"],
                    "appliance_id": request["appliance_id"],
                    "name": request["name"],
                    "request_time": request["request_time"],
                    "deadline": request["deadline"],
                    "latest_start": request["latest_start"],
                    "duration": request["duration"],
                    "power": request["power"],
                }
            )

    return pd.DataFrame.from_records(records)