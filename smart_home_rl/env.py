"""
Smart-home scheduling environment.

This module implements the Markov Decision Process used in the project. Each
episode represents one simulated day. At each hourly time step, an agent decides
whether to start one flexible appliance or wait.

The environment handles:

- valid action computation,
- hard deadline constraints,
- appliance completion and missed deadlines,
- electricity cost and solar usage accounting,
- reward computation,
- discrete state construction for tabular Q-learning,
- episode-level performance metrics.

The environment is intentionally simple enough for tabular reinforcement
learning while still capturing the main trade-offs between electricity cost,
renewable usage, scheduling delay, and deadline satisfaction.
"""
from __future__ import annotations

import numpy as np

from smart_home_rl.config import Config, DEFAULT_APPLIANCES, ApplianceConfig


class SmartHomeEnv:
    """
    Smart home scheduling environment.

    The environment represents one simulated day. At each hour, the agent can
    either do nothing or start one requested appliance.

    Actions:
        0 = do nothing
        1 = start washing machine
        2 = start dishwasher
        3 = start EV charger

    The environment is intentionally simple enough for tabular Q-learning, while
    still capturing the main scheduling trade-offs:
        - electricity cost,
        - solar energy usage,
        - appliance delay,
        - missed deadlines.

    The state also includes simple forecast-aware features. This is realistic in
    a smart home setting because day-ahead electricity prices and solar forecasts
    are often available.
    """

    def __init__(
        self,
        config: Config,
        appliances: list[ApplianceConfig] | None = None,
    ):
        self.config = config
        self.appliances = appliances if appliances is not None else DEFAULT_APPLIANCES
        self.num_appliances = len(self.appliances)

        self.day_data: dict | None = None
        self.current_hour = 0
        self.done = False

        self.requests: list[dict] = []
        self.request_by_appliance: dict[int, dict] = {}

        self.completed: np.ndarray | None = None
        self.missed: np.ndarray | None = None
        self.start_times: list[int | None] = []

        self.scheduled_load: np.ndarray | None = None

        self.total_cost = 0.0
        self.total_grid_energy = 0.0
        self.total_solar_used = 0.0
        self.total_delay = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0

        self.history: list[dict] = []

    def reset(self, day_data: dict) -> tuple:
        """
        Reset the environment to the beginning of a new simulated day.
        """

        self.day_data = day_data
        self.current_hour = 0
        self.done = False

        self.requests = [dict(request) for request in day_data.get("requests", [])]

        for request in self.requests:
            if "latest_start" not in request:
                request["latest_start"] = request["deadline"] - request["duration"]

        self.request_by_appliance = {
            request["appliance_id"]: request for request in self.requests
        }

        self.completed = np.zeros(self.num_appliances, dtype=bool)
        self.missed = np.zeros(self.num_appliances, dtype=bool)
        self.start_times = [None for _ in range(self.num_appliances)]

        self.scheduled_load = np.zeros(self.config.horizon, dtype=float)

        self.total_cost = 0.0
        self.total_grid_energy = 0.0
        self.total_solar_used = 0.0
        self.total_delay = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0

        self.history = []

        if len(self.requests) == 0:
            self.done = True

        return self._get_state()

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """
        Apply one action and move the environment forward by one hour.
        """

        if self.day_data is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.done:
            return self._get_state(), 0.0, True, {"message": "episode already done"}

        hour = self.current_hour
        valid_actions = self.get_valid_actions()

        reward = 0.0
        info = {
            "hour": hour,
            "action": action,
            "valid_actions": valid_actions,
            "was_valid": action in valid_actions,
            "scheduled_appliance": None,
            "cost": 0.0,
            "grid_energy": 0.0,
            "solar_used": 0.0,
            "delay": 0.0,
            "missed_now": [],
        }

        if action not in valid_actions:
            reward -= self.config.invalid_action_penalty
            self.invalid_actions += 1
        elif action != 0:
            scheduling_info = self._start_appliance(action)
            reward -= self.config.cost_weight * scheduling_info["cost"]
            reward += self.config.solar_bonus_weight * scheduling_info["solar_used"]

            info.update(scheduling_info)

        pending_after_action = self._get_pending_appliance_ids()
        waiting_penalty = self.config.delay_weight * len(pending_after_action)
        reward -= waiting_penalty

        info["pending_after_action"] = pending_after_action
        info["waiting_penalty"] = waiting_penalty

        self.current_hour += 1

        missed_now = self._mark_new_missed_deadlines()
        if missed_now:
            missed_penalty = self.config.missed_deadline_penalty * len(missed_now)
            reward -= missed_penalty
            info["missed_now"] = missed_now
            info["missed_penalty"] = missed_penalty
        else:
            info["missed_penalty"] = 0.0

        self.done = self.current_hour >= self.config.horizon or self._all_requests_resolved()

        self.total_reward += reward
        info["reward"] = reward
        info["done"] = self.done

        self.history.append(info)

        return self._get_state(), reward, self.done, info

    def get_valid_actions(self) -> list[int]:
        """
        Return all actions that are valid in the current state.

        Normally, action 0 means 'do nothing' and is valid.

        However, if at least one appliance is at its latest feasible start time,
        doing nothing is no longer considered valid. This models deadlines as hard
        scheduling constraints: when an appliance must start now to be completed on
        time, the controller should not be allowed to intentionally miss it.
        """

        if self.done:
            return []

        scheduling_actions = []
        forced_actions = []

        for request in self.requests:
            appliance_id = request["appliance_id"]

            if self.completed[appliance_id] or self.missed[appliance_id]:
                continue

            if self.current_hour < request["request_time"]:
                continue

            if self.current_hour > request["latest_start"]:
                continue

            if self.current_hour + request["duration"] > self.config.horizon:
                continue

            action = appliance_id + 1
            scheduling_actions.append(action)

            if self.current_hour == request["latest_start"]:
                forced_actions.append(action)

        # If a task must start now to meet its deadline, only forced actions are valid.
        if forced_actions:
            return forced_actions

        # Otherwise, doing nothing remains valid.
        return [0] + scheduling_actions

    def get_metrics(self) -> dict:
        """
        Return cumulative metrics for the current episode.
        """

        completed_count = int(self.completed.sum())
        missed_count = int(self.missed.sum())
        num_requests = len(self.requests)

        completed_energy = 0.0
        requested_energy = 0.0

        for request in self.requests:
            energy = request["duration"] * request["power"]
            requested_energy += energy

            if self.completed[request["appliance_id"]]:
                completed_energy += energy

        if completed_count > 0:
            average_delay = self.total_delay / completed_count
        else:
            average_delay = 0.0

        if completed_energy > 0:
            renewable_usage_ratio = self.total_solar_used / completed_energy
        else:
            renewable_usage_ratio = 0.0

        return {
            "num_requests": num_requests,
            "completed_requests": completed_count,
            "missed_deadlines": missed_count,
            "total_cost": self.total_cost,
            "total_grid_energy": self.total_grid_energy,
            "total_solar_used": self.total_solar_used,
            "requested_energy": requested_energy,
            "completed_energy": completed_energy,
            "renewable_usage_ratio": renewable_usage_ratio,
            "total_delay": self.total_delay,
            "average_delay": average_delay,
            "total_reward": self.total_reward,
            "invalid_actions": self.invalid_actions,
        }

    def _start_appliance(self, action: int) -> dict:
        """
        Start the appliance corresponding to the selected action.
        """

        appliance_id = action - 1
        request = self.request_by_appliance[appliance_id]

        start_hour = self.current_hour
        end_hour = start_hour + request["duration"]

        power = request["power"]
        prices = self.day_data["prices"][start_hour:end_hour]
        solar = self.day_data.get("solar_actual", self.day_data["solar"])[start_hour:end_hour]

        already_scheduled_load = self.scheduled_load[start_hour:end_hour]

        available_solar = np.maximum(solar - already_scheduled_load, 0.0)
        solar_used_per_hour = np.minimum(power, available_solar)
        grid_energy_per_hour = power - solar_used_per_hour

        cost = float(np.sum(grid_energy_per_hour * prices))
        solar_used = float(np.sum(solar_used_per_hour))
        grid_energy = float(np.sum(grid_energy_per_hour))

        self.scheduled_load[start_hour:end_hour] += power

        self.completed[appliance_id] = True
        self.start_times[appliance_id] = start_hour

        delay = float(start_hour - request["request_time"])

        self.total_cost += cost
        self.total_grid_energy += grid_energy
        self.total_solar_used += solar_used
        self.total_delay += delay

        return {
            "scheduled_appliance": request["name"],
            "start_hour": start_hour,
            "end_hour": end_hour,
            "cost": cost,
            "grid_energy": grid_energy,
            "solar_used": solar_used,
            "delay": delay,
        }

    def _get_state(self) -> tuple:
        """
        Build a compact discrete state for tabular Q-learning.

        State components:
            - current hour,
            - current price bin,
            - current solar bin,
            - for each appliance:
                status,
                urgency,
                cost attractiveness of starting now,
                solar attractiveness of starting now.
        """

        if self.done:
            return ("terminal",)

        hour = min(self.current_hour, self.config.horizon - 1)

        price = float(self.day_data["prices"][hour])
        solar = float(self.day_data.get("solar_actual", self.day_data["solar"])[hour])

        state_parts = [
            hour,
            self._price_bin(price),
            self._solar_bin(solar),
        ]

        for appliance_id in range(self.num_appliances):
            status = self._appliance_status(appliance_id)
            urgency = self._appliance_urgency(appliance_id)
            cost_attractiveness = self._cost_attractiveness(appliance_id)
            solar_attractiveness = self._solar_attractiveness(appliance_id)

            state_parts.append(status)
            state_parts.append(urgency)
            state_parts.append(cost_attractiveness)
            state_parts.append(solar_attractiveness)

        return tuple(state_parts)

    def _appliance_status(self, appliance_id: int) -> int:
        """
        Return the discrete status of one appliance.

        0 = not available / no request
        1 = pending
        2 = completed
        3 = missed
        """

        if self.completed[appliance_id]:
            return 2

        if self.missed[appliance_id]:
            return 3

        request = self.request_by_appliance.get(appliance_id)

        if request is None:
            return 0

        if self.current_hour < request["request_time"]:
            return 0

        return 1

    def _appliance_urgency(self, appliance_id: int) -> int:
        """
        Return a discrete urgency level for a pending appliance.

        0 = not pending
        1 = relaxed
        2 = soon
        3 = urgent / latest feasible start
        """

        request = self.request_by_appliance.get(appliance_id)

        if request is None:
            return 0

        if self.completed[appliance_id] or self.missed[appliance_id]:
            return 0

        if self.current_hour < request["request_time"]:
            return 0

        remaining_start_time = request["latest_start"] - self.current_hour

        if remaining_start_time <= 0:
            return 3

        if remaining_start_time <= 2:
            return 2

        return 1

    def _cost_attractiveness(self, appliance_id: int) -> int:
        """
        Compare the cost of starting now with future feasible start times.

        0 = not pending
        1 = expensive compared with future options
        2 = reasonably close to best future option
        3 = cheapest or latest feasible start
        """

        request = self.request_by_appliance.get(appliance_id)

        if not self._is_pending(appliance_id):
            return 0

        if self.current_hour >= request["latest_start"]:
            return 3

        feasible_starts = range(self.current_hour, request["latest_start"] + 1)

        costs = [
            self._estimate_grid_cost(request, start_hour)
            for start_hour in feasible_starts
        ]

        current_cost = costs[0]
        best_cost = min(costs)

        if best_cost <= 1e-8:
            return 3

        relative_gap = (current_cost - best_cost) / best_cost

        if relative_gap <= 0.05:
            return 3

        if relative_gap <= 0.20:
            return 2

        return 1

    def _solar_attractiveness(self, appliance_id: int) -> int:
        """
        Compare the solar coverage of starting now with future feasible starts.

        0 = not pending
        1 = poor solar opportunity
        2 = moderate solar opportunity
        3 = best or near-best solar opportunity
        """

        request = self.request_by_appliance.get(appliance_id)

        if not self._is_pending(appliance_id):
            return 0

        if self.current_hour >= request["latest_start"]:
            return 3

        feasible_starts = range(self.current_hour, request["latest_start"] + 1)

        ratios = [
            self._estimate_solar_ratio(request, start_hour)
            for start_hour in feasible_starts
        ]

        current_ratio = ratios[0]
        best_ratio = max(ratios)

        if best_ratio <= 1e-8:
            return 1

        if current_ratio >= 0.90 * best_ratio:
            return 3

        if current_ratio >= 0.50 * best_ratio:
            return 2

        return 1

    def _is_pending(self, appliance_id: int) -> bool:
        """
        Check whether an appliance is currently requested and unresolved.
        """

        request = self.request_by_appliance.get(appliance_id)

        if request is None:
            return False

        if self.completed[appliance_id] or self.missed[appliance_id]:
            return False

        if self.current_hour < request["request_time"]:
            return False

        return True

    def _estimate_grid_cost(self, request: dict, start_hour: int) -> float:
        """
        Estimate grid electricity cost if a request starts at start_hour.
        """

        end_hour = start_hour + request["duration"]

        prices = self.day_data["prices"][start_hour:end_hour]
        solar = self.day_data.get("solar_forecast", self.day_data["solar"])[start_hour:end_hour]
        already_scheduled_load = self.scheduled_load[start_hour:end_hour]

        available_solar = np.maximum(solar - already_scheduled_load, 0.0)
        solar_used_per_hour = np.minimum(request["power"], available_solar)
        grid_energy_per_hour = request["power"] - solar_used_per_hour

        return float(np.sum(grid_energy_per_hour * prices))

    def _estimate_solar_ratio(self, request: dict, start_hour: int) -> float:
        """
        Estimate solar energy ratio if a request starts at start_hour.
        """

        end_hour = start_hour + request["duration"]

        solar = self.day_data.get("solar_forecast", self.day_data["solar"])[start_hour:end_hour]
        already_scheduled_load = self.scheduled_load[start_hour:end_hour]

        available_solar = np.maximum(solar - already_scheduled_load, 0.0)
        solar_used_per_hour = np.minimum(request["power"], available_solar)

        solar_used = float(np.sum(solar_used_per_hour))
        total_energy = float(request["duration"] * request["power"])

        if total_energy == 0.0:
            return 0.0

        return solar_used / total_energy

    def _get_pending_appliance_ids(self) -> list[int]:
        """
        Return appliance IDs that are currently requested but not yet resolved.
        """

        pending = []

        for request in self.requests:
            appliance_id = request["appliance_id"]

            if self.completed[appliance_id] or self.missed[appliance_id]:
                continue

            if self.current_hour >= request["request_time"]:
                pending.append(appliance_id)

        return pending

    def _mark_new_missed_deadlines(self) -> list[str]:
        """
        Mark appliances that can no longer be started before their deadline.
        """

        missed_now = []

        for request in self.requests:
            appliance_id = request["appliance_id"]

            if self.completed[appliance_id] or self.missed[appliance_id]:
                continue

            if self.current_hour > request["latest_start"]:
                self.missed[appliance_id] = True
                missed_now.append(request["name"])

        return missed_now

    def _all_requests_resolved(self) -> bool:
        """
        Check whether every generated request is either completed or missed.
        """

        for request in self.requests:
            appliance_id = request["appliance_id"]

            if not self.completed[appliance_id] and not self.missed[appliance_id]:
                return False

        return True

    @staticmethod
    def _price_bin(price: float) -> int:
        """
        Discretize electricity price into low, medium or high.
        """

        if price < 0.20:
            return 0

        if price < 0.30:
            return 1

        return 2

    @staticmethod
    def _solar_bin(solar: float) -> int:
        """
        Discretize solar production into low, medium or high.
        """

        if solar < 0.5:
            return 0

        if solar < 1.5:
            return 1

        return 2