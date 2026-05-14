"""
Scheduling agents used in the smart-home energy environment.

This module contains all policies compared in the project:

- a base agent interface,
- heuristic baseline agents,
- a tabular Q-learning agent.

The heuristic baselines are used as interpretable reference policies:
run immediately, cheapest feasible hour, and solar-greedy scheduling. The
Q-learning agent is the main reinforcement learning method. It learns a table
of state-action values and selects actions using epsilon-greedy exploration
during training and greedy action selection during evaluation.
"""
from __future__ import annotations

import numpy as np


class BaseAgent:
    """
    Base interface for all scheduling agents.

    Each agent receives the current environment and chooses one valid action.

    Actions:
        0 = do nothing
        1 = start washing machine
        2 = start dishwasher
        3 = start EV charger
    """

    name = "base_agent"

    def reset(self) -> None:
        """
        Reset internal agent state before a new episode.

        Stateless agents do not need to do anything here.
        """

        pass

    def select_action(self, env, state=None, valid_actions: list[int] | None = None) -> int:
        """
        Select an action from the current environment state.
        """

        raise NotImplementedError

    @staticmethod
    def _get_valid_actions(env, valid_actions: list[int] | None = None) -> list[int]:
        """
        Return valid actions, using the environment if no list is provided.
        """

        if valid_actions is None:
            return env.get_valid_actions()

        return valid_actions

    @staticmethod
    def _valid_scheduling_actions(valid_actions: list[int]) -> list[int]:
        """
        Return all valid actions except action 0, which means 'do nothing'.
        """

        return [action for action in valid_actions if action != 0]


class RunImmediatelyAgent(BaseAgent):
    """
    Baseline that starts requested appliances as soon as possible.

    If several appliances can be started at the same hour, it prioritizes the
    appliance with the earliest deadline.
    """

    name = "run_immediately"

    def select_action(self, env, state=None, valid_actions: list[int] | None = None) -> int:
        valid_actions = self._get_valid_actions(env, valid_actions)
        scheduling_actions = self._valid_scheduling_actions(valid_actions)

        if not scheduling_actions:
            return 0

        def deadline_for_action(action: int) -> int:
            appliance_id = action - 1
            return env.request_by_appliance[appliance_id]["deadline"]

        return min(scheduling_actions, key=deadline_for_action)


class CheapestHourAgent(BaseAgent):
    """
    Heuristic baseline that schedules appliances at the cheapest feasible hour.

    For each pending appliance, the agent estimates the grid electricity cost of
    starting it now and at each future feasible start time before its deadline.

    The appliance is started now if:
        - the current hour is its cheapest remaining feasible start time, or
        - the current hour is its latest feasible start time.

    This is a strong planning heuristic because it uses the full generated daily
    price and solar profiles.
    """

    name = "cheapest_hour"

    def select_action(self, env, state=None, valid_actions: list[int] | None = None) -> int:
        valid_actions = self._get_valid_actions(env, valid_actions)
        scheduling_actions = self._valid_scheduling_actions(valid_actions)

        if not scheduling_actions:
            return 0

        candidate_actions = []

        for action in scheduling_actions:
            appliance_id = action - 1
            request = env.request_by_appliance[appliance_id]

            feasible_starts = range(env.current_hour, request["latest_start"] + 1)

            start_costs = {
                start_hour: self._estimate_grid_cost(env, request, start_hour)
                for start_hour in feasible_starts
            }

            best_start = min(start_costs, key=start_costs.get)
            current_cost = start_costs[env.current_hour]

            is_cheapest_now = env.current_hour == best_start
            is_forced_now = env.current_hour == request["latest_start"]

            if is_cheapest_now or is_forced_now:
                candidate_actions.append(
                    {
                        "action": action,
                        "current_cost": current_cost,
                        "deadline": request["deadline"],
                        "forced": is_forced_now,
                    }
                )

        if not candidate_actions:
            return 0

        # Forced actions are prioritized to avoid missed deadlines.
        candidate_actions.sort(
            key=lambda item: (
                not item["forced"],
                item["current_cost"],
                item["deadline"],
            )
        )

        return candidate_actions[0]["action"]

    @staticmethod
    def _estimate_grid_cost(env, request: dict, start_hour: int) -> float:
        """
        Estimate the electricity cost of starting a request at a given hour.

        The cost accounts for already scheduled load and available solar energy.
        """

        end_hour = start_hour + request["duration"]

        prices = env.day_data["prices"][start_hour:end_hour]
        solar = env.day_data["solar"][start_hour:end_hour]
        already_scheduled_load = env.scheduled_load[start_hour:end_hour]

        available_solar = np.maximum(solar - already_scheduled_load, 0.0)
        solar_used_per_hour = np.minimum(request["power"], available_solar)
        grid_energy_per_hour = request["power"] - solar_used_per_hour

        return float(np.sum(grid_energy_per_hour * prices))


class SolarGreedyAgent(BaseAgent):
    """
    Heuristic baseline that prioritizes renewable energy usage.

    For each pending appliance, the agent estimates how much of its energy could
    be covered by solar production if started now.

    The appliance is started now if:
        - this is its latest feasible start time, or
        - the current solar coverage is good compared with the future feasible
          opportunities.

    This baseline represents a renewable-aware scheduling rule.
    """

    name = "solar_greedy"

    def __init__(self, solar_ratio_threshold: float = 0.35):
        self.solar_ratio_threshold = solar_ratio_threshold

    def select_action(self, env, state=None, valid_actions: list[int] | None = None) -> int:
        valid_actions = self._get_valid_actions(env, valid_actions)
        scheduling_actions = self._valid_scheduling_actions(valid_actions)

        if not scheduling_actions:
            return 0

        candidate_actions = []

        for action in scheduling_actions:
            appliance_id = action - 1
            request = env.request_by_appliance[appliance_id]

            feasible_starts = range(env.current_hour, request["latest_start"] + 1)

            solar_ratios = {
                start_hour: self._estimate_solar_ratio(env, request, start_hour)
                for start_hour in feasible_starts
            }

            current_ratio = solar_ratios[env.current_hour]
            best_future_ratio = max(solar_ratios.values())

            is_forced_now = env.current_hour == request["latest_start"]
            is_good_solar_now = (
                current_ratio >= self.solar_ratio_threshold
                and current_ratio >= 0.9 * best_future_ratio
            )

            if is_forced_now or is_good_solar_now:
                candidate_actions.append(
                    {
                        "action": action,
                        "solar_ratio": current_ratio,
                        "deadline": request["deadline"],
                        "forced": is_forced_now,
                    }
                )

        if not candidate_actions:
            return 0

        # Forced actions are prioritized, then the highest solar ratio.
        candidate_actions.sort(
            key=lambda item: (
                not item["forced"],
                -item["solar_ratio"],
                item["deadline"],
            )
        )

        return candidate_actions[0]["action"]

    @staticmethod
    def _estimate_solar_ratio(env, request: dict, start_hour: int) -> float:
        """
        Estimate the fraction of the appliance energy covered by solar energy.
        """

        end_hour = start_hour + request["duration"]

        solar = env.day_data["solar"][start_hour:end_hour]
        already_scheduled_load = env.scheduled_load[start_hour:end_hour]

        available_solar = np.maximum(solar - already_scheduled_load, 0.0)
        solar_used_per_hour = np.minimum(request["power"], available_solar)

        solar_used = float(np.sum(solar_used_per_hour))
        total_energy = float(request["duration"] * request["power"])

        if total_energy == 0.0:
            return 0.0

        return solar_used / total_energy


class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent.

    The agent learns a Q-value for each discrete state-action pair:

        Q(state, action)

    During training, it uses epsilon-greedy exploration.
    During evaluation, it acts greedily using the learned Q-table.

    The agent only chooses among valid actions, so it should not intentionally
    trigger invalid actions during normal training or evaluation.
    """

    name = "q_learning"

    def __init__(
        self,
        num_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        seed: int = 42,
        initial_q_value: float = 0.0,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple, np.ndarray] = {}
        self.initial_q_value = initial_q_value

        self.training = True

    def reset(self) -> None:
        """
        Reset per-episode state.

        The Q-table is intentionally not reset because it stores what the agent
        has learned across training episodes.
        """

        pass

    def set_training_mode(self) -> None:
        """
        Enable epsilon-greedy exploration.
        """

        self.training = True

    def set_eval_mode(self) -> None:
        """
        Disable exploration and use greedy actions.
        """

        self.training = False

    def select_action(self, env, state=None, valid_actions: list[int] | None = None) -> int:
        valid_actions = self._get_valid_actions(env, valid_actions)

        if not valid_actions:
            return 0

        if state is None:
            state = env._get_state()

        q_values = self._get_q_values(state)

        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        return self._best_valid_action(q_values, valid_actions)

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
        next_valid_actions: list[int],
    ) -> None:
        """
        Apply the Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """

        q_values = self._get_q_values(state)
        current_q = q_values[action]

        if done or not next_valid_actions:
            target = reward
        else:
            next_q_values = self._get_q_values(next_state)
            best_next_q = max(next_q_values[action] for action in next_valid_actions)
            target = reward + self.gamma * best_next_q

        q_values[action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _get_q_values(self, state: tuple) -> np.ndarray:
        """
        Return the Q-values for a state, creating them if necessary.

        Since most rewards in this environment are negative, Q-values are initialized
        with a pessimistic value. This avoids making unseen actions look better only
        because their value is still zero.
        """

        if state not in self.q_table:
            self.q_table[state] = np.full(
                self.num_actions,
                self.initial_q_value,
                dtype=float,
            )

        return self.q_table[state]

    def _best_valid_action(self, q_values: np.ndarray, valid_actions: list[int]) -> int:
        """
        Return a greedy action among valid actions.

        If several actions have the same Q-value, choose randomly among them to
        avoid always selecting the smallest action index.
        """

        valid_q_values = np.array([q_values[action] for action in valid_actions])
        max_q = np.max(valid_q_values)

        best_actions = [
            action
            for action, q_value in zip(valid_actions, valid_q_values)
            if q_value == max_q
        ]

        return int(self.rng.choice(best_actions))