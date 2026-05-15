"""
Experiment, evaluation, plotting, ablation, and robustness utilities.

This module contains the project-level functions used to train and evaluate
agents. It provides utilities for:

- plotting generated daily data,
- running one episode or evaluating agents over many days,
- training the tabular Q-learning agent,
- building comparison tables against baselines,
- generating report-ready plots,
- running reward ablation studies,
- running robustness studies under modified environment conditions.

The functions in this module are called by main.py to produce the CSV files and
figures used for analysis and reporting.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smart_home_rl.agents import (
    CheapestHourAgent,
    QLearningAgent,
    RunImmediatelyAgent,
    SolarGreedyAgent,
)
from smart_home_rl.config import ApplianceConfig, Config, DEFAULT_APPLIANCES
from smart_home_rl.data import build_request_summary, generate_day, generate_days
from smart_home_rl.env import SmartHomeEnv


def plot_example_day(day_data: dict, results_dir: Path) -> list[Path]:
    """
    Save plots for one generated day.

    The plots help verify that the generated data is structured and interpretable.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []
    hours = np.arange(len(day_data["prices"]))

    # Price profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, day_data["prices"], marker="o")
    ax.set_title(
        f"Electricity price profile "
        f"(day {day_data['day_index']}, {day_data['day_type']})"
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Price")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)

    path = results_dir / "example_day_prices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Solar profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, day_data["solar"], marker="o")
    ax.set_title(
        f"Solar production profile "
        f"(weather: {day_data['weather_type']})"
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Solar production")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)

    path = results_dir / "example_day_solar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Request windows
    fig, ax = plt.subplots(figsize=(8, 3))

    if day_data["requests"]:
        labels = []

        for y_pos, request in enumerate(day_data["requests"]):
            labels.append(request["name"])

            ax.hlines(
                y=y_pos,
                xmin=request["request_time"],
                xmax=request["deadline"],
                linewidth=3,
            )
            ax.scatter(
                [request["request_time"], request["latest_start"], request["deadline"]],
                [y_pos, y_pos, y_pos],
                marker="o",
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title("Generated appliance request windows")
        ax.set_xlabel("Hour")
        ax.set_xlim(0, 24)
        ax.set_xticks(hours)
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No appliance requests generated", ha="center", va="center")
        ax.set_axis_off()

    path = results_dir / "example_day_requests.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def plot_dataset_overview(days: list[dict], results_dir: Path) -> list[Path]:
    """
    Save overview plots across several generated days.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []
    hours = np.arange(len(days[0]["prices"]))

    price_matrix = np.vstack([day["prices"] for day in days])
    solar_matrix = np.vstack([day["solar"] for day in days])

    mean_prices = price_matrix.mean(axis=0)
    mean_solar = solar_matrix.mean(axis=0)

    # Average price profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, mean_prices, marker="o")
    ax.set_title(f"Average electricity price over {len(days)} generated days")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average price")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)

    path = results_dir / "average_price_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Average solar profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, mean_solar, marker="o")
    ax.set_title(f"Average solar production over {len(days)} generated days")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average solar production")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)

    path = results_dir / "average_solar_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Request counts by appliance
    request_df = build_request_summary(days)

    fig, ax = plt.subplots(figsize=(7, 4))

    if not request_df.empty:
        counts = request_df["name"].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Number of generated requests over {len(days)} days")
        ax.set_xlabel("Appliance")
        ax.set_ylabel("Number of requests")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No appliance requests generated", ha="center", va="center")
        ax.set_axis_off()

    path = results_dir / "request_counts_by_appliance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def run_episode(agent, env, day_data: dict) -> tuple[dict, list[dict]]:
    """
    Run one full day with a given agent.

    Returns
    -------
    metrics:
        Cumulative episode metrics.
    history:
        Step-by-step environment information.
    """

    if hasattr(agent, "reset"):
        agent.reset()

    state = env.reset(day_data)
    done = env.done

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(env, state=state, valid_actions=valid_actions)

        next_state, reward, done, info = env.step(action)
        state = next_state

    return env.get_metrics(), env.history


def evaluate_agent(agent, env, days: list[dict]) -> dict:
    """
    Evaluate one agent over several simulated days.
    """

    records = []

    for day in days:
        metrics, _ = run_episode(agent, env, day)

        records.append(
            {
                "agent": agent.name,
                "day_index": day["day_index"],
                "day_type": day["day_type"],
                "weather_type": day["weather_type"],
                **metrics,
            }
        )

    per_day = pd.DataFrame.from_records(records)

    metric_columns = [
        "num_requests",
        "completed_requests",
        "missed_deadlines",
        "total_cost",
        "total_grid_energy",
        "total_solar_used",
        "requested_energy",
        "completed_energy",
        "renewable_usage_ratio",
        "total_delay",
        "average_delay",
        "total_reward",
        "invalid_actions",
    ]

    summary = per_day[metric_columns].mean().to_dict()
    summary["agent"] = agent.name

    return {
        "agent": agent.name,
        "summary": summary,
        "per_day": per_day,
    }


def evaluate_agents(agents: list, env, days: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate several agents and return summary and per-day results.
    """

    summaries = []
    per_day_results = []

    for agent in agents:
        result = evaluate_agent(agent, env, days)
        summaries.append(result["summary"])
        per_day_results.append(result["per_day"])

    summary_df = pd.DataFrame.from_records(summaries)
    per_day_df = pd.concat(per_day_results, ignore_index=True)

    first_columns = ["agent"]
    other_columns = [col for col in summary_df.columns if col not in first_columns]
    summary_df = summary_df[first_columns + other_columns]

    return summary_df, per_day_df

def bootstrap_confidence_intervals(
    per_day_df: pd.DataFrame,
    metrics: list[str] | None = None,
    group_column: str = "agent",
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for evaluation metrics.

    The bootstrap resamples evaluation days with replacement within each agent.
    This estimates the uncertainty of the mean performance over the test set.
    """

    if metrics is None:
        metrics = [
            "total_cost",
            "renewable_usage_ratio",
            "average_delay",
            "missed_deadlines",
            "total_reward",
        ]

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level

    records = []

    for group_value, group_df in per_day_df.groupby(group_column):
        n = len(group_df)

        for metric in metrics:
            values = group_df[metric].to_numpy(dtype=float)

            bootstrap_means = np.empty(n_bootstrap)

            for i in range(n_bootstrap):
                sample = rng.choice(values, size=n, replace=True)
                bootstrap_means[i] = sample.mean()

            mean_value = values.mean()
            lower = np.quantile(bootstrap_means, alpha / 2.0)
            upper = np.quantile(bootstrap_means, 1.0 - alpha / 2.0)

            records.append(
                {
                    group_column: group_value,
                    "metric": metric,
                    "mean": mean_value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                    "n_bootstrap": n_bootstrap,
                    "n_observations": n,
                }
            )

    return pd.DataFrame.from_records(records)

def train_q_learning_agent(
    config: Config,
    appliances: list[ApplianceConfig] | None = None,
    num_episodes: int | None = None,
    seed: int | None = None,
) -> tuple[QLearningAgent, pd.DataFrame]:
    """
    Train a tabular Q-learning agent on generated daily episodes.

    Each training episode corresponds to one generated day.
    """

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    if num_episodes is None:
        num_episodes = config.episodes

    if seed is None:
        seed = config.seed + 2_000

    rng = np.random.default_rng(seed)

    env = SmartHomeEnv(config, appliances)

    agent = QLearningAgent(
        num_actions=len(appliances) + 1,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        seed=seed,
        initial_q_value=config.q_initial_value,
    )

    agent.set_training_mode()

    records = []

    for episode in range(num_episodes):
        day_data = generate_day(
            config=config,
            rng=rng,
            appliances=appliances,
            day_index=episode,
        )

        state = env.reset(day_data)
        done = env.done

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(env, state=state, valid_actions=valid_actions)

            next_state, reward, done, info = env.step(action)

            if done:
                next_valid_actions = []
            else:
                next_valid_actions = env.get_valid_actions()

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_valid_actions=next_valid_actions,
            )

            state = next_state

        metrics = env.get_metrics()

        records.append(
            {
                "episode": episode,
                "epsilon": agent.epsilon,
                "q_table_size": len(agent.q_table),
                **metrics,
            }
        )

        agent.decay_epsilon()

    training_history = pd.DataFrame.from_records(records)

    return agent, training_history

def run_multi_seed_experiments(
    config: Config,
    appliances: list[ApplianceConfig] | None = None,
    num_runs: int = 10,
    base_seed: int | None = None,
    fixed_evaluation_days: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full agent comparison over several independent random seeds.

    Each run trains a new Q-learning agent, evaluates it against the heuristic
    baselines, and stores the resulting metrics.

    Parameters
    ----------
    config:
        Global project configuration.

    appliances:
        List of appliances used in the environment.

    num_runs:
        Number of independent runs.

    base_seed:
        Base seed used to derive training and evaluation seeds.

    fixed_evaluation_days:
        If True, all runs are evaluated on the same generated test days.
        This isolates the variability due to Q-learning training.
        If False, each run gets its own evaluation days.

    Returns
    -------
    multi_seed_summary:
        Mean and standard deviation of each metric for each agent.

    multi_seed_per_run:
        Raw per-run summary results for each agent.
    """

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    if base_seed is None:
        base_seed = config.seed + 10_000

    per_run_records = []

    if fixed_evaluation_days:
        shared_evaluation_days = generate_days(
            config=config,
            num_days=config.eval_days,
            seed=base_seed + 1_000,
            appliances=appliances,
        )
    else:
        shared_evaluation_days = None

    for run_id in range(num_runs):
        run_seed = base_seed + run_id

        q_agent, _ = train_q_learning_agent(
            config=config,
            appliances=appliances,
            num_episodes=config.episodes,
            seed=run_seed,
        )

        q_agent.set_eval_mode()

        if fixed_evaluation_days:
            evaluation_days = shared_evaluation_days
        else:
            evaluation_days = generate_days(
                config=config,
                num_days=config.eval_days,
                seed=base_seed + 1_000 + run_id,
                appliances=appliances,
            )

        env = SmartHomeEnv(config, appliances)

        agents = [
            RunImmediatelyAgent(),
            CheapestHourAgent(),
            SolarGreedyAgent(),
            q_agent,
        ]

        run_summary, _ = evaluate_agents(
            agents=agents,
            env=env,
            days=evaluation_days,
        )

        run_summary = run_summary.copy()
        run_summary.insert(0, "run_id", run_id)
        run_summary.insert(1, "seed", run_seed)

        per_run_records.append(run_summary)

    multi_seed_per_run = pd.concat(per_run_records, ignore_index=True)

    metric_columns = [
        column
        for column in multi_seed_per_run.columns
        if column not in ["run_id", "seed", "agent"]
    ]

    summary_parts = []

    for metric in metric_columns:
        metric_summary = (
            multi_seed_per_run
            .groupby("agent")[metric]
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_std",
                }
            )
        )

        summary_parts.append(metric_summary)

    multi_seed_summary = pd.concat(summary_parts, axis=1).reset_index()

    return multi_seed_summary, multi_seed_per_run

def plot_training_curve(training_history: pd.DataFrame, results_dir: Path) -> list[Path]:
    """
    Save training plots for the Q-learning agent.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []

    # Reward curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_history["episode"], training_history["total_reward"], alpha=0.35)

    rolling_reward = training_history["total_reward"].rolling(window=100, min_periods=1).mean()
    ax.plot(training_history["episode"], rolling_reward)

    ax.set_title("Q-learning training reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True, alpha=0.3)

    path = results_dir / "q_learning_training_reward.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Epsilon decay
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_history["episode"], training_history["epsilon"])

    ax.set_title("Q-learning epsilon decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    path = results_dir / "q_learning_epsilon_decay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    # Q-table size
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_history["episode"], training_history["q_table_size"])

    ax.set_title("Number of visited states during Q-learning training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-table size")
    ax.grid(True, alpha=0.3)

    path = results_dir / "q_learning_q_table_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def plot_agent_comparison(summary_df: pd.DataFrame, results_dir: Path) -> list[Path]:
    """
    Save comparison plots for all evaluated agents.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []

    metrics_to_plot = [
        ("total_cost", "Average total cost", "agent_total_cost.png"),
        ("missed_deadlines", "Average missed deadlines", "agent_missed_deadlines.png"),
        ("average_delay", "Average scheduling delay", "agent_average_delay.png"),
        ("renewable_usage_ratio", "Average renewable usage ratio", "agent_renewable_usage_ratio.png"),
        ("total_reward", "Average total reward", "agent_total_reward.png"),
    ]

    for metric, title, filename in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(7, 4))

        summary_df.plot(
            kind="bar",
            x="agent",
            y=metric,
            ax=ax,
            legend=False,
        )

        ax.set_title(title)
        ax.set_xlabel("Agent")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

        path = results_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        saved_paths.append(path)

    return saved_paths


def build_comparison_to_baseline(
    summary_df: pd.DataFrame,
    baseline_agent: str = "run_immediately",
) -> pd.DataFrame:
    """
    Build a compact comparison table against a reference baseline.

    The default reference is the run-immediately policy because it represents
    the naive comfort-oriented strategy.
    """

    if baseline_agent not in set(summary_df["agent"]):
        raise ValueError(f"Baseline agent '{baseline_agent}' not found.")

    baseline = summary_df.loc[summary_df["agent"] == baseline_agent].iloc[0]

    rows = []

    for _, row in summary_df.iterrows():
        cost_reduction_pct = (
            (baseline["total_cost"] - row["total_cost"])
            / baseline["total_cost"]
            * 100.0
        )

        delay_difference_hours = row["average_delay"] - baseline["average_delay"]
        delay_difference_minutes = delay_difference_hours * 60.0

        reward_difference = row["total_reward"] - baseline["total_reward"]
        missed_deadline_difference = row["missed_deadlines"] - baseline["missed_deadlines"]

        rows.append(
            {
                "agent": row["agent"],
                "total_cost": row["total_cost"],
                "cost_reduction_vs_run_immediate_pct": cost_reduction_pct,
                "average_delay": row["average_delay"],
                "delay_difference_vs_run_immediate_hours": delay_difference_hours,
                "delay_difference_vs_run_immediate_minutes": delay_difference_minutes,
                "missed_deadlines": row["missed_deadlines"],
                "missed_deadline_difference": missed_deadline_difference,
                "renewable_usage_ratio": row["renewable_usage_ratio"],
                "total_reward": row["total_reward"],
                "reward_difference": reward_difference,
            }
        )

    return pd.DataFrame.from_records(rows)


def build_scenario_summary(
    per_day_df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Summarize performance by scenario type.

    Examples:
        group_column = "weather_type"
        group_column = "day_type"
    """

    metrics = [
        "completed_requests",
        "missed_deadlines",
        "total_cost",
        "renewable_usage_ratio",
        "average_delay",
        "total_reward",
    ]

    scenario_summary = (
        per_day_df
        .groupby(["agent", group_column], as_index=False)[metrics]
        .mean()
    )

    return scenario_summary


def plot_cost_delay_tradeoff(
    summary_df: pd.DataFrame,
    results_dir: Path,
) -> Path:
    """
    Plot the trade-off between electricity cost and scheduling delay.
    """

    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        summary_df["average_delay"],
        summary_df["total_cost"],
        s=80,
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["agent"],
            (row["average_delay"], row["total_cost"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_title("Cost-delay trade-off")
    ax.set_xlabel("Average delay")
    ax.set_ylabel("Average total cost")
    ax.grid(True, alpha=0.3)

    path = results_dir / "agent_cost_delay_tradeoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_metric_by_group(
    per_day_df: pd.DataFrame,
    group_column: str,
    metric: str,
    title: str,
    filename: str,
    results_dir: Path,
) -> Path:
    """
    Plot one metric grouped by scenario type and agent.
    """

    results_dir.mkdir(exist_ok=True)

    grouped = (
        per_day_df
        .groupby([group_column, "agent"])[metric]
        .mean()
        .unstack("agent")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(group_column)
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Agent", fontsize=8)

    path = results_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_evaluation_overview(
    summary_df: pd.DataFrame,
    per_day_df: pd.DataFrame,
    results_dir: Path,
) -> list[Path]:
    """
    Save additional report-ready evaluation plots.
    """

    saved_paths = []

    saved_paths.append(plot_cost_delay_tradeoff(summary_df, results_dir))

    saved_paths.append(
        plot_metric_by_group(
            per_day_df=per_day_df,
            group_column="weather_type",
            metric="total_cost",
            title="Average cost by weather type",
            filename="agent_cost_by_weather.png",
            results_dir=results_dir,
        )
    )

    saved_paths.append(
        plot_metric_by_group(
            per_day_df=per_day_df,
            group_column="weather_type",
            metric="renewable_usage_ratio",
            title="Renewable usage ratio by weather type",
            filename="agent_renewable_usage_by_weather.png",
            results_dir=results_dir,
        )
    )

    saved_paths.append(
        plot_metric_by_group(
            per_day_df=per_day_df,
            group_column="day_type",
            metric="total_cost",
            title="Average cost by day type",
            filename="agent_cost_by_day_type.png",
            results_dir=results_dir,
        )
    )

    saved_paths.append(
        plot_metric_by_group(
            per_day_df=per_day_df,
            group_column="day_type",
            metric="total_reward",
            title="Average reward by day type",
            filename="agent_reward_by_day_type.png",
            results_dir=results_dir,
        )
    )

    return saved_paths


def run_reward_ablation_study(
    config: Config,
    appliances: list[ApplianceConfig] | None = None,
    evaluation_days: list[dict] | None = None,
    num_episodes: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run reward ablation experiments for Q-learning.

    Each ablation trains a Q-learning agent with a modified reward function.
    All trained agents are then evaluated using the original full reward
    configuration so that results are comparable.
    """

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    if num_episodes is None:
        num_episodes = config.ablation_episodes

    if seed is None:
        seed = config.seed + 5_000

    if evaluation_days is None:
        evaluation_days = generate_days(
            config=config,
            num_days=config.eval_days,
            seed=config.seed + 1_000,
            appliances=appliances,
        )

    ablation_variants = [
        (
            "full_reward",
            {},
        ),
        (
            "no_solar_bonus",
            {
                "solar_bonus_weight": 0.0,
            },
        ),
        (
            "no_delay_penalty",
            {
                "delay_weight": 0.0,
            },
        ),
        (
            "strong_delay_penalty",
            {
                "delay_weight": 0.05,
            },
        ),
    ]

    summary_records = []
    per_day_frames = []
    training_frames = []

    for variant_index, (ablation_name, config_updates) in enumerate(ablation_variants):
        training_config = replace(config, **config_updates)

        agent, training_history = train_q_learning_agent(
            config=training_config,
            appliances=appliances,
            num_episodes=num_episodes,
            seed=seed + 100 * variant_index,
        )

        agent.set_eval_mode()

        # Evaluation uses the original full reward config.
        # This makes total_reward comparable across ablations.
        eval_env = SmartHomeEnv(config, appliances)

        result = evaluate_agent(
            agent=agent,
            env=eval_env,
            days=evaluation_days,
        )

        summary = {
            "ablation": ablation_name,
            **result["summary"],
            "training_reward_first_100": training_history["total_reward"].head(100).mean(),
            "training_reward_last_100": training_history["total_reward"].tail(100).mean(),
            "final_q_table_size": training_history["q_table_size"].iloc[-1],
        }

        summary_records.append(summary)

        per_day = result["per_day"].copy()
        per_day.insert(0, "ablation", ablation_name)
        per_day_frames.append(per_day)

        training_history = training_history.copy()
        training_history.insert(0, "ablation", ablation_name)
        training_frames.append(training_history)

    ablation_summary = pd.DataFrame.from_records(summary_records)
    ablation_per_day = pd.concat(per_day_frames, ignore_index=True)
    ablation_training_history = pd.concat(training_frames, ignore_index=True)

    return ablation_summary, ablation_per_day, ablation_training_history


def plot_reward_ablation_results(
    ablation_summary: pd.DataFrame,
    results_dir: Path,
) -> list[Path]:
    """
    Save report-ready plots for the reward ablation study.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []

    metrics_to_plot = [
        ("total_cost", "Reward ablation: average total cost", "ablation_total_cost.png"),
        ("average_delay", "Reward ablation: average delay", "ablation_average_delay.png"),
        ("missed_deadlines", "Reward ablation: missed deadlines", "ablation_missed_deadlines.png"),
        ("renewable_usage_ratio", "Reward ablation: renewable usage ratio", "ablation_renewable_usage_ratio.png"),
        ("total_reward", "Reward ablation: full-reward evaluation", "ablation_total_reward.png"),
    ]

    for metric, title, filename in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))

        ablation_summary.plot(
            kind="bar",
            x="ablation",
            y=metric,
            ax=ax,
            legend=False,
        )

        ax.set_title(title)
        ax.set_xlabel("Ablation variant")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

        path = results_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        saved_paths.append(path)

    return saved_paths


def run_robustness_study(
    config: Config,
    trained_q_agent: QLearningAgent,
    appliances: list[ApplianceConfig] | None = None,
    num_days: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all policies under several modified environment conditions.

    The Q-learning agent is not retrained for each scenario. This tests whether
    the learned policy trained under normal conditions remains useful when the
    environment changes.
    """

    if appliances is None:
        appliances = DEFAULT_APPLIANCES

    if num_days is None:
        num_days = config.eval_days

    if seed is None:
        seed = config.seed + 8_000

    scenarios = [
        (
            "default",
            "Default price volatility and solar availability",
            {},
        ),
        (
            "low_price_volatility",
            "Lower electricity price variability",
            {
                "price_volatility": 0.03,
            },
        ),
        (
            "high_price_volatility",
            "Higher electricity price variability",
            {
                "price_volatility": 0.14,
            },
        ),
        (
            "low_solar",
            "Lower solar availability",
            {
                "solar_peak": 1.5,
            },
        ),
        (
            "high_solar",
            "Higher solar availability",
            {
                "solar_peak": 4.5,
            },
        ),
    ]

    summary_frames = []
    per_day_frames = []

    trained_q_agent.set_eval_mode()

    for scenario_index, (scenario_name, scenario_description, config_updates) in enumerate(scenarios):
        scenario_config = replace(config, **config_updates)

        scenario_days = generate_days(
            config=scenario_config,
            num_days=num_days,
            seed=seed + 100 * scenario_index,
            appliances=appliances,
        )

        scenario_env = SmartHomeEnv(scenario_config, appliances)

        agents = [
            RunImmediatelyAgent(),
            CheapestHourAgent(),
            SolarGreedyAgent(),
            trained_q_agent,
        ]

        scenario_summary, scenario_per_day = evaluate_agents(
            agents=agents,
            env=scenario_env,
            days=scenario_days,
        )

        scenario_summary.insert(0, "scenario", scenario_name)
        scenario_summary.insert(1, "scenario_description", scenario_description)

        scenario_per_day.insert(0, "scenario", scenario_name)
        scenario_per_day.insert(1, "scenario_description", scenario_description)

        summary_frames.append(scenario_summary)
        per_day_frames.append(scenario_per_day)

    robustness_summary = pd.concat(summary_frames, ignore_index=True)
    robustness_per_day = pd.concat(per_day_frames, ignore_index=True)

    return robustness_summary, robustness_per_day


def plot_robustness_results(
    robustness_summary: pd.DataFrame,
    results_dir: Path,
) -> list[Path]:
    """
    Save robustness plots across environment scenarios.
    """

    results_dir.mkdir(exist_ok=True)

    saved_paths = []

    metrics_to_plot = [
        ("total_cost", "Robustness: average cost by scenario", "robustness_total_cost.png"),
        ("total_reward", "Robustness: average reward by scenario", "robustness_total_reward.png"),
        ("missed_deadlines", "Robustness: missed deadlines by scenario", "robustness_missed_deadlines.png"),
        ("average_delay", "Robustness: average delay by scenario", "robustness_average_delay.png"),
        ("renewable_usage_ratio", "Robustness: renewable usage by scenario", "robustness_renewable_usage_ratio.png"),
    ]

    for metric, title, filename in metrics_to_plot:
        grouped = (
            robustness_summary
            .pivot(index="scenario", columns="agent", values=metric)
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        grouped.plot(kind="bar", ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Agent", fontsize=8)
        ax.tick_params(axis="x", rotation=25)

        path = results_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        saved_paths.append(path)

    return saved_paths


def build_robustness_comparison_to_run_immediate(
    robustness_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each robustness scenario, compare each agent to run-immediately.
    """

    rows = []

    for scenario, scenario_df in robustness_summary.groupby("scenario"):
        baseline = scenario_df.loc[scenario_df["agent"] == "run_immediately"].iloc[0]

        for _, row in scenario_df.iterrows():
            cost_reduction_pct = (
                (baseline["total_cost"] - row["total_cost"])
                / baseline["total_cost"]
                * 100.0
            )

            delay_difference_minutes = (
                row["average_delay"] - baseline["average_delay"]
            ) * 60.0

            reward_difference = row["total_reward"] - baseline["total_reward"]
            missed_deadline_difference = row["missed_deadlines"] - baseline["missed_deadlines"]

            rows.append(
                {
                    "scenario": scenario,
                    "agent": row["agent"],
                    "cost_reduction_vs_run_immediate_pct": cost_reduction_pct,
                    "delay_difference_vs_run_immediate_minutes": delay_difference_minutes,
                    "missed_deadline_difference": missed_deadline_difference,
                    "reward_difference": reward_difference,
                    "total_cost": row["total_cost"],
                    "average_delay": row["average_delay"],
                    "missed_deadlines": row["missed_deadlines"],
                    "total_reward": row["total_reward"],
                }
            )

    return pd.DataFrame.from_records(rows)


def run_ablation_study(config: Config):
    """
    Convenience wrapper for the main reward ablation study.
    """

    return run_reward_ablation_study(config)