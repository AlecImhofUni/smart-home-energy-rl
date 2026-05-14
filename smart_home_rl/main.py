"""
Main entry point for the smart-home reinforcement learning project.

Running this module executes the full experimental pipeline:

1. generate and inspect synthetic daily data,
2. train the tabular Q-learning agent,
3. evaluate heuristic baselines and Q-learning,
4. save comparison tables and plots,
5. run reward ablation experiments,
6. run robustness experiments under modified environment settings,
7. print a compact summary of the main results.

The script is intended to be run from the project root with:

    python -m smart_home_rl.main
"""
from smart_home_rl.agents import (
    CheapestHourAgent,
    RunImmediatelyAgent,
    SolarGreedyAgent,
)
from smart_home_rl.config import Config, DEFAULT_APPLIANCES
from smart_home_rl.data import (
    build_day_summary,
    build_request_summary,
    generate_days,
)
from smart_home_rl.env import SmartHomeEnv
from smart_home_rl.experiments import (
    build_comparison_to_baseline,
    build_robustness_comparison_to_run_immediate,
    build_scenario_summary,
    evaluate_agents,
    plot_agent_comparison,
    plot_dataset_overview,
    plot_evaluation_overview,
    plot_example_day,
    plot_reward_ablation_results,
    plot_robustness_results,
    plot_training_curve,
    run_reward_ablation_study,
    run_robustness_study,
    run_multi_seed_experiments,
    train_q_learning_agent,
    
)


def print_result_interpretation(agent_summary) -> None:
    """
    Print a compact interpretation of the agent comparison.
    """

    best_reward_row = agent_summary.loc[agent_summary["total_reward"].idxmax()]
    best_cost_row = agent_summary.loc[agent_summary["total_cost"].idxmin()]
    best_delay_row = agent_summary.loc[agent_summary["average_delay"].idxmin()]
    best_solar_row = agent_summary.loc[agent_summary["renewable_usage_ratio"].idxmax()]

    q_row = agent_summary.loc[agent_summary["agent"] == "q_learning"].iloc[0]
    immediate_row = agent_summary.loc[agent_summary["agent"] == "run_immediately"].iloc[0]
    cheapest_row = agent_summary.loc[agent_summary["agent"] == "cheapest_hour"].iloc[0]

    reward_difference_vs_immediate = (
        q_row["total_reward"] - immediate_row["total_reward"]
    )

    cost_difference_vs_immediate = (
        q_row["total_cost"] - immediate_row["total_cost"]
    )

    cost_reduction_pct = (
        (immediate_row["total_cost"] - q_row["total_cost"])
        / immediate_row["total_cost"]
        * 100.0
    )

    print("Compact interpretation:")
    print(f"- Best average reward: {best_reward_row['agent']}")
    print(f"- Lowest average cost: {best_cost_row['agent']}")
    print(f"- Lowest average delay: {best_delay_row['agent']}")
    print(f"- Highest renewable usage ratio: {best_solar_row['agent']}")

    print()
    print("Q-learning compared with run-immediately:")
    print(f"- Reward difference: {reward_difference_vs_immediate:.3f}")
    print(f"- Cost difference: {cost_difference_vs_immediate:.3f}")
    print(f"- Cost reduction: {cost_reduction_pct:.2f}%")

    if reward_difference_vs_immediate > 0:
        print("- Q-learning slightly improves reward compared with the naive comfort-oriented baseline.")
    else:
        print("- Q-learning does not improve reward over the naive baseline in this run.")

    if cost_difference_vs_immediate < 0:
        print("- Q-learning reduces electricity cost compared with the naive baseline.")
    else:
        print("- Q-learning does not reduce electricity cost compared with the naive baseline in this run.")

    print()
    print("How to interpret cheapest-hour:")
    print(
        "- Cheapest-hour is a strong planning heuristic because it explicitly checks "
        "future feasible start times using the generated daily profile."
    )
    print(
        "- Therefore, it is acceptable if Q-learning improves over the naive baseline "
        "but does not beat cheapest-hour."
    )

    if cheapest_row["missed_deadlines"] > q_row["missed_deadlines"]:
        print(
            "- In this run, Q-learning also satisfies deadlines more reliably than "
            "the cheapest-hour heuristic."
        )


def main() -> None:
    config = Config()
    config.results_dir.mkdir(exist_ok=True)

    # Generate a small dataset to inspect the synthetic data generation.
    inspection_days = generate_days(
        config=config,
        num_days=30,
        seed=config.seed,
        appliances=DEFAULT_APPLIANCES,
    )

    example_day = inspection_days[0]

    day_summary = build_day_summary(inspection_days)
    request_summary = build_request_summary(inspection_days)

    day_summary_path = config.results_dir / "generated_days_summary.csv"
    request_summary_path = config.results_dir / "generated_requests_summary.csv"

    day_summary.to_csv(day_summary_path, index=False)
    request_summary.to_csv(request_summary_path, index=False)

    saved_plots = []
    saved_plots.extend(plot_example_day(example_day, config.results_dir))
    saved_plots.extend(plot_dataset_overview(inspection_days, config.results_dir))

    # Train Q-learning agent.
    q_agent, training_history = train_q_learning_agent(
        config=config,
        appliances=DEFAULT_APPLIANCES,
        num_episodes=config.episodes,
        seed=config.seed + 2_000,
    )

    training_history_path = config.results_dir / "q_learning_training_history.csv"
    training_history.to_csv(training_history_path, index=False)

    saved_plots.extend(plot_training_curve(training_history, config.results_dir))

    # Evaluate heuristic baselines and trained Q-learning on the same test days.
    evaluation_days = generate_days(
        config=config,
        num_days=config.eval_days,
        seed=config.seed + 1_000,
        appliances=DEFAULT_APPLIANCES,
    )

    env = SmartHomeEnv(config, DEFAULT_APPLIANCES)

    q_agent.set_eval_mode()

    agents = [
        RunImmediatelyAgent(),
        CheapestHourAgent(),
        SolarGreedyAgent(),
        q_agent,
    ]

    agent_summary, agent_per_day = evaluate_agents(
        agents=agents,
        env=env,
        days=evaluation_days,
    )

    agent_summary_path = config.results_dir / "agent_comparison_summary.csv"
    agent_per_day_path = config.results_dir / "agent_comparison_per_day.csv"

    agent_summary.to_csv(agent_summary_path, index=False)
    agent_per_day.to_csv(agent_per_day_path, index=False)

    comparison_table = build_comparison_to_baseline(
        agent_summary,
        baseline_agent="run_immediately",
    )

    comparison_table_path = config.results_dir / "comparison_vs_run_immediately.csv"
    comparison_table.to_csv(comparison_table_path, index=False)

    weather_summary = build_scenario_summary(agent_per_day, group_column="weather_type")
    day_type_summary = build_scenario_summary(agent_per_day, group_column="day_type")

    weather_summary_path = config.results_dir / "scenario_summary_by_weather.csv"
    day_type_summary_path = config.results_dir / "scenario_summary_by_day_type.csv"

    weather_summary.to_csv(weather_summary_path, index=False)
    day_type_summary.to_csv(day_type_summary_path, index=False)

    saved_plots.extend(plot_agent_comparison(agent_summary, config.results_dir))
    saved_plots.extend(
        plot_evaluation_overview(
            summary_df=agent_summary,
            per_day_df=agent_per_day,
            results_dir=config.results_dir,
        )
    )

    # Run multi-seed experiments for more robust agent comparison.
    multi_seed_summary, multi_seed_per_run = run_multi_seed_experiments(
        config=config,
        appliances=DEFAULT_APPLIANCES,
        num_runs=5,
        base_seed=config.seed + 10_000,
        fixed_evaluation_days=True,
    )

    multi_seed_summary_path = config.results_dir / "multi_seed_summary.csv"
    multi_seed_per_run_path = config.results_dir / "multi_seed_per_run.csv"

    multi_seed_summary.to_csv(multi_seed_summary_path, index=False)
    multi_seed_per_run.to_csv(multi_seed_per_run_path, index=False)

    # Run reward ablation study for Q-learning.
    ablation_summary, ablation_per_day, ablation_training_history = run_reward_ablation_study(
        config=config,
        appliances=DEFAULT_APPLIANCES,
        evaluation_days=evaluation_days,
        num_episodes=config.ablation_episodes,
        seed=config.seed + 5_000,
    )

    ablation_summary_path = config.results_dir / "ablation_reward_summary.csv"
    ablation_per_day_path = config.results_dir / "ablation_reward_per_day.csv"
    ablation_training_history_path = config.results_dir / "ablation_reward_training_history.csv"

    ablation_summary.to_csv(ablation_summary_path, index=False)
    ablation_per_day.to_csv(ablation_per_day_path, index=False)
    ablation_training_history.to_csv(ablation_training_history_path, index=False)

    saved_plots.extend(
        plot_reward_ablation_results(
            ablation_summary=ablation_summary,
            results_dir=config.results_dir,
        )
    )

    # Run robustness study under modified environment conditions.
    robustness_summary, robustness_per_day = run_robustness_study(
        config=config,
        trained_q_agent=q_agent,
        appliances=DEFAULT_APPLIANCES,
        num_days=config.eval_days,
        seed=config.seed + 8_000,
    )

    robustness_comparison = build_robustness_comparison_to_run_immediate(
        robustness_summary
    )

    robustness_summary_path = config.results_dir / "robustness_summary.csv"
    robustness_per_day_path = config.results_dir / "robustness_per_day.csv"
    robustness_comparison_path = config.results_dir / "robustness_vs_run_immediately.csv"

    robustness_summary.to_csv(robustness_summary_path, index=False)
    robustness_per_day.to_csv(robustness_per_day_path, index=False)
    robustness_comparison.to_csv(robustness_comparison_path, index=False)

    saved_plots.extend(
        plot_robustness_results(
            robustness_summary=robustness_summary,
            results_dir=config.results_dir,
        )
    )

    print("Smart Home RL project initialized successfully.")
    print()
    print(f"Horizon: {config.horizon} hours")
    print(f"Number of appliances: {len(DEFAULT_APPLIANCES)}")
    print(f"Results directory: {config.results_dir}")
    print()
    print("Generated example day:")
    print(f"- Day index: {example_day['day_index']}")
    print(f"- Day type: {example_day['day_type']}")
    print(f"- Weather type: {example_day['weather_type']}")
    print(f"- Prices shape: {example_day['prices'].shape}")
    print(f"- Solar shape: {example_day['solar'].shape}")
    print("- Requests:")

    if example_day["requests"]:
        for request in example_day["requests"]:
            print(
                f"  {request['name']}: "
                f"request at hour {request['request_time']}, "
                f"latest start at hour {request['latest_start']}, "
                f"deadline at hour {request['deadline']}, "
                f"duration {request['duration']}h"
            )
    else:
        print("  No requests generated for this day.")

    print()
    print("Generated dataset summary:")
    print(f"- Number of simulated days: {len(inspection_days)}")
    print(f"- Average number of requests per day: {day_summary['num_requests'].mean():.2f}")
    print(f"- Average daily requested energy: {day_summary['requested_energy_kwh'].mean():.2f} kWh")
    print()
    print("Saved data generation summary files:")
    print(f"- {day_summary_path}")
    print(f"- {request_summary_path}")

    print()
    print("Q-learning training:")
    print(f"- Training episodes: {len(training_history)}")
    print(f"- Final epsilon: {training_history['epsilon'].iloc[-1]:.3f}")
    print(f"- Final Q-table size: {training_history['q_table_size'].iloc[-1]}")
    print(f"- Mean reward over first 100 episodes: {training_history['total_reward'].head(100).mean():.3f}")
    print(f"- Mean reward over last 100 episodes: {training_history['total_reward'].tail(100).mean():.3f}")
    print(f"- Training history file: {training_history_path}")

    print()
    print("Agent evaluation:")
    print(f"- Number of evaluation days: {len(evaluation_days)}")
    print()
    print(
        agent_summary[
            [
                "agent",
                "completed_requests",
                "missed_deadlines",
                "total_cost",
                "renewable_usage_ratio",
                "average_delay",
                "total_reward",
                "invalid_actions",
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print("Comparison against run-immediately:")
    print(
        comparison_table[
            [
                "agent",
                "cost_reduction_vs_run_immediate_pct",
                "delay_difference_vs_run_immediate_minutes",
                "missed_deadline_difference",
                "reward_difference",
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print_result_interpretation(agent_summary)

    print()
    print("Reward ablation study:")
    print(f"- Training episodes per ablation: {config.ablation_episodes}")
    print(
        ablation_summary[
            [
                "ablation",
                "total_cost",
                "renewable_usage_ratio",
                "average_delay",
                "missed_deadlines",
                "total_reward",
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print("Robustness study:")
    print(f"- Evaluation days per scenario: {config.eval_days}")

    q_learning_robustness = robustness_comparison[
        robustness_comparison["agent"] == "q_learning"
    ].copy()

    print()
    print("Q-learning robustness compared with run-immediately:")
    print(
        q_learning_robustness[
            [
                "scenario",
                "cost_reduction_vs_run_immediate_pct",
                "delay_difference_vs_run_immediate_minutes",
                "missed_deadline_difference",
                "reward_difference",
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print("Robustness interpretation:")
    print("- Positive cost reduction means Q-learning is cheaper than run-immediately.")
    print("- Positive reward difference means Q-learning performs better under the full reward.")
    print("- The full robustness tables are saved as CSV files in the results folder.")

    print()
    print("Saved result files:")
    print("- Main comparison:")
    print(f"  - {agent_summary_path}")
    print(f"  - {agent_per_day_path}")
    print(f"  - {comparison_table_path}")
    print("- Scenario summaries:")
    print(f"  - {weather_summary_path}")
    print(f"  - {day_type_summary_path}")
    print("- Ablation study:")
    print(f"  - {ablation_summary_path}")
    print(f"  - {ablation_per_day_path}")
    print(f"  - {ablation_training_history_path}")
    print("- Robustness study:")
    print(f"  - {robustness_summary_path}")
    print(f"  - {robustness_per_day_path}")
    print(f"  - {robustness_comparison_path}")

    print()
    print("Saved plots:")
    print(f"- {len(saved_plots)} plots saved in {config.results_dir}/")
    print("- Main plots include agent comparison, ablation results, and robustness results.")

    print()
    print(f"Environment object created: {env.__class__.__name__}")


if __name__ == "__main__":
    main()