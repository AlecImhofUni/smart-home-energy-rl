# Smart Home Energy Scheduling with Reinforcement Learning

This project studies how a smart home energy management system can schedule flexible household appliances under uncertain electricity prices and renewable energy availability.

The problem is modeled as a Markov Decision Process (MDP). The goal is to compare simple heuristic scheduling policies with a tabular Q-learning agent in a controlled simulation environment.

The project was developed for the course **Reinforcement Learning and Decision Making Under Uncertainty**.

---

## Project idea

A household may have flexible appliances that do not need to run immediately, but must be completed before a deadline. Examples include:

- a washing machine,
- a dishwasher,
- an electric vehicle charger.

At each hour of the day, the controller observes the current energy situation and decides whether to start one appliance or wait.

The scheduling decision must balance several objectives:

- reducing electricity cost,
- using available solar energy,
- avoiding excessive delays,
- respecting appliance deadlines.

---

## Project structure

```text
smart_home_energy_rl/
│
├── smart_home_rl/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── env.py
│   ├── agents.py
│   ├── experiments.py
│   └── main.py
│
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

### File overview

| File | Role |
|---|---|
| `config.py` | Defines global parameters and default appliance settings |
| `data.py` | Generates synthetic price, solar and appliance request data |
| `env.py` | Implements the smart-home scheduling MDP environment |
| `agents.py` | Defines heuristic agents and the Q-learning agent |
| `experiments.py` | Contains training, evaluation, plotting, ablation and robustness utilities |
| `main.py` | Runs the full experimental pipeline |

---

## Setup (Windows)

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full project:

```powershell
python -m smart_home_rl.main
```

---

## Synthetic data generation

The project uses synthetic data because the objective is to study scheduling behavior in a controlled smart-home environment.

The generated data is stochastic, but not purely random. It follows structured daily patterns.

### Electricity prices

Electricity prices are generated over a 24-hour horizon.

The price profile includes:

- lower prices at night,
- morning and evening price peaks,
- slightly weaker peaks on weekends,
- random daily and hourly noise.

This creates realistic time-of-use price variation while keeping the environment controlled.

### Solar production

Solar production follows a bell-shaped daily curve:

- no production at night,
- increasing production in the morning,
- peak production around midday,
- decreasing production in the evening.

Each day receives a weather type:

- `sunny`,
- `mixed`,
- `cloudy`.

The weather type scales the solar production profile.

### Appliance requests

The project currently uses three flexible appliances:

| Appliance | Duration | Typical request period | Flexibility |
|---|---:|---|---|
| Washing machine | 2h | morning / afternoon / early evening | medium |
| Dishwasher | 2h | evening | medium |
| EV charger | 4h | evening | lower |

Each appliance has:

- a weekday request probability,
- a weekend request probability,
- a realistic request window,
- a fixed duration,
- a deadline derived from its request time.

This avoids unrealistic demand generation and makes the scheduling problem meaningful.

---

## MDP formulation

We model the smart-home scheduling problem as a finite-horizon Markov Decision Process.

- **Episode:** one simulated day of 24 hourly time steps.
- **State:** the current hour, electricity price level, solar production level, and pending appliance requests with their deadlines.
- **Actions:** either wait or start one of the currently available appliance jobs.
- **Transition dynamics:** the environment advances by one hour, updates pending requests, and generates new stochastic appliance demands.
- **Reward:** the reward combines electricity cost, delay penalties, deadline penalties, and incentives for using available solar energy.
- **Objective:** learn a policy that minimizes total daily energy cost while maintaining acceptable user comfort and respecting appliance deadlines.

This formulation assumes that electricity prices and solar availability are known or forecasted at the hourly level.

### State

The state is discrete and includes:

- current hour,
- current electricity price level,
- current solar production level,
- for each appliance:
  - request/completion status,
  - urgency,
  - cost attractiveness of starting now,
  - solar attractiveness of starting now.

The state includes simple forecast-aware features because day-ahead price and solar forecasts are realistic in a smart-home energy setting.

### Actions

The action space is:

| Action | Meaning |
|---:|---|
| 0 | Do nothing |
| 1 | Start washing machine |
| 2 | Start dishwasher |
| 3 | Start EV charger |

Only valid actions are allowed.

If an appliance reaches its latest feasible start time, doing nothing is no longer valid. This models deadlines as hard scheduling constraints.

### Reward

The reward balances electricity cost, user comfort, deadline satisfaction and renewable energy usage.

The reward has the following structure:

```text
reward =
    - electricity cost
    - delay penalty
    - missed deadline penalty
    + solar usage bonus
```

The current reward parameters are:

| Parameter | Value | Interpretation |
|---|---:|---|
| `cost_weight` | 1.0 | Electricity cost is the main optimization objective |
| `delay_weight` | 0.02 | Delays matter, but should not dominate the objective |
| `missed_deadline_penalty` | 5.0 | Missing deadlines is strongly discouraged |
| `solar_bonus_weight` | 0.10 | Solar usage is encouraged, but not more important than feasibility |
| `invalid_action_penalty` | 2.0 | Invalid actions are penalized |

Total rewards are usually negative because the environment is mostly cost-based. A less negative reward means a better policy.

---

## Agents

The project compares four policies.

### Run immediately

Starts an appliance as soon as it becomes available.

If multiple appliances can be started, it chooses the one with the earliest deadline.

This baseline prioritizes user comfort and minimizes delay, but it ignores electricity prices and solar production.

### Cheapest hour

Schedules appliances at the cheapest feasible hour before their deadline.

For each pending appliance, it estimates the grid electricity cost of starting now and at future feasible start times.

This is a strong planning heuristic because it uses the full generated daily price and solar profile.

### Solar greedy

Prioritizes appliance execution when solar coverage is high.

This baseline represents a renewable-aware scheduling rule, but it can perform poorly when appliance requests arrive after the solar peak.

### Q-learning

The main reinforcement learning method is tabular Q-learning.

The agent learns a value table:

```text
Q(state, action)
```

During training, it uses epsilon-greedy exploration. During evaluation, exploration is disabled and the agent chooses the best learned valid action.

The Q-learning update rule is:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

Since most rewards are negative, Q-values are initialized pessimistically. This avoids making unseen actions look artificially good simply because their value is still zero.

---

## Experiments

Running `main.py` executes the full experimental pipeline:

1. generate and inspect synthetic data,
2. train the Q-learning agent,
3. evaluate all agents on the same test days,
4. save comparison tables and plots,
5. run a reward ablation study,
6. run robustness experiments under modified environment conditions.

---

## Main evaluation

The main evaluation compares:

- `run_immediately`,
- `cheapest_hour`,
- `solar_greedy`,
- `q_learning`.

The main metrics are:

| Metric | Meaning |
|---|---|
| `total_cost` | Average electricity cost |
| `renewable_usage_ratio` | Fraction of completed energy covered by solar |
| `average_delay` | Average delay between request time and start time |
| `missed_deadlines` | Average number of missed deadlines |
| `total_reward` | Full reward objective |
| `invalid_actions` | Number of invalid actions |

The project also computes comparison metrics against the naive `run_immediately` baseline:

- cost reduction percentage,
- delay difference in minutes,
- missed deadline difference,
- reward difference.

---

## Reward ablation study

The reward ablation study tests how different reward components affect the learned Q-learning policy.

The tested variants are:

| Variant | Description |
|---|---|
| `full_reward` | Original reward with cost, delay, deadline penalty and solar bonus |
| `no_solar_bonus` | Removes the renewable energy bonus |
| `no_delay_penalty` | Removes the waiting penalty |
| `strong_delay_penalty` | Increases the delay penalty to prioritize user comfort |

Each variant trains a separate Q-learning agent. All ablation-trained agents are evaluated using the original full reward configuration so that final results remain comparable.

This study helps justify the chosen reward design.

---

## Robustness study

The robustness study evaluates how policies behave under modified environment conditions.

The Q-learning agent is trained once under the default environment and then evaluated under several test scenarios.

The tested scenarios are:

| Scenario | Description |
|---|---|
| `default` | Original environment |
| `low_price_volatility` | Electricity prices vary less during the day |
| `high_price_volatility` | Electricity prices vary more during the day |
| `low_solar` | Lower solar production |
| `high_solar` | Higher solar production |

This tests whether the learned policy generalizes beyond the exact training distribution.

---
### Statistical confidence

To make the evaluation more robust, the project reports bootstrap confidence intervals for the main metrics. Instead of relying only on average performance over evaluation days, the confidence intervals estimate the uncertainty around the measured results by resampling the evaluation episodes.

This is used to better assess whether the observed differences between agents are meaningful and reproducible.

---

### Multi-seed evaluation

To evaluate the reproducibility of the learning results, Q-learning is trained and evaluated over multiple random seeds. This allows us to check whether the observed performance is stable or whether it strongly depends on a lucky or unlucky training run.

The heuristic baselines are also included in the multi-seed comparison. Since they are deterministic when evaluated on the same fixed test days, their standard deviation can be zero. This is expected and helps isolate the variability introduced by the learning algorithm.

---

## Result files

All outputs are saved in the `results/` folder.

### Data generation outputs

```text
generated_days_summary.csv
generated_requests_summary.csv
example_day_prices.png
example_day_solar.png
example_day_requests.png
average_price_profile.png
average_solar_profile.png
request_counts_by_appliance.png
```

### Main comparison outputs

```text
agent_comparison_summary.csv
agent_comparison_per_day.csv
comparison_vs_run_immediately.csv
agent_total_cost.png
agent_missed_deadlines.png
agent_average_delay.png
agent_renewable_usage_ratio.png
agent_total_reward.png
agent_cost_delay_tradeoff.png
```

### Statistical confidence outputs

```text
bootstrap_confidence_intervals.csv
```

### Multi-seed outputs

```text
multi_seed_summary.csv
multi_seed_per_run.csv
```

### Scenario analysis outputs

```text
scenario_summary_by_weather.csv
scenario_summary_by_day_type.csv
agent_cost_by_weather.png
agent_renewable_usage_by_weather.png
agent_cost_by_day_type.png
agent_reward_by_day_type.png
```

### Q-learning training outputs

```text
q_learning_training_history.csv
q_learning_training_reward.png
q_learning_epsilon_decay.png
q_learning_q_table_size.png
```

### Ablation outputs

```text
ablation_reward_summary.csv
ablation_reward_per_day.csv
ablation_reward_training_history.csv
ablation_total_cost.png
ablation_average_delay.png
ablation_missed_deadlines.png
ablation_renewable_usage_ratio.png
ablation_total_reward.png
```

### Robustness outputs

```text
robustness_summary.csv
robustness_per_day.csv
robustness_vs_run_immediately.csv
robustness_total_cost.png
robustness_total_reward.png
robustness_missed_deadlines.png
robustness_average_delay.png
robustness_renewable_usage_ratio.png
```

---

## Expected interpretation

The current results support the following interpretation:

- `run_immediately` has the lowest delay and represents a comfort-oriented baseline.
- `cheapest_hour` usually achieves the lowest electricity cost because it is a strong planning heuristic with access to future daily profiles.
- `q_learning` learns a cost-saving policy compared with the naive immediate baseline while keeping missed deadlines close to zero.
- `solar_greedy` can be limited because many appliance requests occur in the evening, after solar production has decreased.
- Q-learning does not need to beat the strongest heuristic to be useful. The important result is that it learns a meaningful trade-off between cost, delay and deadline satisfaction.
- Robustness experiments help identify where the learned policy generalizes well and where it has limitations.

A typical conclusion is:

```text
Q-learning improves over naive immediate scheduling by reducing electricity cost while preserving deadline satisfaction, but it does not consistently outperform a handcrafted planning heuristic that directly uses future price and solar information.
```

---
