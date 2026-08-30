# Approximate Dynamic Programming for Fleet / Inventory

Research-oriented Industrial Engineering / Operations Research benchmark for **dynamic fleet repositioning viewed as inventory control across locations**.

## Research question

Can a low-dimensional approximate value function capture enough future fleet value to improve repositioning decisions over myopic policies, and how does that approximation degrade as fleet size and demand regime shift away from training conditions?

## Current status

**Feature-complete research benchmark.**

The repository implements:

- a two-station closed rental fleet;
- station vehicle counts interpreted as location inventory;
- time-varying directional Poisson demand;
- pre-demand repositioning decisions;
- rental revenue, reposition cost and lost-demand penalty;
- exact finite-horizon dynamic programming as an oracle;
- no-reposition and demand-balance heuristic baselines;
- backward fitted linear value-function approximation;
- Monte Carlo Bellman targets for ADP training;
- three independent ADP training seeds;
- exact expected-value benchmarking;
- common-random-number operational simulation;
- nominal, fleet-size, directional-demand and high-demand OOD blocks;
- model-seed aggregation before statistical inference;
- paired bootstrap confidence intervals and exact sign tests;
- lost-demand, reposition-volume and decision-latency reporting;
- frozen config, tests, final report and CI across Python 3.10–3.12.

## State and action

With total fleet `N`, state `s_t` is the number of vehicles at station A. Station B therefore has `N - s_t` vehicles.

Before demand is observed, action `q_t` repositions vehicles:

```text
q_t > 0  : move vehicles A -> B
q_t < 0  : move vehicles B -> A
```

subject to available fleet and a per-period reposition limit.

After repositioning, random directional rental demand is realized. Served rentals move vehicles between stations, so tomorrow's fleet state is today's post-demand inventory balance.

## Objective

Maximize expected finite-horizon operating contribution:

```text
rental revenue
- reposition cost
- lost-demand penalty
+ future fleet value
```

## Exact DP oracle

For small and medium fleets, the benchmark enumerates every feasible reposition action and a truncated Poisson demand distribution. Backward dynamic programming computes the exact finite-horizon value function and optimal policy.

The exact DP is recomputed for every evaluation scenario and remains the quality reference.

## Approximate dynamic programming

The ADP model stores a separate linear value approximation at each time step:

```text
V_t(s) ~= w_t^T phi(s)
```

with normalized fleet-balance polynomial features. Training proceeds backward. For each state and feasible reposition action, Monte Carlo demand samples estimate a Bellman target using the next-stage approximate value function. Ridge regression then fits the current value approximation.

This is deliberately an auditable value-function approximation benchmark rather than a deep-RL implementation.

## Frozen final campaign

`configs/experiment.json` freezes three ADP training seeds (`0, 1, 2`) and environment seeds `1000-1019`.

Evaluation blocks:

- `nominal_final`: fleet size 12, base demand;
- `fleet_size_ood`: fleet size 18;
- `directional_demand_ood`: 25% stronger directional imbalance;
- `high_demand_ood`: 25% higher demand intensity.

Shifted evaluation models are never used to refit the value approximation.

## Statistical evaluation

Every policy sees the same exogenous demand realization for a given environment seed. ADP results from independent training seeds are first averaged within each environment seed. Only then is paired inference performed across environment seeds.

ADP is compared against demand balance and exact DP using:

- mean paired profit difference;
- paired 95% bootstrap confidence interval;
- win rate;
- exact two-sided sign-test p-value.

This prevents training replicates from being incorrectly counted as independent test observations.

## Metrics

Primary operational metric: **total finite-horizon operating profit**.

Secondary metrics:

- expected-profit gap to exact DP;
- lost demand;
- reposition volume;
- action agreement with exact DP;
- value-function RMSE;
- action-selection latency.

A more complex ADP policy is not preferred automatically. If demand balance is competitive at lower latency, the simpler heuristic remains the recommendation.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m fleet_adp.experiment
python -m fleet_adp.final_campaign
```

## Repository map

```text
src/fleet_adp/
  model.py
  exact_dp.py
  adp.py
  policies.py
  benchmark.py
  simulation.py
  statistics.py
  experiment.py
  final_campaign.py
tests/
  test_model.py
  test_adp.py
  test_completion.py
configs/
  experiment.json
docs/
  experimental_protocol.md
  final_report.md
.github/workflows/
  ci.yml
```

## Scientific acceptance rules

1. exact DP remains the quality oracle wherever tractable;
2. ADP training uses only the base model;
3. final OOD models are evaluation-only;
4. all policies receive identical demand realizations within an environment seed;
5. ADP training seeds are aggregated before inference;
6. profit, lost demand, repositioning and latency are interpreted jointly;
7. expected-value and simulation results are not conflated;
8. negative/null ADP results are retained.

See `docs/final_report.md` for the complete methodological contract.

## Scope boundary

This repository studies a compact two-station closed-fleet inventory system. Multi-station networks, travel-time state, reservations, pricing and deep reinforcement learning are separate research extensions rather than silent scope expansion.

## License

MIT
