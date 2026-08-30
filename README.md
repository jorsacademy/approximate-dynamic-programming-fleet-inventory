# Approximate Dynamic Programming for Fleet / Inventory

Research-oriented Industrial Engineering / Operations Research benchmark for **dynamic fleet repositioning viewed as inventory control across locations**.

## Research question

Can a low-dimensional approximate value function capture enough future fleet value to improve repositioning decisions over myopic policies, and how does that approximation degrade as fleet size and demand regime shift away from training conditions?

## Current status

**Phase 1 implemented: stochastic fleet-inventory model + exact finite-horizon DP + linear ADP benchmark.**

The repository includes:

- a two-station closed rental fleet;
- station vehicle counts interpreted as location inventory;
- time-varying directional Poisson demand;
- pre-demand repositioning decisions;
- rental revenue, reposition cost and lost-demand penalty;
- exact finite-horizon dynamic programming for a transparent oracle;
- no-reposition and demand-balance heuristic baselines;
- backward fitted linear value-function approximation;
- Monte Carlo Bellman targets for ADP training;
- held-out and fleet-size-shift evaluation;
- tests and Python 3.10–3.12 CI.

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

For small and medium fleets, the benchmark enumerates all feasible reposition actions and a truncated Poisson demand distribution. Backward dynamic programming computes the exact finite-horizon value function and optimal policy.

## Approximate dynamic programming

The ADP model stores a separate linear value approximation at each time step:

```text
V_t(s) ~= w_t^T phi(s)
```

with normalized fleet-balance polynomial features. Training proceeds backward. For each state and feasible reposition action, Monte Carlo demand samples estimate a Bellman target using the next-stage approximate value function. Ridge regression then fits the current value approximation.

This is deliberately an auditable value-function approximation benchmark rather than a deep-RL implementation.

## Evaluation

Primary metric: **expected profit gap to exact DP**.

Secondary metrics:

- lost demand;
- reposition volume;
- action agreement with exact DP;
- value-function error;
- policy latency;
- performance under larger fleet size and shifted directional demand.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m fleet_adp.experiment
```

## Repository layout

```text
src/fleet_adp/
  model.py
  exact_dp.py
  adp.py
  policies.py
  benchmark.py
  experiment.py
tests/
  test_model.py
  test_adp.py
configs/
  experiment.json
docs/
  experimental_protocol.md
.github/workflows/
  ci.yml
```

## Scope boundary

This repository studies a compact closed-fleet inventory system. Multi-station networks, travel-time state, reservations, pricing and deep reinforcement learning are separate extensions rather than silent scope expansion.

## License

MIT
