# Final Research Protocol

## Objective

This repository evaluates whether a compact approximate value function can capture enough future fleet value to improve dynamic repositioning decisions relative to myopic policies while remaining computationally light.

The benchmark is intentionally separate from deep reinforcement learning. The approximation is explicit, low-dimensional and auditable.

## Training

Linear value functions are fitted backward on the base fleet model with normalized state features. Three independent ADP training seeds are retained. Each action value during fitting uses Monte Carlo demand samples and the next-stage approximate value function.

No final environment seed is used for fitting or model selection.

## Final evaluation scenarios

Four frozen scenarios are evaluated:

1. `nominal_final`: training-scale fleet and demand;
2. `fleet_size_ood`: larger closed fleet with the learned normalized value function transferred unchanged;
3. `directional_demand_ood`: stronger A-to-B directional demand imbalance;
4. `high_demand_ood`: uniformly higher demand intensity.

The exact finite-horizon dynamic program is recomputed for each evaluation model and remains the oracle reference.

## Common random numbers

Operational simulation uses identical exogenous Poisson demand realizations for every policy under a given environment seed. Demand draws depend only on scenario, time profile and environment seed, not on actions. This creates paired policy comparisons with substantially lower simulation noise.

## Model-seed aggregation

The three ADP training replicates are not treated as independent test observations. For every scenario and environment seed, ADP profit, lost demand, reposition volume and latency are first averaged across training seeds. Statistical inference is then performed across independent environment seeds.

This prevents pseudo-replication of training randomness.

## Metrics

Primary operational metric: total finite-horizon operating profit.

Secondary metrics:

- lost demand;
- reposition volume;
- action-selection latency;
- exact expected-profit gap;
- action agreement with exact DP;
- value-function RMSE.

ADP profit is paired against `demand_balance` and `exact_dp`. Reports include mean paired profit difference, paired 95% bootstrap confidence interval, win rate and exact two-sided sign-test p-value.

## Interpretation rules

ADP does not earn promotion merely by approximating the exact value function closely. It must improve the operating profit/reliability trade-off relative to the simple demand-balance heuristic at acceptable latency.

If demand balance performs as well as ADP, the simpler heuristic is preferred. If exact DP remains practical at the tested scale, exact DP remains the quality reference. OOD degradation must be reported rather than hidden by retraining on shifted evaluation models.

## Scope boundary

The state is one-dimensional because the two-station closed fleet is fully determined by inventory at station A. Multi-station networks, travel-time pipelines, reservations, pricing, stochastic travel times and deep RL are separate research extensions.
