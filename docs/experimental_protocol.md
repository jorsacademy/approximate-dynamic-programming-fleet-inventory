# Experimental Protocol

## Decision process

The system is a two-station closed fleet. State is the number of vehicles at station A; station B inventory follows from fleet conservation. Repositioning occurs before stochastic directional rental demand.

## Exact reference

The exact finite-horizon dynamic program enumerates every feasible reposition action and a truncated Poisson support whose tail probability is aggregated into the final support point. The resulting policy/value function is the reference for expected-profit gap and action agreement.

## ADP training

The linear ADP model is trained only on the base fleet/demand model. At each time and state, Monte Carlo Bellman targets are formed from feasible actions and the next-stage approximate value function. Ridge regression fits normalized polynomial state features.

## Transfer evaluation

The fitted ADP weights are frozen before evaluation. The same weights are evaluated on:

1. the training distribution;
2. a larger fleet-size model;
3. a directional-demand-shift model.

No retraining is performed on the shifted cases.

## Metrics

Primary: expected profit gap to exact DP from a balanced initial fleet state.

Secondary: state-action agreement with exact DP and value-function RMSE.

## Acceptance rules

1. Every reposition action must respect available vehicles and the reposition limit.
2. Fleet conservation must hold after every transition.
3. Exact DP must evaluate all feasible actions.
4. The exact policy must have zero gap to itself within numerical tolerance.
5. Shifted models must not be used to fit ADP weights.
6. A low value RMSE is not sufficient if the induced policy has poor expected profit.
7. Negative or null ADP results are retained.
