from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .model import FleetModel


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    scenario: str
    environment_seed: int
    model_seed: int | None
    total_profit: float
    lost_demand: float
    reposition_volume: float
    mean_decision_latency_ms: float


def simulate_policy(
    model: FleetModel,
    policy_fn,
    *,
    method: str,
    scenario: str,
    environment_seed: int,
    model_seed: int | None = None,
    initial_state: int | None = None,
) -> EpisodeResult:
    """Simulate one policy with exogenous demand common across policies.

    Demand draws depend only on ``environment_seed`` and time, never on the chosen
    action. Reusing a seed therefore implements common random numbers for paired
    policy comparisons.
    """
    state_a = model.fleet_size // 2 if initial_state is None else int(initial_state)
    if not 0 <= state_a <= model.fleet_size:
        raise ValueError("initial_state outside fleet range")

    rng = np.random.default_rng(environment_seed)
    profit = 0.0
    lost_total = 0
    reposition_total = 0
    latencies = []

    for t in range(model.horizon):
        start = perf_counter()
        action = int(policy_fn(model, t, state_a))
        latencies.append((perf_counter() - start) * 1000.0)
        if action not in model.actions(state_a):
            raise ValueError("policy returned infeasible reposition action")
        demand_ab = int(rng.poisson(model.demand_ab[t]))
        demand_ba = int(rng.poisson(model.demand_ba[t]))
        state_a, reward, lost = model.transition(state_a, action, demand_ab, demand_ba)
        profit += reward
        lost_total += lost
        reposition_total += abs(action)

    return EpisodeResult(
        method=method,
        scenario=scenario,
        environment_seed=environment_seed,
        model_seed=model_seed,
        total_profit=float(profit),
        lost_demand=float(lost_total),
        reposition_volume=float(reposition_total),
        mean_decision_latency_ms=float(np.mean(latencies)),
    )
