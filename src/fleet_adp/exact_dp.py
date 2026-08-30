from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np

from .model import FleetModel


@dataclass(frozen=True)
class ExactDPSolution:
    values: np.ndarray
    policy: np.ndarray


def poisson_support(rate: float, max_demand: int = 10) -> list[tuple[int, float]]:
    if rate < 0.0:
        raise ValueError("rate must be nonnegative")
    probs = [exp(-rate) * rate**k / factorial(k) for k in range(max_demand)]
    tail = max(0.0, 1.0 - sum(probs))
    return [(k, float(p)) for k, p in enumerate(probs)] + [(max_demand, float(tail))]


def action_value(
    model: FleetModel,
    t: int,
    state_a: int,
    action: int,
    next_values: np.ndarray,
    *,
    max_demand: int = 10,
) -> float:
    total = 0.0
    for demand_ab, p_ab in poisson_support(model.demand_ab[t], max_demand):
        for demand_ba, p_ba in poisson_support(model.demand_ba[t], max_demand):
            next_a, reward, _ = model.transition(state_a, action, demand_ab, demand_ba)
            total += p_ab * p_ba * (reward + float(next_values[next_a]))
    return float(total)


def solve_exact_dp(model: FleetModel, *, max_demand: int = 10) -> ExactDPSolution:
    values = np.zeros((model.horizon + 1, model.fleet_size + 1), dtype=float)
    policy = np.zeros((model.horizon, model.fleet_size + 1), dtype=int)

    for t in range(model.horizon - 1, -1, -1):
        for state_a in range(model.fleet_size + 1):
            candidates = []
            for action in model.actions(state_a):
                q = action_value(
                    model,
                    t,
                    state_a,
                    action,
                    values[t + 1],
                    max_demand=max_demand,
                )
                candidates.append((q, -abs(action), -action, action))
            best = max(candidates)
            values[t, state_a] = best[0]
            policy[t, state_a] = best[3]
    return ExactDPSolution(values=values, policy=policy)
