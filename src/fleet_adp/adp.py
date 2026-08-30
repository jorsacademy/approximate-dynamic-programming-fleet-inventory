from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FleetModel


@dataclass(frozen=True)
class ADPSolution:
    weights: np.ndarray
    training_seed: int
    samples_per_action: int


def state_features(state_a: int, fleet_size: int) -> np.ndarray:
    frac = state_a / fleet_size
    imbalance = frac - 0.5
    return np.array([1.0, frac, frac * frac, imbalance * imbalance], dtype=float)


def _sample_action_value(
    model: FleetModel,
    weights_next: np.ndarray,
    *,
    t: int,
    state_a: int,
    action: int,
    seed: int,
    samples: int,
) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(samples):
        demand_ab = int(rng.poisson(model.demand_ab[t]))
        demand_ba = int(rng.poisson(model.demand_ba[t]))
        next_a, reward, _ = model.transition(state_a, action, demand_ab, demand_ba)
        total += reward + float(weights_next @ state_features(next_a, model.fleet_size))
    return total / samples


def fit_linear_adp(
    model: FleetModel,
    *,
    seed: int = 0,
    samples_per_action: int = 128,
    ridge: float = 1e-4,
) -> ADPSolution:
    feature_dim = 4
    weights = np.zeros((model.horizon + 1, feature_dim), dtype=float)
    feature_matrix = np.vstack(
        [state_features(state, model.fleet_size) for state in range(model.fleet_size + 1)]
    )

    for t in range(model.horizon - 1, -1, -1):
        targets = []
        for state_a in range(model.fleet_size + 1):
            values = []
            for action in model.actions(state_a):
                local_seed = seed * 1_000_003 + t * 10_007 + state_a * 101 + (action + 50)
                values.append(
                    _sample_action_value(
                        model,
                        weights[t + 1],
                        t=t,
                        state_a=state_a,
                        action=action,
                        seed=local_seed,
                        samples=samples_per_action,
                    )
                )
            targets.append(max(values))
        gram = feature_matrix.T @ feature_matrix + ridge * np.eye(feature_dim)
        rhs = feature_matrix.T @ np.asarray(targets, dtype=float)
        weights[t] = np.linalg.solve(gram, rhs)
    return ADPSolution(weights=weights, training_seed=seed, samples_per_action=samples_per_action)


def adp_action(model: FleetModel, solution: ADPSolution, t: int, state_a: int) -> int:
    candidates = []
    for action in model.actions(state_a):
        local_seed = (
            solution.training_seed * 2_000_003 + t * 20_011 + state_a * 211 + (action + 50)
        )
        value = _sample_action_value(
            model,
            solution.weights[t + 1],
            t=t,
            state_a=state_a,
            action=action,
            seed=local_seed,
            samples=solution.samples_per_action,
        )
        candidates.append((value, -abs(action), -action, action))
    return max(candidates)[3]
