from __future__ import annotations

import numpy as np

from .model import FleetModel


def no_reposition_action(model: FleetModel, t: int, state_a: int) -> int:
    del model, t, state_a
    return 0


def demand_balance_action(model: FleetModel, t: int, state_a: int) -> int:
    total = model.demand_ab[t] + model.demand_ba[t]
    desired_a = model.fleet_size * model.demand_ab[t] / total
    raw = round(state_a - desired_a)
    feasible = list(model.actions(state_a))
    return int(min(feasible, key=lambda action: (abs(action - raw), abs(action), action)))


def normalized_action_histogram(actions: list[int], max_reposition: int) -> np.ndarray:
    support = np.arange(-max_reposition, max_reposition + 1)
    return np.array([np.mean(np.asarray(actions) == action) for action in support], dtype=float)
