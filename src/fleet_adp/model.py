from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FleetModel:
    fleet_size: int
    horizon: int
    max_reposition: int
    rental_revenue: float
    reposition_cost: float
    lost_penalty: float
    demand_ab: tuple[float, ...]
    demand_ba: tuple[float, ...]

    def actions(self, state_a: int) -> range:
        if not 0 <= state_a <= self.fleet_size:
            raise ValueError("invalid fleet state")
        lower = -min(self.max_reposition, self.fleet_size - state_a)
        upper = min(self.max_reposition, state_a)
        return range(lower, upper + 1)

    def transition(self, state_a: int, action: int, demand_ab: int, demand_ba: int) -> tuple[int, float, int]:
        if action not in self.actions(state_a):
            raise ValueError("infeasible reposition action")
        post_a = state_a - action
        post_b = self.fleet_size - post_a
        served_ab = min(post_a, demand_ab)
        served_ba = min(post_b, demand_ba)
        lost = (demand_ab - served_ab) + (demand_ba - served_ba)
        next_a = post_a - served_ab + served_ba
        reward = (
            self.rental_revenue * (served_ab + served_ba)
            - self.reposition_cost * abs(action)
            - self.lost_penalty * lost
        )
        return int(next_a), float(reward), int(lost)


def default_model(
    *,
    fleet_size: int = 12,
    horizon: int = 8,
    demand_scale: float = 1.0,
    directional_shift: float = 0.0,
) -> FleetModel:
    base_ab = np.array([3.0, 4.2, 5.0, 4.5, 3.5, 2.8, 3.4, 4.0], dtype=float)
    base_ba = np.array([4.4, 3.6, 2.8, 3.1, 4.2, 5.0, 4.5, 3.7], dtype=float)
    if horizon != len(base_ab):
        raise ValueError("default demand profile currently uses horizon=8")
    demand_ab = np.maximum(0.2, demand_scale * base_ab * (1.0 + directional_shift))
    demand_ba = np.maximum(0.2, demand_scale * base_ba * (1.0 - directional_shift))
    return FleetModel(
        fleet_size=fleet_size,
        horizon=horizon,
        max_reposition=max(1, round(fleet_size / 4)),
        rental_revenue=8.0,
        reposition_cost=1.5,
        lost_penalty=2.0,
        demand_ab=tuple(float(x) for x in demand_ab),
        demand_ba=tuple(float(x) for x in demand_ba),
    )
