from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adp import ADPSolution, adp_action, fit_linear_adp, state_features
from .exact_dp import ExactDPSolution, action_value, solve_exact_dp
from .model import FleetModel, default_model
from .policies import demand_balance_action, no_reposition_action


@dataclass(frozen=True)
class BenchmarkRow:
    split: str
    method: str
    expected_profit: float
    profit_gap_to_exact: float
    action_agreement: float


def evaluate_policy_exactly(
    model: FleetModel,
    policy_fn,
    *,
    max_demand: int = 10,
) -> np.ndarray:
    values = np.zeros((model.horizon + 1, model.fleet_size + 1), dtype=float)
    for t in range(model.horizon - 1, -1, -1):
        for state_a in range(model.fleet_size + 1):
            action = int(policy_fn(model, t, state_a))
            values[t, state_a] = action_value(
                model,
                t,
                state_a,
                action,
                values[t + 1],
                max_demand=max_demand,
            )
    return values


def _agreement(model: FleetModel, exact: ExactDPSolution, policy_fn) -> float:
    matches = []
    for t in range(model.horizon):
        for state_a in range(model.fleet_size + 1):
            matches.append(int(policy_fn(model, t, state_a)) == int(exact.policy[t, state_a]))
    return float(np.mean(matches))


def _value_rmse(model: FleetModel, exact: ExactDPSolution, adp: ADPSolution) -> float:
    errors = []
    for t in range(model.horizon):
        for state_a in range(model.fleet_size + 1):
            approx = float(adp.weights[t] @ state_features(state_a, model.fleet_size))
            errors.append(approx - exact.values[t, state_a])
    return float(np.sqrt(np.mean(np.square(errors))))


def evaluate_model(
    split: str,
    model: FleetModel,
    adp: ADPSolution,
) -> tuple[list[BenchmarkRow], float]:
    exact = solve_exact_dp(model)
    initial_state = model.fleet_size // 2
    exact_profit = float(exact.values[0, initial_state])

    policies = {
        "no_reposition": no_reposition_action,
        "demand_balance": demand_balance_action,
        "linear_adp": lambda m, t, s: adp_action(m, adp, t, s),
        "exact_dp": lambda m, t, s: int(exact.policy[t, s]),
    }
    rows = []
    for method, policy in policies.items():
        values = evaluate_policy_exactly(model, policy)
        profit = float(values[0, initial_state])
        rows.append(
            BenchmarkRow(
                split=split,
                method=method,
                expected_profit=profit,
                profit_gap_to_exact=exact_profit - profit,
                action_agreement=_agreement(model, exact, policy),
            )
        )
    return rows, _value_rmse(model, exact, adp)


def run_benchmark() -> dict[str, object]:
    train_model = default_model(fleet_size=12)
    adp = fit_linear_adp(train_model, seed=0, samples_per_action=96)

    cases = {
        "in_distribution": train_model,
        "fleet_size_shift": default_model(fleet_size=18),
        "directional_demand_shift": default_model(fleet_size=12, directional_shift=0.20),
    }
    rows: list[BenchmarkRow] = []
    value_rmse = {}
    for split, model in cases.items():
        split_rows, rmse = evaluate_model(split, model, adp)
        rows.extend(split_rows)
        value_rmse[split] = rmse
    return {"adp": adp, "rows": rows, "value_rmse": value_rmse}
