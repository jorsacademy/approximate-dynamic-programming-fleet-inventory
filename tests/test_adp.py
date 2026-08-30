import numpy as np

from fleet_adp.adp import adp_action, fit_linear_adp
from fleet_adp.benchmark import evaluate_model
from fleet_adp.model import default_model


def test_adp_weights_and_actions_are_finite_and_feasible():
    model = default_model(fleet_size=10)
    solution = fit_linear_adp(model, seed=2, samples_per_action=24)
    assert np.all(np.isfinite(solution.weights))
    for t in range(model.horizon):
        for state in [0, model.fleet_size // 2, model.fleet_size]:
            assert adp_action(model, solution, t, state) in model.actions(state)


def test_exact_policy_has_zero_gap_and_adp_gap_is_nonnegative():
    model = default_model(fleet_size=10)
    adp = fit_linear_adp(model, seed=0, samples_per_action=24)
    rows, rmse = evaluate_model("test", model, adp)
    by_method = {row.method: row for row in rows}
    assert abs(by_method["exact_dp"].profit_gap_to_exact) < 1e-8
    assert by_method["linear_adp"].profit_gap_to_exact >= -1e-8
    assert rmse >= 0.0
