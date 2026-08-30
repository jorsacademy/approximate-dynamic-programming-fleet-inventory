import numpy as np

from fleet_adp.exact_dp import poisson_support, solve_exact_dp
from fleet_adp.model import default_model


def test_transition_conserves_fleet():
    model = default_model()
    next_a, reward, lost = model.transition(6, 2, 4, 3)
    assert 0 <= next_a <= model.fleet_size
    assert isinstance(reward, float)
    assert lost >= 0


def test_poisson_support_is_normalized():
    support = poisson_support(4.2, max_demand=10)
    assert abs(sum(prob for _, prob in support) - 1.0) < 1e-10


def test_exact_dp_policy_is_feasible():
    model = default_model(fleet_size=10)
    solution = solve_exact_dp(model)
    assert solution.values.shape == (model.horizon + 1, model.fleet_size + 1)
    for t in range(model.horizon):
        for state in range(model.fleet_size + 1):
            assert int(solution.policy[t, state]) in model.actions(state)
    assert np.all(np.isfinite(solution.values))
