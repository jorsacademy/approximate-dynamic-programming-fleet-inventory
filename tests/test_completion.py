import numpy as np

from fleet_adp.final_campaign import paired_report, run_final_campaign, scenario_models
from fleet_adp.model import default_model
from fleet_adp.policies import demand_balance_action
from fleet_adp.simulation import simulate_policy
from fleet_adp.statistics import paired_bootstrap_profit_difference


def test_common_random_number_simulation_is_reproducible():
    model = default_model(fleet_size=10)
    first = simulate_policy(
        model,
        demand_balance_action,
        method="demand_balance",
        scenario="test",
        environment_seed=7,
    )
    second = simulate_policy(
        model,
        demand_balance_action,
        method="demand_balance",
        scenario="test",
        environment_seed=7,
    )
    assert first.total_profit == second.total_profit
    assert first.lost_demand == second.lost_demand
    assert first.reposition_volume == second.reposition_volume


def test_final_campaign_aggregates_training_seeds_by_environment_seed():
    rows = run_final_campaign(
        environment_seeds=[1000, 1001],
        adp_seeds=(0, 1),
        samples_per_action=8,
    )
    expected_scenarios = set(scenario_models())
    assert {row.scenario for row in rows} == expected_scenarios
    assert {row.method for row in rows} == {
        "no_reposition",
        "demand_balance",
        "linear_adp",
        "exact_dp",
    }
    for scenario in expected_scenarios:
        adp_rows = [r for r in rows if r.scenario == scenario and r.method == "linear_adp"]
        assert len(adp_rows) == 2
        assert all(row.model_seed is None for row in adp_rows)
        assert {row.environment_seed for row in adp_rows} == {1000, 1001}


def test_paired_report_uses_environment_seed_as_unit_of_inference():
    rows = run_final_campaign(
        environment_seeds=[1100, 1101, 1102],
        adp_seeds=(0,),
        samples_per_action=8,
    )
    report = paired_report(
        rows,
        scenario="nominal_final",
        candidate="linear_adp",
        reference="demand_balance",
    )
    assert report["n"] == 3
    assert 0.0 <= report["win_rate"] <= 1.0
    assert 0.0 <= report["sign_test_pvalue"] <= 1.0


def test_paired_bootstrap_profit_difference_is_reproducible():
    candidate = np.array([10.0, 12.0, 11.5, 9.5])
    reference = np.array([9.0, 10.0, 11.0, 9.0])
    first = paired_bootstrap_profit_difference(candidate, reference, seed=4)
    second = paired_bootstrap_profit_difference(candidate, reference, seed=4)
    assert first == second
    assert first[0] > 0.0
