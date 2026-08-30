from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .adp import ADPSolution, adp_action, fit_linear_adp
from .exact_dp import solve_exact_dp
from .model import FleetModel, default_model
from .policies import demand_balance_action, no_reposition_action
from .simulation import EpisodeResult, simulate_policy
from .statistics import paired_profit_report


def scenario_models() -> dict[str, FleetModel]:
    return {
        "nominal_final": default_model(fleet_size=12),
        "fleet_size_ood": default_model(fleet_size=18),
        "directional_demand_ood": default_model(fleet_size=12, directional_shift=0.25),
        "high_demand_ood": default_model(fleet_size=12, demand_scale=1.25),
    }


def train_adp_replicates(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    samples_per_action: int = 64,
) -> dict[int, ADPSolution]:
    train_model = default_model(fleet_size=12)
    return {
        seed: fit_linear_adp(
            train_model,
            seed=seed,
            samples_per_action=samples_per_action,
        )
        for seed in seeds
    }


def _aggregate_adp(rows: list[EpisodeResult]) -> list[EpisodeResult]:
    grouped: dict[tuple[str, int], list[EpisodeResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.environment_seed)].append(row)

    aggregated = []
    for (scenario, environment_seed), selected in sorted(grouped.items()):
        aggregated.append(
            EpisodeResult(
                method="linear_adp",
                scenario=scenario,
                environment_seed=environment_seed,
                model_seed=None,
                total_profit=float(np.mean([r.total_profit for r in selected])),
                lost_demand=int(round(np.mean([r.lost_demand for r in selected]))),
                reposition_volume=int(round(np.mean([r.reposition_volume for r in selected]))),
                mean_decision_latency_ms=float(
                    np.mean([r.mean_decision_latency_ms for r in selected])
                ),
            )
        )
    return aggregated


def run_final_campaign(
    *,
    environment_seeds: list[int] | None = None,
    adp_seeds: tuple[int, ...] = (0, 1, 2),
    samples_per_action: int = 64,
) -> list[EpisodeResult]:
    seeds = list(range(1000, 1020)) if environment_seeds is None else environment_seeds
    adp_models = train_adp_replicates(
        seeds=adp_seeds,
        samples_per_action=samples_per_action,
    )
    rows: list[EpisodeResult] = []
    adp_raw: list[EpisodeResult] = []

    for scenario, model in scenario_models().items():
        exact = solve_exact_dp(model)
        exact_policy = lambda m, t, s, solution=exact: int(solution.policy[t, s])
        for environment_seed in seeds:
            rows.append(
                simulate_policy(
                    model,
                    no_reposition_action,
                    method="no_reposition",
                    scenario=scenario,
                    environment_seed=environment_seed,
                )
            )
            rows.append(
                simulate_policy(
                    model,
                    demand_balance_action,
                    method="demand_balance",
                    scenario=scenario,
                    environment_seed=environment_seed,
                )
            )
            rows.append(
                simulate_policy(
                    model,
                    exact_policy,
                    method="exact_dp",
                    scenario=scenario,
                    environment_seed=environment_seed,
                )
            )
            for model_seed, solution in adp_models.items():
                policy = lambda m, t, s, sol=solution: adp_action(m, sol, t, s)
                adp_raw.append(
                    simulate_policy(
                        model,
                        policy,
                        method="linear_adp",
                        scenario=scenario,
                        environment_seed=environment_seed,
                        model_seed=model_seed,
                    )
                )

    rows.extend(_aggregate_adp(adp_raw))
    return rows


def summarize(rows: list[EpisodeResult]) -> list[dict[str, float | str]]:
    output = []
    keys = sorted({(r.scenario, r.method) for r in rows})
    for scenario, method in keys:
        selected = [r for r in rows if r.scenario == scenario and r.method == method]
        output.append(
            {
                "scenario": scenario,
                "method": method,
                "mean_profit": float(np.mean([r.total_profit for r in selected])),
                "mean_lost_demand": float(np.mean([r.lost_demand for r in selected])),
                "mean_reposition_volume": float(np.mean([r.reposition_volume for r in selected])),
                "mean_latency_ms": float(
                    np.mean([r.mean_decision_latency_ms for r in selected])
                ),
            }
        )
    return output


def paired_report(
    rows: list[EpisodeResult],
    *,
    scenario: str,
    candidate: str,
    reference: str,
) -> dict[str, float | int]:
    candidate_rows = sorted(
        [r for r in rows if r.scenario == scenario and r.method == candidate],
        key=lambda r: r.environment_seed,
    )
    reference_rows = sorted(
        [r for r in rows if r.scenario == scenario and r.method == reference],
        key=lambda r: r.environment_seed,
    )
    if [r.environment_seed for r in candidate_rows] != [r.environment_seed for r in reference_rows]:
        raise ValueError("paired policies must share identical environment seeds")
    return paired_profit_report(
        np.asarray([r.total_profit for r in candidate_rows]),
        np.asarray([r.total_profit for r in reference_rows]),
        seed=41,
    )


def main() -> None:
    rows = run_final_campaign()
    for row in summarize(rows):
        print(
            f"{row['scenario']},{row['method']},profit={row['mean_profit']:.3f},"
            f"lost={row['mean_lost_demand']:.3f},"
            f"reposition={row['mean_reposition_volume']:.3f},"
            f"latency_ms={row['mean_latency_ms']:.4f}"
        )
    for scenario in scenario_models():
        for reference in ["demand_balance", "exact_dp"]:
            report = paired_report(
                rows,
                scenario=scenario,
                candidate="linear_adp",
                reference=reference,
            )
            print(
                f"paired,{scenario},linear_adp-{reference},"
                f"mean_diff={report['mean_profit_difference']:.3f},"
                f"ci95=[{report['ci95_low']:.3f},{report['ci95_high']:.3f}],"
                f"win_rate={report['win_rate']:.3f},"
                f"p={report['sign_test_pvalue']:.4f}"
            )


if __name__ == "__main__":
    main()
