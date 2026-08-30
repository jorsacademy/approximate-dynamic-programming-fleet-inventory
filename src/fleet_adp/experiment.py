from __future__ import annotations

from .benchmark import run_benchmark


def main() -> None:
    result = run_benchmark()
    print("value_rmse")
    for split, rmse in result["value_rmse"].items():
        print(f"{split},{rmse:.6f}")
    print("policy_results")
    for row in result["rows"]:
        print(
            f"{row.split},{row.method},profit={row.expected_profit:.6f},"
            f"gap={row.profit_gap_to_exact:.6f},agreement={row.action_agreement:.3f}"
        )


if __name__ == "__main__":
    main()
