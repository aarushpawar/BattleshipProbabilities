#!/usr/bin/env python3
"""
Battleship Statistics Simulator — AP Stats Final Project
Usage:
    python simulator/main.py              # 1000 games per combination
    python simulator/main.py --games 50   # quick smoke test (~30 s)
"""
import argparse
import pickle
import sys
from pathlib import Path

# When run as  python simulator/main.py  the script dir is added to sys.path
# automatically, so sibling imports work without a package.
sys.path.insert(0, str(Path(__file__).parent))

from simulate import run_simulations, compute_summary
from visualize import generate_all_plots
from guide import print_guide


def main() -> None:
    parser = argparse.ArgumentParser(description="Battleship Statistics Simulator")
    parser.add_argument(
        "--games", type=int, default=1000,
        help="Games per strategy combination (default 1000; use 50 for a quick test)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    total = args.games * 50  # 5 placement × 5 shooting × 2 rules
    print(f"\nRunning {args.games} games × 50 combinations = {total:,} total games …\n")

    df, heatmaps = run_simulations(n_games=args.games, base_seed=args.seed)

    results_path = out / "results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nSaved {results_path}  ({len(df):,} rows)")

    summary = compute_summary(df)
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}  ({len(summary):,} rows)")

    heatmaps_path = out / "heatmaps.pkl"
    with open(heatmaps_path, "wb") as f:
        pickle.dump(heatmaps, f)
    print(f"Saved {heatmaps_path}\n")

    generate_all_plots(df, summary, heatmaps, out / "plots")

    print("\n" + "=" * 78)
    print_guide()

    # Print a quick summary table to console
    print("\n── QUICK SUMMARY: Mean turns taken (all formations combined) ──\n")
    for rules in ("real_life", "online"):
        print(f"  {rules.replace('_',' ').upper()}")
        sub = summary[
            (summary["rules"] == rules)
            & (summary["placement_strategy"] == "all")
            & (summary["shooting_strategy"] != "all")
        ]
        for _, row in sub.iterrows():
            s = row["shooting_strategy"]
            print(
                f"    {s:14s}  x̄ = {row['mean']:5.1f}  "
                f"95% CI [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]"
            )
        print()


if __name__ == "__main__":
    main()
