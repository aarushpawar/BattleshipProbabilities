from __future__ import annotations
import random
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from battleship import Board, GameResult, Rules, FLEET
from strategies import (
    PLACEMENT_STRATEGIES,
    SHOOTING_STRATEGIES,
    compute_formation_spread,
)

PLACEMENT_ORDER = ["random", "edge", "corner", "center", "spread"]
SHOOTING_ORDER = ["random", "parity", "hunt_target", "probability", "edge_first"]
RULE_VARIANTS = [Rules.REAL_LIFE, Rules.ONLINE]


def run_game(
    placement_strategy: str,
    shooting_strategy: str,
    rules: Rules,
    seed: int,
) -> GameResult:
    place_fn = PLACEMENT_STRATEGIES[placement_strategy]
    ShooterClass = SHOOTING_STRATEGIES[shooting_strategy]

    # Retry placement up to 20 times (needed for tight ONLINE + spread combos)
    for attempt in range(20):
        rng = random.Random(seed + attempt * 99991)
        board = Board(rules=rules)
        if place_fn(board, rng):
            break
    else:
        raise RuntimeError(
            f"Could not place ships after 20 attempts "
            f"({placement_strategy}, {rules.value})"
        )

    placement_heatmap = np.zeros((10, 10), dtype=np.float32)
    for ship in board.ships:
        for r, c in ship.cells:
            placement_heatmap[r, c] = 1.0

    formation_spread = compute_formation_spread(board.ships)
    shooter = ShooterClass()
    shot_heatmap = np.zeros((10, 10), dtype=np.float32)
    shots_taken = 0

    while not board.all_sunk() and shots_taken < 100:
        r, c = shooter.next_shot(board, rng)
        board.shoot(r, c)
        shot_heatmap[r, c] = 1.0
        shots_taken += 1

    return GameResult(
        shots_taken=shots_taken,
        shot_heatmap=shot_heatmap,
        placement_heatmap=placement_heatmap,
        placement_strategy=placement_strategy,
        shooting_strategy=shooting_strategy,
        rules=rules.value,
        formation_spread=formation_spread,
    )


def run_simulations(
    n_games: int = 1000,
    base_seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    heatmaps: dict = {}

    total_combos = len(PLACEMENT_ORDER) * len(SHOOTING_ORDER) * len(RULE_VARIANTS)
    combo_idx = 0

    for rules in RULE_VARIANTS:
        for placement in PLACEMENT_ORDER:
            for shooting in SHOOTING_ORDER:
                combo_idx += 1
                agg_shot = np.zeros((10, 10), dtype=np.float64)
                agg_place = np.zeros((10, 10), dtype=np.float64)

                for game_id in range(n_games):
                    seed = base_seed * 100_000 + combo_idx * 10_000 + game_id
                    result = run_game(placement, shooting, rules, seed)

                    records.append(
                        {
                            "rules": result.rules,
                            "placement_strategy": result.placement_strategy,
                            "shooting_strategy": result.shooting_strategy,
                            "shots_taken": result.shots_taken,
                            "formation_spread": round(result.formation_spread, 4),
                            "game_id": game_id,
                            "seed": seed,
                        }
                    )
                    agg_shot += result.shot_heatmap
                    agg_place += result.placement_heatmap

                heatmaps[(rules.value, placement, shooting)] = {
                    "shot": agg_shot / n_games,
                    "placement": agg_place / n_games,
                }

                if verbose:
                    pct = 100 * combo_idx / total_combos
                    print(
                        f"  [{pct:5.1f}%] {rules.value:10s} | {placement:8s} | {shooting:12s}"
                    )

    return pd.DataFrame(records), heatmaps


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add_row(subset: pd.DataFrame, rules: str, placement: str, shooting: str) -> None:
        if subset.empty:
            return
        x = subset["shots_taken"].values.astype(float)
        n = len(x)
        mean = x.mean()
        std = x.std(ddof=1)
        sem = std / np.sqrt(n)
        ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        rows.append(
            {
                "rules": rules,
                "placement_strategy": placement,
                "shooting_strategy": shooting,
                "n": n,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": int(x.min()),
                "q1": round(float(np.percentile(x, 25)), 2),
                "median": round(float(np.median(x)), 2),
                "q3": round(float(np.percentile(x, 75)), 2),
                "max": int(x.max()),
                "ci_lower": round(ci[0], 2),
                "ci_upper": round(ci[1], 2),
            }
        )

    for rules in df["rules"].unique():
        rdf = df[df["rules"] == rules]

        # All 25 cells
        for placement in PLACEMENT_ORDER:
            for shooting in SHOOTING_ORDER:
                sub = rdf[
                    (rdf["placement_strategy"] == placement)
                    & (rdf["shooting_strategy"] == shooting)
                ]
                add_row(sub, rules, placement, shooting)

        # Row marginals: one formation vs. all shooters
        for placement in PLACEMENT_ORDER:
            add_row(rdf[rdf["placement_strategy"] == placement], rules, placement, "all")

        # Column marginals: one shooter vs. all formations
        for shooting in SHOOTING_ORDER:
            add_row(rdf[rdf["shooting_strategy"] == shooting], rules, "all", shooting)

        # Grand total for this rule variant
        add_row(rdf, rules, "all", "all")

    return pd.DataFrame(rows)
