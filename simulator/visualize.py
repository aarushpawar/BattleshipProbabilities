from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from simulate import PLACEMENT_ORDER, SHOOTING_ORDER

_COL_LABELS = [s.replace("_", "\n").title() for s in SHOOTING_ORDER]
_ROW_LABELS = [s.replace("_", "\n").title() for s in PLACEMENT_ORDER]
_RULES_TITLE = {"real_life": "Real-Life Rules", "online": "Online Rules"}

sns.set_theme(style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path.name}")


# ---------------------------------------------------------------------------
# 1. Placement heatmaps (one per rule variant)
# ---------------------------------------------------------------------------

def plot_placement_heatmaps(heatmaps: dict, output_dir: Path, rules: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"Ship Placement Frequency — {_RULES_TITLE[rules]}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    col_letters = list("ABCDEFGHIJ")
    for i, placement in enumerate(PLACEMENT_ORDER):
        hm = np.mean(
            [heatmaps[(rules, placement, s)]["placement"] for s in SHOOTING_ORDER],
            axis=0,
        )
        ax = axes[i]
        im = ax.imshow(hm, cmap="viridis", vmin=0, vmax=hm.max() or 1)
        ax.set_title(placement.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(col_letters, fontsize=7)
        ax.set_yticklabels(range(1, 11), fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, output_dir / f"placement_heatmaps_{rules}.png")


# ---------------------------------------------------------------------------
# 2. Placement heatmaps side-by-side (both rule variants)
# ---------------------------------------------------------------------------

def plot_placement_comparison(heatmaps: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        "Ship Placement Heatmaps: Real-Life Rules vs. Online Rules",
        fontsize=14, fontweight="bold",
    )
    col_letters = list("ABCDEFGHIJ")
    for j, rules in enumerate(("real_life", "online")):
        for i, placement in enumerate(PLACEMENT_ORDER):
            hm = np.mean(
                [heatmaps[(rules, placement, s)]["placement"] for s in SHOOTING_ORDER],
                axis=0,
            )
            ax = axes[j, i]
            im = ax.imshow(hm, cmap="viridis", vmin=0, vmax=hm.max() or 1)
            if j == 0:
                ax.set_title(placement.replace("_", " ").title(), fontsize=11)
            if i == 0:
                ax.set_ylabel(_RULES_TITLE[rules], fontsize=10, labelpad=6)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels(col_letters, fontsize=6)
            ax.set_yticklabels(range(1, 11), fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, output_dir / "placement_heatmaps_rules_compare.png")


# ---------------------------------------------------------------------------
# 3. Shooting heatmaps
# ---------------------------------------------------------------------------

def plot_shooting_heatmaps(heatmaps: dict, output_dir: Path, rules: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"Shot Targeting Frequency — {_RULES_TITLE[rules]}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    col_letters = list("ABCDEFGHIJ")
    for i, shooting in enumerate(SHOOTING_ORDER):
        hm = np.mean(
            [heatmaps[(rules, p, shooting)]["shot"] for p in PLACEMENT_ORDER],
            axis=0,
        )
        ax = axes[i]
        im = ax.imshow(hm, cmap="hot", vmin=0, vmax=hm.max() or 1)
        ax.set_title(shooting.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(col_letters, fontsize=7)
        ax.set_yticklabels(range(1, 11), fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, output_dir / f"shooting_heatmaps_{rules}.png")


# ---------------------------------------------------------------------------
# 4. Box plots: shots taken by shooting strategy
# ---------------------------------------------------------------------------

def plot_shots_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Distribution of Turns Needed to Win by Shooting Strategy",
        fontsize=14, fontweight="bold",
    )
    palette = sns.color_palette("Set2", 5)
    for j, rules in enumerate(("real_life", "online")):
        sub = df[df["rules"] == rules]
        ax = axes[j]
        sns.boxplot(
            data=sub,
            x="shooting_strategy",
            y="shots_taken",
            hue="shooting_strategy",
            order=SHOOTING_ORDER,
            hue_order=SHOOTING_ORDER,
            palette=palette,
            legend=False,
            ax=ax,
            linewidth=1.2,
        )
        ax.set_title(_RULES_TITLE[rules], fontsize=12)
        ax.set_xlabel("Shooting Strategy", fontsize=11)
        ax.set_ylabel("Turns Taken" if j == 0 else "", fontsize=11)
        ax.set_xticks(range(len(SHOOTING_ORDER)))
        ax.set_xticklabels(_COL_LABELS, fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "shots_boxplot.png")


# ---------------------------------------------------------------------------
# 5. Bar chart with 95 % CI: mean shots per shooting strategy
# ---------------------------------------------------------------------------

def plot_shots_barplot_ci(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[
        (summary["placement_strategy"] == "all") & (summary["shooting_strategy"] != "all")
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.35
    palette = sns.color_palette("Set2", 2)

    for j, rules in enumerate(("real_life", "online")):
        sub = data[data["rules"] == rules].set_index("shooting_strategy")
        means = [sub.loc[s, "mean"] for s in SHOOTING_ORDER]
        lo = [sub.loc[s, "mean"] - sub.loc[s, "ci_lower"] for s in SHOOTING_ORDER]
        hi = [sub.loc[s, "ci_upper"] - sub.loc[s, "mean"] for s in SHOOTING_ORDER]
        ax.bar(
            x + j * width, means, width,
            label=_RULES_TITLE[rules],
            color=palette[j], alpha=0.87,
            yerr=[lo, hi], capsize=5,
            error_kw={"elinewidth": 1.5},
        )

    ax.set_xlabel("Shooting Strategy", fontsize=12)
    ax.set_ylabel("Mean Turns Taken (x̄)", fontsize=12)
    ax.set_title(
        "Mean Turns to Win by Shooting Strategy\nwith 95 % Confidence Intervals",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(_COL_LABELS, fontsize=10)
    ax.legend(title="Rules")
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / "shots_barplot_ci.png")


# ---------------------------------------------------------------------------
# 6. 5×5 strategy matrix heatmap
# ---------------------------------------------------------------------------

def plot_strategy_matrix(summary: pd.DataFrame, output_dir: Path, rules: str) -> None:
    sub = summary[
        (summary["rules"] == rules)
        & (summary["placement_strategy"] != "all")
        & (summary["shooting_strategy"] != "all")
    ]
    matrix = sub.pivot(
        index="placement_strategy", columns="shooting_strategy", values="mean"
    ).reindex(index=PLACEMENT_ORDER, columns=SHOOTING_ORDER)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        matrix,
        annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, linecolor="lightgray",
        cbar_kws={"label": "Mean Turns Taken"},
        ax=ax,
    )
    ax.set_title(
        f"Mean Turns — Formation Strategy vs. Shooting Strategy\n{_RULES_TITLE[rules]}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Shooting Strategy", fontsize=12)
    ax.set_ylabel("Formation Strategy", fontsize=12)
    ax.set_xticklabels(_COL_LABELS, rotation=0, fontsize=10)
    ax.set_yticklabels(_ROW_LABELS, rotation=0, fontsize=10)
    fig.tight_layout()
    _save(fig, output_dir / f"strategy_matrix_heatmap_{rules}.png")


# ---------------------------------------------------------------------------
# 7. Best vs. worst formation violin plot
# ---------------------------------------------------------------------------

def plot_best_vs_worst_formation(
    df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, rules: str
) -> None:
    marginal = summary[
        (summary["rules"] == rules)
        & (summary["placement_strategy"] != "all")
        & (summary["shooting_strategy"] == "all")
    ]
    best = marginal.loc[marginal["mean"].idxmax(), "placement_strategy"]
    worst = marginal.loc[marginal["mean"].idxmin(), "placement_strategy"]

    sub = df[
        (df["rules"] == rules) & df["placement_strategy"].isin([best, worst])
    ].copy()

    palette = {best: "#e74c3c", worst: "#3498db"}
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=sub,
        x="shooting_strategy", y="shots_taken",
        hue="placement_strategy",
        order=SHOOTING_ORDER,
        hue_order=[best, worst],
        palette=palette,
        inner="box",
        ax=ax,
    )
    ax.set_title(
        f"Hardest vs. Easiest Formation to Defend — {_RULES_TITLE[rules]}\n"
        f"Hardest (most turns): {best.title()}   ·   Easiest (fewest turns): {worst.title()}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Shooting Strategy", fontsize=11)
    ax.set_ylabel("Turns Taken", fontsize=11)
    ax.set_xticks(range(len(SHOOTING_ORDER)))
    ax.set_xticklabels(_COL_LABELS, fontsize=9)
    ax.legend(title="Formation", loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / f"best_vs_worst_formation_{rules}.png")


# ---------------------------------------------------------------------------
# 8. Real-life vs. online bar chart
# ---------------------------------------------------------------------------

def plot_rules_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[
        (summary["placement_strategy"] == "all") & (summary["shooting_strategy"] != "all")
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.35
    palette = sns.color_palette("Set2", 2)

    for j, rules in enumerate(("real_life", "online")):
        sub = data[data["rules"] == rules].set_index("shooting_strategy")
        means = [sub.loc[s, "mean"] for s in SHOOTING_ORDER]
        lo = [sub.loc[s, "mean"] - sub.loc[s, "ci_lower"] for s in SHOOTING_ORDER]
        hi = [sub.loc[s, "ci_upper"] - sub.loc[s, "mean"] for s in SHOOTING_ORDER]
        ax.bar(
            x + j * width, means, width,
            label=_RULES_TITLE[rules],
            color=palette[j], alpha=0.87,
            yerr=[lo, hi], capsize=5,
        )

    ax.set_xlabel("Shooting Strategy", fontsize=12)
    ax.set_ylabel("Mean Turns Taken", fontsize=12)
    ax.set_title(
        "Real-Life Rules vs. Online Rules\nMean Turns per Shooting Strategy with 95 % CI",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(_COL_LABELS, fontsize=10)
    ax.legend(title="Rules Variant")
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / "real_life_vs_online.png")


# ---------------------------------------------------------------------------
# 9. Regression: formation spread vs. turns
# ---------------------------------------------------------------------------

def plot_regression(df: pd.DataFrame, output_dir: Path, rules: str) -> None:
    sub = df[(df["rules"] == rules) & (df["shooting_strategy"] == "random")]
    x = sub["formation_spread"].values
    y = sub["shots_taken"].values

    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    r2 = r ** 2
    fitted = slope * x + intercept
    residuals = y - fitted

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Linear Regression: Formation Spread vs. Turns Needed\n"
        f"(Random Shooter, {_RULES_TITLE[rules]})",
        fontsize=12, fontweight="bold",
    )

    # Scatterplot + LSRL
    ax = axes[0]
    ax.scatter(x, y, alpha=0.06, s=6, color="steelblue", rasterized=True)
    xl = np.linspace(x.min(), x.max(), 200)
    ax.plot(xl, slope * xl + intercept, "r-", linewidth=2,
            label=f"ŷ = {slope:.3f}x + {intercept:.1f}\nr = {r:.3f},  r² = {r2:.3f}\np = {p:.4e}")
    ax.set_xlabel("Formation Spread (mean pairwise Manhattan distance)", fontsize=11)
    ax.set_ylabel("Turns Taken", fontsize=11)
    ax.set_title("Scatterplot with LSRL", fontsize=11)
    ax.legend(fontsize=9)

    # Residual plot
    ax2 = axes[1]
    ax2.scatter(fitted, residuals, alpha=0.06, s=6, color="darkorange", rasterized=True)
    ax2.axhline(0, color="red", linewidth=1.5)
    ax2.set_xlabel("Fitted Values (ŷ)", fontsize=11)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.set_title("Residual Plot", fontsize=11)

    fig.tight_layout()
    _save(fig, output_dir / f"regression_spread_vs_shots_{rules}.png")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    heatmaps: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating plots …")

    for rules in ("real_life", "online"):
        plot_placement_heatmaps(heatmaps, output_dir, rules)
        plot_shooting_heatmaps(heatmaps, output_dir, rules)
        plot_strategy_matrix(summary, output_dir, rules)
        plot_best_vs_worst_formation(df, summary, output_dir, rules)
        plot_regression(df, output_dir, rules)

    plot_placement_comparison(heatmaps, output_dir)
    plot_shots_boxplot(df, output_dir)
    plot_shots_barplot_ci(summary, output_dir)
    plot_rules_comparison(summary, output_dir)

    print("All plots saved.")
