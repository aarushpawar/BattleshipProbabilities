GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           BATTLESHIP STATISTICS — AP STATS TESTING GUIDE                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO READ YOUR DATA
─────────────────────────────────────────────────────────────────────────────
The simulation ran N games for every combination of:
  5 formation strategies × 5 shooting strategies × 2 rule variants
  = 50 unique combinations

  output/results.csv   — one row per game
  output/summary.csv   — summary statistics for every group

The data forms a 5 × 5 TABLE per rule variant:

         │  random  parity  hunt_target  probability  edge_first
─────────┼──────────────────────────────────────────────────────
 random  │  [...]   [...]    [...]         [...]        [...]
 edge    │  [...]   [...]    [...]         [...]        [...]
 corner  │  [...]   [...]    [...]         [...]        [...]
 center  │  [...]   [...]    [...]         [...]        [...]
 spread  │  [...]   [...]    [...]         [...]        [...]

Each cell contains N values of shots_taken.

You can "flatten" the table in two ways:
  ▸ Fix a COLUMN (shooting strategy), compare rows
      → Which formation is hardest for THIS shooter?
  ▸ Fix a ROW (formation), compare columns
      → Which shooter beats THIS formation fastest?

summary.csv rows where placement_strategy == "all" are the column totals.
summary.csv rows where shooting_strategy  == "all" are the row totals.

─────────────────────────────────────────────────────────────────────────────
TEST 1 — Two-Sample t-Test: Best Shooter vs. Random (Fixed Formation)
─────────────────────────────────────────────────────────────────────────────
Question  Does the PROBABILITY shooter need significantly fewer turns than the
          RANDOM shooter when facing a RANDOM formation?

Filter    results.csv → placement_strategy == "random"
  Group 1 (n = N): shooting_strategy == "probability"  → shots_taken values
  Group 2 (n = N): shooting_strategy == "random"        → shots_taken values

Test      Two-sample t-test for a difference in means (μ₁ − μ₂)
  H₀:  μ_probability − μ_random = 0
  Hₐ:  μ_probability − μ_random < 0   (one-tailed)

Conditions to verify before running the test:
  ✓ Random       — each game is an independent simulation
  ✓ Large sample — n ≫ 30; by the CLT, x̄ is approximately Normal
  ✓ Independent  — the two groups do not overlap

What to report
  x̄₁, x̄₂, s₁, s₂, t-statistic, degrees of freedom, p-value,
  95 % CI for μ₁ − μ₂, conclusion written in context.

Repeat for every other formation (row) to see if the result is consistent.

─────────────────────────────────────────────────────────────────────────────
TEST 2 — Two-Sample t-Test: Real-Life Rules vs. Online Rules
─────────────────────────────────────────────────────────────────────────────
Question  Using the HUNT-AND-TARGET shooter, do online rules (no ships may
          touch) lead to a different mean number of turns than real-life rules?

Filter    results.csv → shooting_strategy == "hunt_target"
  Group 1 (n = 5N): rules == "real_life"   → shots_taken
  Group 2 (n = 5N): rules == "online"      → shots_taken
  (Each group pools all 5 formation strategies for maximum sample size.)

Test      Two-sample t-test for μ_real_life − μ_online
  H₀:  μ_real_life − μ_online = 0
  Hₐ:  μ_real_life − μ_online ≠ 0   (two-tailed — direction is unclear)

Why interesting  Under online rules, sinking a ship reveals that ALL
  adjacent cells are guaranteed empty.  The test quantifies whether that
  extra information meaningfully changes the number of turns needed.

What to report
  Both x̄ values, t-statistic, p-value, 95 % CI for the difference, conclusion.

─────────────────────────────────────────────────────────────────────────────
TEST 3 — Chi-Squared Goodness-of-Fit: Is Random Placement Truly Uniform?
─────────────────────────────────────────────────────────────────────────────
Question  Under the RANDOM placement strategy, are ships equally likely to
          occupy any of the 100 cells, or do edge effects create a non-uniform
          distribution?

Data      The aggregated placement heatmap for placement_strategy == "random"
          is in output/plots/placement_heatmaps_real_life.png (first panel).
          Load the underlying counts from the heatmaps .pkl file, or compute
          them from results.csv if you add a per-cell column.

          Total ship-cell occupations across N games:
            Fleet size = 5+4+3+3+2 = 17 cells per game
            Total = 17 × N   (spread across 100 cells)
          Expected count per cell (if uniform) = 17N / 100 = 0.17N

Test      χ² Goodness-of-Fit
  H₀:  All 100 cells are equally likely to be occupied (p = 1/100 each)
  Hₐ:  Some cells are more likely than others
  df = 100 − 1 = 99
  Condition: all expected counts ≥ 5  ✓  (true for N ≥ 30)

What to report
  χ² statistic, p-value, which cells deviate most (compare to the heatmap),
  conclusion.

─────────────────────────────────────────────────────────────────────────────
TEST 4 — Two-Proportion z-Test: Quick-Win Rate
─────────────────────────────────────────────────────────────────────────────
Question  Against the SPREAD formation, is the PROBABILITY shooter more
          likely to win in 50 turns or fewer than the RANDOM shooter?

Filter    results.csv → placement_strategy == "spread"
  Group 1 (n = N): shooting_strategy == "probability"
  Group 2 (n = N): shooting_strategy == "random"
  Create: quick_win = 1 if shots_taken ≤ 50, else 0
  Compute p̂₁ = proportion of quick wins in Group 1
          p̂₂ = proportion of quick wins in Group 2

Test      Two-proportion z-test
  H₀:  p_probability − p_random = 0
  Hₐ:  p_probability − p_random > 0   (one-tailed)

Conditions
  ✓ Random (independent simulations)
  ✓ 10 % condition (not sampling from a finite pool)
  ✓ Verify n·p̂ ≥ 10 and n·(1 − p̂) ≥ 10 for both groups with actual values

What to report
  p̂₁, p̂₂, pooled p̂, z-statistic, p-value, 95 % CI for p₁ − p₂, conclusion.

─────────────────────────────────────────────────────────────────────────────
TEST 5 — Linear Regression: Does Formation Spread Predict Turns Needed?
─────────────────────────────────────────────────────────────────────────────
Question  Is there a linear relationship between how spread out a formation is
          (formation_spread) and how many turns the RANDOM shooter needs to win?

Filter    results.csv → shooting_strategy == "random"
  x = formation_spread  (mean pairwise Manhattan distance between all ship cells)
  y = shots_taken
  Scatterplot → output/plots/regression_spread_vs_shots_real_life.png

Test      Linear regression t-test for slope β
  H₀:  β = 0  (formation spread does not linearly predict turns)
  Hₐ:  β > 0  (more spread → more turns needed; one-tailed)

Conditions (check all four plots):
  • Scatterplot — does the pattern appear linear?
  • Residual plot — residuals randomly scattered around 0?
    (checks independence + equal spread / homoscedasticity)
  • Normal probability plot of residuals — approximately a straight line?

What to report
  LSRL equation  ŷ = a + bx  (with units on both axes),
  r (correlation coefficient),
  r² with interpretation ("__ % of the variation in turns needed is
    explained by the linear relationship with formation spread"),
  t-statistic for slope, p-value, residual plot, conclusion.

─────────────────────────────────────────────────────────────────────────────
QUICK-REFERENCE TABLE
─────────────────────────────────────────────────────────────────────────────
 #   Question                         Filter CSV to…                  Test
 1   Best shooter vs. random?         placement_strategy=="random"    Two-sample t-test
 2   Rules change game length?        shooting_strategy=="hunt_target" Two-sample t-test
 3   Is random placement uniform?     placement_strategy=="random"    χ² Goodness-of-Fit
 4   Quick-win rate by strategy?      placement_strategy=="spread"    Two-proportion z-test
 5   Does spread predict turns?       shooting_strategy=="random"     Linear regression
─────────────────────────────────────────────────────────────────────────────
"""


def print_guide() -> None:
    print(GUIDE)
