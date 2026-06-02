from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Rules(Enum):
    REAL_LIFE = "real_life"
    ONLINE = "online"


FLEET = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]

TOTAL_SHIP_CELLS = sum(size for _, size in FLEET)  # 17


class Ship:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.cells: list[tuple[int, int]] = []
        self.hits: set[tuple[int, int]] = set()

    def place(self, cells: list[tuple[int, int]]) -> None:
        self.cells = cells
        self.hits = set()

    def is_sunk(self) -> bool:
        return len(self.hits) == self.size


class Board:
    """
    10x10 battleship board.
    Grid states: 0=unshot, 1=miss, 2=hit (partial), 3=sunk, 4=excluded
    State 4 (excluded) is used under ONLINE rules when a ship sinks — its
    orthogonal neighbors are guaranteed empty, so we mark them to inform
    future probability calculations.
    """

    def __init__(self, rules: Rules = Rules.REAL_LIFE):
        self.rules = rules
        self.grid = np.zeros((10, 10), dtype=np.int8)
        self.ships: list[Ship] = []

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def _orthogonal_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        result = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < 10 and 0 <= c < 10:
                result.append((r, c))
        return result

    def place_ship(self, ship: Ship, row: int, col: int, horizontal: bool) -> bool:
        cells = (
            [(row, col + i) for i in range(ship.size)]
            if horizontal
            else [(row + i, col) for i in range(ship.size)]
        )

        # Bounds check
        if any(not (0 <= r < 10 and 0 <= c < 10) for r, c in cells):
            return False

        # Overlap check
        placed: set[tuple[int, int]] = set()
        for s in self.ships:
            placed.update(s.cells)
        if any(cell in placed for cell in cells):
            return False

        # No-adjacency check (ONLINE rules)
        if self.rules == Rules.ONLINE:
            for cell in cells:
                for nb in self._orthogonal_neighbors(*cell):
                    if nb in placed:
                        return False

        ship.place(cells)
        self.ships.append(ship)
        return True

    # ------------------------------------------------------------------
    # Shooting
    # ------------------------------------------------------------------

    def shoot(self, row: int, col: int) -> str:
        state = self.grid[row, col]
        if state != 0:
            return "already_shot"

        for ship in self.ships:
            if (row, col) in ship.cells:
                ship.hits.add((row, col))
                if ship.is_sunk():
                    for r, c in ship.cells:
                        self.grid[r, c] = 3
                    if self.rules == Rules.ONLINE:
                        for r, c in ship.cells:
                            for nr, nc in self._orthogonal_neighbors(r, c):
                                if self.grid[nr, nc] == 0:
                                    self.grid[nr, nc] = 4
                    return "sunk"
                self.grid[row, col] = 2
                return "hit"

        self.grid[row, col] = 1
        return "miss"

    def all_sunk(self) -> bool:
        return all(s.is_sunk() for s in self.ships)

    # ------------------------------------------------------------------
    # Probability map  (port of JS calculateAllProbs)
    # ------------------------------------------------------------------

    def get_probability_map(self) -> np.ndarray:
        """
        For every remaining (un-sunk) ship, slide it horizontally and
        vertically over every board position.  A placement is valid if none
        of its cells are in state 1 (miss), 3 (sunk), or 4 (excluded).
        Count how often each cell is covered by a valid placement, then
        normalise to [0, 1].

        Under ONLINE rules the excluded cells (state 4) already encode the
        adjacency information from sunk ships, so no additional adjacency
        check is needed here — any unshot cell (state 0) is at least one
        step away from all excluded cells and therefore satisfies the
        no-adjacency constraint.

        Uses numpy cumsum vectorisation: O(ships × 10 array ops) per call.
        """
        prob_map = np.zeros((10, 10), dtype=np.float64)
        bad = np.isin(self.grid, (1, 3, 4)).astype(np.int32)

        for ship in self.ships:
            if ship.is_sunk():
                continue
            s = ship.size

            # --- horizontal ---
            row_cs = np.cumsum(bad, axis=1)           # (10, 10)
            right = row_cs[:, s - 1:]                 # (10, 10-s+1)
            left = np.zeros((10, 10 - s + 1), dtype=np.int32)
            if s > 1:
                left[:, 1:] = row_cs[:, : 10 - s]
            valid_h = (right - left == 0).astype(np.float64)  # (10, 10-s+1)

            padded = np.zeros((10, 10), dtype=np.float64)
            padded[:, : 10 - s + 1] = valid_h
            cum = np.cumsum(padded, axis=1)
            lc = np.zeros((10, 10), dtype=np.float64)
            lc[:, s:] = cum[:, :-s]
            prob_map += cum - lc

            # --- vertical (transpose trick) ---
            bad_t = bad.T
            col_cs = np.cumsum(bad_t, axis=1)
            right_v = col_cs[:, s - 1:]
            left_v = np.zeros((10, 10 - s + 1), dtype=np.int32)
            if s > 1:
                left_v[:, 1:] = col_cs[:, : 10 - s]
            valid_v = (right_v - left_v == 0).astype(np.float64)

            padded_v = np.zeros((10, 10), dtype=np.float64)
            padded_v[:, : 10 - s + 1] = valid_v
            cum_v = np.cumsum(padded_v, axis=1)
            lc_v = np.zeros((10, 10), dtype=np.float64)
            lc_v[:, s:] = cum_v[:, :-s]
            prob_map += (cum_v - lc_v).T

        mx = prob_map.max()
        if mx > 0:
            prob_map /= mx
        return prob_map


@dataclass
class GameResult:
    shots_taken: int
    shot_heatmap: np.ndarray
    placement_heatmap: np.ndarray
    placement_strategy: str
    shooting_strategy: str
    rules: str
    formation_spread: float
