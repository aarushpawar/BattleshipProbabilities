from __future__ import annotations
import random
from battleship import Board, Ship, Rules, FLEET


# ===========================================================================
# Helper
# ===========================================================================

def compute_formation_spread(ships: list[Ship]) -> float:
    """Mean pairwise Manhattan distance between every pair of ship cells."""
    cells = [cell for ship in ships for cell in ship.cells]
    if len(cells) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            total += abs(cells[i][0] - cells[j][0]) + abs(cells[i][1] - cells[j][1])
            count += 1
    return total / count


# ===========================================================================
# Placement strategies
# ===========================================================================

def _try_place(board: Board, ship: Ship, rng: random.Random, attempts: int = 1000) -> bool:
    for _ in range(attempts):
        r = rng.randint(0, 9)
        c = rng.randint(0, 9)
        h = rng.choice((True, False))
        if board.place_ship(ship, r, c, h):
            return True
    return False


def place_random(board: Board, rng: random.Random) -> bool:
    for name, size in FLEET:
        if not _try_place(board, Ship(name, size), rng):
            return False
    return True


def _edge_weight(r: int, c: int) -> float:
    return float(max(1, 5 - min(r, 9 - r, c, 9 - c)))


def _corner_weight(r: int, c: int) -> float:
    d = min(r + c, r + (9 - c), (9 - r) + c, (9 - r) + (9 - c))
    return float(max(1, 9 - d))


def _center_weight(r: int, c: int) -> float:
    d = max(0.0, abs(r - 4.5) - 1.0) + max(0.0, abs(c - 4.5) - 1.0)
    return float(max(1, 7 - int(d)))


def _weighted_place(board: Board, rng: random.Random, wfn) -> bool:
    candidates = [(r, c, h) for r in range(10) for c in range(10) for h in (True, False)]
    weights = [wfn(r, c) for r, c, _ in candidates]
    for name, size in FLEET:
        ship = Ship(name, size)
        placed = False
        for _ in range(2000):
            r, c, h = rng.choices(candidates, weights=weights, k=1)[0]
            if board.place_ship(ship, r, c, h):
                placed = True
                break
        if not placed:
            return False
    return True


def place_edge(board: Board, rng: random.Random) -> bool:
    return _weighted_place(board, rng, _edge_weight)


def place_corner(board: Board, rng: random.Random) -> bool:
    return _weighted_place(board, rng, _corner_weight)


def place_center(board: Board, rng: random.Random) -> bool:
    return _weighted_place(board, rng, _center_weight)


def place_spread(board: Board, rng: random.Random) -> bool:
    """Each ship is placed to maximise its minimum Manhattan distance to
    cells already occupied by earlier ships."""
    for name, size in FLEET:
        ship = Ship(name, size)
        existing = [cell for s in board.ships for cell in s.cells]

        if not existing:
            if not _try_place(board, ship, rng):
                return False
            continue

        # Sample 300 candidate positions and rank by min-distance to existing
        scored: list[tuple[float, int, int, bool]] = []
        for _ in range(300):
            r = rng.randint(0, 9)
            c = rng.randint(0, 9)
            h = rng.choice((True, False))
            cells = (
                [(r, c + i) for i in range(size)]
                if h
                else [(r + i, c) for i in range(size)]
            )
            if any(not (0 <= rr < 10 and 0 <= cc < 10) for rr, cc in cells):
                continue
            min_d = min(
                abs(rr - er) + abs(cc - ec)
                for rr, cc in cells
                for er, ec in existing
            )
            scored.append((min_d, r, c, h))

        scored.sort(reverse=True)
        placed = False
        for _, r, c, h in scored:
            if board.place_ship(ship, r, c, h):
                placed = True
                break
        if not placed and not _try_place(board, ship, rng):
            return False

    return True


PLACEMENT_STRATEGIES: dict[str, callable] = {
    "random": place_random,
    "edge": place_edge,
    "corner": place_corner,
    "center": place_center,
    "spread": place_spread,
}


# ===========================================================================
# Shooting strategies
# ===========================================================================

def _unshot(board: Board) -> list[tuple[int, int]]:
    return [(r, c) for r in range(10) for c in range(10) if board.grid[r, c] == 0]


def _active_hits(board: Board) -> list[tuple[int, int]]:
    return [(r, c) for r in range(10) for c in range(10) if board.grid[r, c] == 2]


def _hit_neighbors(board: Board) -> list[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for r, c in _active_hits(board):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board.grid[nr, nc] == 0:
                targets.add((nr, nc))
    return list(targets)


class RandomShooter:
    def next_shot(self, board: Board, rng: random.Random) -> tuple[int, int]:
        return rng.choice(_unshot(board))


class ParityShooter:
    """Checkerboard hunt (only even-parity cells) + target mode on hits."""

    def next_shot(self, board: Board, rng: random.Random) -> tuple[int, int]:
        neighbors = _hit_neighbors(board)
        if neighbors:
            return rng.choice(neighbors)
        # Hunt phase: prefer (row+col) % 2 == 0 cells
        parity = [(r, c) for r, c in _unshot(board) if (r + c) % 2 == 0]
        if parity:
            return rng.choice(parity)
        fallback = _unshot(board)
        return rng.choice(fallback)


class HuntTargetShooter:
    """Random until a hit; then target orthogonal neighbours until ship sinks."""

    def next_shot(self, board: Board, rng: random.Random) -> tuple[int, int]:
        neighbors = _hit_neighbors(board)
        if neighbors:
            return rng.choice(neighbors)
        return rng.choice(_unshot(board))


class ProbabilityShooter:
    """Recompute the full placement-probability map each turn; shoot the
    highest-probability unshot cell.  Ports the JS calculateAllProbs logic
    via Board.get_probability_map()."""

    def next_shot(self, board: Board, rng: random.Random) -> tuple[int, int]:
        prob = board.get_probability_map()
        best, best_p = [], -1.0
        for r in range(10):
            for c in range(10):
                if board.grid[r, c] == 0:
                    p = prob[r, c]
                    if p > best_p:
                        best_p, best = p, [(r, c)]
                    elif p == best_p:
                        best.append((r, c))
        if best:
            return rng.choice(best)
        return rng.choice(_unshot(board))


class EdgeFirstShooter:
    """Exhausts the outermost ring of the board before moving inward."""

    @staticmethod
    def _ring(r: int, c: int) -> int:
        return min(r, 9 - r, c, 9 - c)

    def next_shot(self, board: Board, rng: random.Random) -> tuple[int, int]:
        candidates = [(self._ring(r, c), r, c) for r, c in _unshot(board)]
        if not candidates:
            return (0, 0)
        min_ring = min(x[0] for x in candidates)
        best = [(r, c) for ring, r, c in candidates if ring == min_ring]
        return rng.choice(best)


SHOOTING_STRATEGIES: dict[str, type] = {
    "random": RandomShooter,
    "parity": ParityShooter,
    "hunt_target": HuntTargetShooter,
    "probability": ProbabilityShooter,
    "edge_first": EdgeFirstShooter,
}
