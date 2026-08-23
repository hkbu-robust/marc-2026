"""Unified Stage 2 costmap: static occupancy base plus overlay costs.

Pure module: accepts numpy occupancy arrays, no ROS or SDK I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

WorldXY = Tuple[float, float]
CellRC = Tuple[int, int]

OCC_FREE = 0
OCC_OCCUPIED = 100
OCC_UNKNOWN = -1

# Overlay / effective cost: values >= this are treated as blocked.
BLOCKED_COST = 100


@dataclass(frozen=True)
class CostmapConfig:
    """Planning geometry defaults aligned with the organizer demo."""

    inflation_m: float = 0.25
    unknown_is_blocked: bool = True
    margin_m: float = 12.0
    snap_m: float = 1.5


@dataclass
class CostmapView:
    """Cropped planning window over a parent costmap."""

    trav: np.ndarray  # bool (H, W), True = traversable after inflation
    resolution: float
    origin_xy: WorldXY  # world origin of the *parent* map
    rmin: int
    cmin: int
    parent: "Costmap"

    @property
    def shape(self) -> Tuple[int, int]:
        return self.trav.shape

    @property
    def ready(self) -> bool:
        return self.trav.size > 0

    def world_to_local(self, wx: float, wy: float) -> CellRC:
        row, col = self.parent.world_to_cell(wx, wy)
        return row - self.rmin, col - self.cmin

    def local_to_world(self, row: int, col: int) -> WorldXY:
        return self.parent.cell_to_world(row + self.rmin, col + self.cmin)

    def local_to_parent(self, row: int, col: int) -> CellRC:
        return row + self.rmin, col + self.cmin

    def in_bounds(self, row: int, col: int) -> bool:
        h, w = self.trav.shape
        return 0 <= row < h and 0 <= col < w

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and bool(self.trav[row, col])


class Costmap:
    """Static road occupancy plus sensor/obstacle overlay costs."""

    def __init__(self, config: Optional[CostmapConfig] = None):
        self.config = config or CostmapConfig()
        self._base: Optional[np.ndarray] = None  # int8 (H, W)
        self._overlay: Optional[np.ndarray] = None  # int16 (H, W), 0 = no overlay
        self._res: Optional[float] = None
        self._ox: Optional[float] = None
        self._oy: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self._base is not None

    @property
    def resolution(self) -> Optional[float]:
        return self._res

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        if self._base is None:
            return None
        return self._base.shape

    def load_occupancy(
        self,
        grid: np.ndarray,
        resolution: float,
        origin_xy: WorldXY,
    ) -> None:
        """Load a 2D occupancy grid (0 free, 100 occupied, -1 unknown)."""
        if grid.ndim != 2:
            raise ValueError("occupancy grid must be 2-D")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        arr = np.asarray(grid, dtype=np.int8)
        self._base = arr
        self._overlay = np.zeros(arr.shape, dtype=np.int16)
        self._res = float(resolution)
        self._ox = float(origin_xy[0])
        self._oy = float(origin_xy[1])

    def world_to_cell(self, wx: float, wy: float) -> CellRC:
        if not self.ready:
            raise RuntimeError("costmap has no occupancy")
        col = int((wx - self._ox) / self._res)
        row = int((wy - self._oy) / self._res)
        return row, col

    def cell_to_world(self, row: int, col: int) -> WorldXY:
        if not self.ready:
            raise RuntimeError("costmap has no occupancy")
        wx = self._ox + (col + 0.5) * self._res
        wy = self._oy + (row + 0.5) * self._res
        return wx, wy

    def in_bounds(self, row: int, col: int) -> bool:
        if self._base is None:
            return False
        h, w = self._base.shape
        return 0 <= row < h and 0 <= col < w

    def effective_cost(self, row: int, col: int) -> int:
        """Merged base + overlay cost; >= BLOCKED_COST means blocked."""
        if not self.in_bounds(row, col):
            return BLOCKED_COST
        base = int(self._base[row, col])
        overlay = int(self._overlay[row, col])
        if overlay > 0:
            return max(base if base >= 0 else 0, overlay)
        if base == OCC_UNKNOWN:
            return BLOCKED_COST if self.config.unknown_is_blocked else OCC_FREE
        if base == OCC_OCCUPIED or base > 0:
            return max(base, BLOCKED_COST) if base == OCC_OCCUPIED else base
        return OCC_FREE

    def is_base_blocked(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        base = int(self._base[row, col])
        if base == OCC_UNKNOWN:
            return self.config.unknown_is_blocked
        return base != OCC_FREE

    def set_overlay_cost(self, row: int, col: int, cost: int) -> List[CellRC]:
        """Set overlay cost for one parent cell. Returns changed parent cells."""
        if not self.ready:
            return []
        if not self.in_bounds(row, col):
            return []
        cost = int(cost)
        prev = int(self._overlay[row, col])
        if prev == cost:
            return []
        self._overlay[row, col] = cost
        return [(row, col)]

    def clear_overlay(self, cells: Optional[Iterable[CellRC]] = None) -> List[CellRC]:
        """Clear overlay costs (all or listed cells). Returns changed cells."""
        if not self.ready:
            return []
        changed: List[CellRC] = []
        if cells is None:
            rows, cols = np.nonzero(self._overlay)
            for r, c in zip(rows.tolist(), cols.tolist()):
                self._overlay[r, c] = 0
                changed.append((r, c))
            return changed
        for row, col in cells:
            if self.in_bounds(row, col) and int(self._overlay[row, col]) != 0:
                self._overlay[row, col] = 0
                changed.append((row, col))
        return changed

    def apply_overlay_disc(
        self,
        wx: float,
        wy: float,
        radius_m: float,
        cost: int,
    ) -> List[CellRC]:
        """Mark a world-space disc on the overlay. Returns changed parent cells."""
        if not self.ready:
            return []
        cr, cc = self.world_to_cell(wx, wy)
        rad_c = int(round(float(radius_m) / self._res))
        if rad_c <= 0:
            return self.set_overlay_cost(cr, cc, cost)
        h, w = self._base.shape
        r0, r1 = max(0, cr - rad_c), min(h, cr + rad_c + 1)
        c0, c1 = max(0, cc - rad_c), min(w, cc + rad_c + 1)
        changed: List[CellRC] = []
        cost = int(cost)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if (r - cr) ** 2 + (c - cc) ** 2 <= rad_c ** 2:
                    prev = int(self._overlay[r, c])
                    if prev != cost:
                        self._overlay[r, c] = cost
                        changed.append((r, c))
        return changed

    def traversable_mask(self) -> np.ndarray:
        """Full-map free mask before inflation."""
        if not self.ready:
            raise RuntimeError("costmap has no occupancy")
        h, w = self._base.shape
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                mask[r, c] = self.effective_cost(r, c) < BLOCKED_COST
        return mask

    def crop_window(
        self,
        start_xy: WorldXY,
        goal_xy: WorldXY,
        margin_m: Optional[float] = None,
        inflation_m: Optional[float] = None,
    ) -> CostmapView:
        """Crop around start/goal, apply inflation, return a planning view."""
        if not self.ready:
            raise RuntimeError("costmap has no occupancy")
        margin_m = self.config.margin_m if margin_m is None else float(margin_m)
        inflation_m = (
            self.config.inflation_m if inflation_m is None else float(inflation_m)
        )

        sr, sc = self.world_to_cell(*start_xy)
        gr, gc = self.world_to_cell(*goal_xy)
        h, w = self._base.shape
        margin_c = int(np.ceil(margin_m / self._res))
        rmin = max(0, min(sr, gr) - margin_c)
        rmax = min(h, max(sr, gr) + margin_c + 1)
        cmin = max(0, min(sc, gc) - margin_c)
        cmax = min(w, max(sc, gc) + margin_c + 1)

        sub_h, sub_w = rmax - rmin, cmax - cmin
        trav = np.zeros((sub_h, sub_w), dtype=bool)
        for r in range(rmin, rmax):
            for c in range(cmin, cmax):
                trav[r - rmin, c - cmin] = self.effective_cost(r, c) < BLOCKED_COST

        infl_c = int(round(inflation_m / self._res))
        if infl_c > 0:
            trav = _erode(trav, infl_c)

        return CostmapView(
            trav=trav,
            resolution=self._res,
            origin_xy=(self._ox, self._oy),
            rmin=rmin,
            cmin=cmin,
            parent=self,
        )


def _erode(mask: np.ndarray, iters: int) -> np.ndarray:
    """8-neighbor erosion; border cells become blocked."""
    g = mask
    for _ in range(iters):
        e = np.zeros_like(g)
        if g.shape[0] < 3 or g.shape[1] < 3:
            return e
        e[1:-1, 1:-1] = (
            g[1:-1, 1:-1]
            & g[:-2, 1:-1]
            & g[2:, 1:-1]
            & g[1:-1, :-2]
            & g[1:-1, 2:]
            & g[:-2, :-2]
            & g[:-2, 2:]
            & g[2:, :-2]
            & g[2:, 2:]
        )
        g = e
    return g


def snap_to_free(
    trav: np.ndarray,
    cell: CellRC,
    snap_m: float,
    resolution: float,
) -> Optional[CellRC]:
    """Spiral search for the nearest free cell within snap_m."""
    r, c = cell
    h, w = trav.shape
    if not (0 <= r < h and 0 <= c < w):
        return None
    if trav[r, c]:
        return cell
    max_rad = int(max(1, round(snap_m / max(resolution, 1e-6))))
    for rad in range(1, max_rad + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and trav[nr, nc]:
                    return (nr, nc)
    return None
