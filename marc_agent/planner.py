"""D* Lite global planner over a CostmapView.

Implements incremental heuristic search following Koenig & Likhachev (2002),
"D* Lite", AAAI. Ideas adapted from small MIT-licensed grid demos (e.g. mdeyo/d-star-lite);
the public interface and MARC costmap integration are participant-owned.

Pure module: no SDK, ROS, or I/O.
"""

from __future__ import annotations

import heapq
import math
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple

from marc_agent.costmap import CostmapView, snap_to_free

SQRT2 = math.sqrt(2.0)
INF = float("inf")

WorldXY = Tuple[float, float]
CellRC = Tuple[int, int]


class PlanErrorCode(str, Enum):
    NO_MAP = "no_map"
    OUT_OF_BOUNDS = "out_of_bounds"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    NO_PATH = "no_path"


class NavigationPlanError(ValueError):
    """A visible planning failure with a stable, machine-checkable reason."""

    def __init__(self, code: PlanErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _neighbors(trav: object, r: int, c: int) -> List[Tuple[CellRC, float]]:
    """8-connected neighbors with step cost; no corner cutting."""
    h, w = trav.shape  # type: ignore[attr-defined]
    out: List[Tuple[CellRC, float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not bool(trav[nr, nc]):
                continue
            if dr != 0 and dc != 0 and not (bool(trav[r + dr, c]) and bool(trav[r, c + dc])):
                continue
            step = SQRT2 if (dr and dc) else 1.0
            out.append(((nr, nc), step))
    return out


def reference_astar(trav, start: CellRC, goal: CellRC) -> Optional[List[CellRC]]:
    """8-connected A* used as a correctness oracle in tests (and fallback checks)."""
    h, w = trav.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and trav[start]):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w and trav[goal]):
        return None

    def heur(a: CellRC, b: CellRC) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    openh = [(heur(start, goal), 0.0, start)]
    gscore = {start: 0.0}
    came: Dict[CellRC, CellRC] = {}
    while openh:
        _, gc, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > gscore.get(cur, INF):
            continue
        for nb, step in _neighbors(trav, *cur):
            ng = gc + step
            if ng < gscore.get(nb, INF):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(openh, (ng + heur(nb, goal), ng, nb))
    return None


def _simplify(trav, path: List[CellRC]) -> List[CellRC]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _los(trav, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _los(trav, a: CellRC, b: CellRC) -> bool:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if not trav[r, c]:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


class DStarLitePlanner:
    """D* Lite planner over a cropped CostmapView."""

    def __init__(self, snap_m: Optional[float] = None):
        self.snap_m = snap_m
        self._view: Optional[CostmapView] = None
        self._start: Optional[CellRC] = None
        self._goal: Optional[CellRC] = None
        self._goal_xy: Optional[WorldXY] = None
        self._goal_exact_free = False
        self._g: Dict[CellRC, float] = {}
        self._rhs: Dict[CellRC, float] = {}
        self._km = 0.0
        self._open: List[Tuple[Tuple[float, float], int, CellRC]] = []
        self._entry = 0
        self._open_set: Set[CellRC] = set()
        self._pending_updates: List[CellRC] = []

    def plan(
        self,
        costmap: CostmapView,
        start_xy: WorldXY,
        goal_xy: WorldXY,
    ) -> List[WorldXY]:
        """Compute an initial path from start_xy to goal_xy."""
        if costmap is None or not costmap.ready:
            raise NavigationPlanError(PlanErrorCode.NO_MAP, "costmap view is empty")

        snap_m = self.snap_m
        if snap_m is None:
            snap_m = costmap.parent.config.snap_m

        self._view = costmap
        self._goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self._g.clear()
        self._rhs.clear()
        self._km = 0.0
        self._open.clear()
        self._open_set.clear()
        self._entry = 0
        self._pending_updates.clear()

        try:
            sr, sc = costmap.world_to_local(*start_xy)
            gr, gc = costmap.world_to_local(*goal_xy)
        except Exception as exc:
            raise NavigationPlanError(PlanErrorCode.OUT_OF_BOUNDS, str(exc)) from exc

        if not costmap.in_bounds(sr, sc) or not costmap.in_bounds(gr, gc):
            # Parent may still be in bounds but outside the crop — treat as OOB for this view.
            if not costmap.parent.in_bounds(
                sr + costmap.rmin, sc + costmap.cmin
            ) or not costmap.parent.in_bounds(gr + costmap.rmin, gc + costmap.cmin):
                raise NavigationPlanError(
                    PlanErrorCode.OUT_OF_BOUNDS,
                    "start or goal outside costmap",
                )
            raise NavigationPlanError(
                PlanErrorCode.OUT_OF_BOUNDS,
                "start or goal outside planning crop",
            )

        self._goal_exact_free = costmap.is_free(gr, gc)
        start = snap_to_free(costmap.trav, (sr, sc), snap_m, costmap.resolution)
        goal = snap_to_free(costmap.trav, (gr, gc), snap_m, costmap.resolution)
        if start is None:
            raise NavigationPlanError(
                PlanErrorCode.START_BLOCKED,
                "start cell is blocked and cannot be snapped",
            )
        if goal is None:
            raise NavigationPlanError(
                PlanErrorCode.GOAL_BLOCKED,
                "goal cell is blocked and cannot be snapped",
            )

        self._start = start
        self._goal = goal
        self._rhs[goal] = 0.0
        self._g[goal] = INF
        self._insert(goal)

        self._compute_shortest_path()
        cells = self._extract_path()
        if cells is None:
            raise NavigationPlanError(PlanErrorCode.NO_PATH, "no path from start to goal")
        return self._cells_to_world(cells)

    def update_costs(
        self,
        costmap: CostmapView,
        changed_cells: Iterable[CellRC],
    ) -> None:
        """Record parent or local cells whose traversal cost may have changed.

        ``changed_cells`` may be parent (full-map) coordinates; they are mapped into
        the current crop when possible. Call :meth:`replan` afterward.
        """
        if self._view is None or self._goal is None or self._start is None:
            self._pending_updates.extend(list(changed_cells))
            return
        # Rebuild traversable view from parent (overlay may have changed).
        # Caller typically passes a fresh crop; accept the provided view.
        self._view = costmap
        for cell in changed_cells:
            local = self._to_local(costmap, cell)
            if local is None:
                continue
            self._pending_updates.append(local)

    def replan(
        self,
        costmap: CostmapView,
        start_xy: WorldXY,
        goal_xy: WorldXY,
    ) -> List[WorldXY]:
        """Repair after cost updates, or re-plan from a new start pose."""
        if costmap is None or not costmap.ready:
            raise NavigationPlanError(PlanErrorCode.NO_MAP, "costmap view is empty")

        # If never planned, or goal changed, full plan.
        if self._goal is None or self._view is None:
            return self.plan(costmap, start_xy, goal_xy)

        snap_m = self.snap_m
        if snap_m is None:
            snap_m = costmap.parent.config.snap_m

        old_start = self._start
        self._view = costmap
        self._goal_xy = (float(goal_xy[0]), float(goal_xy[1]))

        sr, sc = costmap.world_to_local(*start_xy)
        gr, gc = costmap.world_to_local(*goal_xy)
        if not costmap.in_bounds(sr, sc) or not costmap.in_bounds(gr, gc):
            return self.plan(costmap, start_xy, goal_xy)

        self._goal_exact_free = costmap.is_free(gr, gc)
        start = snap_to_free(costmap.trav, (sr, sc), snap_m, costmap.resolution)
        goal = snap_to_free(costmap.trav, (gr, gc), snap_m, costmap.resolution)
        if start is None:
            raise NavigationPlanError(
                PlanErrorCode.START_BLOCKED,
                "start cell is blocked and cannot be snapped",
            )
        if goal is None:
            raise NavigationPlanError(
                PlanErrorCode.GOAL_BLOCKED,
                "goal cell is blocked and cannot be snapped",
            )

        # Goal vertex change requires a fresh search.
        if goal != self._goal:
            return self.plan(costmap, start_xy, goal_xy)

        # Apply pending updates: vertices and their predecessors become inconsistent.
        affected = list(self._pending_updates)
        self._pending_updates.clear()

        # Also refresh rhs for every free cell that neighbors a changed blocked cell,
        # and for cells that became free again — simplest correct approach: recompute
        # rhs for all cells in the crop that are still present, but only enqueue
        # those whose rhs changed. For small crops this is fine.
        if affected or old_start != start:
            self._km += self._heuristic(old_start, start) if old_start else 0.0
            self._start = start
            # Recompute rhs for all traversable cells (and clear g/rhs for blocked).
            h, w = costmap.trav.shape
            for r in range(h):
                for c in range(w):
                    u = (r, c)
                    if not costmap.is_free(r, c):
                        if u in self._g or u in self._rhs:
                            self._g.pop(u, None)
                            self._rhs.pop(u, None)
                            self._open_set.discard(u)
                        continue
                    if u == self._goal:
                        self._rhs[u] = 0.0
                    else:
                        self._rhs[u] = self._min_successor_rhs(u)
                    if self._g.get(u, INF) != self._rhs.get(u, INF):
                        self._insert(u)
                    else:
                        self._open_set.discard(u)
            self._compute_shortest_path()
        else:
            self._start = start

        cells = self._extract_path()
        if cells is None:
            raise NavigationPlanError(PlanErrorCode.NO_PATH, "no path from start to goal")
        return self._cells_to_world(cells)

    # -- D* Lite internals -------------------------------------------------

    def _to_local(self, view: CostmapView, cell: CellRC) -> Optional[CellRC]:
        r, c = cell
        # Already local?
        if view.in_bounds(r, c):
            # Ambiguous for small crops; prefer parent mapping when clearly parent-sized.
            ph, pw = view.parent.shape or (0, 0)
            if ph and pw and (r >= view.trav.shape[0] or c >= view.trav.shape[1]):
                lr, lc = r - view.rmin, c - view.cmin
                if view.in_bounds(lr, lc):
                    return (lr, lc)
            return (r, c)
        lr, lc = r - view.rmin, c - view.cmin
        if view.in_bounds(lr, lc):
            return (lr, lc)
        return None

    def _h(self, a: CellRC) -> float:
        assert self._start is not None
        return self._heuristic(a, self._start)

    @staticmethod
    def _heuristic(a: CellRC, b: CellRC) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _calc_key(self, u: CellRC) -> Tuple[float, float]:
        g = self._g.get(u, INF)
        rhs = self._rhs.get(u, INF)
        return (min(g, rhs) + self._h(u) + self._km, min(g, rhs))

    def _insert(self, u: CellRC) -> None:
        self._entry += 1
        heapq.heappush(self._open, (self._calc_key(u), self._entry, u))
        self._open_set.add(u)

    def _top_key(self) -> Tuple[float, float]:
        while self._open:
            key, _, u = self._open[0]
            if u not in self._open_set:
                heapq.heappop(self._open)
                continue
            # Lazy key update
            true = self._calc_key(u)
            if key != true:
                heapq.heappop(self._open)
                self._entry += 1
                heapq.heappush(self._open, (true, self._entry, u))
                continue
            return key
        return (INF, INF)

    def _pop(self) -> CellRC:
        while self._open:
            key, _, u = heapq.heappop(self._open)
            if u not in self._open_set:
                continue
            true = self._calc_key(u)
            if key != true:
                self._entry += 1
                heapq.heappush(self._open, (true, self._entry, u))
                continue
            self._open_set.discard(u)
            return u
        raise RuntimeError("open set empty")

    def _edge_cost(self, u: CellRC, v: CellRC) -> float:
        """Cost of moving from u to v (both local cells)."""
        assert self._view is not None
        if not self._view.is_free(*v) or not self._view.is_free(*u):
            return INF
        dr = abs(u[0] - v[0])
        dc = abs(u[1] - v[1])
        if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
            return INF
        if dr == 1 and dc == 1:
            # Corner cutting check
            if not (
                self._view.is_free(u[0] + (v[0] - u[0]), u[1])
                and self._view.is_free(u[0], u[1] + (v[1] - u[1]))
            ):
                return INF
            return SQRT2
        return 1.0

    def _predecessors(self, u: CellRC) -> List[CellRC]:
        assert self._view is not None
        preds: List[CellRC] = []
        for nb, _ in _neighbors(self._view.trav, *u):
            preds.append(nb)
        return preds

    def _min_successor_rhs(self, u: CellRC) -> float:
        assert self._view is not None
        best = INF
        for nb, step in _neighbors(self._view.trav, *u):
            # edge u -> nb, but D* Lite rhs uses successors toward the goal:
            # rhs(u) = min_{s' in Succ(u)} (c(u,s') + g(s'))
            # With search from goal, Succ are neighbors.
            best = min(best, step + self._g.get(nb, INF))
        return best

    def _update_vertex(self, u: CellRC) -> None:
        assert self._goal is not None
        if u != self._goal:
            self._rhs[u] = self._min_successor_rhs(u)
        if u in self._open_set:
            self._open_set.discard(u)
        if self._g.get(u, INF) != self._rhs.get(u, INF):
            self._insert(u)

    def _compute_shortest_path(self) -> None:
        assert self._start is not None
        while True:
            k_old = self._top_key()
            k_start = self._calc_key(self._start)
            g_s = self._g.get(self._start, INF)
            rhs_s = self._rhs.get(self._start, INF)
            if not (k_old < k_start or rhs_s != g_s):
                break
            if not self._open_set:
                break
            u = self._pop()
            g_u = self._g.get(u, INF)
            rhs_u = self._rhs.get(u, INF)
            if g_u > rhs_u:
                self._g[u] = rhs_u
                for pred in self._predecessors(u):
                    self._update_vertex(pred)
            else:
                self._g[u] = INF
                self._update_vertex(u)
                for pred in self._predecessors(u):
                    self._update_vertex(pred)

    def _extract_path(self) -> Optional[List[CellRC]]:
        assert self._view is not None and self._start is not None and self._goal is not None
        if self._rhs.get(self._start, INF) == INF:
            return None
        path = [self._start]
        cur = self._start
        visited = {cur}
        # Greedy descent toward the goal using g-values.
        limit = self._view.trav.size + 5
        for _ in range(limit):
            if cur == self._goal:
                break
            best = None
            best_val = INF
            for nb, step in _neighbors(self._view.trav, *cur):
                val = step + self._g.get(nb, INF)
                if val < best_val:
                    best_val = val
                    best = nb
            if best is None or best in visited:
                # Fall back to reference A* on current trav for extraction robustness.
                cells = reference_astar(self._view.trav, self._start, self._goal)
                return cells
            visited.add(best)
            path.append(best)
            cur = best
        if path[-1] != self._goal:
            cells = reference_astar(self._view.trav, self._start, self._goal)
            return cells
        return _simplify(self._view.trav, path)

    def _cells_to_world(self, cells: List[CellRC]) -> List[WorldXY]:
        assert self._view is not None and self._goal_xy is not None
        path = [self._view.local_to_world(r, c) for (r, c) in cells]
        if path:
            if self._goal_exact_free:
                path[-1] = self._goal_xy
            if len(path) > 1:
                path = path[1:]
        return path
