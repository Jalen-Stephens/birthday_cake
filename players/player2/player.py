from __future__ import annotations
from typing import List, Tuple, Optional
from math import inf
import time

from shapely import Point, Polygon
from players.player import Player, PlayerException
from src.cake import Cake


class Player2(Player):
    """
    Expanded-search equal-target player.

    Goal: every final piece area ≈ (area(cake)/children) within ±0.05 cm² when geometrically possible,
    and interior ratio near the global target ratio as a soft tie-breaker.

    Strategy per cut:
      - Always split the CURRENT LARGEST PIECE.
      - Run a dense boundary search to "peel" one target-sized piece:
        * Uniformly sample many anchor points along the boundary.
        * For each anchor, sweep the opposite endpoint densely around the boundary.
        * Keep top K sweeps per anchor (lowest area error) and run adaptive local refinement
          (bracket-shrink) to drive error down to the strict tolerance if possible.
      - If strict target not achievable due to geometry/validity, take the closest by (area, ratio).
      - Odd n: first make a 1/n vs (n-1)/n cut on the whole cake using the same search, then proceed.

    A global time budget (TIME_BUDGET_SEC) caps total work; within it we bias toward precision.
    """

    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(children, cake, cake_path)
        self.debug = debug

        # Targets
        self.total_area = self.cake.exterior_shape.area
        self.target_area = self.total_area / max(1, self.children)
        self.target_ratio = (
            self.cake.interior_shape.area / self.cake.exterior_shape.area
            if self.cake.exterior_shape.area > 0
            else 0.0
        )

        # Strict spec tolerances
        self.strict_tol = 0.05  # cm^2
        self.ratio_tol = 0.05  # 5% (soft, for tie-break only)

        # Dense search parameters (tuned for ~1 minute worst-case)
        self.ANCHOR_SAMPLES = 120  # boundary anchors per search
        self.SWEEP_SAMPLES = 720  # coarse sweep positions per anchor
        self.TOP_SWEEPS_PER_ANCHOR = 4  # refine multiple best candidates per anchor
        self.REFINE_ITERS = 6  # bracket-shrink iterations
        self.REFINE_GRID = 21  # samples per refine iteration in [L,R]

        # Bisection-like shrink factor
        self.REFINE_SHRINK = 0.35

        # Validity guard for near-adjacent endpoints (normalized by boundary length)
        self.MIN_SEP_FRAC = 1.0 / 600.0

        # Time budget (wall clock) to approximate CPU minute
        self.TIME_BUDGET_SEC = 55.0

    # -------------------- public entry --------------------
    def get_cuts(self) -> List[Tuple[Point, Point]]:
        if self.children <= 1:
            return []

        start_time = time.time()
        deadline = start_time + self.TIME_BUDGET_SEC

        work = self.cake.copy()
        cuts: List[Tuple[Point, Point]] = []

        # Odd: first 1/n peel from the whole cake
        if self.children % 2 == 1:
            whole = self._largest_piece(work)
            first = self._target_search(
                work, whole, frac=1.0 / self.children, deadline=deadline
            )
            if first is None:
                # fallback: any valid cut to proceed
                first = self._any_valid_cut_on_piece(work, whole)
            if first is None:
                raise PlayerException("Player2: no valid first cut for odd strategy")
            cuts.append(first)
            work.cut(*first)

        # Continue peeling target shares until n pieces exist
        while len(work.get_pieces()) < self.children:
            if time.time() > deadline:
                # Time nearly up: be pragmatic
                cut = self._best_equal_area_or_any(work)
            else:
                largest = self._largest_piece(work)
                cut = self._target_search(
                    work, largest, frac=None, deadline=deadline
                )  # None => absolute target
                if cut is None:
                    # fallback if strict search fails
                    cut = self._best_equal_area_or_any(work)

            if cut is None:
                raise PlayerException(
                    f"Player2: ran out of valid cuts at {len(work.get_pieces())} pieces"
                )

            cuts.append(cut)
            work.cut(*cut)

        if len(cuts) != self.children - 1:
            raise PlayerException(
                f"Player2: expected {self.children - 1} cuts, got {len(cuts)}"
            )

        return cuts

    # -------------------- main search --------------------
    def _target_search(
        self,
        cake: Cake,
        piece: Polygon,
        frac: Optional[float],
        deadline: float,
    ) -> Optional[Tuple[Point, Point]]:
        """
        Search a boundary-to-boundary chord to peel a target-sized area from `piece`.
        If frac is None -> target = self.target_area, else target = frac * piece.area.
        Returns a cut (Point, Point) or None if no valid chord found.
        """
        bound = piece.boundary
        L = bound.length
        if L <= 0 or piece.area <= 0:
            return None

        target = (
            self.target_area
            if frac is None
            else max(0.0, min(piece.area, frac * piece.area))
        )
        min_sep = self.MIN_SEP_FRAC

        def pt(t: float) -> Point:
            # normalized arclength -> boundary point
            return bound.interpolate((t % 1.0) * L)

        best_cut: Optional[Tuple[Point, Point]] = None
        best_err = inf
        best_score = inf

        # Loop anchors
        for ia in range(self.ANCHOR_SAMPLES):
            if time.time() > deadline:
                break

            ta = ia / self.ANCHOR_SAMPLES
            A = pt(ta)

            # coarse sweep; keep top-K candidates for refinement
            candidates: List[
                Tuple[float, float, float]
            ] = []  # (primary_err, tb, score)

            for jb in range(self.SWEEP_SAMPLES):
                if time.time() > deadline:
                    break

                tb = jb / self.SWEEP_SAMPLES
                sep = min((tb - ta) % 1.0, (ta - tb) % 1.0)
                if sep < min_sep:
                    continue

                B = pt(tb)
                if not self._cut_valid_for_piece(cake, piece, A, B):
                    continue

                primary, score = self._target_err_for_cut(cake, piece, A, B, target)
                if primary < self.strict_tol:
                    # perfect enough — return immediately
                    return (A, B)

                # keep a small candidate set to refine
                if len(candidates) < self.TOP_SWEEPS_PER_ANCHOR:
                    candidates.append((primary, tb, score))
                    candidates.sort(key=lambda x: (x[0], x[2]))
                else:
                    # replace worst if better
                    if (primary, score) < (candidates[-1][0], candidates[-1][2]):
                        candidates[-1] = (primary, tb, score)
                        candidates.sort(key=lambda x: (x[0], x[2]))

            # refine each candidate locally
            for primary, tb0, _ in candidates:
                if time.time() > deadline:
                    break

                tb_best = tb0
                err_best = primary
                score_best = inf
                # local bracket around tb0
                half_window = 1.0 / max(24.0, float(self.SWEEP_SAMPLES))
                left = (tb_best - half_window) % 1.0
                right = (tb_best + half_window) % 1.0

                for _ in range(self.REFINE_ITERS):
                    if time.time() > deadline:
                        break

                    # sample a grid between left..right (wrap-aware)
                    grid = self._linspace_wrap(left, right, self.REFINE_GRID)
                    improved = False
                    for tb in grid:
                        sep = min((tb - ta) % 1.0, (ta - tb) % 1.0)
                        if sep < min_sep:
                            continue
                        B = pt(tb)
                        if not self._cut_valid_for_piece(cake, piece, A, B):
                            continue
                        e, s = self._target_err_for_cut(cake, piece, A, B, target)
                        if e < err_best or (
                            abs(e - err_best) <= 1e-12 and s < score_best
                        ):
                            err_best = e
                            score_best = s
                            tb_best = tb
                            improved = True
                            if err_best < self.strict_tol:
                                return (A, pt(tb_best))

                    # shrink the bracket around the best tb
                    half_window *= self.REFINE_SHRINK
                    left = (tb_best - half_window) % 1.0
                    right = (tb_best + half_window) % 1.0
                    if not improved:
                        break

                # track global best
                if err_best < best_err or (
                    abs(err_best - best_err) <= 1e-12 and score_best < best_score
                ):
                    best_err = err_best
                    best_score = score_best
                    best_cut = (A, pt(tb_best))

        return best_cut

    # -------------------- scoring & validity --------------------
    def _target_err_for_cut(
        self, cake: Cake, piece: Polygon, A: Point, B: Point, target: float
    ) -> Tuple[float, float]:
        """
        Returns:
          primary = |area(chosen_side) - target|
          score   = primary + 0.1 * |ratio(chosen_side) - target_ratio|
        The chosen side is whichever piece is closer to the target area.
        """
        split = cake.cut_piece(piece, A, B)
        if len(split) != 2:
            return (inf, inf)
        P, Q = split
        side = P if abs(P.area - target) <= abs(Q.area - target) else Q
        primary = abs(side.area - target)
        ratio = self._piece_ratio(cake, side)
        score = primary + 0.1 * abs(ratio - self.target_ratio)
        return (primary, score)

    def _cut_valid_for_piece(
        self, cake: Cake, piece: Polygon, A: Point, B: Point
    ) -> bool:
        ok, _ = cake.cut_is_valid(A, B)
        if not ok:
            return False
        # ensure A and B lie on THIS piece's boundary
        return cake.point_lies_on_piece_boundary(
            A, piece
        ) and cake.point_lies_on_piece_boundary(B, piece)

    # -------------------- fallbacks --------------------
    def _best_equal_area_or_any(self, cake: Cake) -> Optional[Tuple[Point, Point]]:
        """Try best equal-area on the largest piece; else any valid cut anywhere."""
        largest = self._largest_piece(cake)
        cut = self._best_equal_area_on_piece(cake, largest)
        if cut is not None:
            return cut
        return self._any_valid_cut_anywhere(cake)

    def _best_equal_area_on_piece(
        self, cake: Cake, piece: Polygon
    ) -> Optional[Tuple[Point, Point]]:
        pts = self._candidate_points(piece)
        best = None
        best_score = inf
        half = piece.area / 2.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = cake.cut_is_valid(a, b)
                if not ok:
                    continue
                split = cake.cut_piece(piece, a, b)
                if len(split) != 2:
                    continue
                p, q = split
                area_err = abs(p.area - half) + abs(q.area - half)
                r_err = 0.5 * (
                    abs(self._piece_ratio(cake, p) - self.target_ratio)
                    + abs(self._piece_ratio(cake, q) - self.target_ratio)
                )
                score = area_err + 0.1 * r_err
                if score < best_score:
                    best_score = score
                    best = (a, b)
        return best

    def _any_valid_cut_on_piece(
        self, cake: Cake, piece: Polygon
    ) -> Optional[Tuple[Point, Point]]:
        pts = self._candidate_points(piece)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = cake.cut_is_valid(a, b)
                if ok:
                    return (a, b)
        return None

    def _any_valid_cut_anywhere(self, cake: Cake) -> Optional[Tuple[Point, Point]]:
        for piece in cake.get_pieces():
            cut = self._any_valid_cut_on_piece(cake, piece)
            if cut:
                return cut
        return None

    # -------------------- geometry helpers --------------------
    def _largest_piece(self, cake: Cake) -> Polygon:
        return max(cake.get_pieces(), key=lambda p: p.area)

    def _piece_ratio(self, cake: Cake, poly: Polygon) -> float:
        if poly.is_empty or poly.area <= 0:
            return 0.0
        return cake.get_piece_ratio(poly)

    def _candidate_points(self, poly: Polygon) -> List[Point]:
        """Vertices + all edge midpoints (including closing edge) for fallback paths."""
        verts = list(poly.exterior.coords[:-1])
        pts = [Point(v) for v in verts]
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            pts.append(Point((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return pts

    def _linspace_wrap(self, left: float, right: float, m: int) -> List[float]:
        """
        Sample m points from [left, right] on a unit circle parameter, handling wrap-around.
        """
        if m <= 1:
            return [left]
        vals: List[float] = []
        if left <= right:
            step = (right - left) / (m - 1)
            for k in range(m):
                vals.append(left + k * step)
        else:
            # wrapped interval
            total = (1.0 - left) + right
            step = total / (m - 1)
            acc = 0.0
            for _ in range(m):
                t = (left + acc) % 1.0
                vals.append(t)
                acc += step
        return vals
