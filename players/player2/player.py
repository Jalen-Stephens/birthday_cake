# players/player2.py
from __future__ import annotations
from typing import List, Tuple, Optional
from math import inf
from shapely import Point
from shapely.geometry import LineString, Polygon

from players.player import Player, PlayerException
import src.constants as c
from src.cake import Cake, extend_line


class Player2(Player):
    """
    divide-&-Conquer (beam) planner + robust sequential

    plan phase (on subcakes):
      - recursively split target piece counts (prefer near-halves)
      - evaluate chords built from rich boundary candidates
      - keep top-K (beam) by squared error vs. target areas
      - return an ordered list of cuts (root cut, then cuts for each subpiece)

    phase (on a working copy of the real cake):
      - apply cuts in that order
      - for each planned pair, SNAP both endpoints to the boundary of the
        actual piece it should cut (try all pieces, choose the first that yields a
        valid split under simulator rules)
      - if a planned cut can't be realized, use a greedy best cut on the
        largest current piece; if that fails, use any-valid fallback.

    Focus: equal areas. (Crust ratio only lightly penalized in scoring;
    you can dial that in later.)
    """

    # ------------------ init ------------------

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

        base_poly = self.cake.get_pieces()[0]
        self.total_area = base_poly.area
        self.target_area = self.total_area / max(1, self.children)
        self.avg_ratio = self.cake.interior_shape.area / max(self.total_area, 1e-9)

        self.beam_width = 5  # widen for better search, narrow for speed
        self.sample_count = 64  # boundary samples per piece (in addition to vertices)
        self.area_tol = getattr(c, "PIECE_SPAN_TOL", 0.5)  # cm^2
        self.ratio_w = 0.10  # small weight on ratio to break ties (can set 0)

    # ------------------ public ------------------

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        if self.children <= 1:
            return []

        # on a sandbox copy (subcakes)
        plan_cake = self.cake.copy()
        root_piece = plan_cake.get_pieces()[0]
        _, planned = self._search(
            self._wrap_subcake(plan_cake, root_piece),
            pieces_needed=self.children,
            best_so_far=inf,
            depth=0,
        )

        # planned list sequentially on a fresh working cake,
        # snapping each cut to the current geometry
        realized = self._realize_sequential(self.cake.copy(), planned)

        # if we didn't get enough, greedily add equal-area cuts
        while len(realized) < self.children - 1:
            add = self._best_greedy_cut_for(self.cake.copy(), realized)
            if add is None:
                break
            realized.append(add)

        # if still short, fill with any valid cuts
        if len(realized) != self.children - 1:
            if not self._fill_with_any_valid(self.cake.copy(), realized):
                raise PlayerException(
                    f"Player2: could not assemble {self.children - 1} cuts (got {len(realized)})."
                )

        return realized

    # ------------------ search (plan) ------------------

    def _search(
        self,
        cake: Cake,  # sandbox subcake with a single piece
        pieces_needed: int,
        best_so_far: float,
        depth: int,
    ) -> tuple[float, List[Tuple[Point, Point]]]:
        piece = cake.get_pieces()[0]
        if pieces_needed == 1:
            return self._leaf_score(cake), []

        m = pieces_needed
        preferred = [(m // 2, m - (m // 2))]
        neighbors: list[tuple[int, int]] = []
        if m > 2:
            a0 = preferred[0][0]
            neighbors.append((max(1, a0 - 1), m - max(1, a0 - 1)))
            neighbors.append((min(m - 1, a0 + 1), m - min(m - 1, a0 + 1)))

        splits: list[tuple[int, int]] = []
        seen = set()
        for a, b in preferred + neighbors:
            if a <= 0 or b <= 0:
                continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                splits.append((a, b))
                seen.add(key)

        # rich candidate endpoints on THIS piece boundary
        cand_pts = self._candidate_points(piece)

        cand: list[
            tuple[float, Tuple[Point, Point], Tuple[Polygon, Polygon], Tuple[int, int]]
        ] = []

        for i in range(len(cand_pts)):
            for j in range(i + 1, len(cand_pts)):
                pA, pB = cand_pts[i], cand_pts[j]
                ok, _ = cake.cut_is_valid(pA, pB)
                if not ok:
                    continue
                parts = cake.cut_piece(piece, pA, pB)
                if len(parts) != 2:
                    continue
                P, Q = parts
                aP, aQ = P.area, Q.area

                # score each desired split (a,b)
                for a, b in splits:
                    tP = a * self.target_area
                    tQ = b * self.target_area
                    err = (aP - tP) ** 2 + (aQ - tQ) ** 2
                    if self.ratio_w > 0:
                        rP = self._ratio_v_parent(cake, P)
                        rQ = self._ratio_v_parent(cake, Q)
                        err += self.ratio_w * (
                            (rP - self.avg_ratio) ** 2 + (rQ - self.avg_ratio) ** 2
                        )
                    cand.append((err, (pA, pB), (P, Q), (a, b)))

        if not cand:
            return inf, []

        cand.sort(key=lambda x: x[0])
        cand = cand[: self.beam_width]

        best_score = best_so_far
        best_seq: List[Tuple[Point, Point]] = []

        for cut_err, (from_p, to_p), (P, Q), (a, b) in cand:
            if cut_err >= best_score:
                continue

            # prune very poor fits
            tP = a * self.target_area
            tQ = b * self.target_area
            if (
                abs(P.area - tP) > 10 * self.area_tol
                and abs(Q.area - tQ) > 10 * self.area_tol
            ):
                continue

            cakeP = self._wrap_subcake(cake, P)
            cakeQ = self._wrap_subcake(cake, Q)

            sP, seqP = self._search(cakeP, a, best_score - cut_err, depth + 1)
            if cut_err + sP >= best_score:
                continue

            sQ, seqQ = self._search(cakeQ, b, best_score - cut_err - sP, depth + 1)
            total = cut_err + sP + sQ

            if total < best_score:
                best_score = total
                best_seq = [(from_p, to_p)] + seqP + seqQ

        return best_score, best_seq

    # ------------------ realization (snap & apply) ------------------

    def _realize_sequential(
        self, sim: Cake, planned: List[Tuple[Point, Point]]
    ) -> List[Tuple[Point, Point]]:
        realized: List[Tuple[Point, Point]] = []

        for p_raw, q_raw in planned:
            snapped = self._snap_to_some_piece(sim, p_raw, q_raw)
            if snapped is None:
                # planned cut can’t be realized on current geometry; try greedy instead
                g = self._best_greedy_cut_for(sim, realized)
                if g is None:
                    # try any valid
                    g = self._any_valid_on(sim)
                if g is None:
                    # give up on this slot; continue (fill later)
                    continue
                p_use, q_use = g
            else:
                p_use, q_use = snapped

            realized.append((p_use, q_use))
            try:
                sim.cut(p_use, q_use)
            except Exception:
                # if simulator still rejects, skip but keep count
                pass

        return realized

    def _snap_to_some_piece(
        self, sim: Cake, p: Point, q: Point
    ) -> Optional[Tuple[Point, Point]]:
        """
        Try every current piece: project p and q to that piece boundary.
        If extend_line(boundary-snapped segment) validly splits the piece into two
        >= MIN_PIECE_AREA polygons, return snapped endpoints.
        """
        for piece in sim.get_pieces():
            bound = piece.boundary
            a = bound.interpolate(bound.project(p))
            b = bound.interpolate(bound.project(q))

            # Build the extended line and test piece-level split predicate
            line = extend_line(LineString([a, b]))
            good, _ = sim.does_line_cut_piece_well(line, piece)
            if not good:
                continue

            # Also make sure global validity passes (single-piece, within cake)
            ok, _ = sim.cut_is_valid(a, b)
            if not ok:
                continue

            return a, b
        return None

    # ------------------ greedy & fallbacks ------------------

    def _best_greedy_cut_for(
        self, original: Cake, already_realized: List[Tuple[Point, Point]]
    ) -> Optional[Tuple[Point, Point]]:
        """
        Given a fresh copy of the original cake and the cuts already realized,
        simulate those cuts to reproduce current geometry, then find the best
        approximate equal-area cut on the largest current piece.
        """
        sim = original.copy()
        # replay already_realized to mirror current state
        for a, b in already_realized:
            try:
                sim.cut(a, b)
            except Exception:
                # Try one snap pass if it fails
                snapped = self._snap_to_some_piece(sim, a, b)
                if snapped is None:
                    return None
                sim.cut(snapped[0], snapped[1])

        target_piece = max(sim.get_pieces(), key=lambda p: p.area)
        # We aim to split remaining children roughly in half
        # Determine remaining children from area ratio
        remaining_area = sum(p.area for p in sim.get_pieces())
        remaining_children = max(2, round(remaining_area / self.target_area))
        a = remaining_children // 2
        b = remaining_children - a
        tP = a * self.target_area
        tQ = b * self.target_area

        best = None
        best_err = inf

        pts = self._candidate_points(target_piece)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                pA, pB = pts[i], pts[j]
                ok, _ = sim.cut_is_valid(pA, pB)
                if not ok:
                    continue
                parts = sim.cut_piece(target_piece, pA, pB)
                if len(parts) != 2:
                    continue
                P, Q = parts
                err = (P.area - tP) ** 2 + (Q.area - tQ) ** 2
                if err < best_err:
                    best_err = err
                    best = (pA, pB)

        return best

    def _any_valid_on(self, sim: Cake) -> Optional[Tuple[Point, Point]]:
        """Pick any valid cut on the largest current piece."""
        piece = max(sim.get_pieces(), key=lambda p: p.area)
        pts = self._candidate_points(piece)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = sim.cut_is_valid(a, b)
                if ok:
                    return a, b
        return None

    def _fill_with_any_valid(
        self, original: Cake, cuts: List[Tuple[Point, Point]]
    ) -> bool:
        """Keep adding any valid cuts until we reach n-1."""
        sim = original.copy()
        for a, b in cuts:
            try:
                sim.cut(a, b)
            except Exception:
                snapped = self._snap_to_some_piece(sim, a, b)
                if snapped is None:
                    continue
                sim.cut(snapped[0], snapped[1])

        while len(cuts) < self.children - 1:
            g = self._any_valid_on(sim)
            if g is None:
                return False
            cuts.append(g)
            try:
                sim.cut(g[0], g[1])
            except Exception:
                # if even this fails, try snap once
                snapped = self._snap_to_some_piece(sim, g[0], g[1])
                if snapped is None:
                    return False
                sim.cut(snapped[0], snapped[1])
        return True

    # ------------------ points / scoring / subcakes ------------------

    def _candidate_points(self, poly: Polygon) -> List[Point]:
        """
        Vertices + uniform samples along boundary length.
        This avoids 'corner fan' and provides many chords that truly cross the piece.
        """
        pts: List[Point] = [Point(xy) for xy in list(poly.exterior.coords[:-1])]
        L = poly.boundary.length
        if L <= c.TOL:
            return pts
        step = L / self.sample_count
        # start at small offset to avoid degeneracy with actual vertices
        offset = 0.5 * step
        for k in range(self.sample_count):
            s = (offset + k * step) % L
            pts.append(poly.boundary.interpolate(s))
        return pts

    def _leaf_score(self, cake: Cake) -> float:
        piece = cake.get_pieces()[0]
        area_err = (piece.area - self.target_area) ** 2
        if self.ratio_w <= 0:
            return area_err
        ratio = self._ratio_v_parent(cake, piece)
        ratio_err = (ratio - self.avg_ratio) ** 2
        return area_err + self.ratio_w * ratio_err

    def _wrap_subcake(self, parent: Cake, poly: Polygon) -> Cake:
        new = object.__new__(Cake)
        new.exterior_shape = poly
        new.interior_shape = parent.interior_shape
        new.exterior_pieces = [poly]
        new.sandbox = True
        return new

    def _ratio_v_parent(self, parent: Cake, poly: Polygon) -> float:
        if poly.is_empty or poly.area <= 0:
            return 0.0
        inter = poly.intersection(parent.interior_shape)
        return 0.0 if inter.is_empty else inter.area / poly.area