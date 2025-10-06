# players/player2/player.py
from __future__ import annotations
from typing import List, Tuple, Optional
from math import inf

from shapely import Point, Polygon

from players.player import Player, PlayerException
import src.constants as c
from src.cake import Cake


class Player2(Player):
    """
    Equal-share player:
      - Even n: repeatedly bisect the current largest piece.
      - Odd  n: cut 1/n vs (n-1)/n first, then bisect largest until n pieces.

    Primary objective: piece area ≈ target_area
    Secondary (soft):  interior ratio ≈ target_ratio
    """

    def __init__(self, children: int, cake: Cake, cake_path: Optional[str] = None, debug: bool = False) -> None:
        super().__init__(children, cake, cake_path)
        self.debug = debug

        self.total_area = self.cake.exterior_shape.area
        self.target_area = self.total_area / max(1, self.children)
        self.target_ratio = (
            self.cake.interior_shape.area / self.cake.exterior_shape.area
            if self.cake.exterior_shape.area > 0 else 0.0
        )

        # tolerances from spec / constants
        self.area_tol = getattr(c, "PIECE_SPAN_TOL", 0.5)  # cm^2 tolerance for “same size”
        self.ratio_tol = 0.05  # 5%

    # ---------------- public ----------------
    def get_cuts(self) -> List[Tuple[Point, Point]]:
        if self.children <= 1:
            return []

        work = self.cake.copy()
        cuts: List[Tuple[Point, Point]] = []

        # ODD: peel off one 1/n share first (then even strategy on the rest)
        if self.children % 2 == 1:
            whole = self._largest_piece(work)
            cut = self._find_best_fractional_cut(work, whole, frac=1.0 / self.children)
            if not cut:
                cut = self._any_valid_cut_on_piece(work, whole)
            if not cut:
                raise PlayerException("Player2: no valid first cut for odd strategy")
            cuts.append(cut)
            work.cut(*cut)

        # EVEN (and remainder after odd’s first cut):
        # keep splitting the current largest piece until we have n pieces
        while len(work.get_pieces()) < self.children:
            piece_to_split = self._next_piece_to_split(work)
            cut = self._find_best_bisect_cut_on_piece(work, piece_to_split)
            if not cut:
                # try next largest piece if current one doesn’t admit any valid bisection
                cut = self._bisect_fallback_try_others(work, piece_to_split)
            if not cut:
                # ultimate fallback: any valid cut on the largest piece
                cut = self._any_valid_cut_on_piece(work, piece_to_split)
            if not cut:
                # last resort: any valid cut anywhere
                cut = self._any_valid_cut_anywhere(work)

            if not cut:
                raise PlayerException(
                    f"Player2: ran out of valid cuts at {len(work.get_pieces())} pieces"
                )

            cuts.append(cut)
            work.cut(*cut)

        if len(cuts) != self.children - 1:
            raise PlayerException(f"Player2: expected {self.children - 1} cuts, got {len(cuts)}")

        return cuts

    # --------------- selection policy ---------------
    def _next_piece_to_split(self, cake: Cake) -> Polygon:
        """
        Choose the piece to split next:
        - Prefer the largest piece whose area > target_area + tol.
        - If all are near/below target but we still need more pieces,
          pick the absolute largest.
        """
        pieces = list(cake.get_pieces())
        pieces.sort(key=lambda p: p.area, reverse=True)

        for p in pieces:
            if p.area > self.target_area + self.area_tol:
                return p
        return pieces[0]  # all are ~target; still need more pieces → split largest lightly

    def _bisect_fallback_try_others(self, cake: Cake, first_piece: Polygon) -> Optional[Tuple[Point, Point]]:
        """If the largest piece had no valid bisection, try the next largest, etc."""
        pieces = list(cake.get_pieces())
        pieces.sort(key=lambda p: p.area, reverse=True)
        for p in pieces:
            if p is first_piece:
                continue
            cut = self._find_best_bisect_cut_on_piece(cake, p)
            if cut:
                return cut
        return None

    # --------------- cut scoring & search ---------------
    def _find_best_bisect_cut_on_piece(self, cake: Cake, piece: Polygon) -> Optional[Tuple[Point, Point]]:
        """
        Pick the valid perimeter-to-perimeter cut on THIS piece that best splits it in half.
        (Area primary, ratio secondary). If no valid cut at all, return None.
        """
        pts = self._candidate_points(piece)
        best: Optional[Tuple[Point, Point]] = None
        best_score = inf

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = cake.cut_is_valid(a, b)
                if not ok:
                    continue
                score = self._bisect_score(cake, piece, a, b)
                if score < best_score:
                    best_score = score
                    best = (a, b)
        return best

    def _bisect_score(self, cake: Cake, piece: Polygon, a: Point, b: Point) -> float:
        split = cake.cut_piece(piece, a, b)
        if len(split) != 2:
            return inf
        P, Q = split
        half = piece.area / 2.0

        # Primary: area closeness to equal halves (squared error)
        area_err = (P.area - half) ** 2 + (Q.area - half) ** 2

        # Secondary: interior ratio closeness to target_ratio (absolute error avg)
        rP = self._piece_ratio(cake, P)
        rQ = self._piece_ratio(cake, Q)
        ratio_err = (abs(rP - self.target_ratio) + abs(rQ - self.target_ratio)) / 2.0

        return area_err + 0.1 * ratio_err  # area dominates; ratio breaks ties

    def _find_best_fractional_cut(self, cake: Cake, piece: Polygon, frac: float) -> Optional[Tuple[Point, Point]]:
        """
        For odd n: cut piece into (frac, 1-frac) of its area.
        """
        pts = self._candidate_points(piece)
        targetA = frac * piece.area
        targetB = (1.0 - frac) * piece.area

        best: Optional[Tuple[Point, Point]] = None
        best_score = inf

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                ok, _ = cake.cut_is_valid(a, b)
                if not ok:
                    continue
                split = cake.cut_piece(piece, a, b)
                if len(split) != 2:
                    continue
                P, Q = split
                area_err = (P.area - targetA) ** 2 + (Q.area - targetB) ** 2
                rP = self._piece_ratio(cake, P)
                rQ = self._piece_ratio(cake, Q)
                ratio_err = (abs(rP - self.target_ratio) + abs(rQ - self.target_ratio)) / 2.0
                score = area_err + 0.1 * ratio_err
                if score < best_score:
                    best_score = score
                    best = (a, b)

        return best

    # --------------- last-resort helpers ---------------
    def _any_valid_cut_on_piece(self, cake: Cake, piece: Polygon) -> Optional[Tuple[Point, Point]]:
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

    # --------------- geometry utilities ---------------
    def _largest_piece(self, cake: Cake) -> Polygon:
        return max(cake.get_pieces(), key=lambda p: p.area)

    def _piece_ratio(self, cake: Cake, poly: Polygon) -> float:
        if poly.is_empty or poly.area <= 0:
            return 0.0
        return cake.get_piece_ratio(poly)

    def _candidate_points(self, poly: Polygon) -> List[Point]:
        """
        Vertices + all edge midpoints (including the closing edge).
        This gives enough coverage while keeping search fast.
        """
        verts = list(poly.exterior.coords[:-1])
        pts = [Point(v) for v in verts]
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            pts.append(Point((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return pts
