from shapely.geometry import Point, Polygon
from typing import Optional
from math import cos, sin, radians
import random

from players.player import Player
from src.cake import Cake, get_polygon_angles


class Player8(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        working_cake = self.cake.copy()
        all_cuts = []

        def divide(pieces_needed: int, target_piece: Polygon):
            if pieces_needed == 1:  # base case
                return

            cut = self.find_good_cut(target_piece, working_cake)  # find good cut
            if not cut:
                # no valid cut found for this piece; stop dividing this branch
                return

            from_p, to_p = cut
            new_pieces = working_cake.cut_piece(target_piece, from_p, to_p)
            all_cuts.append((from_p, to_p))

            # recursively divide each new piece
            divide(pieces_needed // 2, new_pieces[0])
            divide(pieces_needed - pieces_needed // 2, new_pieces[1])

        piece = working_cake.get_pieces()[0]
        divide(self.children, piece)

        return all_cuts

    # -------------------------------------------------------------------------
    # CUTTING STRATEGIES
    # -------------------------------------------------------------------------

    def find_good_cut(
        self, piece: Polygon, working_cake: Cake
    ) -> Optional[tuple[Point, Point]]:
        """Adaptive cut finder:
        1. Try centroid-based method first
        2. If no cut found or polygon is complex, try boundary pairs
        3. If still no cut, try random angles as last resort
        """
        # 1. Try centroid-based strategy first
        best_cut = self._find_cut_by_centroid(piece, working_cake)

        # 2. Check polygon complexity and maybe switch to fallback
        if best_cut is None or self._is_complex_polygon(piece):
            boundary_cut = self._find_cut_by_boundary_pairs(piece, working_cake)
            if boundary_cut is not None:
                best_cut = boundary_cut

        # 3. Optional random fallback
        if best_cut is None:
            random_cut = self._find_cut_by_random_angles(piece, working_cake)
            if random_cut is not None:
                best_cut = random_cut

        return best_cut

    # -------------------------------------------------------------------------
    # HELPER: Centroid-based cut search
    # -------------------------------------------------------------------------
    def _find_cut_by_centroid(
        self, piece: Polygon, working_cake: Cake
    ) -> Optional[tuple[Point, Point]]:
        centroid = piece.centroid
        radius = max(centroid.distance(Point(p)) for p in piece.exterior.coords) * 2

        def get_cut_points(angle_deg: float) -> tuple[Point, Point]:
            angle_rad = radians(angle_deg)
            dx = cos(angle_rad)
            dy = sin(angle_rad)
            return (
                Point(centroid.x - dx * radius, centroid.y - dy * radius),
                Point(centroid.x + dx * radius, centroid.y + dy * radius),
            )

        best_score = (2, float("inf"))
        best_cut = None
        best_angle = None

        # Coarse search (every 10 degrees)
        for angle in range(0, 180, 10):
            from_p, to_p = get_cut_points(angle)
            score = self._score_cut(piece, working_cake, from_p, to_p)
            if score < best_score:
                best_score = score
                best_cut = (from_p, to_p)
                best_angle = angle

        # Refined search around best angle, if found
        if best_angle is not None:
            for angle in range(max(0, best_angle - 10), min(180, best_angle + 10)):
                from_p, to_p = get_cut_points(angle)
                score = self._score_cut(piece, working_cake, from_p, to_p)
                if score < best_score:
                    best_score = score
                    best_cut = (from_p, to_p)

        # Snap to boundary
        if best_cut is not None:
            bound = piece.boundary
            from_p = bound.interpolate(bound.project(best_cut[0]))
            to_p = bound.interpolate(bound.project(best_cut[1]))
            best_cut = (from_p, to_p)

        return best_cut

    # -------------------------------------------------------------------------
    # HELPER: Boundary-to-boundary pair search
    # -------------------------------------------------------------------------
    def _find_cut_by_boundary_pairs(
        self, piece: Polygon, working_cake: Cake
    ) -> Optional[tuple[Point, Point]]:
        coords = list(piece.exterior.coords)
        n = len(coords)

        best_score = (2, float("inf"))
        best_cut = None

        # To avoid O(n^2) explosion on very large polygons, subsample boundary
        step = max(1, n // 40)  # e.g. at most ~40 points
        sampled_indices = range(0, n, step)

        for i in sampled_indices:
            for j in sampled_indices:
                if j <= i:
                    continue
                fp = Point(coords[i])
                tp = Point(coords[j])
                score = self._score_cut(piece, working_cake, fp, tp)
                if score < best_score:
                    best_score = score
                    best_cut = (fp, tp)

                    # if we found a perfect area-balanced cut, stop early
                    if best_score[0] == 0:
                        return best_cut

        return best_cut

    # -------------------------------------------------------------------------
    # HELPER: Random angle fallback
    # -------------------------------------------------------------------------
    def _find_cut_by_random_angles(
        self, piece: Polygon, working_cake: Cake
    ) -> Optional[tuple[Point, Point]]:
        centroid = piece.centroid
        radius = max(centroid.distance(Point(p)) for p in piece.exterior.coords) * 2

        def get_cut_points(angle_deg: float) -> tuple[Point, Point]:
            angle_rad = radians(angle_deg)
            dx = cos(angle_rad)
            dy = sin(angle_rad)
            return (
                Point(centroid.x - dx * radius, centroid.y - dy * radius),
                Point(centroid.x + dx * radius, centroid.y + dy * radius),
            )

        best_score = (2, float("inf"))
        best_cut = None

        for _ in range(100):
            angle = random.uniform(0, 180)
            from_p, to_p = get_cut_points(angle)
            score = self._score_cut(piece, working_cake, from_p, to_p)
            if score < best_score:
                best_score = score
                best_cut = (from_p, to_p)
                if best_score[0] == 0:  # found a good area-balanced cut
                    break

        return best_cut

    # -------------------------------------------------------------------------
    # HELPER: Cut scoring
    # -------------------------------------------------------------------------
    def _score_cut(
        self, piece: Polygon, working_cake: Cake, from_p: Point, to_p: Point
    ) -> tuple[int, float]:
        bound = piece.boundary
        from_p = bound.interpolate(bound.project(from_p))
        to_p = bound.interpolate(bound.project(to_p))

        is_valid, _ = working_cake.cut_is_valid(from_p, to_p)
        if not is_valid:
            return (2, float("inf"))

        try:
            cut_pieces = working_cake.cut_piece(piece, from_p, to_p)
            if len(cut_pieces) != 2:
                return (2, float("inf"))

            areas = [p.area for p in cut_pieces]
            ratios = [working_cake.get_piece_ratio(p) for p in cut_pieces]

            area_diff = abs(areas[0] - areas[1])
            ratio_diff = abs(ratios[0] - ratios[1])

            if area_diff < 0.5:  # tolerance for equal area
                return (0, ratio_diff)
            else:
                return (1, area_diff)
        except Exception:
            return (2, float("inf"))

    # -------------------------------------------------------------------------
    # HELPER: Polygon complexity check
    # -------------------------------------------------------------------------
    def _is_complex_polygon(self, piece: Polygon) -> bool:
        hull = piece.convex_hull
        hull_ratio = piece.area / hull.area if hull.area != 0 else 0
        if hull_ratio < 0.85:
            return True

        angles = get_polygon_angles(piece)
        num_reflex = sum(1 for a in angles if a < 180)
        return num_reflex > 3
