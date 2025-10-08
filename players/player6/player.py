from dataclasses import dataclass
from players.player import Player
from src.cake import Cake
from shapely.geometry import Polygon, LineString, Point
from shapely import MultiLineString, intersection
from shapely.ops import split
from typing import cast
import numpy as np
from math import hypot

import src.constants as c


@dataclass
class CutResult:
    polygons: list[Polygon]
    points: tuple[Point, Point]


def extend_line(line: LineString, fraction: float = 0.05) -> LineString:
    coords = list(line.coords)
    if len(coords) != 2:
        return line

    (x1, y1), (x2, y2) = coords
    dx, dy = (x2 - x1), (y2 - y1)
    L = hypot(dx, dy)
    if L == 0:
        return line

    ux, uy = dx / L, dy / L

    a = fraction * L

    x1n, y1n = x1 - a * ux, y1 - a * uy
    x2n, y2n = x2 + a * ux, y2 + a * uy

    return LineString([(x1n, y1n), (x2n, y2n)])


class Player6(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.target_area = self.cake.get_area() / self.children
        self.target_ratio = (
            self.cake.interior_shape.area / self.cake.exterior_shape.area
        )

    def find_common_points(self, pieces: list[Polygon]) -> tuple[tuple[float, float]]:
        """Find the vertices of the new cake polygon once cut"""
        return tuple(set(pieces[0].exterior.coords) & set(pieces[1].exterior.coords))

    def __cut_is_within_cake(self, cut: LineString, polygon: Polygon) -> bool:
        outside = cut.difference(polygon.buffer(c.TOL * 2))
        return outside.is_empty

    def get_intersecting_pieces_from_point(self, p: Point, polygons: list[Polygon]):
        touched_pieces = [
            piece for piece in polygons if self.point_lies_on_piece_boundary(p, piece)
        ]

        return touched_pieces

    def get_scaled_vertex_points(self):
        x_offset, y_offset = self.get_offsets()
        xys = self.get_boundary_points()

        ext_points = []
        for xy in xys:
            x, y = xy.coords[0]
            ext_points.extend(
                [x * c.CAKE_SCALE + x_offset, y * c.CAKE_SCALE + y_offset]
            )

        int_xys = self.get_interior_points()

        int_points = []
        for int_xy in int_xys:
            ip = []
            for xy in int_xy:
                x, y = xy.coords[0]
                ip.extend([x * c.CAKE_SCALE + x_offset, y * c.CAKE_SCALE + y_offset])
            int_points.append(ip)

        return ext_points, int_points

    def point_lies_on_piece_boundary(self, p: Point, piece: Polygon):
        return p.distance(piece.boundary) <= c.TOL

    def get_cuttable_piece(self, from_p: Point, to_p: Point, polygons: list[Polygon]):
        a_pieces = self.get_intersecting_pieces_from_point(from_p, polygons)
        b_pieces = self.get_intersecting_pieces_from_point(to_p, polygons)

        contenders = set(a_pieces).intersection(set(b_pieces))

        if len(contenders) > 1:
            return None, "line can cut multiple pieces, should only cut one"

        if len(contenders) == 0:
            return None, "line doesn't cut any piece of cake well"

        piece = list(contenders)[0]

        # snap points to piece boundary
        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = extend_line(LineString([a, b]))

        piece_well_cut, reason = self.does_line_cut_piece_well(line, piece)
        if not piece_well_cut:
            return None, reason

        return piece, ""

    def does_line_cut_piece_well(self, line: LineString, piece: Polygon):
        """Checks whether line cuts piece in two valid (large enough) pieces"""
        if piece.touches(line):
            return False, "cut lies on piece boundary"

        if not line.crosses(piece):
            return False, "line does not cut through piece"

        cut_pieces = split(piece, line)
        if len(cut_pieces.geoms) != 2:
            return False, f"line cuts piece in {len(cut_pieces.geoms)}, not 2"

        all_sizes_are_good = all([p.area >= c.MIN_PIECE_AREA for p in cut_pieces.geoms])

        if not all_sizes_are_good:
            return False, "line cuts a piece that's too small"

        return True, ""

    def cut_is_valid(
        self, from_p: Point, to_p: Point, polygon: Polygon
    ) -> tuple[bool, str]:
        """Check whether a cut from `from_p` to `to_p` is valid.

        If invalid, the method returns the reason as the second argument.
        """
        line = LineString([from_p, to_p])

        if not self.__cut_is_within_cake(line, polygon):
            return False, "cut is not within cake"

        cuttable_piece, reason = self.get_cuttable_piece(from_p, to_p, [polygon])

        if not cuttable_piece:
            return False, reason

        return True, "valid"

    def virtual_cut(self, piece: Polygon, cut: LineString) -> CutResult:
        """
        Make a virtual cut over our existing cake(piece)
        """

        # --------
        # >>> from shapely import LineString
        # >>> x, y = LineString([(0, 0), (1, 1)]).xy
        # >>> list(x)
        # [0.0, 1.0]
        # >>> list(y)
        # [0.0, 1.0]
        # x, y = cut.coords
        coords = list(cut.coords)
        from_p, to_p = Point(coords[0]), Point(coords[1])

        is_valid, reason = self.cut_is_valid(from_p, to_p, piece)
        # NOTE: catch this later
        if not is_valid:
            return None

        # as this cut is valid, we will have exactly one cuttable piece
        target_piece, _ = self.get_cuttable_piece(from_p, to_p, [piece])
        if not target_piece:
            return None

        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        # line = extend_line(line)
        # cut the cake
        split_piece = split(piece, line)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]
        # point1, point2 = line.coords
        # output = CutResult(polygons=split_pieces, points=(Point(point1), Point(point2)))
        output = CutResult(polygons=split_pieces, points=(a, b))

        return output

    def current_polygon(self, cut: CutResult) -> set[Polygon]:
        cut_points, polygons = cut.points, cut.polygons
        res: set[Polygon] = set()
        for polygon in polygons:
            for x, y in polygon.exterior.coords:
                for point in cut_points:
                    if point.x == x and point.y == y:
                        res.add(polygon)
        return res

    def get_piece_ratio(self, piece: Polygon):
        if piece.intersects(self.cake.interior_shape):
            inter = piece.intersection(self.cake.interior_shape)
            return inter.area / piece.area if not inter.is_empty else 0
        return 0

    def score_cut(self, cut: CutResult) -> tuple[float, float]:
        if cut is None:
            return (float("inf"), float("inf"))

        polygons = self.current_polygon(cut)

        if len(polygons) != 2:
            return (float("inf"), float("inf"))

        area_scores = [abs(polygon.area - self.target_area) for polygon in polygons]
        ratio_scores = [
            abs(self.get_piece_ratio(polygon) - self.target_ratio)
            for polygon in polygons
        ]

        area_score = min(area_scores)
        ratio_score = ratio_scores[area_scores.index(min(area_scores))]

        # If difference from target area < 0.125, treat it as equal → rely on ratio
        if area_score <= 0.125:
            area_score = 0.0
        if ratio_score <= 0.05:
            ratio_score = 0.0
        return (area_score, ratio_score)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Adaptive coarse-to-fine search for near-optimal cuts with convergence criteria."""
        result: list[tuple[Point, Point]] = []

        while len(result) < self.children - 1:
            largest_piece = self.get_max_piece(self.cake.exterior_pieces)
            min_x, min_y, max_x, max_y = largest_piece.exterior.bounds
            cut_fracs = np.linspace(0.01, 1, 1000)
            best_slice, best_score = None, (float("inf"), float("inf"))

            for frac in cut_fracs:
                for try_fn in (self._try_x_slice, self._try_y_slice):
                    cuts = try_fn(
                        frac * self.children, min_x, max_x, min_y, max_y, largest_piece
                    )
                    if not cuts:
                        continue
                    for cut in cuts:
                        if cut is None:
                            continue
                        score = self.score_cut(cut)
                        if score < best_score:
                            best_score, best_slice = score, cut

            if not best_slice:
                break

            if best_slice:
                result.append(best_slice.points)
                self.cake.cut(best_slice.points[0], best_slice.points[1])

        return result

    def _try_x_slice(
        self,
        iteration: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
    ) -> list[CutResult] | None:
        """Attempt to make a vertical cut at the given iteration."""
        x_span = max_x - min_x
        x_position = iteration * x_span / self.children + min_x
        x_slice = LineString([[x_position, min_y], [x_position, max_y]])

        # Find intersection with the piece
        intersections = intersection(x_slice, piece)
        if isinstance(intersections, MultiLineString) and not intersections.is_empty:
            return [self.virtual_cut(piece, line) for line in intersections.geoms]
        if isinstance(intersections, LineString) and not intersections.is_empty:
            return [self.virtual_cut(piece, intersections)]
        else:
            return None

    def _try_y_slice(
        self,
        iteration: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
    ) -> list[CutResult] | None:
        """Attempt to make a horizontal cut at the given iteration."""
        y_span = max_y - min_y
        y_position = iteration * y_span / self.children + min_y
        y_slice = LineString([[min_x, y_position], [max_x, y_position]])

        # Find intersection with the piece
        intersections = intersection(y_slice, piece)
        if isinstance(intersections, MultiLineString) and not intersections.is_empty:
            return [self.virtual_cut(piece, line) for line in intersections.geoms]
        if isinstance(intersections, LineString) and not intersections.is_empty:
            return [self.virtual_cut(piece, intersections)]
        else:
            return None

    def get_max_piece(self, pieces: list[Polygon]) -> Polygon:
        return max(pieces, key=lambda piece: piece.area)
