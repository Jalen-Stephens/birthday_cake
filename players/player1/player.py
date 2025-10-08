from __future__ import annotations
from typing import List, Tuple, Iterable

from shapely import Point  # endpoints the engine expects
from shapely.geometry import Point as GeoPoint
from shapely.geometry import Polygon as ShapelyPolygon, LinearRing as ShapelyLinearRing
from shapely.geometry.base import BaseGeometry

from players.player import Player
from src.cake import Cake

# ----------------- Types & eps -----------------
Point2D = Tuple[float, float]
Poly = List[Point2D]
EPS = 1e-12


# ----------------- Polygon access -----------------
def _as_points(obj: Iterable) -> Poly:
    pts: Poly = []
    for p in obj:
        x, y = float(p[0]), float(p[1])
        pts.append((x, y))
    return pts


def _extract_polygon(cake: Cake) -> Poly:
    """Extract the cake outline in cm as list[(x,y)]."""
    # 1) Preferred: Shapely exterior on cake.exterior_shape
    ext = getattr(cake, "exterior_shape", None)
    if isinstance(ext, (ShapelyPolygon, ShapelyLinearRing, BaseGeometry)):
        coords = (
            list(ext.exterior.coords)
            if hasattr(ext, "exterior") and ext.exterior
            else list(ext.coords)
        )
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]  # drop duplicate closing vertex
        pts = _as_points(coords)
        if len(pts) >= 3:
            return pts

    # 2) Fallback: boundary points API (already in cm)
    if hasattr(cake, "get_boundary_points") and callable(cake.get_boundary_points):
        pts = _as_points(cake.get_boundary_points())
        if len(pts) >= 3:
            return pts

    # 3) Rare: interior shape
    intr = getattr(cake, "interior_shape", None)
    if isinstance(intr, (ShapelyPolygon, ShapelyLinearRing, BaseGeometry)):
        coords = (
            list(intr.exterior.coords)
            if hasattr(intr, "exterior") and intr.exterior
            else list(intr.coords)
        )
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        pts = _as_points(coords)
        if len(pts) >= 3:
            return pts

    raise ValueError("Player1: Could not find polygon points on Cake.")


# ----------------- Geometry core -----------------
def _area(poly: Poly) -> float:
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a  # signed


def _ensure_ccw(poly: Poly) -> Poly:
    return poly if _area(poly) > 0 else list(reversed(poly))


def _bbox(pts: Poly):
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


# --- clip polygon to half-plane x <= t (Sutherland–Hodgman) ---
def _clip_x_le(poly: Poly, t: float) -> Poly:
    def inside(p: Point2D) -> bool:
        return p[0] <= t + 1e-12

    def intersect(a: Point2D, b: Point2D) -> Point2D:
        x1, y1 = a
        x2, y2 = b
        dx = x2 - x1
        if abs(dx) < EPS:
            return (t, y1)  # vertical segment at x ~= t
        s = (t - x1) / dx
        return (t, y1 + s * (y2 - y1))

    out: Poly = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[(i - 1) % n]
        ci, pi = inside(cur), inside(prv)
        if ci:
            if not pi:
                out.append(intersect(prv, cur))
            out.append(cur)
        elif pi:
            out.append(intersect(prv, cur))
    return out


def _area_left_of_x(poly_ccw: Poly, x: float) -> float:
    return abs(_area(_clip_x_le(poly_ccw, x)))


# --- vertical intersections & chords ---
def _vertical_intersections(poly: Poly, x: float) -> List[float]:
    ys: List[float] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            dx = x2 - x1
            if abs(dx) < EPS:
                ys.extend([y1, y2])  # collect both ends of a vertical edge
            else:
                s = (x - x1) / dx
                ys.append(y1 + s * (y2 - y1))
    ys.sort()
    out: List[float] = []
    for y in ys:
        if not out or abs(y - out[-1]) > 1e-9:  # dedupe vertex touches
            out.append(y)
    return out


def _vertical_chords(poly: Poly, x: float) -> list[tuple[float, float]]:
    ys = _vertical_intersections(poly, x)
    chords: list[tuple[float, float]] = []
    for i in range(0, len(ys) - 1, 2):
        y0, y1 = ys[i], ys[i + 1]
        if y1 - y0 > 1e-9:
            chords.append((y0, y1))
    return chords


def _find_cut_x(poly_ccw: Poly, target_area: float, lo: float, hi: float) -> float:
    for _ in range(60):  # ~double precision
        mid = 0.5 * (lo + hi)
        a = _area_left_of_x(poly_ccw, mid)
        if a < target_area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- ask the engine if a chord is “good” (no slivers etc.) ---
def _is_good_cut(cake: Cake, x: float, y0: float, y1: float) -> bool:
    a, b = Point(x, y0), Point(x, y1)
    if hasattr(cake, "cut_is_valid"):
        try:
            ok, reason = cake.cut_is_valid(a, b)
            if not ok:
                return False
        except Exception:
            return False
    if hasattr(cake, "does_line_cut_piece_well"):
        if not cake.does_line_cut_piece_well(a, b):
            return False
    return True


# ----------------- Player -----------------
class Player1(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Return children-1 valid vertical cuts (concave-safe, near equal-area)."""
        n = int(self.children)
        if n <= 1:
            return []

        # polygon in cm
        poly = _extract_polygon(self.cake)
        poly = _ensure_ccw(poly)

        total = abs(_area(poly))
        if total < 1e-10:
            return []

        xmin, ymin, xmax, ymax = _bbox(poly)
        W = xmax - xmin
        if W <= 0:
            return []

        # helpers to choose a feasible chord at x
        def try_x(xx: float) -> tuple[float, float] | None:
            chords = _vertical_chords(poly, xx)
            if not chords:
                return None
            # Prefer longer chords first (less likely to make slivers)
            chords.sort(key=lambda ab: ab[1] - ab[0], reverse=True)

            # 1) engine validation if available
            for yl, yh in chords:
                if _is_good_cut(self.cake, xx, yl, yh):
                    return (yl, yh)

            # 2) shapely containment of chord midpoint as a fallback
            ext = getattr(self.cake, "exterior_shape", None)
            if isinstance(ext, BaseGeometry):
                for yl, yh in chords:
                    mid = 0.5 * (yl + yh)
                    if ext.contains(GeoPoint(xx, mid)):
                        return (yl, yh)

            # 3) last resort pick the longest span
            return chords[0] if chords else None

        cuts: list[tuple[Point, Point]] = []

        # small neighborhood to dodge tips/vertices (relative to cake width)
        nudges = [
            0.0,
            0.001 * W,
            -0.001 * W,
            0.005 * W,
            -0.005 * W,
            0.01 * W,
            -0.01 * W,
            0.02 * W,
            -0.02 * W,
            0.05 * W,
            -0.05 * W,
        ]

        for i in range(1, n):
            # ideal area target
            target = total * (i / n)
            x0 = _find_cut_x(poly, target, xmin, xmax)

            chosen_x = None
            chosen = None

            # 1) try ideal x with nudges
            for dx in nudges:
                xx = x0 + dx
                res = try_x(xx)
                if res and _is_good_cut(self.cake, xx, res[0], res[1]):
                    chosen_x, chosen = xx, res
                    break

            # 2) if still nothing, relax area target slightly (±0.5% → ±3%)
            if chosen is None:
                for pct in (0.005, -0.005, 0.015, -0.015, 0.03, -0.03):
                    adj = total * max(min((i / n) + pct, 1.0), 0.0)
                    x2 = _find_cut_x(poly, adj, xmin, xmax)
                    for dx in nudges:
                        xx = x2 + dx
                        res = try_x(xx)
                        if res and _is_good_cut(self.cake, xx, res[0], res[1]):
                            chosen_x, chosen = xx, res
                            break
                    if chosen is not None:
                        break

            # 3) as a final fallback, scan a coarse grid across the cake
            if chosen is None:
                grid = [xmin + k * (W / 40.0) for k in range(41)]
                for xx in grid:
                    res = try_x(xx)
                    if res and _is_good_cut(self.cake, xx, res[0], res[1]):
                        chosen_x, chosen = xx, res
                        break

            # if truly nothing feasible, skip (avoid invalid move)
            if chosen is None or chosen_x is None:
                continue

            yl, yh = chosen
            cuts.append((Point(chosen_x, yl), Point(chosen_x, yh)))

        return cuts
