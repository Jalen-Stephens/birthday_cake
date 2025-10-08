from shapely import Point, Polygon
from shapely.geometry import LineString
from src.cake import Cake


def get_perimeter_points(cake, num_samples: int):
    """Get points on the perimeter for potential cutting."""
    # get the current largest polygon piece (the cake body)
    largest_piece = max(cake.get_pieces(), key=lambda piece: piece.area)

    boundary = largest_piece.exterior
    points = [
        boundary.interpolate(i / num_samples, normalized=True)
        for i in range(num_samples)
    ]
    return points  # Return Point objects directly


def _would_cut_along_boundary(p1: Point, p2: Point, piece: Polygon) -> bool:
    """Check if a cut between two points would lie along the piece boundary."""

    # Get the piece boundary
    boundary = piece.boundary

    # Check if both points are very close to the boundary
    p1_on_boundary = p1.distance(boundary) < 0.1
    p2_on_boundary = p2.distance(boundary) < 0.1

    if not (p1_on_boundary and p2_on_boundary):
        return False

    # Create the cut line
    cut_line = LineString([p1, p2])

    # Check if the line is very close to the boundary
    line_distance_to_boundary = cut_line.distance(boundary)

    # Additional check: if the line is mostly along the boundary, skip it
    # This is a more sophisticated check that looks at the angle
    if line_distance_to_boundary < 0.1:
        # Check if the line is roughly parallel to the boundary
        # by sampling points along the line and checking their distance to boundary
        line_length = cut_line.length
        if line_length > 0:
            # Sample points along the line
            num_samples = max(3, int(line_length / 0.5))  # Sample every 0.5 units
            all_close_to_boundary = True

            for i in range(num_samples + 1):
                t = i / num_samples
                sample_point = cut_line.interpolate(t, normalized=True)
                if sample_point.distance(boundary) > 0.2:  # Not close to boundary
                    all_close_to_boundary = False
                    break

            return all_close_to_boundary

    return False


def get_areas(
    cake: Cake,
    xy1: Point,
    xy2: Point,
    target_ratio: float = 0.5,
    original_area: float = None,
    acceptable_error: float = 1.0,
) -> tuple[bool, float | None]:
    """Find cuts that produce pieces with target ratio. Enhanced with configurable tolerance."""
    valid, _ = cake.cut_is_valid(xy1, xy2)

    if not valid:
        return False, None

    target_piece, _ = cake.get_cuttable_piece(xy1, xy2)

    if target_piece is None:
        return False, None

    split_pieces = cake.cut_piece(target_piece, xy1, xy2)

    areas = [piece.area for piece in split_pieces]

    # check the ratio with whole cake area
    target_area = original_area * target_ratio

    # check if one piece is the target area of 1/children
    area_diffs = [abs(area - target_area) for area in areas]
    min_diff = min(area_diffs)

    if min_diff <= acceptable_error:
        return True, min_diff

    return False, min_diff


# probably can use binary search to make this faster and optimize our search route instead of n^2 time complexity going through each point
def find_valid_cuts(
    cake: Cake,
    perim_points: list[Point] | None = None,
    target_ratio: float = 0.5,
    original_area: float = None,
    acceptable_error: float = 1.0,
) -> list[tuple[Point, Point, Polygon]]:
    valid_cuts = []
    skipped_boundary_cuts = 0

    for piece in cake.exterior_pieces:
        # try all pairs
        for i, xy1 in enumerate(perim_points):
            for xy2 in perim_points[i + 1 :]:
                # OPTIMIZATION: Skip cuts that would lie along the boundary (performance improvement)
                if _would_cut_along_boundary(xy1, xy2, piece):
                    skipped_boundary_cuts += 1
                    continue

                if cake.cut_is_valid(xy1, xy2):
                    areas_valid, area_diff = get_areas(
                        cake, xy1, xy2, target_ratio, original_area, acceptable_error
                    )

                    if areas_valid:
                        valid_cuts.append((xy1, xy2, piece, area_diff))

    # sort by almost to least accurate just in terms of area (future incorporate ratio)
    valid_cuts.sort(key=lambda x: x[3])

    # Optional: Print optimization stats
    if skipped_boundary_cuts > 0:
        print(f"Optimization: Skipped {skipped_boundary_cuts} boundary cuts")

    return [(xy1, xy2, piece) for xy1, xy2, piece, _ in valid_cuts]
