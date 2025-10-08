from shapely import Point

from players.player import Player, PlayerException
from src.cake import Cake


class Player7(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.target_area = cake.get_area() / children

    def evaluate_cut(self, from_p: Point, to_p: Point) -> float:
        """Evaluate how good a cut is by measuring deviation from even multiples of target area.
        Returns the sum of squared differences from the nearest even multiple of target area."""

        # Check if cut is valid first
        is_valid, _ = self.cake.cut_is_valid(from_p, to_p)
        if not is_valid:
            return float("inf")  # Invalid cuts get worst possible score

        try:
            # Make a copy of the cake to evaluate the cut
            cake_copy = self.cake.copy()
            cake_copy.cut(from_p, to_p)

            # Get the piece sizes after the cut
            piece_sizes = cake_copy.get_piece_sizes()

            # Calculate total deviation from even multiples of target area
            total_deviation = 0.0
            for piece_size in piece_sizes:
                # Find the nearest even multiple of target area
                multiple = round(piece_size / self.target_area)
                nearest_even_multiple = multiple * self.target_area
                deviation = abs(piece_size - nearest_even_multiple)
                total_deviation += deviation * deviation  # Sum of squared deviations

            return total_deviation

        except Exception:
            return float("inf")

    def get_sample_points(self, piece) -> list[Point]:
        """Get sample points along the piece boundary including vertices, midpoints, and quarter points."""
        coords = list(piece.exterior.coords[:-1])  # Exclude the duplicate last point
        sample_points = []

        for i in range(len(coords)):
            # Add vertex
            sample_points.append(Point(coords[i]))

            # Add midpoint and quarter points of the edge to next vertex
            next_i = (i + 1) % len(coords)
            start = coords[i]
            end = coords[next_i]

            # Quarter point (1/4 along the edge)
            quarter_x = start[0] + 0.25 * (end[0] - start[0])
            quarter_y = start[1] + 0.25 * (end[1] - start[1])
            sample_points.append(Point(quarter_x, quarter_y))

            # Midpoint (1/2 along the edge)
            mid_x = start[0] + 0.5 * (end[0] - start[0])
            mid_y = start[1] + 0.5 * (end[1] - start[1])
            sample_points.append(Point(mid_x, mid_y))

            # Three-quarter point (3/4 along the edge)
            three_quarter_x = start[0] + 0.75 * (end[0] - start[0])
            three_quarter_y = start[1] + 0.75 * (end[1] - start[1])
            sample_points.append(Point(three_quarter_x, three_quarter_y))

        return sample_points

    def find_best_cut(self) -> tuple[Point, Point]:
        """Find the cut that minimizes deviation from target area."""
        best_cut = None
        best_score = float("inf")

        # Get all pieces that could potentially be cut
        pieces = self.cake.get_pieces()

        for piece in pieces:
            # Only consider pieces that are larger than twice the minimum area
            # to ensure we can make a valid cut
            # Skip pieces that are too small to cut or already close to target area
            target_area = self.cake.get_area() / self.children
            if (
                abs(piece.area - target_area) < 0.5
            ):  # Skip pieces already close to target
                continue

            # Get sample points along the piece boundary
            sample_points = self.get_sample_points(piece)

            # Try cuts between different sample points
            for i in range(len(sample_points)):
                for j in range(i + 1, len(sample_points)):
                    from_p = sample_points[i]
                    to_p = sample_points[j]

                    score = self.evaluate_cut(from_p, to_p)

                    if score < best_score:
                        best_score = score
                        best_cut = (from_p, to_p)

        if best_cut is None:
            raise PlayerException("could not find a valid cut")

        return best_cut

    def get_boundary_direction(self, piece, point: Point) -> tuple[float, float]:
        """Get the direction vector along the piece boundary at the given point."""
        boundary = piece.boundary

        # Find the closest point on the boundary
        closest_point = boundary.interpolate(boundary.project(point))

        # Get a small offset along the boundary in both directions
        distance = boundary.project(closest_point)
        offset = 0.01

        # Get points slightly before and after
        before_point = boundary.interpolate((distance - offset) % boundary.length)
        after_point = boundary.interpolate((distance + offset) % boundary.length)

        # Calculate direction vector
        dx = after_point.x - before_point.x
        dy = after_point.y - before_point.y

        # Normalize the direction vector
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            return dx / length, dy / length
        else:
            return 0.0, 0.0

    def optimize_cut(
        self, from_p: Point, to_p: Point, iterations: int = 50
    ) -> tuple[Point, Point]:
        """Optimize a cut by moving points along the boundary direction."""
        best_cut = (from_p, to_p)
        best_score = self.evaluate_cut(from_p, to_p)

        # If the initial cut is invalid, return it as-is
        if best_score == float("inf"):
            return best_cut

        # Find the piece that this cut would affect
        cuttable_piece, _ = self.cake.get_cuttable_piece(from_p, to_p)
        if not cuttable_piece:
            return best_cut

        current_from = Point(from_p.x, from_p.y)
        current_to = Point(to_p.x, to_p.y)

        # Step size for optimization (start larger, decrease over time)
        initial_step_size = 0.2

        for iteration in range(iterations):
            # Calculate step size (decreases over iterations)
            step_size = initial_step_size * (1 - iteration / iterations)

            improved = False

            # Try moving each point along the boundary in both directions
            for direction in [-1, 1]:
                # Get boundary direction at current from point
                from_dx, from_dy = self.get_boundary_direction(
                    cuttable_piece, current_from
                )

                # Move the from point along the boundary
                new_from = Point(
                    current_from.x + direction * step_size * from_dx,
                    current_from.y + direction * step_size * from_dy,
                )

                new_score = self.evaluate_cut(new_from, current_to)

                if new_score < best_score:
                    best_score = new_score
                    best_cut = (new_from, current_to)
                    current_from = new_from
                    improved = True
                    break

                # Get boundary direction at current to point
                to_dx, to_dy = self.get_boundary_direction(cuttable_piece, current_to)

                # Move the to point along the boundary
                new_to = Point(
                    current_to.x + direction * step_size * to_dx,
                    current_to.y + direction * step_size * to_dy,
                )

                new_score = self.evaluate_cut(current_from, new_to)

                if new_score < best_score:
                    best_score = new_score
                    best_cut = (current_from, new_to)
                    current_to = new_to
                    improved = True
                    break

            # If no improvement found, continue with next iteration
            if not improved:
                continue

        return best_cut

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        for _ in range(self.children - 1):
            from_p, to_p = self.find_best_cut()

            # Optimize the cut with 20 iterations of improvement
            optimized_from_p, optimized_to_p = self.optimize_cut(
                from_p, to_p, iterations=20
            )

            moves.append((optimized_from_p, optimized_to_p))

            # Simulate the cut on our cake to maintain accurate state
            self.cake.cut(optimized_from_p, optimized_to_p)

        return moves
