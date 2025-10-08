from shapely import Point, LineString

# import random
from players.player import Player
from players.random_player import RandomPlayer
from src.cake import Cake
import src.constants as c


class CrustOptimizingPlayer(Player):
    """
    Strategy:
    - Samples random points along the cake crust (outer edge)
    - Computes crust density to find crust-heavy regions
    - Picks starting point p1 from densest crust points
    - Chooses p2 and evaluates cut quality using weighted scoring
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.random_player = RandomPlayer(children, cake, cake_path)
        self.valid_line_pair = dict()
        self.tolerance = 0.05
        self.max_crust_points = 3  # how many top crust-heavy points to consider

    # ---------- Utility metrics ----------

    def get_crust_length(self, piece):
        """Length of piece boundary that lies on the exterior crust."""
        return piece.boundary.intersection(self.cake.exterior_shape.boundary).length

    def get_crust_density(self, piece, p: Point):
        """Approximate local crust density near a boundary point. This is only meaningful for the initial cut."""
        crust_section = self.cake.exterior_shape.boundary.buffer(c.CRUST_SIZE * 2)
        nearby = piece.boundary.intersection(crust_section)
        return nearby.length

    def get_piece(self, p1, p2):
        """Find which piece both points belong to."""
        for piece in self.cake.get_pieces():
            if self.cake.point_lies_on_piece_boundary(
                p1, piece
            ) and self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

    # Inside CrustOptimizingPlayer class

    def check_precision(self, p1, p2, Area_list, piece, crust_ratio):
        """Return a tuple containing the absolute difference between the piece area and the closest
        target area, and the absolute difference between the new crust ratio and the target crust ratio."""

        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) < 2:
            return float("inf"), float("inf")  # Return a tuple for invalid cuts

        if split_pieces[0].area < split_pieces[1].area:
            Area_piece = split_pieces[0].area
            closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
            cake_precision = (
                abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
            )

            new_piece1_crust_ratio = self.cake.get_piece_ratio(split_pieces[0])

            crust_precision = abs(new_piece1_crust_ratio - crust_ratio)
            return cake_precision, crust_precision
        else:
            Area_piece = split_pieces[1].area
            closest_target_area = min(Area_list, key=lambda a: abs(a - Area_piece))
            cake_precision = (
                abs(Area_piece - closest_target_area) / self.cake.exterior_shape.area
            )

            new_piece1_crust_ratio = self.cake.get_piece_ratio(split_pieces[1])

            crust_precision = abs(new_piece1_crust_ratio - crust_ratio)
            return cake_precision, crust_precision

    def get_weight(self, p1, p2, piece, Goal_ratio):
        """Compute weight for a given cut — based on area ratio deviation."""
        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) != 2:
            return 0
        r1 = self.cake.get_piece_ratio(split_pieces[0])
        r2 = self.cake.get_piece_ratio(split_pieces[1])
        weight = 0.5 * (1 - abs(Goal_ratio - r1)) + 0.5 * (1 - abs(Goal_ratio - r2))
        return weight

    # ---------- Main Algorithm ----------

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        # num_p1 = 125
        # num_p2 = 300

        piece = max(self.cake.get_pieces(), key=lambda p: p.area)
        crust_ratio = self.cake.get_piece_ratio(piece)

        Total_Area = self.cake.exterior_shape.area
        Area_list = []
        for i in range(1, self.children):
            Area_list += [(Total_Area / self.children) * i]
        for k in range(self.children - 1):
            print(f"Cut {k + 1}/{self.children - 1}")

            best_line_list = []
            best_line = [100, None, None, 100]
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            piece_boundary = piece.boundary

            num_candidates = 240  # You can adjust this number
            step_size = piece_boundary.length / num_candidates
            candidates = [
                piece_boundary.interpolate(i * step_size) for i in range(num_candidates)
            ]

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = candidates[i]
                    p2 = candidates[j]
                    # print(p1, p2)
                    good, _ = self.cake.does_line_cut_piece_well(
                        LineString((p1, p2)), piece
                    )

                    if not good:
                        # print(_)
                        continue

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        # print(p1, p2)
                        continue
                    # print(p1,p2)
                    cake_precision, crust_precision = self.check_precision(
                        p1, p2, Area_list, piece, crust_ratio
                    )
                    if cake_precision == float("inf"):
                        continue

                    if best_line[0] > cake_precision:
                        best_line = [cake_precision, p1, p2, crust_precision]

                    if cake_precision < 0.001:
                        best_line_list.append((cake_precision, p1, p2, crust_precision))

            # print(best_line)
            print(len(best_line_list))

            if len(best_line_list) > 0:
                best_line_list.sort(key=lambda x: (x[3] + 0.001) * (x[0] + 0.1))
                bestline = best_line_list[0]
            elif best_line[1] is not None:
                print("No optimal crust found")
                bestline = best_line
            else:
                print("random strategy")
                a, b = self.random_player.find_random_cut()
                bestline = [100, a, b, 100]
            # --- Step 4: Execute cut ---
            # print(bestline)
            moves.append((bestline[1], bestline[2]))
            self.cake.cut(bestline[1], bestline[2])

        return moves
