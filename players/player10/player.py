from shapely.geometry import LineString, Point, Polygon
from players.player import Player
from src.cake import Cake
from shapely.ops import split


class Player10(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        # is area within some tolerance of the target area
        self.target_area_tolerance = 0.0

    def find_line(self, x: float, peice: Polygon):
        """Make a line through some x coordinate that gices equal areas"""

        # bounding box of piece, left and rightmos peices, highest and lowest
        leftmost, lowest, rightmost, highest = peice.bounds
        height = highest - lowest

        # STILL NEEDED: Angle functionality
        # could be done doing some cosine/sine geometry on the bottom or top point

        # cut line much longer than actual piece
        bottom_point = (x, lowest - height)
        top_point = (x, highest + height)
        cut_line = LineString([bottom_point, top_point])
        return cut_line

    def find_cuts(self, line: LineString, piece: Polygon):
        """This function finds the ctual points where the cut line goes through cake"""
        intersection = line.intersection(piece.boundary)

        # What is the intersections geometry? - want it to be at least two points
        if intersection.is_empty or intersection.geom_type == "Point":
            return None
        points = []
        if intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)
        elif intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            points = [Point(c) for c in coords]
        elif intersection.geom_type == "MultiLineString":
            # FIX: Handle when line crosses multiple times (non-convex case)
            all_coords = []
            for line_segment in intersection.geoms:
                all_coords.extend(list(line_segment.coords))
            points = [Point(c) for c in all_coords]
        elif intersection.geom_type == "GeometryCollection":
            # FIX: Sometimes Shapely returns a mixed collection
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    points.append(geom)
                elif geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    points.extend([Point(c) for c in coords])

        # return the points where the sweeping line intersects with the cake
        return (points[0], points[1])

    def calculate_piece_area(self, piece: Polygon, x: float):
        """Determines the area of the peices we cut from the left side"""
        line = self.find_line(x, piece)
        pieces = split(piece, line)
        left_x, _, right_x, _ = piece.bounds
        # we should get two peices if not, line didnt cut properly
        if len(pieces.geoms) != 2:
            # if we ever attempt to cut to the left of the piece return 0
            if x <= left_x:
                return 0.0
            elif x >= right_x:
                return piece.area
            else:
                return 0.0
        peice1, piece2 = pieces.geoms

        # want the smaller peice / leftmost peice
        if peice1.centroid.x < piece2.centroid.x:
            return peice1.area
        else:
            return piece2.area

    def binary_search(self, piece: Polygon, target_area: float):
        """Use binary search to find x-position that cuts off target_area from the left."""

        left_x, _, right_x, _ = piece.bounds
        best_x = None
        best_error = float("inf")

        # try for best cut for 50 iterationd
        for iteration in range(50):
            # try middle first
            mid_x = (left_x + right_x) / 2

            # get the area of that prospective x position
            left_area = self.calculate_piece_area(piece, mid_x)

            if left_area == 0:
                # Too far left, move right
                left_x = mid_x
                continue

            if left_area >= piece.area:
                # Too far right, move left
                right_x = mid_x
                continue

            # how far away from the target value
            error = abs(left_area - target_area)

            # Track best
            if error < best_error:
                best_error = error
                best_x = mid_x

            # Check if its good enough
            if error < self.target_area_tolerance:
                return mid_x

            # Adjust search based on distance from target area
            if left_area < target_area:
                left_x = mid_x  # Need more, move right
            else:
                right_x = mid_x  # Too much, move left

        return best_x

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic function"""
        print(f"__________Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        print(f"TARGET AREA: {target_area}")
        cuts = []
        # important to make a copy so we can test without cutting real cake
        cake_copy = self.cake.copy()

        # Lots of print for checking
        # go through alg for n - 1 children as we need n-1 cuts
        for cut in range(self.children - 1):
            print(f"cut {cut + 1} out of {self.children - 1}")

            current_cake_pieces = cake_copy.get_pieces()
            # always want to cut the biggest peice as we are sweeping
            cutting_piece = max(current_cake_pieces, key=lambda pc: pc.area)
            print(f"cutting peice area = {cutting_piece.area}")
            print(f"target amount = {target_area}")

            # find the bext x value using binary search
            x = self.binary_search(cutting_piece, target_area)

            # if we cant find an x abort
            if x is None:
                print("Failed no x")
                break

            # find the points to cut given that x value
            cut_points = self.find_cuts(self.find_line(x, cutting_piece), cutting_piece)

            # if we cant find an cut points abort
            if cut_points is None:
                print("Failed no cut points")
                break

            # define our cut points
            from_p, to_p = cut_points

            # check that the cut is valid
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                print(f"invalid cut: {why}")
                continue

            # try to cut the cake and see what areas we get
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))
                print(f"cut at {x}")
                print(f"Pieces = {len(cake_copy.get_pieces())}")
                areas = [p.area for p in cake_copy.get_pieces()]
                print(f"  Current areas: {[f'{a:.2f}' for a in sorted(areas)]}")

            except Exception as e:
                print(f"Error {e}")
                break
        print(f"\nFinal: Made {len(cuts)} out of {self.children - 1} cuts")
        # Check to make sure it makes it to the end of the code (will comment out)
        if len(cake_copy.get_pieces()) == self.children:
            areas = [p.area for p in cake_copy.get_pieces()]
            print(f"Final piece areas: {[f'{a:.2f}' for a in sorted(areas)]}")
            print(
                f"Min: {min(areas):.2f}, Max: {max(areas):.2f}, Diff: {max(areas) - min(areas):.2f}"
            )

        return cuts
