from shapely import LineString, Point
import random

from players.player import Player
from src.cake import Cake

from players.random_player import RandomPlayer


class WeightedRandom(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.valid_line_pair = dict()
        self.random = RandomPlayer(children, cake, cake_path)

    def valid_piece_size(self, p1, p2, Area_list, piece):
        split_pieces = self.cake.cut_piece(piece, p1, p2)
        Area_piece = split_pieces[0].area
        for valid_area in Area_list:
            if (
                abs(Area_piece - valid_area) <= 0.1 * valid_area
                and abs(Area_piece - valid_area) > 0.1 * Area_piece
            ):
                return True
        return False

    def get_weight(self, p1, p2, piece, Goal_ratio):
        split_pieces = self.cake.cut_piece(piece, p1, p2)
        if len(split_pieces) == 1:
            return 0
        # print("split_pieces", split_pieces, len(split_pieces))
        r1 = self.cake.get_piece_ratio(split_pieces[0])
        r2 = self.cake.get_piece_ratio(split_pieces[1])
        weight = 0
        weight += 0.5 * (1 - abs(Goal_ratio - r1))
        weight += 0.5 * (1 - abs(Goal_ratio - r2))
        return weight

    def choose_edge_weight(self):
        elements = list(self.valid_line_pair.keys())
        weights = list(self.valid_line_pair.values())
        return random.choices(elements, weights=weights, k=1)[0]

    def get_piece(self, p1, p2):
        for piece in self.cake.get_pieces():
            if self.cake.point_lies_on_piece_boundary(
                p1, piece
            ) and self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

    def reset_par_dict(self):
        self.valid_line_pair = dict()
        line_to_piece = dict()

        for piece in self.cake.get_pieces():
            vertices = list(piece.exterior.coords[:-1])
            lines = [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
            for Line1 in lines:
                Line1 = LineString(Line1)
                line_to_piece[Line1] = piece
                for Line2 in lines:
                    Line2 = LineString(Line2)
                    line_to_piece[Line2] = piece
                    mid_Line1 = Line1.centroid
                    mid_Line2 = Line2.centroid
                    if self.cake.cut_is_valid(mid_Line1, mid_Line2):
                        self.valid_line_pair[(Line1, Line2)] = 1

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        Goal_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        Total_Area = self.cake.exterior_shape.area
        Area_list = []
        for i in range(1, self.children):
            Area_list += [(Total_Area / self.children) * i]

        for k in range(self.children - 1):
            print(k)
            self.reset_par_dict()

            # print(self.valid_line_pair.values())

            best_line = None
            best_value = 0

            for _ in range(4000):
                # print(self.valid_line_pair.values())
                # print(k)
                line1, line2 = self.choose_edge_weight()
                w1 = random.random()
                w2 = random.random()
                p1 = line1.interpolate(w1, normalized=True)
                p2 = line2.interpolate(w2, normalized=True)

                piece = self.get_piece(p1, p2)

                if piece is None:
                    continue

                good, _ = self.cake.does_line_cut_piece_well(
                    LineString((p1, p2)), piece
                )
                if not good:
                    self.valid_line_pair[(line1, line2)] /= 2
                    continue

                if self.cake.cut_is_valid(p1, p2):
                    if self.valid_piece_size(p1, p2, Area_list, piece):
                        # print("here")
                        # print(p1, p2)
                        weight = self.get_weight(p1, p2, piece, Goal_ratio)
                        # print("weight", weight)
                        if weight > best_value:
                            best_value = weight
                            self.valid_line_pair[(line1, line2)] += weight
                            best_line = (p1, p2)
                    else:
                        # print("this should not happen")
                        # input()
                        continue
                else:
                    self.valid_line_pair[(line1, line2)] /= 2

            # print("there", best_line[0], best_line[1])

            if best_line is None:
                print("random", k)
                best_line = self.random.find_random_cut()

            moves.append(best_line)
            self.cake.cut(best_line[0], best_line[1])
        return moves
