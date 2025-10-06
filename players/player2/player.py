from shapely import Point

from players.player import Player
from src.cake import Cake


class Player2(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        cuts = []
        for _ in range(self.children):
            largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
            from_p = largest_piece.centroid
            to_p = Point(largest_piece.exterior.coords[0] + largest_piece.exterior.coords[-1]) / 2
            is_valid, _ = self.cake.cut_is_valid(from_p, to_p)
            if is_valid:
                cuts.append((from_p, to_p))
                self.cake.cut(from_p, to_p)
        return cuts

