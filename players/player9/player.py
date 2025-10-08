from shapely import Point

# from players.player9.Weighted_Random import WeightedRandom
from players.player9.crust_optimizing_player import CrustOptimizingPlayer

from players.player import Player
from src.cake import Cake


class Player9(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.WR_player = CrustOptimizingPlayer(children, cake, cake_path)
        # self.WR_player = WeightedRandom(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        return self.WR_player.get_cuts()
