from __future__ import annotations
from shapely import Point, Polygon, LineString
from shapely.geometry import MultiPolygon
from typing import List, Tuple
from players.player import Player
from src.cake import Cake
import src.constants as c


class Player2(Player):
    """
    Area-Targeted Cutting Strategy:
    - Calculates exact target piece sizes
    - Samples boundary points to find cuts that produce correct-sized pieces
    - Prioritizes area accuracy over crust ratio
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.target_piece_area = self.cake.exterior_shape.area / self.children
        self.target_ratio = self.cake.get_piece_ratio(self.cake.get_pieces()[0])

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        """
        Generate cuts targeting specific piece sizes.
        """
        moves = []
        
        # Build list of target cumulative areas for pieces we want to cut off
        total_area = self.cake.exterior_shape.area
        area_targets = []
        for i in range(1, self.children):
            area_targets.append(self.target_piece_area * i)
        
        for cut_num in range(self.children - 1):
            print(f"Cut {cut_num + 1}/{self.children - 1}")
            
            # Get largest piece
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            
            # Determine what size piece we should cut off
            # Always aim for exactly target_piece_area
            target_cut_area = self.target_piece_area
            
            print(f"  Piece area: {piece.area:.2f}, Target cut: {target_cut_area:.2f}, Target piece: {self.target_piece_area:.2f}")
            
            # Find cut that produces this target area
            cut = self._find_target_area_cut(piece, target_cut_area, area_targets)
            
            if cut is None:
                print(f"Warning: Failed to find cut at iteration {cut_num}")
                break
            
            p1, p2 = cut
            moves.append((p1, p2))
            
            try:
                self.cake.cut(p1, p2)
            except Exception as e:
                print(f"Error applying cut {cut_num}: {e}")
                break
        
        return moves

    def _find_target_area_cut(
        self, 
        piece: Polygon, 
        target_area: float,
        area_targets: List[float]
    ) -> Tuple[Point, Point] | None:
        """
        Find a cut that produces a piece close to target_area.
        """
        piece_boundary = piece.boundary
        boundary_length = piece_boundary.length
        
        if boundary_length < 2.0:
            return None
        
        # Sample points along boundary
        # Use aggressive sampling for best accuracy
        num_candidates = min(250, max(80, int(boundary_length / 0.6)))
        step_size = boundary_length / num_candidates
        candidates = [
            piece_boundary.interpolate(i * step_size) 
            for i in range(num_candidates)
        ]
        
        best_cut = None
        best_score = float('inf')
        excellent_cuts = []  # Store cuts with very good area accuracy
        
        # Try pairs of boundary points - exhaustive for small pieces
        sample_step = max(1, num_candidates // 70)  # Very dense sampling
        
        for i in range(0, num_candidates, sample_step):
            for j in range(i + 3, num_candidates, sample_step):
                p1 = candidates[i]
                p2 = candidates[j]
                
                # Quick checks
                if p1.distance(p2) < 0.5:
                    continue
                
                # Validate cut
                cut_line = LineString([p1, p2])
                good, _ = self.cake.does_line_cut_piece_well(cut_line, piece)
                
                if not good:
                    continue
                
                valid, _ = self.cake.cut_is_valid(p1, p2)
                if not valid:
                    continue
                
                # Score this cut
                area_error, crust_error = self._score_cut(
                    p1, p2, piece, target_area, area_targets
                )
                
                if area_error == float('inf'):
                    continue
                
                # Combined score - prioritize area accuracy more
                # Area within 0.5cm² is critical per spec
                score = area_error * 40.0 + crust_error * 8.0
                
                # Track excellent cuts (within 3% of target area - tighter threshold)
                if area_error < 0.03:
                    excellent_cuts.append((area_error, crust_error, p1, p2))
                
                if score < best_score:
                    best_score = score
                    best_cut = (p1, p2)
        
        # If we have cuts with excellent area accuracy, pick best crust ratio among them
        if len(excellent_cuts) > 3:
            print(f"  Found {len(excellent_cuts)} excellent cuts")
            
            # First, filter to only the very best area cuts (within 1.5% if possible)
            very_best_area = [cut for cut in excellent_cuts if cut[0] < 0.015]
            
            if len(very_best_area) > 2:
                # Among the very best area cuts, pick best crust
                very_best_area.sort(key=lambda x: x[1])
                _, crust_err, p1, p2 = very_best_area[0]
                print(f"    Best area+crust: crust_err={crust_err:.4f}")
                return (p1, p2)
            else:
                # Otherwise just optimize crust among all excellent cuts
                excellent_cuts.sort(key=lambda x: x[1])
                _, crust_err, p1, p2 = excellent_cuts[0]
                print(f"    Best crust error: {crust_err:.4f}")
                return (p1, p2)
        
        # Even if we don't have many "excellent" cuts, if we have some good ones
        # prioritize crust ratio
        if len(excellent_cuts) > 0:
            print(f"  Found {len(excellent_cuts)} good cuts, optimizing crust")
            excellent_cuts.sort(key=lambda x: x[1])
            _, crust_err, p1, p2 = excellent_cuts[0]
            print(f"    Crust error: {crust_err:.4f}")
            return (p1, p2)
        
        if best_cut and best_score < float('inf'):
            # Show how good the cut is
            area_component = best_score / 30.0  # Approximate area component
            print(f"  Best cut score: {best_score:.2f}")
            return best_cut
        
        # Fallback: if no cut found, try less strict approach
        print(f"  Using fallback strategy")
        return self._find_any_reasonable_cut(piece, target_area)

    def _score_cut(
        self, 
        p1: Point, 
        p2: Point, 
        piece: Polygon,
        target_area: float,
        area_targets: List[float]
    ) -> Tuple[float, float]:
        """
        Score cut based on area accuracy and crust ratio.
        Returns (area_error, crust_error) where lower is better.
        """
        try:
            split_pieces = self.cake.cut_piece(piece, p1, p2)
            
            if len(split_pieces) != 2:
                return float('inf'), float('inf')
            
            piece_a, piece_b = split_pieces
            
            # We want to cut off a piece close to target_area
            # The smaller piece is what we're "cutting off"
            smaller_piece = piece_a if piece_a.area < piece_b.area else piece_b
            larger_piece = piece_b if piece_a.area < piece_b.area else piece_a
            
            # Calculate area error as percentage of target
            area_diff = abs(smaller_piece.area - target_area)
            area_error = area_diff / self.target_piece_area
            
            # Also check if it matches any of our cumulative targets well
            if area_targets:
                min_target_error = min(
                    abs(smaller_piece.area - t) / self.target_piece_area 
                    for t in area_targets
                )
                area_error = min(area_error, min_target_error)
            
            # Crust ratio score - both pieces should be close to target
            ratio_a = self.cake.get_piece_ratio(piece_a)
            ratio_b = self.cake.get_piece_ratio(piece_b)
            
            # Calculate deviation from target for each piece
            deviation_a = abs(ratio_a - self.target_ratio)
            deviation_b = abs(ratio_b - self.target_ratio)
            
            # Total crust error - want both pieces to match target
            crust_error = deviation_a + deviation_b
            
            return area_error, crust_error
            
        except Exception:
            return float('inf'), float('inf')

    def _calculate_crust_proportion(self, poly: Polygon) -> float:
        """
        Calculate the proportion of the polygon that is 'crust'.
        """
        try:
            interior = poly.buffer(-c.CRUST_SIZE)
            
            if isinstance(interior, MultiPolygon):
                interior_area = sum(p.area for p in interior.geoms if isinstance(p, Polygon))
            elif isinstance(interior, Polygon):
                interior_area = max(0.0, interior.area)
            else:
                interior_area = 0.0
            
            crust_area = max(0.0, poly.area - interior_area)
            
            if poly.area > 0:
                return min(1.0, crust_area / poly.area)
            return 1.0
        except Exception:
            return 0.5

    def _find_any_reasonable_cut(self, piece: Polygon, target_area: float) -> Tuple[Point, Point] | None:
        """
        Fallback: find any cut that's reasonably close to target area.
        Less strict validation, more aggressive sampling.
        """
        boundary = list(piece.exterior.coords)[:-1]
        n = len(boundary)
        
        if n < 3:
            return None
        
        best_cut = None
        best_area_diff = float('inf')
        
        # Try many boundary point pairs
        step = max(1, n // 40)
        
        for i in range(0, n, step):
            for j in range(i + 2, n, step):
                p1 = Point(boundary[i])
                p2 = Point(boundary[j])
                
                if p1.distance(p2) < 0.3:
                    continue
                
                # Check validity
                try:
                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue
                    
                    cut_line = LineString([p1, p2])
                    good, _ = self.cake.does_line_cut_piece_well(cut_line, piece)
                    
                    if not good:
                        continue
                    
                    # Check area
                    split_pieces = self.cake.cut_piece(piece, p1, p2)
                    if len(split_pieces) == 2:
                        smaller = min(split_pieces[0].area, split_pieces[1].area)
                        area_diff = abs(smaller - target_area)
                        
                        if area_diff < best_area_diff:
                            best_area_diff = area_diff
                            best_cut = (p1, p2)
                except Exception:
                    continue
        
        return best_cut
