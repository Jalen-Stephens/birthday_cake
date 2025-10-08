import sys
import os
import glob

# Add the project root to Python path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.game import Game
from src.args import Args


def test_all_cakes():
    """Test Player6 with all available cake files"""

    # Get all cake files
    cake_files = glob.glob("cakes/players/player*/*.csv")

    results = []

    for cake_file in cake_files:
        # Extract cake name for logging
        cake_name = os.path.basename(cake_file).replace(".csv", "")
        player_dir = os.path.basename(os.path.dirname(cake_file))

        print(f"Testing: {player_dir}/{cake_name}")
        try:
            Game(
                Args(
                    gui=False,
                    player=6,  # Player6
                    import_cake=cake_file,
                    seed=42,
                    children=8,
                    export_cake=None,
                    debug=False,
                    sandbox=False,
                )
            )
            print("Success")
        except Exception as e:
            print(f"Failed: {e}")
            print("Failed")

    return results


if __name__ == "__main__":
    os.chdir(
        "/Users/soumilbaldota/birthday_cake"
    )  # Ensure we're in the right directory
    results = test_all_cakes()
