from pathlib import Path


SRC_PATH: Path = Path(__file__).parent

# Root of the project (src/frog_mjlab -> src -> project root).
_PROJECT_ROOT: Path = SRC_PATH.parents[1]

# Dedicated directory for robot models (MJCF/XML + mesh assets).
MODEL_PATH: Path = _PROJECT_ROOT / "model"


def main() -> None:
    print("Hello from frog-mjlab!")
