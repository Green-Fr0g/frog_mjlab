from pathlib import Path


SRC_PATH: Path = Path(__file__).parent

# Root of the project (source/frog_mjlab/frog_mjlab -> source/frog_mjlab -> source -> project root).
_PROJECT_ROOT: Path = SRC_PATH.parents[2]

# Dedicated directory for robot models (MJCF/XML + mesh assets).
MODEL_PATH: Path = _PROJECT_ROOT / "model"

# Top-level raw motion data directory (aligned with frog_lab layout).
MOTION_DATA_PATH: Path = _PROJECT_ROOT / "motion_data"


def main() -> None:
    print("Hello from frog-mjlab!")
