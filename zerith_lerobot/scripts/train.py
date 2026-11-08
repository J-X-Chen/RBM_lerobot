"""Run the Zerith-local LeRobot training implementation from any working directory."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
sys.path.insert(0, str(SRC))

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
