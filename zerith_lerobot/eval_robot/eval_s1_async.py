"""Legacy filename forwarding to the H1 evaluator; uses H1 CLI arguments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zerith_lerobot.eval_robot.eval_h1 import main

if __name__ == '__main__':
    if '--async-prefetch' not in sys.argv:
        sys.argv.append('--async-prefetch')
    main()
