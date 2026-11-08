"""Same H1 evaluation state machine with asynchronous chunk prefetch enabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zerith_lerobot.eval_robot.eval_h1 import main

if __name__ == '__main__':
    if '--async-prefetch' not in sys.argv:
        sys.argv.append('--async-prefetch')
    main()
