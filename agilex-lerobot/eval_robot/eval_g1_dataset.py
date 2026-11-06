"""Offline ALOHA evaluation; old G1 hardware options are intentionally unsupported."""
import sys
from eval_aloha import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--mode=dataset"])
