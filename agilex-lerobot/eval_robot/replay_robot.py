"""The copied G1 replay controller must not command Piper robots."""
if __name__ == "__main__":
    raise SystemExit("G1 replay is disabled. Use eval_aloha.py --mode=dataset to inspect actions without moving hardware.")
