"""The copied Unitree asynchronous control loop is not valid for Piper hardware."""
if __name__ == "__main__":
    raise SystemExit("Use eval_aloha.py. ALOHA currently supports synchronous action chunks only; G1 async control is disabled.")
