#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Activate the H1 Python 3.10 environment before launching; no implicit SDK initialization.
exec "${ZERITH_PYTHON:-python}" -m zerith_image_service.server "$@"
