"""Put the repo root on sys.path so `pytest` finds the `aegis` package."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
