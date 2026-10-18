"""Pytest bootstrap: put the repo root on sys.path so tests can import the
sibling scripts (``fake_policy_client``, ``chunked_policy``) directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
