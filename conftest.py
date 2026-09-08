# conftest.py — ensure `src/` is on sys.path so tests can import app.* and src.* packages.
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
