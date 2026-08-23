"""Bootstrap: put the monorepo root on sys.path so `import xnch` resolves
regardless of whether pytest is rooted here or at the repo root."""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
