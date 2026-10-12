"""Locate shared contracts for source checkouts and independent upload ZIPs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SHARED_ROOT = HERE if (HERE / "shared").is_dir() else HERE.parent
if not (SHARED_ROOT / "shared").is_dir():
    raise ImportError("Missing shared/: upload a generated server ZIP or the complete checkout")
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))
