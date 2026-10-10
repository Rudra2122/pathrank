"""PathRank: multi-stage job recommendation + notification system."""
import os as _os
import sys as _sys
import warnings as _w

if _sys.platform == "darwin":
    # torch, faiss-cpu and lightgbm macOS wheels each bundle their own libomp; two OpenMP
    # runtimes in one process segfault. Must be set before any of them is imported.
    # Override per run, e.g. OMP_NUM_THREADS=4 make smoke.
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

# pandas 2.x + numpy 2.5 emit this on every Timestamp - Timedelta; harmless, very noisy
_w.filterwarnings("ignore", message=".*'generic' unit for NumPy timedelta", category=DeprecationWarning)

__version__ = "0.1.0"
