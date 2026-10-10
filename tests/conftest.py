import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CFG = "configs/ci.yaml"


@pytest.fixture(scope="session")
def built():
    os.chdir(ROOT)
    for d in ("data/raw_ci", "data/processed_ci", "artifacts_ci"):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--config", CFG, "--stages", "all"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    from pathrank.config import load_config
    return load_config(CFG)


