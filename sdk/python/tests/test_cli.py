from __future__ import annotations

import json
import subprocess
import sys

import pytest

from evalwright.cli import main

GOOD = '''
import evalwright as ew
from evalwright.scorers import ExactMatch
ew.Eval("cli-demo", [{"input": "a", "expected": "A"}, {"input": "b", "expected": "B"}], {task}, [ExactMatch()])
'''


def write(tmp_path, name, task):
    p = tmp_path / name
    p.write_text(GOOD.replace("{task}", task))
    return p


def test_run_compare_ls_and_gates(tmp_path, capsys):
    store = str(tmp_path / "store")
    f = write(tmp_path, "upper_eval.py", "str.upper")
    assert main(["run", str(tmp_path), "--store-dir", store, "--min", "exact_match=0.9"]) == 0
    assert "cli-demo" in capsys.readouterr().out
    f.write_text(GOOD.replace("{task}", "lambda x: 'A'"))
    report = tmp_path / "r.json"
    from evalwright.evals import core
    core._registry.clear()
    rc = main(["run", str(f), "--store-dir", store, "--max-regression", "0.1", "--json", str(report)])
    err = capsys.readouterr().err
    assert rc == 1 and "GATE FAILED" in err and "regressed" in err
    assert json.loads(report.read_text())[0]["gate_failures"]
    assert main(["compare", "cli-demo", "--store-dir", store]) == 0
    assert '"regressed": 1' in capsys.readouterr().out
    assert main(["ls", "--store-dir", store]) == 0
    assert capsys.readouterr().out.count("cli-demo") == 2


def test_broken_file_and_missing(tmp_path, capsys):
    (tmp_path / "broken_eval.py").write_text("raise RuntimeError('nope')\n")
    assert main(["run", str(tmp_path), "--store-dir", str(tmp_path / "s")]) == 2
    assert "nope" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["run", str(empty)]) == 2
    assert main(["compare", "nothing", "--store-dir", str(tmp_path / "s")]) == 2
    with pytest.raises(SystemExit):
        main(["run", str(tmp_path / "x.txt")])
    with pytest.raises(SystemExit):
        main(["run", str(tmp_path), "--min", "bad"])


def test_module_entrypoint_and_version():
    out = subprocess.run([sys.executable, "-m", "evalwright", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("evalwright ")
