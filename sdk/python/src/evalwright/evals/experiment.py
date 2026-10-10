"""Experiment storage (local, append-only JSONL) and case-by-case comparison."""
from __future__ import annotations

import json
import os
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .stats import bootstrap_ci, mean


def exp_dir(store: Path, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return store / safe


def new_experiment_id() -> str:
    ns = time.time_ns()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(ns / 1e9)) + f"{(ns // 1_000_000) % 1000:03d}-" \
        + secrets.token_hex(2)


def save_experiment(store: Path, name: str, exp_id: str, cases: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    d = exp_dir(store, name) / exp_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":"), default=str) + "\n")
    summary = {**summary, "created_ns": time.time_ns()}       # ordering key (ids can share a millisecond)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    tmp = exp_dir(store, name) / f".latest.{os.getpid()}"
    tmp.write_text(exp_id)
    os.replace(tmp, exp_dir(store, name) / "LATEST")      # atomic pointer update
    return d


def list_experiments(store: Path, name: str) -> list[str]:
    d = exp_dir(store, name)
    if not d.exists():
        return []
    found = []
    for p in d.iterdir():
        f = p / "summary.json"
        if p.is_dir() and f.exists():
            try:
                created = int(json.loads(f.read_text()).get("created_ns", 0))
            except (ValueError, OSError):
                created = 0
            found.append((created, p.name))
    return [name for _, name in sorted(found)]       # oldest -> newest by creation time


def resolve(store: Path, name: str, ref: str) -> str | None:
    if ref == "latest":
        p = exp_dir(store, name) / "LATEST"
        return p.read_text().strip() if p.exists() else None
    if ref == "previous":
        ids = list_experiments(store, name)
        return ids[-2] if len(ids) >= 2 else None
    return ref if (exp_dir(store, name) / ref / "summary.json").exists() else None


def load_experiment(store: str | Path, name: str, ref: str = "latest") -> dict[str, Any] | None:
    store = Path(store)
    exp_id = resolve(store, name, ref)
    if exp_id is None:
        return None
    d = exp_dir(store, name) / exp_id
    cases = [json.loads(line) for line in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"id": exp_id, "cases": cases, "summary": json.loads((d / "summary.json").read_text())}


def _per_case(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """case_id -> {scorer: mean over trials} (None scores skipped)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        for k, v in (c.get("scores") or {}).items():
            if v is not None:
                acc[c["case_id"]][k].append(float(v))
    return {cid: {k: sum(v) / len(v) for k, v in d.items()} for cid, d in acc.items()}


def compare_experiments(baseline: dict[str, Any], current: dict[str, Any], eps: float = 1e-9) -> dict[str, Any]:
    """Paired comparison on cases present in both runs."""
    b, c = _per_case(baseline["cases"]), _per_case(current["cases"])
    common = sorted(set(b) & set(c))
    scorers = sorted({k for d in c.values() for k in d} | {k for d in b.values() for k in d})
    out: dict[str, Any] = {"baseline": baseline["id"], "current": current["id"], "common_cases": len(common),
                           "new_cases": len(set(c) - set(b)), "missing_cases": len(set(b) - set(c)), "scores": {}}
    for s in scorers:
        diffs = [c[i][s] - b[i][s] for i in common if s in c[i] and s in b[i]]
        if not diffs:
            continue
        out["scores"][s] = {"baseline": mean([b[i][s] for i in common if s in b[i] and s in c[i]]),
                            "current": mean([c[i][s] for i in common if s in b[i] and s in c[i]]),
                            "diff": mean(diffs), "ci95": bootstrap_ci(diffs),
                            "improved": sum(d > eps for d in diffs), "regressed": sum(d < -eps for d in diffs),
                            "unchanged": sum(abs(d) <= eps for d in diffs), "n": len(diffs)}
    return out
