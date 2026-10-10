"""Built-in scorers. A scorer is any callable taking any subset of
(input, output, expected, metadata) as keyword arguments and returning a float in [0, 1], a bool,
a Score, a list of Scores, a {name: value} dict, or None (= not applicable, excluded from averages).
"""
from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .evals.types import Score

__all__ = ["Contains", "ExactMatch", "JSONValid", "LLMJudge", "Levenshtein", "NumericCloseness", "RegexMatch",
           "Score"]


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)


class ExactMatch:
    def __init__(self, case_sensitive: bool = True, strip: bool = True, name: str = "exact_match") -> None:
        self.case_sensitive, self.strip, self.name = case_sensitive, strip, name

    def __call__(self, output: Any, expected: Any) -> float | None:
        if expected is None:
            return None
        a, b = _text(output), _text(expected)
        if self.strip:
            a, b = a.strip(), b.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        return float(a == b)


class Contains:
    """1 if the output contains the expected substring (or all/any of a list of substrings)."""

    def __init__(self, mode: str = "all", case_sensitive: bool = False, name: str = "contains") -> None:
        if mode not in ("all", "any", "fraction"):
            raise ValueError("mode must be all | any | fraction")
        self.mode, self.case_sensitive, self.name = mode, case_sensitive, name

    def __call__(self, output: Any, expected: Any) -> float | None:
        if expected is None:
            return None
        needles = expected if isinstance(expected, (list, tuple)) else [expected]
        hay = _text(output)
        if not self.case_sensitive:
            hay = hay.lower()
        hits = [(n if self.case_sensitive else str(n).lower()) in hay for n in needles]
        if not hits:
            return None
        if self.mode == "all":
            return float(all(hits))
        if self.mode == "any":
            return float(any(hits))
        return sum(hits) / len(hits)


class JSONValid:
    """1 if the output is (or parses as) a JSON object containing the required keys."""

    def __init__(self, required_keys: tuple[str, ...] = (), name: str = "json_valid") -> None:
        self.required_keys, self.name = required_keys, name

    def __call__(self, output: Any) -> float:
        obj = output
        if isinstance(output, str):
            try:
                obj = json.loads(output)
            except json.JSONDecodeError:
                return 0.0
        if self.required_keys:
            return float(isinstance(obj, Mapping) and all(k in obj for k in self.required_keys))
        return float(isinstance(obj, (Mapping, list)))


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Levenshtein:
    """Normalized edit similarity: 1 - distance / max(len). Inputs capped at max_len chars."""

    def __init__(self, max_len: int = 2000, name: str = "levenshtein") -> None:
        self.max_len, self.name = max_len, name

    def __call__(self, output: Any, expected: Any) -> float | None:
        if expected is None:
            return None
        a, b = _text(output)[: self.max_len], _text(expected)[: self.max_len]
        if not a and not b:
            return 1.0
        return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


class NumericCloseness:
    """max(0, 1 - |output - expected| / max(|expected|, eps))."""

    def __init__(self, eps: float = 1e-9, name: str = "numeric_closeness") -> None:
        self.eps, self.name = eps, name

    def __call__(self, output: Any, expected: Any) -> float | None:
        if expected is None:
            return None
        try:
            o, e = float(output), float(expected)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - abs(o - e) / max(abs(e), self.eps))


class RegexMatch:
    def __init__(self, pattern: str, flags: int = 0, name: str = "regex_match") -> None:
        self.rx, self.name = re.compile(pattern, flags), name

    def __call__(self, output: Any) -> float:
        return float(self.rx.search(_text(output)) is not None)


LLM = Callable[[str], "str | Awaitable[str]"]

_JUDGE_TEMPLATE = """You are grading the output of an AI system.

Criteria: {rubric}

Input:
{input}

Expected (may be empty):
{expected}

Output:
{output}

Reply with a short rationale, then on the final line write exactly "Choice: X" where X is one of: {choices}.
"""


class LLMJudge:
    """Model-agnostic LLM-as-judge. `llm` is any function (sync or async) prompt -> text, so it
    works with any provider client. The final 'Choice: X' line maps to a score via `choices`."""

    def __init__(self, llm: LLM, rubric: str, choices: Mapping[str, float] | None = None,
                 name: str = "llm_judge", template: str = _JUDGE_TEMPLATE) -> None:
        self.llm, self.rubric, self.name, self.template = llm, rubric, name, template
        self.choices = dict(choices or {"A": 1.0, "B": 0.5, "C": 0.0})

    async def __call__(self, input: Any, output: Any, expected: Any = None) -> Score:
        prompt = self.template.format(rubric=self.rubric, input=_text(input), output=_text(output),
                                      expected="" if expected is None else _text(expected),
                                      choices=", ".join(self.choices))
        reply = self.llm(prompt)
        if inspect.isawaitable(reply):
            reply = await reply
        text = str(reply)
        m = re.findall(r"Choice:\s*([A-Za-z0-9_]+)", text)
        choice = m[-1] if m else None
        if choice not in self.choices:
            return Score(self.name, None, {"error": "unparseable judge reply", "reply": text[-500:]})
        return Score(self.name, self.choices[choice], {"choice": choice, "rationale": text[-1000:]})
