from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..serialize import dumps, to_jsonable


@dataclass
class Score:
    name: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None:
            s = float(self.score)
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"score {self.name}={s} outside [0, 1]")
            self.score = s


@dataclass
class EvalCase:
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def case_id(self) -> str:
        """Stable id (content hash) so experiments can be compared case by case."""
        if self.id is not None:
            return self.id
        blob = dumps(to_jsonable({"input": self.input, "expected": self.expected}))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]
