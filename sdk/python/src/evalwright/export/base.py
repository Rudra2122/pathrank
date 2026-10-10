"""Exporter protocol + result type."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExportResult:
    success: bool
    retryable: bool = False
    retry_after: float | None = None     # seconds, from a Retry-After header
    error: str | None = None

    @staticmethod
    def ok() -> ExportResult:
        return ExportResult(True)

    @staticmethod
    def retry(error: str, retry_after: float | None = None) -> ExportResult:
        return ExportResult(False, True, retry_after, error)

    @staticmethod
    def fail(error: str) -> ExportResult:
        return ExportResult(False, False, None, error)


@runtime_checkable
class Exporter(Protocol):
    def export(self, batch: list[dict[str, Any]]) -> ExportResult: ...

    def close(self) -> None: ...
