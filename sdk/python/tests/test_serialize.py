from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from evalwright.serialize import REDACTED, Limits, default_redactor, dumps, to_jsonable


class Color(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class Point:
    x: int
    y: float


class FakeArray:                      # numpy-like duck type, no numpy dependency in tests
    def __init__(self, data):
        self.data, self.dtype, self.size, self.shape = data, "float32", len(data), (len(data),)

    def tolist(self):
        return list(self.data)

    def reshape(self, *_):
        return self

    def __getitem__(self, sl):
        return FakeArray(self.data[sl])


class Model:                          # pydantic-v2-like
    def model_dump(self):
        return {"a": 1}


class Weird:
    __slots__ = ()                    # no __dict__, and repr itself fails

    def __repr__(self):
        raise RuntimeError("boom")


class NoSlice(FakeArray):
    def __getitem__(self, sl):
        raise TypeError("no slicing")


def test_common_types():
    out = to_jsonable({"p": Point(1, 2.5), "c": Color.RED, "t": dt.datetime(2026, 1, 2, 3, 4),
                       "d": dt.timedelta(seconds=90), "path": Path("/a/b"), "b": b"xyz", "s": {1, 2},
                       "e": ValueError("bad"), "nan": float("nan"), "big": 2 ** 80, "m": Model()})
    assert out["p"] == {"x": 1, "y": 2.5} and out["c"] == "red" and out["d"] == 90.0
    assert out["b"] == "<bytes len=3>" and sorted(out["s"]) == [1, 2]
    assert out["e"] == {"type": "ValueError", "message": "bad"} and out["nan"] == "nan"
    assert out["big"] == str(2 ** 80) and out["m"] == {"a": 1}
    json.dumps(out)


def test_cycles_depth_and_size_limits():
    a: dict = {"x": 1}
    a["self"] = a
    assert "cycle" in to_jsonable(a)["self"]
    deep: list = []
    cur = deep
    for _ in range(50):
        cur.append([])
        cur = cur[0]
    s = dumps(to_jsonable(deep, Limits(max_depth=5)))
    assert "max depth" in s
    big = to_jsonable(list(range(1000)), Limits(max_items=10))
    assert len(big) == 11 and "990 more" in big[-1]
    long = to_jsonable("x" * 50_000, Limits(max_string=100))
    assert len(long) < 140 and "truncated" in long


def test_numpy_like_large_array_summarized():
    out = to_jsonable(FakeArray(list(range(1000))), Limits(max_items=5))
    assert out["shape"] == [1000] and len(out["head"]) <= 6


def test_unrepresentable_object_never_raises():
    assert to_jsonable(Weird()).startswith("<unserializable Weird")
    out = to_jsonable(NoSlice(list(range(100))), Limits(max_items=5))
    assert out["shape"] == [100] and out["head"] is None


def test_redaction():
    r = default_redactor(redact_emails=True)
    out = to_jsonable({"api_key": "abc", "headers": {"Authorization": "Bearer xyz"}, "password": "p",
                       "note": "key sk-ABCDEFGHIJKLMNOPQRSTUV and mail a.b@c.com", "token": None}, redactor=r)
    assert out["api_key"] == REDACTED and out["headers"]["Authorization"] == REDACTED
    assert out["password"] == REDACTED and out["token"] is None
    assert "sk-" not in out["note"] and "[EMAIL]" in out["note"]


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True, allow_infinity=True) | st.text()
    | st.binary(max_size=20),
    lambda children: st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=8), children, max_size=6)
    | st.tuples(children, children) | st.frozensets(st.integers(), max_size=4),
    max_leaves=40)


@settings(max_examples=300, deadline=None)
@given(json_values)
def test_property_total_and_json_safe(x):
    out = to_jsonable(x, Limits(max_depth=4, max_string=50, max_items=5), default_redactor())
    s = dumps(out)                    # must always be strict JSON (no NaN/Infinity)
    assert isinstance(json.loads(s), (dict, list, str, int, float, bool, type(None)))


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=500), st.integers(min_value=1, max_value=60))
def test_property_string_bound(s, n):
    out = to_jsonable(s, Limits(max_string=n))
    assert len(out) <= n + len(f"...[truncated {len(s)} chars]")
    if len(s) <= n:
        assert out == s


def test_floats_inf():
    assert to_jsonable(math.inf) == "inf"
