"""Run with the CLI (stores an experiment and compares to the last one):
    evalwright run examples/qa_eval.py --max-regression 0.05 --min exact_match=0.5
"""
import evalwright as ew
from evalwright.scorers import Contains, ExactMatch, Levenshtein

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Kenya": "Nairobi", "Peru": "Lima", "Canada": "Ottawa"}


def answer(country: str) -> str:            # the "model" under test
    guesses = {"Canada": "Toronto"}           # one deliberate mistake
    return guesses.get(country, CAPITALS[country])


ew.Eval("capitals", [{"input": c, "expected": cap} for c, cap in CAPITALS.items()], answer,
        [ExactMatch(), Levenshtein(), Contains(case_sensitive=False)], metadata={"model": "lookup-v1"})
