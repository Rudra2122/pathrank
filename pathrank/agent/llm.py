"""LLM backends. `anthropic` when ANTHROPIC_API_KEY is set, else a deterministic template
so the agent graph, guardrails and eval all run offline (CI, no key)."""
from __future__ import annotations

import json
import os
import re

import evalwright as ew


class TemplateLLM:
    name = "template"

    def draft(self, profile: dict, recs: list[dict]) -> dict:
        top = recs[:3]
        first = profile.get("first_line") or "New jobs picked for you"
        def where(r):   # interaction-only datasets (XING, OLX) have no locations
            loc = ", ".join(x for x in (r["city"].title(), r["state"]) if x)
            return f" in {loc}" if loc else ""
        lines = [f'- "{r["title"]}"{where(r)}: {r["reason"]}' for r in top]
        return {"subject": f"{len(recs)} new matches, starting with \"{top[0]['title']}\"",
                "body": f"{first}.\n" + "\n".join(lines) + "\nSee all your matches and apply in one tap.",
                "job_ids": [r["job_id"] for r in top]}

    def judge(self, profile: dict, recs: list[dict], msg: dict) -> dict:
        body = msg.get("body", "")
        s = {"relevance": 5 if msg.get("job_ids") else 1, "specificity": 4 if '"' in body else 2,
             "tone": 4, "call_to_action": 5 if re.search(r"\b(apply|see|view)\b", body, re.I) else 2}
        return {"scores": s, "overall": sum(s.values()) / len(s), "rationale": "heuristic rubric (no LLM key)"}


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def _json(self, prompt: str) -> dict:
        r = self.client.messages.create(model=self.model, max_tokens=600,
                                        messages=[{"role": "user", "content": prompt}])
        usage = getattr(r, "usage", None)
        if usage is not None:            # lands on the current span (draft or judge node)
            ew.log(metrics={"input_tokens": float(getattr(usage, "input_tokens", 0) or 0),
                            "output_tokens": float(getattr(usage, "output_tokens", 0) or 0)},
                   metadata={"model": self.model})
        txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {}

    def draft(self, profile: dict, recs: list[dict]) -> dict:
        prompt = (
            "Write a short job-alert notification for a student/early-career job seeker.\n"
            f"Profile: {json.dumps(profile)}\n"
            f"Recommended jobs (the ONLY jobs you may mention): {json.dumps(recs)}\n"
            "Rules: mention 1-3 of these jobs, put each job title in double quotes exactly as given, "
            "do not invent jobs, employers, salaries or facts, subject <= 70 chars, body <= 400 chars, "
            "end with a call to action.\n"
            'Return JSON only: {"subject": str, "body": str, "job_ids": [ids of jobs mentioned]}')
        return self._json(prompt)

    def judge(self, profile: dict, recs: list[dict], msg: dict) -> dict:
        prompt = (
            "You are grading a job-alert notification. Score 1-5 on relevance (fits the profile), "
            "specificity, tone (friendly, not spammy) and call_to_action.\n"
            f"Profile: {json.dumps(profile)}\nAllowed jobs: {json.dumps(recs)}\nMessage: {json.dumps(msg)}\n"
            'Return JSON only: {"scores": {"relevance": n, "specificity": n, "tone": n, "call_to_action": n}, '
            '"rationale": str}')
        out = self._json(prompt)
        s = out.get("scores", {})
        out["overall"] = sum(s.values()) / max(len(s), 1)
        return out


def get_llm(provider: str, model: str):
    if provider == "template":
        return TemplateLLM()
    if provider in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM(model)
        except ImportError:
            if provider == "anthropic":
                raise
    if provider == "anthropic":
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return TemplateLLM()
