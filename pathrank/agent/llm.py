"""LLM backends for the notification agent.

provider=auto picks Anthropic if ANTHROPIC_API_KEY is set, else OpenAI if OPENAI_API_KEY is set,
else a deterministic template so the agent graph, guardrails and eval all run offline (CI, no key).
provider=anthropic | openai forces one (and fails loudly if its key is missing).
Both real backends use the same prompts, so their eval numbers are comparable."""
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


def _extract_json(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def draft_prompt(profile: dict, recs: list[dict]) -> str:
    return (
        "Write a short job-alert notification for a student/early-career job seeker.\n"
        f"Profile: {json.dumps(profile)}\n"
        f"Recommended jobs (the ONLY jobs you may mention): {json.dumps(recs)}\n"
        "Rules: mention 1-3 of these jobs, put each job title in double quotes exactly as given, "
        "do not invent jobs, employers, salaries or facts, subject <= 70 chars, body <= 400 chars, "
        "end with a call to action.\n"
        'Return JSON only: {"subject": str, "body": str, "job_ids": [ids of jobs mentioned]}')


def judge_prompt(profile: dict, recs: list[dict], msg: dict) -> str:
    return (
        "You are grading a job-alert notification. Score 1-5 on relevance (fits the profile), "
        "specificity, tone (friendly, not spammy) and call_to_action.\n"
        f"Profile: {json.dumps(profile)}\nAllowed jobs: {json.dumps(recs)}\nMessage: {json.dumps(msg)}\n"
        'Return JSON only: {"scores": {"relevance": n, "specificity": n, "tone": n, "call_to_action": n}, '
        '"rationale": str}')


def _overall(out: dict) -> dict:
    s = out.get("scores") or {}
    vals = [float(v) for v in s.values() if isinstance(v, (int, float))]
    out["overall"] = sum(vals) / max(len(vals), 1)
    return out


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
                   metadata={"model": self.model, "provider": self.name})
        return _extract_json("".join(b.text for b in r.content if getattr(b, "type", "") == "text"))

    def draft(self, profile: dict, recs: list[dict]) -> dict:
        return self._json(draft_prompt(profile, recs))

    def judge(self, profile: dict, recs: list[dict], msg: dict) -> dict:
        return _overall(self._json(judge_prompt(profile, recs, msg)))


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str, client=None):
        if client is None:
            import openai
            client = openai.OpenAI()     # reads OPENAI_API_KEY
        self.client = client
        self.model = model

    def _json(self, prompt: str) -> dict:
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=600, temperature=0,
            response_format={"type": "json_object"},          # JSON mode: the reply is a JSON object
            messages=[{"role": "user", "content": prompt}])
        usage = getattr(r, "usage", None)
        if usage is not None:
            ew.log(metrics={"input_tokens": float(getattr(usage, "prompt_tokens", 0) or 0),
                            "output_tokens": float(getattr(usage, "completion_tokens", 0) or 0)},
                   metadata={"model": self.model, "provider": self.name})
        return _extract_json(r.choices[0].message.content if r.choices else "")

    def draft(self, profile: dict, recs: list[dict]) -> dict:
        return self._json(draft_prompt(profile, recs))

    def judge(self, profile: dict, recs: list[dict], msg: dict) -> dict:
        return _overall(self._json(judge_prompt(profile, recs, msg)))


def get_llm(provider: str, model: str, openai_model: str | None = None):
    """provider: auto | anthropic | openai | template (see module docstring)."""
    openai_model = os.environ.get("OPENAI_MODEL") or openai_model or "gpt-4o-mini"
    if provider not in ("auto", "anthropic", "openai", "template"):
        raise ValueError(f"unknown agent provider {provider!r} (auto | anthropic | openai | template)")
    if provider == "template":
        return TemplateLLM()
    for name, key, make in (("anthropic", "ANTHROPIC_API_KEY", lambda: AnthropicLLM(model)),
                            ("openai", "OPENAI_API_KEY", lambda: OpenAILLM(openai_model))):
        if provider not in ("auto", name):
            continue
        if not os.environ.get(key):
            if provider == name:
                raise RuntimeError(f"provider={name} but {key} is not set")
            continue
        try:
            return make()
        except ImportError:
            if provider == name:
                raise
    return TemplateLLM()
