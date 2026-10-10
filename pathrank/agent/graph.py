"""LangGraph notification agent.

  check_cap -> get_profile -> get_recs -> draft -> guardrails --pass--> judge -> END
      |capped                                ^          |fail (retries left)
      v                                      +----------+
     END                                     fail (no retries) -> blocked -> END

Tools are plain functions injected at build time (profile lookup, recommender, frequency cap)
so the same graph runs against the offline store or the live API.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

import evalwright as ew
from langgraph.graph import END, StateGraph

from . import guardrails


class AgentState(TypedDict, total=False):
    user_id: int
    capped: bool
    profile: dict
    recs: list
    message: dict
    errors: list
    attempts: int
    judge: dict
    status: str


def build_agent(get_profile: Callable[[int], dict], get_recs: Callable[[int], list],
                check_frequency_cap: Callable[[int], bool], llm, max_retries: int = 1):

    def n_cap(s: AgentState):
        return {"capped": not check_frequency_cap(s["user_id"]), "attempts": 0}

    def n_profile(s: AgentState):
        return {"profile": get_profile(s["user_id"])}

    def n_recs(s: AgentState):
        return {"recs": get_recs(s["user_id"])}

    def n_draft(s: AgentState):
        try:
            msg = llm.draft(s["profile"], s["recs"])
        except Exception as e:  # noqa: BLE001  (network/parse errors become guardrail failures)
            msg = {"error": str(e)}
        return {"message": msg, "attempts": s.get("attempts", 0) + 1}

    def n_guard(s: AgentState):
        return {"errors": guardrails.check(s["message"], s["recs"])}

    def n_judge(s: AgentState):
        return {"judge": llm.judge(s["profile"], s["recs"], s["message"]), "status": "ready"}

    def n_blocked(s: AgentState):
        return {"status": "blocked"}

    def r_cap(s: AgentState) -> str:
        return "end" if s["capped"] else "go"

    def r_recs(s: AgentState) -> str:
        return "go" if s["recs"] else "end"

    def r_guard(s: AgentState) -> str:
        if not s["errors"]:
            return "pass"
        return "retry" if s["attempts"] <= max_retries else "block"

    g = StateGraph(AgentState)
    # every node is an evalwright span (a no-op unless evalwright.init() was called)
    nodes = [("check_cap", n_cap, "tool"), ("get_profile", n_profile, "tool"), ("get_recs", n_recs, "retrieval"),
             ("draft", n_draft, "llm"), ("guardrails", n_guard, "function"), ("judge", n_judge, "llm"),
             ("blocked", n_blocked, "function")]
    for name, fn, kind in nodes:
        g.add_node(name, ew.traced(name=name, type=kind, capture_args=False)(fn))
    g.set_entry_point("check_cap")
    g.add_conditional_edges("check_cap", r_cap, {"end": END, "go": "get_profile"})
    g.add_edge("get_profile", "get_recs")
    g.add_conditional_edges("get_recs", r_recs, {"end": END, "go": "draft"})
    g.add_edge("draft", "guardrails")
    g.add_conditional_edges("guardrails", r_guard, {"pass": "judge", "retry": "draft", "block": "blocked"})
    g.add_edge("judge", END)
    g.add_edge("blocked", END)
    return g.compile()


def run_agent(agent, user_id: int) -> dict[str, Any]:
    with ew.span("notification_agent", type="task", input={"user_id": int(user_id)}) as span:
        out = agent.invoke({"user_id": int(user_id)})
        if out.get("capped"):
            out["status"] = "capped"
        elif not out.get("recs"):
            out["status"] = "no_recs"
        span.log(output={"status": out.get("status"), "message": out.get("message")},
                 metrics={"attempts": float(out.get("attempts") or 0)},
                 tags=[f"status:{out.get('status')}"])
    return out
