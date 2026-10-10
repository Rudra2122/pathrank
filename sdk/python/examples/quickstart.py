"""Tracing quickstart: python examples/quickstart.py  (writes .evalwright/traces/quickstart.jsonl)"""
import asyncio

import evalwright as ew

ew.init("quickstart")                     # or EVALWRIGHT_EXPORT=otlp:http://localhost:4318/v1/traces


@ew.traced(type="retrieval")
def retrieve(query: str) -> list[str]:
    return [f"doc about {query}", "another doc"]


@ew.traced(type="llm")
async def generate(query: str, docs: list[str]) -> str:
    await asyncio.sleep(0.01)             # stand-in for a model call
    ew.log(metrics={"prompt_tokens": 42, "completion_tokens": 7}, metadata={"model": "demo-model"})
    return f"Answer to {query!r} using {len(docs)} docs"


async def answer(query: str) -> str:
    async with ew.span("answer", type="task", input=query) as s:
        docs = retrieve(query)
        out = await generate(query, docs)
        s.log(output=out)
        return out


if __name__ == "__main__":
    print(asyncio.run(answer("what is a span?")))
    ew.flush()
    print(ew.get_tracer().stats())
