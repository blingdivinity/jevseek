"""Decide: TypeSafe's Jev, via OpenRouter's System One endpoint, picks among the
proposed tokens.

Jev is a decision model -- it answers typed questions about a `state`, it does
not generate. Here the state is the transcript and the question is a `choice`
whose criteria are deepseek's candidate tokens. Criteria order biases the answer,
so the same candidates are asked several times under different orderings, each
ordering as its own question in the same call, and the caller combines them.
"""

from __future__ import annotations

import asyncio

import httpx

API_URL = "https://openrouter.ai/api/v1/systemone"
MODEL = "typesafe/jev-1.13"
INSTRUCTIONS = "Next token of the assistant's reply?"


class Usage:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.dollars = 0.0


def client_for(key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jev-seek",
            "X-Title": "jev-seek",
        },
        timeout=120,
    )


async def decide(
    client: httpx.AsyncClient,
    usage: Usage,
    state: str,
    orderings: dict[str, list[str]],
    *,
    model: str = MODEL,
    instructions: str = INSTRUCTIONS,
    nouls: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """({ordering name: {token: probability}}, {noul name: P(yes)}).

    One `choice` question per ordering, plus any yes/no `nouls` asked of the
    same state in the same call -- "is it finished?", "is it rambling?".
    Stopping is its own decision: a token spelled `<END>` is not something jev,
    asked what a person would type next, will ever pick.

    Empty criteria descriptions: jevgpt probed these to rank identically to
    label-as-description at about half the input tokens.
    """
    questions = {
        name: {"type": "choice", "instructions": instructions, "criteria": {t: "" for t in toks}}
        for name, toks in orderings.items()
    }
    for name, q in (nouls or {}).items():
        questions[f"__{name}__"] = {"type": "noul", "instructions": q}
    body = {"model": model, "state": state, "questions": questions}
    for attempt in range(5):
        r = await client.post(API_URL, json=body)
        if r.status_code < 400:
            break
        await asyncio.sleep(2 + 3 * attempt)
    r.raise_for_status()
    data = r.json()
    usage.calls += 1
    usage.input_tokens += data["usage"].get("input_tokens", 0)
    usage.dollars += data["usage"].get("cost", 0.0)
    answers = data["answers"]
    judged = {name: answers.pop(f"__{name}__")["noul"] for name in (nouls or {})}
    return {name: ans["probabilities"] for name, ans in answers.items()}, judged
