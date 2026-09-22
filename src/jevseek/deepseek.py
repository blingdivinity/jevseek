"""Propose: deepseek-flash's next-token distribution for a chat prefix.

One call per step: `max_tokens=1`, `logprobs=true`, `top_logprobs=20`, thinking
off, and the reply so far sent as an assistant message with `prefix: true`
(the beta chat-prefix-completion endpoint). The prefix is retokenised server
side, so it does not matter that we grow it one token at a time.

When the greedy token is end-of-sequence the API returns `finish_reason=stop`
with no logprobs at all, so END is synthesised there with probability 1.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass

import httpx

API_URL = "https://api.deepseek.com/beta/chat/completions"
RAW_URL = "https://api.deepseek.com/beta/completions"
MODEL = "deepseek-flash"
EOS_TOKENS = {"<｜｜end▁of▁sentence｜｜>", "<｜end▁of▁sentence｜>"}
END = "<END>"
# Off-peak flash prices per Mtok: cache-hit input, cache-miss input, output.
PRICE_HIT, PRICE_MISS, PRICE_OUT = 0.003, 0.15, 0.6


@dataclass
class Candidate:
    text: str          # decoded token text, or END
    logprob: float

    @property
    def p(self) -> float:
        return math.exp(self.logprob)


@dataclass
class Usage:
    calls: int = 0
    hit: int = 0
    miss: int = 0
    out: int = 0

    @property
    def dollars(self) -> float:
        return (self.hit * PRICE_HIT + self.miss * PRICE_MISS + self.out * PRICE_OUT) / 1e6


def client_for(key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=60,
    )


def _decode(entry: dict) -> str | None:
    """Token text from the byte array; None for fragments that are not valid
    UTF-8 on their own (multi-byte characters split across tokens)."""
    raw = entry.get("bytes")
    if raw is None:
        text = entry["token"]
        return None if re.search(r"\\x[0-9a-f]{2}", text) else text
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _logaddexp(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


async def _post(client, usage, url, body) -> dict:
    for attempt in range(5):
        r = await client.post(url, json=body)
        if r.status_code < 400:
            break
        await asyncio.sleep(1 + 2 * attempt)
    r.raise_for_status()
    data = r.json()
    u = data.get("usage", {})
    usage.calls += 1
    usage.hit += u.get("prompt_cache_hit_tokens", 0)
    usage.miss += u.get("prompt_cache_miss_tokens", 0)
    usage.out += u.get("completion_tokens", 0)
    return data


def _candidates(entries: list[dict], top_k: int, temperature: float = 1.0) -> list[Candidate]:
    """Merge duplicate spellings and, when the call was made at a temperature
    other than 1, undo it: the API reports log_softmax(logits / T), verified to
    two decimals, so T x logprob renormalised over the returned set is the true
    distribution restricted to that set."""
    merged: dict[str, float] = {}
    for entry in entries:
        text = _decode(entry)
        if text is None or text == "":
            continue
        if text in EOS_TOKENS:
            text = END
        lp = entry["logprob"] * temperature
        merged[text] = _logaddexp(merged[text], lp) if text in merged else lp
    if temperature != 1.0 and merged:
        m = max(merged.values())
        z = m + math.log(sum(math.exp(v - m) for v in merged.values()))
        merged = {t: v - z for t, v in merged.items()}
    cands = sorted((Candidate(t, lp) for t, lp in merged.items()), key=lambda c: -c.logprob)
    return cands[:top_k]


# The API samples its one token and, when that sample is EOS, returns no
# logprobs at all -- even when EOS was only 6% likely. So a step that comes back
# empty is re-asked at a high temperature: any non-EOS sample carries the full
# top-k, including <END> at its true probability once un-tempered, and jev gets
# to decide whether the reply is over. Only a near-certain EOS survives this.
FISH_TEMPERATURE = 2.0
FISH_TRIES = 4


async def _fish(fetch, top_k: int, tries: int) -> list[Candidate]:
    entries = await fetch(1.0)
    if entries:
        return _candidates(entries, top_k)
    for _ in range(tries):
        entries = await fetch(FISH_TEMPERATURE)
        if entries:
            return _candidates(entries, top_k, FISH_TEMPERATURE)
    return [Candidate(END, 0.0)]


async def propose(
    client: httpx.AsyncClient,
    usage: Usage,
    messages: list[dict],
    prefix: str,
    *,
    model: str = MODEL,
    top_k: int = 20,
    fish_tries: int = FISH_TRIES,
) -> list[Candidate]:
    """Chat mode: the reply so far rides as an assistant prefix."""
    async def fetch(temperature: float):
        body = {
            "model": model,
            "messages": [*messages, {"role": "assistant", "content": prefix, "prefix": True}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": min(max(top_k, 1), 20),
            "thinking": {"type": "disabled"},
            "temperature": temperature,
        }
        data = await _post(client, usage, API_URL, body)
        lp = data["choices"][0].get("logprobs")
        if not lp or not lp.get("content"):
            return None
        return lp["content"][0]["top_logprobs"]
    return await _fish(fetch, top_k, fish_tries)


async def propose_raw(
    client: httpx.AsyncClient,
    usage: Usage,
    text: str,
    *,
    model: str = MODEL,
    top_k: int = 20,
    fish_tries: int = FISH_TRIES,
) -> list[Candidate]:
    """Raw mode: plain text continuation on the beta completions endpoint, no
    chat template, no system prompt -- deepseek prompted like a base model. The
    distribution here is far flatter than in chat mode."""
    async def fetch(temperature: float):
        body = {"model": model, "prompt": text, "max_tokens": 1,
                "logprobs": min(max(top_k, 1), 20), "temperature": temperature}
        data = await _post(client, usage, RAW_URL, body)
        tops = (data["choices"][0].get("logprobs") or {}).get("top_logprobs") or []
        if not tops or not tops[0]:
            return None
        # Legacy logprobs shape: [{token: logprob, ...}] with no byte arrays.
        return [{"token": t, "logprob": p} for t, p in tops[0].items()]
    return await _fish(fetch, top_k, fish_tries)
