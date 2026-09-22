"""Score a reply with Pangram's AI-text detector.

POST /task, then poll /task/{id} until STAGE_SUCCESS. The result is per-window;
we report the word-weighted mean `ai_assistance_score` (0 = human, 1 = AI) and
the document-level label.
"""

from __future__ import annotations

import asyncio

import httpx

API = "https://text.external-api.pangram.com"


def client_for(key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"x-api-key": key, "Content-Type": "application/json"}, timeout=60)


async def score(client: httpx.AsyncClient, text: str, *, model: str = "default") -> dict:
    if len(text.split()) < 10:
        return {"ai": None, "label": "too short", "words": len(text.split())}
    r = await client.post(f"{API}/task", json={"text": text, "model": model})
    if r.status_code >= 400:
        return {"ai": None, "label": f"http {r.status_code}", "error": r.text[:200]}
    task_id = r.json()["task_id"]
    for _ in range(60):
        await asyncio.sleep(1.5)
        r = await client.get(f"{API}/task/{task_id}")
        r.raise_for_status()
        data = r.json()
        stage = data.get("stage") or data.get("status")
        if stage == "STAGE_SUCCESS":
            wins = data.get("windows") or []
            words = sum(w.get("word_count", 0) for w in wins) or 1
            ai = sum(w.get("ai_assistance_score", 0) * w.get("word_count", 0) for w in wins) / words
            return {
                "ai": round(ai, 4),
                "label": data.get("prediction_short") or (wins[0].get("label") if wins else "?"),
                "confidence": wins[0].get("confidence") if wins else None,
                "words": words,
            }
        if stage == "STAGE_FAILED":
            return {"ai": None, "label": "failed"}
    return {"ai": None, "label": "timeout"}
