"""The loop: deepseek proposes a top-k, jev chooses, repeat.

Orderings
---------
Jev's answer depends on the order the criteria are listed in, so each step asks
the same question several times, one ordering per question, in a single call:

  deepseek   candidates in deepseek's own order, most likely first
  reverse    least likely first
  alpha      alphabetical -- arbitrary but stable, unrelated to either model
  random:N   N seeded shuffles

The per-ordering distributions are combined (`mean` of probabilities or
`geo`metric mean) into one distribution, optionally tilted toward deepseek's
own probabilities (`mix` is an exponent on p_deepseek: 0 = pure jev, 1 = the
product of the two), and decoded greedily or by temperature sampling.

Every step also records how much the orderings disagreed, which is the
experiment: how strong is the position bias, and does averaging shuffles
actually cancel it.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field

from . import deepseek, jev
from .deepseek import END, Candidate

DEFAULT_ORDERS = "deepseek,reverse,random:3"
DEFAULT_SYSTEM = ""
INSTRUCTIONS = {
    "next": "Which token should come next?",
    "assistant": "Next token of the assistant's reply?",
    "human": "Which token would a real person, not a chatbot, write next?",
}
STOP_QUESTION = "Is this message finished -- would the writer send it now rather than keep typing?"
RAMBLE_QUESTION = ("Has the assistant's message started rambling, repeating itself or padding -- "
                   "gone on past the point where it should have been sent?")
ESSAY_STOP_QUESTION = "Is this post finished -- is it complete as a published essay, with its argument landed?"
ESSAY_RAMBLE_QUESTION = ("Has the post started padding, repeating its point, or losing the thread -- "
                         "gone on past where it should have ended?")


@dataclass
class Config:
    """Defaults are the chat configuration that held up best: deepseek continues
    a plain `User:/Assistant:` transcript on the completions endpoint, jev is
    asked the plainest question, candidates below 1/1000 of deepseek's top are
    dropped, and jev's own "finished?" / "rambling?" judgements end the reply.
    See PRESETS for the other two shapes that work."""
    mode: str = "base"             # chat | raw | base
    instruction: str = "next"      # jev's question: a key of INSTRUCTIONS or free text
    system: str = DEFAULT_SYSTEM   # deepseek's system prompt in chat; jev-only in base; unused in raw
    jev_preamble: str = ""         # text jev sees above the transcript and deepseek never sees
    orders: str = DEFAULT_ORDERS   # criteria orderings, all asked in one jev call
    combine: str = "mean"          # mean | geo
    top_k: int = 10                # deepseek candidates offered to jev, API max 20
    min_p: float = 0.001           # drop candidates below min_p x deepseek's top probability
    merge: str = "space"           # none | space | case -- fold token variants into one criterion
    mix: float = 0.0               # exponent on p_deepseek in the score when there is no gate
    gate: str = "none"             # none | top1 | entropy -- jev's weight from deepseek's confidence
    gate_skip: float = 0.0         # skip the jev call when its weight is below this; 0 = never
    gate_power: float = 1.0        # w = w ** gate_power; below 1 hands jev more say at branch points
    temperature: float = 0.0       # 0 = greedy over the final score
    repeat_penalty: float = 1.6    # divisor per prior occurrence in the window; 1 = off
    repeat_window: int = 16
    max_tokens: int = 120
    min_tokens: int = 4            # neither END nor "finished" may fire before this many tokens
    stop_noul: float = 0.7         # ask jev "is it finished?" each step; stop at this probability. 0 = off
    stop_question: str = STOP_QUESTION
    ramble_noul: float = 0.6       # ask jev "has it started rambling?"; stop (and trim) at this probability. 0 = off
    ramble_question: str = RAMBLE_QUESTION
    fish_tries: int = deepseek.FISH_TRIES  # re-asks at high temperature when a sampled EOS hid the logprobs; 0 = off
    nudge: bool = True             # raw/base: supply "\n\n" when deepseek EOSes after an unterminated line
    seed: int = 0
    deepseek_model: str = deepseek.MODEL
    jev_model: str = jev.MODEL


PRESETS: dict[str, dict] = {
    # group-chat replies with the gate: jev chooses where deepseek is unsure
    # (w = (1 - p_top)^0.5, not asked below 0.05), deepseek carries the rest
    "gated": dict(gate="top1", gate_skip=0.05, gate_power=0.5),
    # group-chat replies, jev on every token with full weight
    "chat": {},
    # long-form: deepseek continues a document, jev chooses only where deepseek
    # is unsure (w = (1 - p_top)^0.5, not asked below 0.05), essay stop questions
    "essay": dict(mode="raw", top_k=15, repeat_penalty=1.4, max_tokens=1400, min_tokens=500,
                  gate="top1", gate_skip=0.05, gate_power=0.5,
                  stop_noul=0.75, stop_question=ESSAY_STOP_QUESTION,
                  ramble_noul=0.75, ramble_question=ESSAY_RAMBLE_QUESTION),
    # the original experiment: chat endpoint, assistant framing, no floor, no
    # stop questions -- jev alone, for reproducing the early findings
    "pure": dict(mode="chat", instruction="assistant",
                 system="You are a helpful assistant. Reply in plain prose, no markdown.",
                 min_p=0.0, repeat_penalty=1.0, min_tokens=0, stop_noul=0.0, ramble_noul=0.0),
}


@dataclass
class Step:
    prefix: str
    chosen: str | None
    candidates: list[dict]                   # [{token, p_deepseek, p_jev, score}]
    per_order: dict[str, dict[str, float]]
    orderings: dict[str, list[str]]          # the criteria order each question was asked in
    stats: dict
    latency_ms: float


@dataclass
class Result:
    text: str
    steps: list[Step]
    stop: str
    seconds: float
    ds_usage: deepseek.Usage
    jev_usage: jev.Usage
    summary: dict = field(default_factory=dict)


def parse_orders(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, n = part.partition(":")
        if name not in {"deepseek", "reverse", "alpha", "random"}:
            raise ValueError(f"unknown ordering {name!r}")
        out.append((name, int(n) if n else 1))
    return out


def orderings(cands: list[Candidate], spec: list[tuple[str, int]], rng: random.Random) -> dict[str, list[str]]:
    by_p = [c.text for c in sorted(cands, key=lambda c: -c.logprob)]
    out: dict[str, list[str]] = {}
    for name, n in spec:
        for i in range(n):
            key = name if n == 1 else f"{name}{i}"
            if name == "deepseek":
                out[key] = list(by_p)
            elif name == "reverse":
                out[key] = by_p[::-1]
            elif name == "alpha":
                out[key] = sorted(by_p, key=str.lower)
            else:
                perm = list(by_p)
                rng.shuffle(perm)
                out[key] = perm
    return out


def combine(per_order: dict[str, dict[str, float]], tokens: list[str], how: str) -> dict[str, float]:
    n = len(per_order)
    if how == "geo":
        agg = {t: math.exp(sum(math.log(d.get(t, 0.0) + 1e-6) for d in per_order.values()) / n) for t in tokens}
    else:
        agg = {t: sum(d.get(t, 0.0) for d in per_order.values()) / n for t in tokens}
    z = sum(agg.values()) or 1.0
    return {t: v / z for t, v in agg.items()}


def order_stats(per_order: dict[str, dict[str, float]], orderings_: dict[str, list[str]], combined: dict[str, float]) -> dict:
    """How much did the order matter at this step?"""
    winner = max(combined, key=combined.get)
    tops = {name: max(d, key=d.get) for name, d in per_order.items()}
    n = len(per_order)
    return {
        # share of orderings whose own winner was the combined winner
        "agree": sum(t == winner for t in tops.values()) / n,
        # share of orderings that picked whatever happened to be listed first
        "first_wins": sum(tops[name] == orderings_[name][0] for name in per_order) / n,
        # mean total-variation distance of each ordering from the combined distribution
        "tv": sum(0.5 * sum(abs(d.get(t, 0.0) - combined[t]) for t in combined) for d in per_order.values()) / n,
        "distinct_winners": len(set(tops.values())),
    }


def jev_weight(cands: list[Candidate], gate: str) -> float:
    """How much say jev gets at this step, from how sure deepseek is.

    Measured over the recorded runs: when deepseek's top token is >= 0.99, jev
    still overrides it about half the time, and nearly every such override is a
    grammar, case or glue error (`sea` -> `Sea`, `against the` -> `against
    shore`, `capital of` -> `capital is`). Where deepseek is unsure the
    overrides are the content choices we wanted jev for. So:

      none     1 -- jev decides, deepseek only proposes
      top1     1 - p(top token): sure token -> deepseek, toss-up -> jev
      entropy  normalised entropy of the top-k: 0 when one token has it all,
               1 when they are level; softer than top1 in the middle
    """
    if gate == "none" or len(cands) < 2:
        return 1.0
    ps = [c.p for c in cands]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    if gate == "top1":
        return 1.0 - max(ps)
    if gate == "entropy":
        return -sum(p * math.log(p) for p in ps if p > 0) / math.log(len(ps))
    raise ValueError(f"unknown gate {gate!r}")

def _penalty(tokens: list[str], tok: str, cfg: Config) -> float:
    """Jev loops when pushed past where it wanted to stop (`yeah... yeah...`), so
    every prior occurrence of a token in the window divides its score.
    Punctuation counts too: `...` twenty times is the failure, one `.` per
    sentence costs little."""
    if cfg.repeat_penalty <= 1.0 or not tok.strip():
        return 1.0
    key = tok.strip().lower()
    return cfg.repeat_penalty ** sum(t.strip().lower() == key for t in tokens[-cfg.repeat_window:])


def _pick(scores: dict[str, float], temperature: float, rng: random.Random) -> str:
    if temperature <= 0:
        return max(scores, key=scores.get)
    toks = list(scores)
    weights = [max(scores[t], 1e-12) ** (1.0 / temperature) for t in toks]
    return rng.choices(toks, weights=weights, k=1)[0]


def transcript(system: str, history: list[tuple[str, str]], message: str, reply: str) -> str:
    """The state jev judges: a plain rendering of the whole conversation."""
    parts = [f"System: {system}"] if system else []
    for u, a in history:
        parts.append(f"User: {u}\nAssistant: {a}")
    parts.append(f"User: {message}\nAssistant: {reply}")
    return "\n\n".join(parts)


def chat_messages(system: str, history: list[tuple[str, str]], message: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": message})
    return msgs

def merge_variants(cands: list[Candidate], level: str) -> tuple[list[Candidate], dict[str, str]]:
    """Jev is whitespace-blind: offered ` capital` and `capital` it picks either,
    and `Thecapital` follows. So variants that differ only by surrounding
    whitespace (` Paris` / `Paris`, `.` / `.\\n\\n`) are one criterion, holding
    their summed probability, and deepseek's likelier spelling is what gets
    emitted. `case` also folds ` Sea` into ` sea`. Whitespace-only tokens stay
    as they are.

    Returns the grouped candidates (label, logsumexp) and label -> surface token.
    """
    if level == "none":
        return cands, {c.text: c.text for c in cands}
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        key = c.text if c.text == END or not c.text.strip() else c.text.strip()
        if level == "case":
            key = key.lower()
        groups.setdefault(key, []).append(c)
    grouped, surface = [], {}
    for label, members in groups.items():
        lp = members[0].logprob
        for m in members[1:]:
            lp = deepseek._logaddexp(lp, m.logprob)
        grouped.append(Candidate(label, lp))
        surface[label] = max(members, key=lambda m: m.logprob).text
    return sorted(grouped, key=lambda c: -c.logprob), surface


def floor_candidates(cands: list[Candidate], min_p: float) -> list[Candidate]:
    """Drop candidates deepseek rates below `min_p` x its top probability. The
    grammar breaks all came from jev picking 1e-4 tokens; the floor removes
    those and leaves jev free among what is actually plausible."""
    if min_p <= 0 or not cands:
        return cands
    cut = math.log(min_p) + max(c.logprob for c in cands)
    return [c for c in cands if c.logprob >= cut or c.text == END]


def jev_state(cfg: Config, history, message: str, prefix: str) -> str:
    body = message + prefix if cfg.mode == "raw" else transcript(cfg.system, history, message, prefix)
    return f"{cfg.jev_preamble}\n\n{body}" if cfg.jev_preamble else body


def base_prompt(history, message: str, prefix: str) -> str:
    """`base` mode: what deepseek continues -- the transcript as plain text, no
    chat template and no system prompt, so the candidates are base-model-ish.
    Jev sees the same transcript with the system prompt on top and plays the
    assistant; the persona lives in the chooser, not the proposer."""
    return transcript("", history, message, prefix)


TURN_MARKERS = ("\nUser:", "\n\nUser:", "\nSystem:")
BOUNDARY = (".", "!", "?", "…", "\n")


def trim_to_boundary(text: str) -> str:
    """Cut a reply back to its last sentence boundary, if it has one."""
    stripped = text.rstrip()
    cut = max(stripped.rfind(b) for b in BOUNDARY)
    return stripped[:cut + 1].rstrip() if cut > 0 else stripped


async def reply(
    ds_client, jev_client, history: list[tuple[str, str]], message: str, cfg: Config,
    *, on_token=None, ds_usage: deepseek.Usage | None = None, jev_usage: jev.Usage | None = None,
) -> Result:
    ds_usage = ds_usage or deepseek.Usage()
    jev_usage = jev_usage or jev.Usage()
    spec = parse_orders(cfg.orders)
    rng = random.Random(cfg.seed)
    msgs = chat_messages(cfg.system, history, message)
    tokens: list[str] = []
    steps: list[Step] = []
    stop = "max-tokens"
    t_start = time.perf_counter()
    nudged = False

    for _ in range(cfg.max_tokens):
        t0 = time.perf_counter()
        prefix = "".join(tokens)
        kw = dict(model=cfg.deepseek_model, top_k=cfg.top_k, fish_tries=cfg.fish_tries)
        if cfg.mode == "raw":
            raw = await deepseek.propose_raw(ds_client, ds_usage, message + prefix, **kw)
        elif cfg.mode == "base":
            raw = await deepseek.propose_raw(ds_client, ds_usage, base_prompt(history, message, prefix), **kw)
        else:
            raw = await deepseek.propose(ds_client, ds_usage, msgs, prefix, **kw)

        # In completion mode deepseek-flash is near-certain the document is over
        # after a bare section header ("III. The Aftermath") -- EOS survives
        # fishing at T=2. Supplying the newline it would have written under
        # the header gets it writing again every time. One nudge per line; if
        # EOS comes straight back, the document really is over.
        if (cfg.mode != "chat" and cfg.nudge and len(raw) == 1 and raw[0].text == END
                and not nudged and prefix.rstrip() and not prefix.rstrip().endswith(BOUNDARY)):
            tokens.append("\n\n")
            nudged = True
            if on_token:
                on_token("\n\n")
            steps.append(Step(prefix=prefix, chosen="\n\n", candidates=[{"token": "\n\n", "surface": "\n\n",
                              "p_deepseek": 0.0, "p_jev": 0.0, "score": 0.0}], per_order={}, orderings={},
                              stats={"agree": 1.0, "first_wins": 1.0, "tv": 0.0, "distinct_winners": 1, "nudged": True},
                              latency_ms=round((time.perf_counter() - t0) * 1000, 1)))
            continue
        nudged = False
        cands, surface = merge_variants(floor_candidates(raw, cfg.min_p), cfg.merge)
        p_ds = {c.text: c.p for c in cands}

        w = jev_weight(cands, cfg.gate) ** cfg.gate_power
        judged: dict[str, float] = {}
        if len(cands) == 1 or (cfg.gate != "none" and w < cfg.gate_skip):
            # Nothing to choose, or deepseek is sure enough that jev is not asked.
            ords, per_order = {}, {}
            combined = {t: p / sum(p_ds.values()) for t, p in p_ds.items()}
            stats = {"agree": 1.0, "first_wins": 1.0, "tv": 0.0, "distinct_winners": 1}
        else:
            ords = orderings(cands, spec, rng)
            state = jev_state(cfg, history, message, prefix)
            nouls = {}
            if cfg.stop_noul > 0:
                nouls["finished"] = cfg.stop_question
            if cfg.ramble_noul > 0:
                nouls["rambling"] = cfg.ramble_question
            per_order, judged = await jev.decide(
                jev_client, jev_usage, state, ords, model=cfg.jev_model,
                instructions=INSTRUCTIONS.get(cfg.instruction, cfg.instruction), nouls=nouls)
            combined = combine(per_order, [c.text for c in cands], cfg.combine)
            stats = order_stats(per_order, ords, combined)
        stats.update({k: round(v, 3) for k, v in judged.items()})

        # Stopping is jev's own decision, asked alongside the token question.
        # "Finished" only counts at a sentence boundary; "rambling" stops at
        # once and trims back to the last boundary before the padding began.
        verdict = None
        if (len(tokens) >= cfg.min_tokens and judged.get("finished", 0) >= cfg.stop_noul > 0
                and prefix.rstrip().endswith(BOUNDARY)):
            verdict = "END:finished"
        elif judged.get("rambling", 0) >= cfg.ramble_noul > 0:
            # The floor does not apply: a reply that is rambling is over.
            verdict = "END:rambling"
            tokens[:] = [trim_to_boundary(prefix)]
        if verdict:
            steps.append(Step(prefix=prefix, chosen=None, candidates=[], per_order=per_order, orderings=ords,
                              stats=stats, latency_ms=round((time.perf_counter() - t0) * 1000, 1)))
            stop = verdict
            break
        stats["jev_weight"] = round(w, 3)
        stats["jev_asked"] = bool(per_order)

        if cfg.gate == "none":
            score = lambda t: combined[t] * (p_ds[t] ** cfg.mix)
        else:
            score = lambda t: (combined[t] ** w) * (p_ds[t] ** (1 - w))
        scores = {t: score(t) / _penalty(tokens, t, cfg) for t in combined
                  if not (t == END and len(tokens) < cfg.min_tokens and len(combined) > 1)}
        z = sum(scores.values()) or 1.0
        scores = {t: s / z for t, s in scores.items()}
        chosen = _pick(scores, cfg.temperature, rng)

        steps.append(Step(
            prefix=prefix,
            chosen=None if chosen == END else surface[chosen],
            candidates=[{"token": c.text, "surface": surface[c.text], "p_deepseek": round(c.p, 5),
                         "p_jev": round(combined[c.text], 4), "score": round(scores.get(c.text, 0.0), 4)} for c in cands],
            per_order=per_order,
            orderings=ords,
            stats=stats,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        ))
        if chosen == END:
            # deepseek's greedy EOS comes back with no logprobs at all, so there
            # is nothing for jev to choose from; the floor cannot hold there.
            stop = "END:deepseek" if len(cands) == 1 else "END:jev"
            break
        tokens.append(surface[chosen])
        if on_token:
            on_token(surface[chosen])
        if cfg.mode == "base":
            # A base-model continuation ends the assistant turn by starting the next one.
            text = "".join(tokens)
            cut = min((text.find(m) for m in TURN_MARKERS if m in text), default=-1)
            if cut >= 0:
                tokens[:] = [text[:cut].rstrip()]
                stop = "END:turn"
                break

    res = Result("".join(tokens), steps, stop, time.perf_counter() - t_start, ds_usage, jev_usage)
    res.summary = summarize(res)
    return res


def summarize(res: Result) -> dict:
    judged = [s for s in res.steps if s.per_order]
    n = len(judged) or 1
    # how often jev's pick was deepseek's own greedy pick
    agree_ds = sum(
        (s.chosen or END) == max(s.candidates, key=lambda c: c["p_deepseek"])["surface"]
        for s in res.steps if s.candidates
    ) / (len(res.steps) or 1)
    return {
        "tokens": len(res.steps) - res.stop.startswith("END"),
        "stop": res.stop,
        "seconds": round(res.seconds, 1),
        "ms_per_token": round(res.seconds * 1000 / (len(res.steps) or 1)),
        "jev_agrees_with_deepseek_top1": round(agree_ds, 3),
        "order_agree": round(sum(s.stats["agree"] for s in judged) / n, 3),
        "first_position_wins": round(sum(s.stats["first_wins"] for s in judged) / n, 3),
        "mean_tv": round(sum(s.stats["tv"] for s in judged) / n, 3),
        "steps_with_split_orders": sum(s.stats["distinct_winners"] > 1 for s in judged),
        "mean_jev_weight": round(sum(s.stats.get("jev_weight", 1.0) for s in res.steps) / (len(res.steps) or 1), 3),
        "jev_skipped": sum(not s.stats.get("jev_asked", True) for s in res.steps),
        "deepseek_calls": res.ds_usage.calls,
        "deepseek_prompt_tokens": res.ds_usage.hit + res.ds_usage.miss,
        "deepseek_cache_hit_ratio": round(res.ds_usage.hit / ((res.ds_usage.hit + res.ds_usage.miss) or 1), 3),
        "jev_calls": res.jev_usage.calls,
        "jev_input_tokens": res.jev_usage.input_tokens,
        "dollars": round(res.ds_usage.dollars + res.jev_usage.dollars, 4),
    }
