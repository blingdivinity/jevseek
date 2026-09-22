"""jevseek: deepseek proposes the next token, jev chooses it.

  jevseek                                   interactive REPL (/reset, /quit)
  jevseek --say "do you ever get bored in here"
  jevseek --preset essay --say-file prompt.txt --record run.json
  jevseek --say "..." --pangram -v          detector score + per-step tables

Keys: DEEPSEEK_API_KEY, OPENROUTER_API_KEY (jev via OpenRouter's System One
endpoint), PANGRAM_API_KEY (optional). A .env file in the working directory
is read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from . import deepseek, jev, pangram
from .sampler import INSTRUCTIONS, PRESETS, Config, Result, reply


def _env(*names: str) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None


def _keys() -> tuple[str, str]:
    load_dotenv()
    ds = _env("DEEPSEEK_API_KEY", "deepseek")
    orr = _env("OPENROUTER_API_KEY", "openrouter")
    missing = [n for n, v in (("DEEPSEEK_API_KEY", ds), ("OPENROUTER_API_KEY", orr)) if not v]
    if missing:
        sys.exit(f"missing: {', '.join(missing)} (export them or put them in .env; see .env.example)")
    return ds, orr


def _show_step(step, orders_n: int):
    """One line per candidate: deepseek's p, jev's combined p, the score, and a
    mark on what each side wanted."""
    ds_top = max(step.candidates, key=lambda c: c["p_deepseek"])["surface"]
    print(f"\n  after {step.prefix[-40:]!r}", file=sys.stderr)
    for c in sorted(step.candidates, key=lambda c: -c["score"])[:8]:
        marks = ("*" if c["surface"] == (step.chosen or deepseek.END) else " ") + ("d" if c["surface"] == ds_top else " ")
        print(f"  {marks} {c['surface']!r:22} ds={c['p_deepseek']:<8.4f} jev={c['p_jev']:<7.3f} score={c['score']:.3f}",
              file=sys.stderr)
    s = step.stats
    print(f"    orders agree={s['agree']:.2f} first_wins={s['first_wins']:.2f} tv={s['tv']:.2f} "
          f"winners={s['distinct_winners']}/{orders_n} jev_w={s.get('jev_weight', 1.0):.2f}"
          f"{'' if s.get('jev_asked', True) else ' (skipped)'} {step.latency_ms:.0f}ms", file=sys.stderr)


def _summary_line(res: Result) -> str:
    s = res.summary
    return (f"{s['tokens']} tokens · stopped: {s['stop']} · {s['seconds']}s ({s['ms_per_token']}ms/tok) · "
            f"jev=deepseek-top1 {s['jev_agrees_with_deepseek_top1']:.0%} · orders agree {s['order_agree']:.0%} · "
            f"first-listed wins {s['first_position_wins']:.0%} · tv {s['mean_tv']:.2f} · "
            f"jev weight {s['mean_jev_weight']:.2f} ({s['jev_skipped']} skipped) · ${s['dollars']:.4f}")


async def run_turn(ds_client, jev_client, history, message, cfg, *, verbose, echo=True, pg_client=None) -> Result:
    if echo:
        print("jev › ", end="", flush=True)
    res = await reply(ds_client, jev_client, history, message, cfg,
                      on_token=(lambda t: print(t, end="", flush=True)) if echo else None)
    if echo:
        print()
        if res.stop in ("END:rambling", "END:turn"):
            print(f"  ↳ kept: {res.text.strip()!r}", file=sys.stderr)
    if verbose:
        n = max((len(s.per_order) for s in res.steps), default=1)
        for step in res.steps:
            _show_step(step, n)
    if pg_client is not None:
        pg = await pangram.score(pg_client, (message if cfg.mode == "raw" else "") + res.text)
        res.summary["pangram"] = pg
        tag = f"pangram AI={pg['ai']:.3f} ({pg['label']}, {pg.get('confidence')})" if pg["ai"] is not None else f"pangram: {pg['label']}"
        print(f"      {tag}", file=sys.stderr)
    print(f"      {_summary_line(res)}", file=sys.stderr)
    return res


def _record(path: Path, turns: list[dict], cfg: Config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "turns": turns}, indent=1, ensure_ascii=False))


def _turn_json(message: str, res: Result) -> dict:
    return {
        "message": message,
        "reply": res.text,
        "summary": res.summary,
        "steps": [asdict(s) for s in res.steps],
    }


def build_config(args) -> Config:
    """Preset first, then every flag the user actually passed on top."""
    overrides = {k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__ and v is not None}
    if "no_nudge" in vars(args) and args.no_nudge:
        overrides["nudge"] = False
    return Config(**{**PRESETS[args.preset], **overrides})


async def amain(args):
    ds_key, or_key = _keys()
    cfg = build_config(args)
    messages = list(args.say or [])
    if args.say_file:
        messages.append(Path(args.say_file).read_text())
    turns: list[dict] = []
    pg_key = _env("PANGRAM_API_KEY", "pangram")
    if args.pangram and not pg_key:
        sys.exit("--pangram needs PANGRAM_API_KEY")
    pg_client = pangram.client_for(pg_key) if args.pangram else None
    async with deepseek.client_for(ds_key) as ds_client, jev.client_for(or_key) as jev_client:
        history: list[tuple[str, str]] = []

        async def turn(message: str):
            res = await run_turn(ds_client, jev_client, history, message, cfg, verbose=args.verbose, pg_client=pg_client)
            history.append((message, res.text))
            turns.append(_turn_json(message, res))
            if args.record:
                _record(Path(args.record), turns, cfg)

        if messages:
            for message in messages:
                print(f"you › {message}")
                await turn(message)
            return

        print(f"jevseek — deepseek proposes, jev chooses (preset: {args.preset}). /reset clears, /quit exits.", file=sys.stderr)
        while True:
            try:
                message = input("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message == "/quit":
                break
            if message == "/reset":
                history.clear()
                continue
            await turn(message)


def main():
    d = Config()  # defaults, shown in --help; argparse itself defaults to None so presets can fill in
    p = argparse.ArgumentParser(prog="jevseek", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS), default="gated",
                   help="gated: chat replies, jev chooses where deepseek is unsure (default). chat: jev on every token. "
                        "essay: long-form document with the gate. pure: the original chat-endpoint experiment, no floor, no stop questions")
    p.add_argument("--say", action="append", metavar="MSG", help="scripted turn; repeat for a conversation")
    p.add_argument("--say-file", metavar="PATH", help="a turn read from a file (long prompts, essay headers)")

    g = p.add_argument_group("what each model sees")
    g.add_argument("--mode", choices=["chat", "raw", "base"],
                   help=f"chat: deepseek's chat endpoint with a system prompt. raw: continue the text as a document on "
                        f"the completions endpoint. base: deepseek continues a plain User:/Assistant: transcript on that "
                        f"endpoint; jev sees it with --system on top (default {d.mode})")
    g.add_argument("--instruction", help=f"jev's question: one of {', '.join(INSTRUCTIONS)} or free text (default {d.instruction})")
    g.add_argument("--system", help="deepseek's system prompt in chat mode; jev-only persona in base mode (default none)")
    g.add_argument("--jev-preamble", help="text placed above the transcript for jev only")

    g = p.add_argument_group("choosing")
    g.add_argument("--orders", help=f"comma list of deepseek | reverse | alpha | random:N, all asked in one call (default {d.orders})")
    g.add_argument("--combine", choices=["mean", "geo"], help=f"how per-ordering distributions merge (default {d.combine})")
    g.add_argument("--top-k", type=int, help=f"deepseek candidates offered to jev, API max 20 (default {d.top_k})")
    g.add_argument("--min-p", type=float, help=f"drop candidates below this fraction of deepseek's top probability (default {d.min_p})")
    g.add_argument("--merge", choices=["none", "space", "case"], help=f"fold token variants into one criterion (default {d.merge})")
    g.add_argument("--mix", type=float, help=f"without a gate: exponent on p_deepseek in the score; 0 = pure jev (default {d.mix})")
    g.add_argument("--gate", choices=["none", "top1", "entropy"],
                   help="jev's weight per step from deepseek's confidence: score = jev^w * deepseek^(1-w)")
    g.add_argument("--gate-skip", type=float, help="with --gate: do not call jev at all when its weight is below this")
    g.add_argument("--gate-power", type=float, help="with --gate: w = w ** power; below 1 gives jev more say")
    g.add_argument("--temperature", type=float, help=f"sampling over the final score; 0 = greedy (default {d.temperature})")
    g.add_argument("--repeat-penalty", type=float, help=f"divisor per prior occurrence in the last 16 tokens; 1 = off (default {d.repeat_penalty})")
    g.add_argument("--no-repeat-ngram", type=int,
                   help=f"block a token that would repeat an n-gram already in the text -- stops jev re-choosing a whole sentence; 0 = off (default {d.no_repeat_ngram}; essay preset 4)")
    g.add_argument("--seed", type=int, help="shuffle and sampling seed")

    g = p.add_argument_group("stopping")
    g.add_argument("--max-tokens", type=int, help=f"(default {d.max_tokens})")
    g.add_argument("--min-tokens", type=int, help=f"no ending before this many tokens (default {d.min_tokens})")
    g.add_argument("--stop-noul", type=float,
                   help=f"each step also asks jev 'is this message finished?'; stop at a sentence boundary when P >= this; 0 = off (default {d.stop_noul})")
    g.add_argument("--stop-question", help="wording of that yes/no question")
    g.add_argument("--ramble-noul", type=float,
                   help=f"also asks 'has this started rambling?'; stop and trim to the last sentence when P >= this; 0 = off (default {d.ramble_noul})")
    g.add_argument("--ramble-question", help="wording of that yes/no question")
    g.add_argument("--ramble-patience", type=int,
                   help=f"the rambling verdict must hold this many judged steps in a row before it counts (default {d.ramble_patience}; essay preset 6)")
    g.add_argument("--fish-tries", type=int,
                   help=f"when deepseek's sampled token is EOS it returns no logprobs; re-ask at T=2 this many times and "
                        f"un-temper so <END> is a candidate jev can weigh; 0 = let deepseek end alone (default {d.fish_tries})")
    g.add_argument("--no-nudge", action="store_true",
                   help="raw/base: do not supply a newline when deepseek EOSes right after a header-like line")

    g = p.add_argument_group("output")
    g.add_argument("--pangram", action="store_true", help="score each reply with Pangram's AI-text detector (PANGRAM_API_KEY)")
    g.add_argument("--record", metavar="PATH", help="JSON trace: every step's candidates, per-ordering jev distributions, stop judgements")
    g.add_argument("--verbose", "-v", action="store_true", help="print each step's candidate table to stderr")
    g.add_argument("--deepseek-model", help=f"(default {deepseek.MODEL})")
    g.add_argument("--jev-model", help=f"(default {jev.MODEL})")
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
