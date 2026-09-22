"""Render recorded runs into one static page.

    python -m jevseek.site                          # experiments/runs -> docs/index.html
    python -m jevseek.site runs site/index.html     # any dir of *.json / *.json.gz

Each run shows the settings that differ from the defaults, then every turn:
prompt, reply, what was cut and why, Pangram's verdict, the per-reply stats,
and the trace of jev's "finished?" / "rambling?" judgements.
"""

from __future__ import annotations

import gzip
import html
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .sampler import Config

DEFAULTS = asdict(Config())
SKIP = {"system", "stop_question", "ramble_question", "jev_preamble"}  # shown separately, long
RETIRED = {"end_bias", "group_whitespace", "repeat_window", "deepseek_model", "jev_model"}  # not worth a chip
ORDER = ["mode", "instruction", "orders", "combine", "mix", "gate", "gate_skip", "gate_power", "min_p", "top_k",
         "merge", "repeat_penalty", "min_tokens", "stop_noul", "ramble_noul", "temperature", "seed",
         "max_tokens", "fish_tries", "nudge"]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def diff_config(cfg: dict) -> list[tuple[str, str]]:
    out = []
    for k in ORDER + [k for k in cfg if k not in ORDER]:
        if k in SKIP or k not in cfg or k in RETIRED:
            continue
        if cfg[k] != DEFAULTS.get(k):
            out.append((k, cfg[k]))
    return out


def pangram_badge(s: dict) -> str:
    pg = s.get("pangram")
    if not pg:
        return '<span class="badge muted">pangram —</span>'
    if pg.get("ai") is None:
        return f'<span class="badge muted">pangram {esc(pg.get("label"))}</span>'
    ai = pg["ai"]
    cls = "human" if ai < 0.2 else ("mixed" if ai < 0.8 else "ai")
    return f'<span class="badge {cls}">pangram {ai:.3f} · {esc(pg.get("label"))}</span>'


def untrimmed(t: dict) -> str:
    """The full text the sampler emitted before any trim: the last step's
    prefix plus whatever it chose. Equals `reply` unless a stop trimmed it."""
    steps = t.get("steps") or []
    if not steps:
        return t.get("reply", "")
    last = steps[-1]
    return (last.get("prefix") or "") + (last.get("chosen") or "")


def failure_note(t: dict) -> str:
    """What went wrong, when something did: the tail the ramble/turn stop cut
    off (struck through), the noul readings at that moment, and a flag when a
    run hit max-tokens (which in practice means the sampler never stopped)."""
    s = t.get("summary", {})
    stop = s.get("stop", "")
    steps = t.get("steps") or []
    last = steps[-1] if steps else {}
    st = last.get("stats", {})
    kept = t.get("reply", "")
    full = untrimmed(t)
    parts = []
    if stop in ("END:rambling", "END:turn") and full.strip() != kept.strip() and full.startswith(kept.rstrip()[:len(full)]):
        tail = full[len(kept.rstrip()):]
        why = "jev's rambling noul" if stop == "END:rambling" else "deepseek starting the next turn"
        readings = " · ".join(f'{k} {st[k]:.2f}' for k in ("finished", "rambling") if k in st)
        parts.append(
            f'<div class="cut"><span class="who">cut ›</span>{esc(tail.strip())}'
            f'<div class="cutwhy">stopped by {why} at {len(steps)} steps{" — " + esc(readings) if readings else ""}</div></div>'
        )
    elif stop == "max-tokens":
        readings = " · ".join(f'{k} {st[k]:.2f}' for k in ("finished", "rambling") if k in st)
        parts.append(f'<div class="cutwhy">never stopped — hit max-tokens{" — last readings: " + esc(readings) if readings else ""}</div>')
    return "".join(parts)


def noul_trace(t: dict) -> str:
    rows = []
    for i, s in enumerate(t.get("steps") or []):
        st = s.get("stats", {})
        if "finished" not in st and "rambling" not in st:
            continue
        fin = f'{st["finished"]:.2f}' if "finished" in st else "  — "
        ram = f'{st["rambling"]:.2f}' if "rambling" in st else "  — "
        tail = (s.get("prefix") or "")[-48:].replace("\n", "⏎")
        rows.append(f"{i:4}  fin {fin}  ram {ram}  …{tail}")
    if not rows:
        return ""
    return f'<details><summary>noul trace ({len(rows)} steps: finished / rambling, with the text so far)</summary><pre>{esc(chr(10).join(rows))}</pre></details>'


def render_turn(t: dict, cfg: dict) -> str:
    s = t.get("summary", {})
    stop = s.get("stop", "?")
    stop_cls = "bad" if stop in ("max-tokens", "END:rambling") else ("warn" if stop in ("END:deepseek", "END:turn") else "")
    facts = [
        f'{s.get("tokens", "?")} tokens',
        f'<span class="stop {stop_cls}">stop {esc(stop)}</span>',
        f'jev=ds-top1 {s.get("jev_agrees_with_deepseek_top1", 0):.0%}',
        f'jev weight {s.get("mean_jev_weight", 1.0):.2f}',
        f'{s.get("jev_skipped", 0)} skipped',
        f'{s.get("seconds", 0)}s',
        f'${s.get("dollars", 0):.4f}',
    ]
    label = "orders" if "orders" in t else None
    head = f'<div class="ordtag">orders = {esc(t["orders"])}</div>' if label else ""
    return f"""
<article class="turn">
  {head}
  <div class="prompt"><span class="who">you ›</span> {esc(t.get("message", ""))}</div>
  <div class="reply"><span class="who">jev ›</span>{esc(t.get("reply", ""))}</div>
  {failure_note(t)}
  <div class="facts">{pangram_badge(s)} {" · ".join(facts)}</div>
  {noul_trace(t)}
</article>"""


def render_run(path: Path, data: dict, stem: str) -> str:
    cfg = data.get("config", {})
    diffs = diff_config(cfg)
    chips = "".join(f'<span class="chip"><b>{esc(k)}</b> {esc(v)}</span>' for k, v in diffs) or '<span class="chip muted">defaults</span>'
    longs = "".join(
        f'<details><summary>{esc(k)}</summary><pre>{esc(cfg[k])}</pre></details>'
        for k in ("system", "jev_preamble", "stop_question", "ramble_question")
        if cfg.get(k) and cfg[k] != DEFAULTS.get(k)
    )
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    turns = "".join(render_turn(t, cfg) for t in data.get("turns", []))
    return f"""
<section class="run" id="{esc(stem)}">
  <h2>{esc(stem)} <small>{when} · {len(data.get("turns", []))} turn(s)</small></h2>
  <div class="chips">{chips}</div>
  {longs}
  {turns}
</section>"""


CSS = """
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1c1c1c;background:#fafaf7}
h1{font-weight:600} h2{font-size:1.1rem;margin:2.5rem 0 .5rem;border-top:1px solid #ddd;padding-top:1rem}
h2 small{color:#777;font-weight:400;margin-left:.5rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:.5rem}
.chip{background:#eee;border-radius:4px;padding:.1rem .45rem;font-size:.8rem;font-family:ui-monospace,Menlo,monospace}
.chip.muted,.badge.muted{color:#888}
details{margin:.25rem 0} summary{cursor:pointer;font-size:.85rem;color:#555}
pre{white-space:pre-wrap;background:#f1f1ee;padding:.5rem;border-radius:4px;font-size:.8rem}
.turn{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:.75rem 1rem;margin:.75rem 0}
.who{color:#999;font-family:ui-monospace,Menlo,monospace;font-size:.8rem;margin-right:.4rem}
.prompt{color:#444;white-space:pre-wrap;margin-bottom:.5rem}
.reply{white-space:pre-wrap}
.facts{margin-top:.6rem;font-size:.8rem;color:#666}
.badge{border-radius:4px;padding:.05rem .4rem;font-weight:600;margin-right:.4rem}
.badge.human{background:#dff5e1;color:#1b6e2a} .badge.ai{background:#fde2e2;color:#a12626} .badge.mixed{background:#fff3cd;color:#7a5b00}
.ordtag{font-size:.8rem;color:#555;font-family:ui-monospace,Menlo,monospace;margin-bottom:.3rem}
.cut{white-space:pre-wrap;margin-top:.4rem;color:#a12626}
.cutwhy{font-size:.8rem;color:#a12626;margin-top:.2rem;white-space:normal}
.stop.bad{color:#a12626;font-weight:600} .stop.warn{color:#7a5b00;font-weight:600}
nav{font-size:.85rem;line-height:1.9} nav a{margin-right:.6rem;white-space:nowrap}
.intro{color:#555}
"""


def _load(p: Path) -> dict | None:
    try:
        raw = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
        return json.loads(raw)
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile):
        return None


def _stem(p: Path) -> str:
    return p.name[: -len(".json.gz")] if p.name.endswith(".json.gz") else p.stem


def build(runs_dir: Path, out: Path) -> int:
    files = sorted([*runs_dir.glob("*.json"), *runs_dir.glob("*.json.gz")], key=lambda p: p.stat().st_mtime, reverse=True)
    sections, nav = [], []
    for p in files:
        data = _load(p)
        if not data or not data.get("turns"):
            continue
        sections.append(render_run(p, data, _stem(p)))
        nav.append(f'<a href="#{esc(_stem(p))}">{esc(_stem(p))}</a>')
    page = f"""<!doctype html><meta charset="utf-8"><title>jevseek runs</title><style>{CSS}</style>
<h1>jevseek — recorded runs</h1>
<p class="intro">deepseek proposes the next token, TypeSafe's jev chooses. Each section is one recorded run;
chips show the settings that differ from the current defaults. Pangram scores are 0 = human, 1 = AI. Newest first.
Red <i>cut ›</i> lines are text jev's own "rambling" judgement removed.</p>
<nav>{" ".join(nav)}</nav>
{"".join(sections)}
<p class="intro">generated {datetime.now().strftime("%Y-%m-%d")} from {len(sections)} run files</p>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return len(sections)


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/runs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/index.html")
    n = build(runs, out)
    print(f"wrote {out} ({n} runs)")


if __name__ == "__main__":
    main()
