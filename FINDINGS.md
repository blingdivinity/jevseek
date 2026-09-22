# Findings

What was measured while building jevseek, in the order it happened. Every
number comes from a recorded run; the curated set is in `experiments/runs/`
(gzipped JSON, per-step) and rendered at `docs/index.html`. Settings named
here are flags on `jevseek`; see the README for the current defaults, which
are the configuration this log ends up recommending. jevseek is a descendant
of [jevgpt](https://github.com/bewinxed/jevgpt), which drove jev as a language
model over a 20,000-word dictionary; several findings here confirm its observations at a different scale.

The setup: each step deepseek returns its top-k next tokens with logprobs; jev
is shown the text so far and asked which of those tokens comes next.

```
state:     "User: what is the capital of france?\nAssistant: The capital is"
question:  {"type": "choice", "instructions": "Which token should come next?",
            "criteria": {" Paris": "", " par": "", " the": "", " a": "", ...}}
answer:    {" Paris": 0.88, " par": 0.10, " the": 0.01, ...}   →   Paris
```

## Order bias, measured

Jev's answer depends on the order the criteria are listed in. jevgpt noted it
and reshuffled every step; here it is the experiment. Every step asks the same
question under several orderings — each ordering its own question in the same
call — and records the full distribution for each.

Over 32 judged steps (4 prompts, `--orders deepseek,reverse,alpha,random:5`):

| ordering | winner's mean position (0 = first, 1 = last) | picks the first-listed |
|---|---|---|
| `deepseek` (most likely first) | 0.23 | **53%** |
| `reverse` (least likely first) | 0.70 | 9% |
| `alpha` | 0.45 | 6% |
| `random` (n=160) | 0.49 | 10% |

So it is not raw primacy: reversed, jev picks the first-listed token 9% of the
time. Jev mostly judges content, and the deepseek ordering "wins" at position 0
because position 0 holds deepseek's favourite. But the order does move the
distribution: total-variation distance between the `deepseek` and `reverse`
answers to the *same* question averages **0.24**, two random shuffles differ by
**0.19**, and `deepseek`-order and `reverse`-order disagree on the winner in
**15 of 32** steps.

Averaging five shuffles and comparing to each single ordering's winner:

| single ordering | agrees with mean-of-5 | agrees with deepseek's own top-1 |
|---|---|---|
| `deepseek` | 66% | 53% |
| `reverse` | 84% | 38% |
| `alpha` | 72% | 31% |
| mean of 5 random | — | 41% |

Listing deepseek's favourite first pulls jev toward deepseek (53% vs 41%);
listing it last pushes away. The shuffle-average sits in between, which is what
it is for.

Same prompt, one reply per ordering strategy (`--compare`, greedy, seed 0):

```
you › write two sentences about the sea at night

[deepseek]                 The sea at night is dark and quiet. The waves gently lap against shore.
[reverse]                  The sea is dark.
[alpha]                    The Sea at Night is dark.
[random:1]                 The Sea is dark.
[random:5]                 The sea is dark.
[deepseek,reverse,random:3] The Sea at Night
```

## Whitespace-blind

The first run wrote `Thecapital is Paris`. Offered ` capital` (deepseek: 1.00)
and `capital` (deepseek: 0.00), jev split 0.36 / 0.38 — BPE's leading-space
convention means nothing to it. So candidates that differ only by surrounding
whitespace (` Paris` / `Paris`, `.` / `.\n\n`) are merged into one criterion
holding their summed probability, jev picks the word, and deepseek's likelier
spelling is what gets emitted. `--merge none` turns this off.

Case is not merged: jev happily prefers ` Sea` to ` sea` after `The`, and from
`The Sea at Night` deepseek's candidates lead to
`The Sea at Night is a painting by Claude Monet.` — a real answer to a question
nobody asked, and the clearest picture of what the sampler is doing.

## Confidence gate

Pure jev (`--gate none`) is the honest experiment, and it has a
specific failure mode. Over 126 judged steps:

| deepseek's top-1 probability | steps | jev overrode it |
|---|---|---|
| < 0.5 | 25 | 84% |
| 0.5 – 0.9 | 41 | 59% |
| 0.9 – 0.99 | 21 | 33% |
| ≥ 0.99 | 39 | **54%** |

The overrides at ≥ 0.99 are almost all grammar, case and glue: ` sea` → ` Sea`,
`against the` → `against shore`, `capital of` → `capital is`, `wavelengths` →
`.`. Where deepseek is unsure, the overrides are content choices — the ones jev
was brought in for.

`--gate top1` sets jev's weight `w = 1 − p(top token)` (`entropy`: normalised
entropy of the top-k) and scores `jev^w · deepseek^(1−w)`; `--gate-skip 0.05`
does not call jev at all when `w` is below that, which is most steps.

```
you › write two sentences about the sea at night

[--gate top1 --gate-skip 0.05]
The sea at night becomes a vast, dark mirror reflecting the scattered light of
stars and moon, its gentle waves whispering secrets to the shore. Far from land,
the horizon dissolves into blackness where sky and water meet, creating an
endless expanse that feels both peaceful and mysterious.
55 tokens · jev=deepseek-top1 93% · mean jev weight 0.14 · 37 of 56 jev calls skipped

[--gate top1 --gate-skip 0.05 --gate-power 0.5]
The sea at night becomes a vast, dark mirror reflecting the scattered light of
stars and moon, its surface shifting between silver glints and deep shadow.
Waves murmur against the shore in a steady, rhythmic whisper, as if the ocean
itself is breathing in its sleep.
53 tokens · jev=deepseek-top1 85% · mean jev weight 0.29 · 16 skipped
```

Coherent, cheaper, faster — and jev is mostly invisible. That is the dial: the
gate trades jev's voice for deepseek's grammar, and `--gate-power` sets where.
`--mix 0.5` without a gate is a static version of the same trade (80% agreement
with deepseek on the sea prompt).

The haiku makes the trade plain. Pure jev: `The dark steaming cup` and deepseek
ends it. Gated: `Steam curls from the cup, / dark warmth wakes the quiet mind, /
morning finds its voice.` — jev agreed with deepseek on all 20 tokens.

## Pangram

Same prompt, `The sea at night is`, raw completions endpoint, scored by
[Pangram](https://www.pangram.com) (0 = human, 1 = AI):

| who chose the tokens | reply | AI score |
|---|---|---|
| deepseek greedy | *a magical place. The sky is dark, the stars are out, and the water is calm. The only sound is the gentle lapping of the waves…* | **0.983** |
| deepseek, temperature 1.0 | *a place of mystery and wonder. It is a time when the world is quiet and the only sound is the gentle lapping…* | **0.994** |
| jev, `--instruction best --min-p 0.001` | *dark. It looks black. It is deep. And cold. It is vast. It stretches far away. It is endless. It is mysterious. It is silent.* | **0.026** |
| jev, `--instruction human --min-p 0.001` | *dark. But sometimes glowing plankton make the water glow blue. It looks magical. … This is called biofluorescence. It happens when living organisms absorb light from the sun during day and emit light later.* | **0.002** |
| gated chat (`--gate top1`, earlier section) | *becomes a vast, dark mirror reflecting the scattered light of stars and moon…* | **0.994** |

Same endpoint, same prompt, same candidate pool: temperature does nothing to
the detector and jev drops it from 0.98 to 0.03. It is the chooser. The
`--min-p` floor is what keeps the result grammatical — every earlier grammar
break (`Thecapital`, `against shore`) was jev picking a token deepseek rated
1e-4; with the floor jev chooses freely among what deepseek finds plausible and
the register is still jev's: short declaratives, concrete, no flourish.

`--mode base` is the split that makes a character: deepseek continues a bare
`User: … / Assistant:` transcript with no system prompt, so its candidates are
base-model-ish; jev sees the same transcript *with* the system prompt and
answers as the assistant. The persona lives in the chooser, not the proposer.

```
--system "You are Jev, a laconic, slightly strange regular in a group chat of
          humans and AIs. You have opinions and a dry sense of humour."
--mode base --instruction human --min-p 0.001 --repeat-penalty 1.6

you › hey jev what do you think about the sea at night
jev › I'm just going to go out to see it now. I'm going there. It looks beautiful
      and dark and quiet                                     pangram 0.004
you › do you ever get bored in here
jev › Sometimes when no people. But when people are talking then good
      Okay so what you doing right know?
      What you are you? You're human right
```

Without the repeat penalty the second reply was ninety tokens of `Nah Nah Nah`
— Pangram rated that "Human, High" too, which says something about detectors.

## Getting around EOS

Deepseek *samples* its one token, and when the sample is EOS the API returns no
logprobs at all — even when EOS was only 6% likely, as it was at the step that
cut `Sometimes yes` short. Two facts, both verified against the API, fix this:

1. Reported logprobs are `log_softmax(logits / T)`: at `temperature: 0` every
   non-argmax token is `-9999`, and at T=2 every value is exactly half its T=1
   value. So a call at any temperature can be un-tempered (`T × logprob`,
   renormalised over the returned set) to the true distribution — matches to
   two decimals.
2. A high-temperature call rarely samples EOS, and any non-EOS sample carries
   the full top-k including EOS itself.

So a step that comes back empty is re-asked at T=2 up to `--fish-tries` times
and un-tempered. `<END>` becomes an ordinary candidate with its true
probability, jev decides whether the reply is over, and `--min-tokens` holds.
Only a near-certain EOS gets through, and that one is `END:deepseek` in the
trace.

## Stopping is its own question

With no floor at all, nine out of nine `base`-mode replies ran to `--max-tokens`,
and the back half of each was vamping: *lol anyway whatever yeah anyway sup*.
The trace shows why. Deepseek, continuing a transcript, rarely offers EOS; and
when it does, jev — asked *which token would a real person write next?* —
gives a token spelled `<END>` at most 0.39. No real person types `<END>`.
Stopping is not a next-token question.

So it is asked as its own question, riding on the same jev call. Two `noul`s:

```
finished:  Is this message finished -- would the writer send it now rather than keep typing?
rambling:  Has the assistant's message started rambling, repeating itself or padding --
           gone on past the point where it should have been sent?
```

Both are well calibrated. On a reply left to run, `finished` sat at 0.81 after
`Nah.`, dropped to ~0.15 mid-clause, peaked at 0.85 after *…go out to play and
leave ya hanging*, and then its peaks decayed — 0.74, 0.71, 0.64, 0.59 — as the
vamp set in. `rambling` climbed 0.13 → 0.36 → 0.50 (*yeah lol???*) → 0.67 →
0.86 → 0.95 over the same stretch while `finished` never rose. `finished` only
counts at a sentence boundary; `rambling` stops at once and trims back to the
last boundary.

```
--stop-noul 0.7 --ramble-noul 0.6 --min-tokens 6

you › do you ever get bored in here
jev › Nah. Not much. You?                                        END:finished, 7 tokens
you › what's the best thing you've seen today
jev › Honestly... nothing too big so far.                       END:finished, 8 tokens
you › explain why the sky is blue but make it interesting
jev › Because it is actually not. Well... it looks like blue... but... well...
      I guess because light gets scattered and stuff like...    END:rambling, 29 tokens
you › ok real question. are you happy
jev › I'm... uh... well... yeah? Maybe? Why do you ask that... what about you?
                                                                END:turn, 22 tokens
```

`finished`'s absolute level depends on the reply — a slang answer that never
actually explained the sky never rises above 0.46 — which is why `rambling` is
the one that matters for the failure mode, and why a single fixed threshold on
`finished` is not enough.

## Long form: two essays "by Scott Alexander"

The chat results raised the obvious question: is the "human" score a property
of jev's taste, or just of short slangy messages a detector cannot get purchase
on? So: two posts, raw document mode (title, byline, date, the premise in an
epistemic-status line, `I.`), three arms each. `experiments/essay.sh <navier-stoked|mech-interp>
<control|gated|pure>`.

| arm | who chooses | LW: *Navier-Stoked* | ACX: *Mech interp* |
|---|---|---|---|
| control | deepseek greedy | 1250 tokens, coherent, good — **0.993 AI** | 1114 tokens — **0.989 AI** |
| gated | jev at branch points (`--gate top1 --gate-power 0.5 --gate-skip 0.05`) | 683 tokens, coherent, distinct voice — **0.993 AI** | 156 tokens, listicle — **0.979 AI** |
| pure | jev every token | ~60 tokens of argument, then *"Stop. Sorry. Let me back up before I start to explain the thing about the thing again"*, then `Anyway Anyway Anyanyayay` — **0.003 Human** | ~80 tokens, then `In In In Part Part Part` for 400 more — **0.003 Human** |

So, honestly: **at essay length the "human" score was coming from the
breakage, not the voice.** Pure jev holds a register for about a hundred
tokens and then runs out of things to classify, exactly as jevgpt found. The
gated arm is the interesting one — it is coherent for 700 tokens and it does
*not* sound like the control:

> The agents are not conscious. They are not a hive mind. They are a
> bureaucracy. They are a very fast bureaucracy. They are a bureaucracy that
> does not sleep.

versus the control's

> The population is managed by a meta-controller that does not understand the
> mathematics and does not need to. It understands which agents are making
> progress and which are not, and it allocates attention accordingly.

— jev's stacked declaratives against deepseek's subordinate clauses. Pangram
flags both at 0.99. A distinctive voice is not the same thing as an undetected
one, and stacked anaphora is itself a pattern detectors know.

Two things the runs fixed along the way:

- **Header EOS.** In completion mode deepseek-flash is ~100% certain the
  document is over right after a bare section header (`III. The Aftermath`);
  EOS survives fishing at T=2. Supplying the `\n\n` it would have written
  under the header gets it writing again every time (`--no-nudge` to disable).
- **The ramble noul must ignore `--min-tokens`.** Gated by the floor, it
  could not trim the pure-jev collapse at 100 tokens and waited until 500.
  Fixed; on the gated LW post it fired exactly where it should, at
  *"The graph is 1 million nodes. The graph is 5 million nodes."*

Costs: ~$0.04 per 700-token gated essay, ~$0.04 per 1250-token control.
Deepseek caches in 64-token blocks with a ~5 s build and never counts the last
~200 tokens, so a growing one-token-per-second prefix gets ~70% hits at essay
length and none in short chats. Jev has no cache; its state is billed once per
call and extra questions cost ~80 tokens each, which is why all orderings and
both nouls ride on one call.

## The question is the whole steering surface

Three wordings of jev's next-token question, same group-chat prompts, `base`
mode, no persona:

| question | typical reply | failure |
|---|---|---|
| *Next token of the assistant's reply?* | `The capital is Paris` / `The sea is dark.` | ends after one clause: an assistant that has answered stops |
| *Which token would a real person, not a chatbot, write next?* | `Nah man you're not bored you're not you you're…` | performs humanness: `lol anyway wait hold up`; never stops, because no real person types `<END>` |
| *Which token should come next?* | `I've been watching people's conversations. The interesting things are the ones that are unexpected. Like someone saying something surprising and then getting surprised themselves.` | none of the above; all four replies ended on their own |

The plain question won, and so did *no* persona: the same question with a
"laconic strange regular" system prompt gave `hold up let me check my log
files` — the persona being acted out. The voice was already in the sampler.
That configuration is now the default.

## Does averaging five orderings give a clear signal?

Over 3,072 recorded steps with the default five orderings:

| | |
|---|---|
| all five agree on the winner | 54% |
| winner survives dropping any one ordering | 86% |
| a lone random shuffle matches the average | 83% |
| mean winner margin (p1 − p2) | 0.17, median 0.13 |

When the orderings split, the average's margin collapses to ~0.06 — those are
genuine near-ties in jev's judgement, and the order is what breaks them.
Averaging does not manufacture a preference; it makes the tie-break
reproducible instead of an artifact of listing order. In those tied steps the
average sides with deepseek's own favourite only 28% of the time, so the
branch points where jev's voice is exercised are exactly the ones where its
opinion is weakest. Five orderings is about right: one shuffle flips 17% of
choices, and ten would not sharpen a 0.06 margin into a preference.

## Caveats

- **Pushed past where it wanted to stop, jev vamps.** `--repeat-penalty 1.6`
  kills verbatim loops (`yes yes yes`); the two stop nouls kill the
  conversational kind. A hard `--min-tokens` floor is the surest way to cause
  it.
- **Tokens, not words.** Jev is judging ` par` against ` Paris`, which is not
  a question it was trained on. Whitespace merging helps; `--min-p` helps
  more; sub-word fragments remain.
- **Long form is not jev's regime.** Per-token taste holds a paragraph. Past
  that, either deepseek carries the sentences (`--preset essay`) or the text
  dissolves.
- Byte-fragment tokens (a multi-byte character split across tokens) are dropped
  from the candidate set.
- `base` mode ends the turn when deepseek starts the next `User:` line
  (`END:turn`).

## Trace format

`--record` writes JSON with, per step: the candidates with deepseek's and jev's
probabilities and the final score, every ordering that was asked and the
distribution it returned, the order-bias stats (`agree`, `first_wins`, `tv`,
`distinct_winners`), jev's weight, whether jev was asked, and its `finished` /
`rambling` judgements. `python -m jevseek.site <dir> <out.html>` renders a
directory of them.
