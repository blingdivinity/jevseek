# jevseek

DeepSeek proposes the next token. Jev chooses it.

TypeSafe's [jev](https://typesafe.ai) is a *decision* model: it cannot generate
text, it can only answer typed questions about a piece of state — pick one of
these, yes or no, score this. jevseek makes it a sampler. Each step,
`deepseek-flash` returns its top-k next tokens with logprobs; jev is shown the
conversation and asked *"Which token should come next?"*; the token it picks
is appended; repeat. Jev also decides when the reply is finished, and when it
has started rambling.

The result has a voice that is not deepseek's. Same prompt, deepseek alone:

> a place of mystery and wonder. It is a time when the world is quiet and the
> only sound is the gentle lapping of the waves against the shore.

Deepseek's candidates, jev's choices:

> dark. It looks black. It is deep. And cold. It is vast. It stretches far
> away. It is endless. It is mysterious. It is silent.

```
you › what's the best thing you've seen today
jev › I've been watching people's conversations. The interesting things are the
      ones that are unexpected. Like someone saying something surprising and then
      getting surprised themselves.
you › ok real question. are you happy
jev › I'm fine.
```

It is a direct descendant of [jevgpt](https://github.com/bewinxed/jevgpt),
which drove jev as a language model over a 20,000-word dictionary at six calls
a word. jevseek asks it ten candidates instead, at one call a token, about a
second and $0.00005 each.

## Install

```bash
pip install git+https://github.com/blingdivinity/jevseek     # or: uv tool install ...
cp .env.example .env                                     # or export the three keys
```

You need a [DeepSeek](https://platform.deepseek.com) key and an
[OpenRouter](https://openrouter.ai) key (jev is served there as
`typesafe/jev-1.13` on the System One endpoint). A
[Pangram](https://www.pangram.com) key is optional, for `--pangram`.

## Use

```bash
jevseek                                            # REPL; /reset, /quit
jevseek --say "do you ever get bored in here"
jevseek --say "..." -v                             # per-token table: what deepseek offered, what jev chose
jevseek --say "..." --pangram --record run.json    # detector score + full per-step trace
jevseek --preset essay --say-file post.txt         # long form
```

Four presets:

| preset | what it does | when |
|---|---|---|
| `gated` (default) | like `chat`, but jev's weight per token is `(1 − p_top)^0.5` and it is not asked at all when deepseek is ≥99.7% sure; deepseek carries grammar and glue, jev takes the branch points | replies, group chats, characters |
| `chat` | deepseek continues a plain `User:/Assistant:` transcript on the completions endpoint, no system prompt; jev chooses every token and decides when to stop | jev with the most say; the voice at its strongest and least stable |
| `essay` | deepseek continues a document (title, byline, body); jev's weight per token is `(1 − p_top)^0.5` and it is not asked at all when deepseek is ≥99.7% sure; essay-worded stop questions | long form, where jev on every token dissolves after a paragraph |
| `pure` | the original experiment: chat endpoint, "assistant" framing, no floor, no stop questions | reproducing the early findings |

Every preset value is overridable; `jevseek --help` lists the flags with their
defaults. The ones that matter most:

- `--instruction` — jev's question. `next` (default), `assistant`, `human`, or
  any text. This is most of the steering there is.
- `--system` — in `chat`, a persona only jev sees. Off by default; the plain
  question with no persona gave the best replies.
- `--min-p 0.001` — drop candidates below 1/1000 of deepseek's top pick. This
  is what keeps the output grammatical; jev picking 1e-4 tokens is where
  `Thecapital` came from.
- `--orders deepseek,reverse,random:3` — jev's answer depends on the order the
  options are listed in, so it is asked five times in one call and averaged.
- `--stop-noul 0.7` / `--ramble-noul 0.6` — jev's own "is it finished?" and
  "has it started rambling?" judgements, asked on the same call. Stopping is
  not a token; a decision model asked what a person would type next will never
  choose `<END>`. `--ramble-patience 6` (essay) makes the rambling verdict
  hold for six judged steps before it counts: one spike on a repeated phrase
  is not a verdict.
- `--repeat-penalty 1.6` / `--no-repeat-ngram 4` — the per-token penalty
  stops `yes yes yes`; the n-gram block stops jev re-choosing a whole sentence
  (*"The proof is not understandable."* four times, each word only mildly
  repeated).
- `--gate top1 --gate-skip 0.05 --gate-power 0.5` — the gate (on in `gated`
  and `essay`): deepseek carries grammar, jev takes the branch points.
  `--gate none` hands every token to jev.

## What it does and doesn't do

The measurements are in [FINDINGS.md](FINDINGS.md); the recorded runs, with
every candidate and every judgement, are rendered at
[docs/index.html](docs/index.html). The short version:

- The voice is the chooser's. On the same endpoint and prompt, deepseek greedy
  and deepseek at temperature 1.0 both score ~0.99 on Pangram's AI detector;
  jev-chosen replies score ~0.003. Temperature does nothing; the chooser does.
- It holds for a paragraph. Jev on every token runs out of things to classify
  after ~100 tokens and dissolves (`Anyway Anyway Anyanyayay`). The `essay`
  preset stays coherent for 700 tokens with a distinct staccato voice — and
  Pangram flags it at 0.99 anyway. A distinctive voice is not an undetected
  one.
- Jev is whitespace-blind (` capital` vs `capital`), eager to end when framed
  as an assistant, and vamps when pushed past where it wanted to stop. Each has
  a fix; each fix is a flag.
- DeepSeek's API reports `log_softmax(logits / T)` and returns no logprobs at
  all when its sampled token is EOS. jevseek re-asks at T=2 and un-tempers, so
  `<END>` becomes a candidate with its true probability that jev can weigh.

The prompts in `experiments/prompts/` carry a fake "Scott Alexander" byline.
That is an experiment in shaping deepseek's candidate pool with a known voice;
nothing generated here is by him.

## Development

```bash
git clone https://github.com/blingdivinity/jevseek && cd jevseek
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
sh experiments/chat.sh                       # the four group-chat questions, defaults
sh experiments/essay.sh navier-stoked gated  # one arm of the long-form comparison
python -m jevseek.site                       # experiments/runs -> docs/index.html
```

MIT. Jev is TypeSafe's; deepseek-flash is DeepSeek's; the idea of making jev
write is [bewinxed](https://github.com/bewinxed)'s.
