#!/bin/sh
# Long-form comparison on one document prompt: deepseek alone vs jev at the
# branch points vs jev on every token.
#
#   sh experiments/essay.sh navier-stoked control
#   sh experiments/essay.sh navier-stoked gated
#   sh experiments/essay.sh mech-interp pure
#
# Prompts live in experiments/prompts/<name>.txt. Both carry a fake
# "Scott Alexander" byline: the byline shapes deepseek's candidate pool, which
# is the point of the experiment. Nothing here is by him.
cd "$(dirname "$0")/.." || exit 1
NAME="$1"; ARM="$2"; OUT="experiments/runs/$NAME-$ARM.json"
PROMPT="experiments/prompts/$NAME.txt"
[ -f "$PROMPT" ] || { echo "no such prompt: $PROMPT"; exit 1; }
case "$ARM" in
  control) jevseek --preset essay --gate-skip 1.0 --pangram --record "$OUT" --say-file "$PROMPT" ;;
  gated)   jevseek --preset essay --pangram --record "$OUT" --say-file "$PROMPT" ;;
  pure)    jevseek --preset essay --gate none --min-tokens 200 --max-tokens 700 --pangram --record "$OUT" --say-file "$PROMPT" ;;
  *) echo "usage: $0 <prompt-name> control|gated|pure"; exit 1 ;;
esac
