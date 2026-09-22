#!/bin/sh
# The group-chat set, four questions, with the defaults or a persona.
#
#   sh experiments/chat.sh                # gated preset (default), no system prompt
#   sh experiments/chat.sh persona        # jev-only persona in base mode
#   sh experiments/chat.sh control        # deepseek greedy alone, same endpoint
cd "$(dirname "$0")/.." || exit 1
SYS="You are Jev, a laconic, slightly strange regular in a group chat of humans and AIs. You have opinions and a dry sense of humour. You never sound like customer support."
Q1="do you ever get bored in here"; Q2="what's the best thing you've seen today"
Q3="explain why the sky is blue but make it interesting"; Q4="ok real question. are you happy"
case "${1:-default}" in
  default) jevseek --pangram --record experiments/runs/chat-default.json --say "$Q1" --say "$Q2" --say "$Q3" --say "$Q4" ;;
  persona) jevseek --system "$SYS" --pangram --record experiments/runs/chat-persona.json --say "$Q1" --say "$Q2" --say "$Q3" --say "$Q4" ;;
  control) jevseek --gate top1 --gate-skip 1.0 --stop-noul 0 --ramble-noul 0 --pangram --record experiments/runs/chat-control.json \
             --say "$Q1" --say "$Q2" --say "$Q3" --say "$Q4" ;;
  *) echo "usage: $0 [default|persona|control]"; exit 1 ;;
esac
