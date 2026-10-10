#!/usr/bin/env bash
# A short guided tour of the optHermes engine in the isolated profile.
#
# Costs a few short model calls on the profile's own route (opencode-go /
# deepseek-v4-flash). Nothing touches the default profile.
#
#   scripts/tour.sh            run three turns, then print the engine's state
#   scripts/tour.sh --state    print the state only, no model calls
set -uo pipefail

P="$HOME/.hermes/profiles/opthermes-dev"
RUN="$HOME/.local/bin/opthermes-dev"

if [ ! -d "$P" ]; then
  echo "No profile at $P. Create it first:"
  echo "  hermes profile create opthermes-dev --clone"
  echo "  hermes -p opthermes-dev config set context.engine optchat"
  echo "  hermes -p opthermes-dev config set memory.memory_enabled false"
  echo "  hermes -p opthermes-dev config set memory.user_profile_enabled false"
  echo "  hermes -p opthermes-dev config set platforms.whatsapp.enabled false"
  echo "  ln -s ~/Documents/coding/opthermes/optchat $P/plugins/optchat"
  exit 1
fi

if [ "${1:-}" != "--state" ]; then
  echo "== turn 1: teach it something =="
  "$RUN" chat -q "Remember: the project codename is ORCHID-17. Reply with just OK." 2>&1 | tail -2

  echo
  echo "== turn 2: ask for it back. Each turn is a fresh call; the only continuity is the"
  echo "   engine's log and view, so an answer here came from the view or from zoom =="
  "$RUN" chat -q "What codename did I give you? Answer from memory." 2>&1 | tail -2

  echo
  echo "== turn 3: a long message, which needs a model-written summary =="
  LONG=$(python3 -c "print('Deployment notes for the staging cluster. ' * 20)")
  "$RUN" chat -q "$LONG" 2>&1 | tail -2
fi

echo
echo "== the log: every message, verbatim, append-only, one line each =="
python3 - "$P" <<'PY'
import json, sys, glob
files = sorted(glob.glob(sys.argv[1] + "/optchat/main/*.jsonl"))
if not files:
    print("  (empty - the engine has not seen a turn yet)")
for f in files:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        print(f"  {r['id']:>3}  {r['kind']:<6} {r['size']:>5}B  {r['text'][:64]!r}")
PY

echo
echo "== the tree: one line per node. l=0 is a message; a node at l>0 merges 2**l of them =="
python3 - "$P" <<'PY'
import json, sys, glob
files = sorted(glob.glob(sys.argv[1] + "/optchat/tree/*.jsonl"))
if not files:
    print("  (empty - no nodes built yet)")
for f in files:
    for line in open(f, encoding="utf-8"):
        n = json.loads(line)
        print(f"  l={n['l']} i={n['i']:<3} {n['size']:>4}B  {n['text'][:64]!r}")
PY

echo
echo "== the view: what the model sees instead of the transcript (ids as <start>+<count>) =="
cat "$P/optchat/view.json" 2>/dev/null; echo
echo "   (it merges only past 128 KB, so a short tour stays all level 0)"

echo
echo "== usage per call. cache_read is the win; task='optchat' is a compaction =="
python3 - "$P" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1] + "/state.db")
rows = c.execute(
    "select task, api_call_count, input_tokens, output_tokens, cache_read_tokens "
    "from session_model_usage order by last_seen desc limit 8").fetchall()
if not rows:
    print("  (no usage recorded yet)")
for t, calls, i, o, cr in reversed(rows):
    print(f"  task={t or '(turn)':<14} calls={calls}  in={i:<7} out={o:<6} cache_read={cr}")
PY
