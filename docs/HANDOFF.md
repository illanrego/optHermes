You are resuming the optHermes project. Read these first, in order:

  ~/Documents/coding/opthermes/docs/PLAN-next.md    the plan, revision 2 (the current work)
  ~/Documents/coding/opthermes/docs/FINDINGS.md     the tracker (WP1-WP9, the seams, the upstream asks)
  ~/Documents/coding/opthermes/docs/ANALYSIS.md     the assistant's analysis against the design
  ~/Documents/coding/opthermes/CONTRACT.md          the frozen interfaces, revisions 1-11
  ~/Documents/coding/opthermes/docs/optchat-spec.md the canonical design (Taelin's OptChat)

What this project is: Hermes keeps the body (tools, providers, skills, gateway, delegation); the
memory and context subsystem is Taelin's design - one permanent append-only log, a binary summary
tree, a constant-size view injected in place of the transcript, zoom/date navigation, compactions
shaped like turns so they read the turns' cache entry, and Hermes' own lossy compression never
firing.

Where it stands:

- WP1-WP9 are implemented, verified (209 tests) and merged to main in ~/Documents/coding/opthermes.
- Three host seams are built in the host fork worktree ~/Documents/coding/opthermes-seam on branch
  feat/context-engine-seams (commit cf1743d63a): tools to select_context, ContextSelectionBlocked,
  and on_message. They are NOT merged into any running host, and there are NO PRs - the user's rule
  is that nothing goes upstream until he has tested it on his terms.
- Orders 1, 2 and 4 of the plan are **DONE and merged to main** (Order 1: 07d5fa8 → f548c0d; Order 2:
 6dcfa92; Order 4: 0fa9f67 plus the host gate 81af99c011 in the seam worktree). Main is at HEAD with
 329 tests green, CONTRACT revisions 12-18 in order. The second challenge's D2/D4 verdicts are
 applied (56e857c Order 2b, dbd4633 Order 2c), and Order 3 landed as b27e94c - UID turn boundaries
 with conformance checks and a durable, frozen selected view per logical turn and scope; Order 5's engine side landed as 609b527 (CONTRACT
 revision 19 - compaction roles shaped like a turn, engine-side prefix overlaps measured).
- The fourth seam - the logical-turn boundary (`turn_boundary`) - landed on the seam worktree branch
  `feat/context-engine-seams` at `202e30e78f`.
- **P3b has been redone** (commit 8fa8cb9, CONTRACT revision 15): the rebuild is cut, and a missing view
  state is an explicit unavailable outcome - ViewStateUnavailable, nothing written, the engine steps
  aside for the session. Exact-state recovery of a missing view.json is impossible from the surviving
  artifacts; revision 15 records that as a decision, not an omission.
- **Order 0's decisions are written and challenged twice** - `docs/DECISIONS.md`, revisions 2-3 (the
  second pass's D2/D4 verdicts are applied). Read them before touching orders 3/5:
  - **D5 reversed**: Order 1's P3b reconstructs the view from the tree's saved nodes, which IS
    reconstruction and is forbidden by AGENTS.md and plan revision 2. **P3b is redone** as the
    explicit unavailable-state outcome (commit 8fa8cb9); CONTRACT revision 12's clause is superseded by
    revision 15.
  - D2 needs a host-owned occurrence identity or a durable correlation mechanism; the suffix
    invariant is supported by this assembly but not guaranteed by the interface.
  - D4's count delta is unsound alone (refused positions count; `notes.py` can interleave) - it must
    become a validated transaction with provable range ownership and explicit recovery failure.
  - D3 is one snapshot per logical turn and reading scope; D1 describes capabilities per route.

FIRST, check that state:

  cd ~/Documents/coding/opthermes && git log --oneline -6 && git status --short && git worktree list

What is next, in the plan's order:

- The LIVE RUN: the first pass is DONE and green (2026-10-12). Four turns in one continuous
  session on the seam build; every turn bound a `TurnBoundary`; snapshots froze and retired;
  zoom/date round-tripped; wire numbers: turn->turn 9216 cache-read tokens,
  turn->compaction 7936 across 8 compactions. What
  remains: your own session on your terms (the upstream gate), a compaction->compaction wire
  number, and then the seam-merge/PR decision.
- Live-run recipe: `~/Documents/coding/opthermes-seam/.hermes/bin/hermes -p opthermes-dev -t
  context_engine` (the checkout's own launcher; bare `hermes` runs the main build with no seams).
  Continue the ONE continuous session with `--resume <id> chat -q "..."`; a fresh `-q` session on a
  used engine root is an explicit unavailable state by design (revision 16). Fresh root: rename
  `<profile>/optchat` aside first. The working session: `--resume 20261012_130047_ff611e`.

Rules the user holds me to:

- Commit after verification, never before; one commit per step; push when a work package lands.
- Never trust a delegate's summary: read the diff, run the suite myself, and prove each new test
  fails without the fix.
- No PRs, no upstream pushes. Everything stays local.
- Codex is rate-limited in 5-hour windows: `codex exec -m gpt-6-astra -c model_reasoning_effort="high"
  --approve-for-me -C <repo> -o <out.md> "$(cat <brief>)"`. Use sol for mechanical work, astra for
  the subtle; delegate to Hermes subagents when Codex is out.
- English correction at the start of each reply (this fresh chat resumes it).

The three findings that justify the plan:

1. A compaction cannot read a turn's cache entry: it sends [system][user: view + task] where a turn
   sends [system][assistant: view][user: turn]. The roles differ at the same position, so the
   prefixes diverge before tools or cache marks can matter. **Fixed:** 609b527 (CONTRACT revision 19 -
   the envelope now matches; the engine-side prefix numbers are recorded there, the live-wire
   measurement rides the live run).
2. The turn-boundary fix (WP1) is inert against the real host: rows carry stamped persistence fields,
   the API copy has them stripped, and the engine is handed an unstripped clone - so identity and
   equality both fail and it falls back to the last user row. The durable field is message_uid
   (present in conversation_messages, stripped from the API copy). Fix: P2a.
3. WP9 made the live view contain the current turn, which is then served alongside the same turn
   tail: the turn appears twice and the view is no longer rendered before the message is logged.
   Fix: P3a - freeze the rendered bytes actually served, captured before the turn's rows are
   ingested.
