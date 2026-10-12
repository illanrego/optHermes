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
- Orders 1, 2 and 4 of the plan are **DONE and merged to main** (Order 1: 07d5fa8 → f548c0d;
  Order 2: 6dcfa92; Order 4: 0fa9f67 plus the host gate 81af99c011 in the seam worktree). Their
  worktrees are removed and nothing is in flight. Main is at HEAD with 240 tests green, CONTRACT
  revisions 12-14 in order.
- **P3b must be redone** (see D5's reversal below): Order 1's implementation reconstructs the view
  from the tree's saved nodes, which is forbidden reconstruction. Redo it as exact-state recovery or
  an explicit unavailable-state outcome, and amend CONTRACT revision 12's clause.
- **Order 0's decisions are written and then challenged** - `docs/DECISIONS.md`, revision 2. Read it
  before touching orders 2/3/5, because the challenge reversed one of them:
  - **D5 reversed**: Order 1's P3b reconstructs the view from the tree's saved nodes, which IS
    reconstruction and is forbidden by AGENTS.md and plan revision 2. **P3b must be redone** as
    exact-state recovery or an explicit unavailable-state outcome, and CONTRACT revision 12's clause
    amended.
  - D2 needs a host-owned occurrence identity or a durable correlation mechanism; the suffix
    invariant is supported by this assembly but not guaranteed by the interface.
  - D4's count delta is unsound alone (refused positions count; `notes.py` can interleave) - it must
    become a validated transaction with provable range ownership and explicit recovery failure.
  - D3 is one snapshot per logical turn and reading scope; D1 describes capabilities per route.

FIRST, check that state:

  cd ~/Documents/coding/opthermes && git log --oneline -6 && git status --short && git worktree list

What is next, in the plan's order:

- Order 0's decisions (D1-D5 in the plan). Hold their challenge pass for Codex's reset; the two that
  genuinely need it are D2 (the host's current-turn suffix invariant) and D4 (the recovery protocol).
- Orders 2-5: the durability protocol (P1b/P1d); the turn boundary and the frozen view (P2a/P3a);
  truthful retrieval (P6b, the navigation gate #16, and a real transcript-swap acceptance test); then
  the compaction's request identity (P4a) and the cache measurements (P4b).
- Then a LIVE RUN. The engine has not been exercised inside a real host since WP4, and that run is
  also the gate for merging the host seams.

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
   prefixes diverge before tools or cache marks can matter. Fix: P4a.
2. The turn-boundary fix (WP1) is inert against the real host: rows carry stamped persistence fields,
   the API copy has them stripped, and the engine is handed an unstripped clone - so identity and
   equality both fail and it falls back to the last user row. The durable field is message_uid
   (present in conversation_messages, stripped from the API copy). Fix: P2a.
3. WP9 made the live view contain the current turn, which is then served alongside the same turn
   tail: the turn appears twice and the view is no longer rendered before the message is logged.
   Fix: P3a - freeze the rendered bytes actually served, captured before the turn's rows are
   ingested.
