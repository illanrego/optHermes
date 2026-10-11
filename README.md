# optHermes

Hermes keeps the body. We replace the memory and context subsystem with
OptChat's philosophy, through public seams.

- **Canonical spec:** Victor Taelin's OptChat / UniiChat gist. Our restatement
  lives in `docs/optchat-spec.md`; that file is what briefs point at.
- **Host:** Hermes Agent. The engine ships as a `ContextEngine` plugin, so
  Hermes keeps every capability (tools, MCP, skills, gateway, cron, delegation,
  desktop, TUI) and every upstream fix.
- **Prior art:** `ottobunge/hermes-optchat` and `fourseven/hermes-optchat` exist.
  They are read-only reference for Hermes internals only. We implement the gist
  ourselves; no code is copied from either.

**Status:** working. The engine runs live in an isolated profile; the upstream seam is
open as PR #136135.

## Known behaviour (measured, not assumed)

- **The cache benefit requires a continuous session.** Inside one session, a turn's second
  API call read 16,000 of ~16,363 input tokens from cache. Across separate sessions the
  prefix never matches and cache reads are zero. One endless chat is the intended shape.
- The engine replaces the **history**, not the host's system prompt or tool schemas
  (~16k tokens before the view), so the saving scales with how much history the session
  would otherwise carry.
- **Keep the compaction on the same provider and model as the turns, and lower its reasoning
  effort instead.** One compaction on a reasoning model burned 9,483 output tokens for a single
  512-byte line - but pinning a *different* model would break the shared prompt cache the design
  rests on, since a compaction is supposed to read the turns' cache entry through its
  tools+system prefix. `auxiliary.optchat.reasoning_effort: minimal` is the lever, not a
  different model. With `provider: auto` (the default) the compaction follows whatever model the
  turns use - a local one included.

## Layout

```
CONTRACT.md            frozen interfaces, module boundaries, byte budgets
PLAN.md                the build chain and the agent budget
AGENTS.md              rules for delegates and future sessions
docs/optchat-spec.md   our restatement of the canonical spec
optchat/               the plugin package (empty until slice 2)
tests/                 our tests
```

## The two deliverables

1. **The engine** (this repo): a Hermes `ContextEngine` plugin. Own repo, own
   tests, catalog-eligible once the seams land.
2. **The seams** (a fork of `hermes-agent`, branch `feat/context-engine-cache-seam`):
   small, generic additions to the `ContextEngine` ABC, PR'd upstream. They are
   upgrades, not prerequisites. The engine runs on stock Hermes using an interim
   shim until they land.
