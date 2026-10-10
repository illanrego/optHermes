# OptChat for Hermes (working name)

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

**Status:** planning. Contract frozen, no code yet.

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
