## Summary

A context engine that replaces the per-request context via
`ContextEngine.select_context()` cannot control where that request's prompt-cache
breakpoints land. The host strips every `cache_control` mark and re-applies its own
plan afterwards (`_redecorate_prompt_cache_for_provider`, `agent/conversation_loop.py`),
so a replaced context is re-sent uncached on every turn. For an engine whose whole
point is a constant-size, cache-friendly prefix, that is the difference between cheap
and roughly 10x.

Proposal: one generic, additive hook, so the engine that selected the context can also
supply the plan for it.

## The gap

`_apply_context_engine_selection()` (conversation_loop.py) runs `select_context()` and
honours the returned list (RFC #36765). Then
`_redecorate_prompt_cache_for_provider()` calls
`strip_anthropic_cache_control(messages)` and rebuilds a plan with
`build_prompt_cache_plan(...)`, computed from the host's own view of the layout. The
engine's structural intent - where its blocks begin and end - is invisible to that
pass, and on routes where the host's planner adds no marks, the replaced context gets
none at all.

`select_context()` is documented as the only verb that can swap which messages enter a
request. This is the same class of gap one step later: the engine owns the selection,
but not the caching of it.

## Who it affects

Engines that *replace* the context, rather than merely observing turns. Two working
examples in the wild, both ports of the same public design (Victor Taelin's OptChat
spec), so the demand is a class rather than one plugin:

- `fourseven/hermes-optchat` - wraps `_redecorate_prompt_cache_for_provider` to put its
  own marks back. A private-function patch, which also makes the plugin ineligible for
  the plugin catalog ("no patching core code at runtime").
- `ottobunge/hermes-optchat` - carries the same need as an unmerged core branch.

## Proposed seam (minimal, additive)

Add to the `ContextEngine` ABC:

```python
def cache_plan(self, messages, tools, **kwargs):
    """Return a cache plan for THIS request, or ``None`` to keep the host's."""
    return None
```

and call it in `_redecorate_prompt_cache_for_provider()` after the host's plan is
applied, gated by the existing `_engine_overrides_hook(engine, "cache_plan")` helper,
and fail-open on any exception. An engine that does not implement it stays
byte-identical and pays nothing per request - the same contract shape as
`select_context()`.

One thing I would rather the maintainers decide than assume: the hook only fires where
the host's own caching is enabled (`agent._use_prompt_caching`). Should an engine be
able to opt in when the host's caching is off, or is that deliberately a separate seam?

## Status

I have an implementation of exactly this on a branch - 5 files, +147/-2, with tests
covering the ABC default, a non-implementing engine staying byte-identical, an
implementing engine's plan being used, and a raising hook failing open. Happy to open a
PR if the direction is wanted; not opening it yet, per the issue-first guidance in
CONTRIBUTING.md.
