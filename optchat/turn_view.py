"""Bounded, durable rendered selection; never reconstruct a lost snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .persistence import atomic_json


class TurnViewUnavailable(RuntimeError):
    """There is no provable pre-turn rendering for this selection."""


class TurnView:
    """One live logical turn, with reading scopes sharing immutable UTF-8 bytes.

    Publish the association before the payload, and both before ingestion. A
    crash between publications is unavailable, not permission to render again.
    Notifications do not carry turn IDs: their pending association is the first
    unconsumed user UID, validated against admission at first selection.
    """

    def __init__(self, root: Path):
        self.state_path = root / "turn-selection.json"
        self.render_path = root / "turn-render.json"

    def state(self) -> dict[str, Any] | None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if self.render_path.exists():
                raise TurnViewUnavailable("snapshot association is missing")
            return None
        except (OSError, ValueError) as exc:
            raise TurnViewUnavailable("snapshot association is unreadable") from exc
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("phase") not in ("pending", "selected", "complete")
                or not isinstance(state.get("session"), str)
                or type(state.get("count")) is not int or state["count"] < 0
                or not isinstance(state.get("scopes"), list)
                or not all(isinstance(scope, str) and scope for scope in state["scopes"])
                or (state.get("uid") is not None and not isinstance(state["uid"], str))
                or (state.get("turn") is not None and not isinstance(state["turn"], str))
                or (state["phase"] == "pending" and
                    (state.get("turn") is not None or state["scopes"]))
                or (state["phase"] == "selected" and
                    (not state.get("turn") or not state.get("uid") or not state["scopes"]))
                or (state["phase"] != "complete" and not isinstance(state.get("digest"), str))):
            raise TurnViewUnavailable("snapshot association is malformed")
        return state

    def render(self, state: dict[str, Any]) -> str:
        try:
            rendered = json.loads(self.render_path.read_text(encoding="utf-8"))
            if (not isinstance(rendered, str)
                    or hashlib.sha256(rendered.encode("utf-8")).hexdigest() != state["digest"]):
                raise ValueError("render digest mismatch")
        except (OSError, ValueError, KeyError) as exc:
            raise TurnViewUnavailable("selected render is missing, unreadable or mismatched") from exc
        return rendered

    def capture(self, rendered: str, *, session: str, uid: str | None, count: int,
                turn: str | None = None, scope: str | None = None) -> str:
        state = {"version": 1, "phase": "selected" if turn is not None else "pending",
                 "session": session, "uid": uid, "turn": turn,
                 "scopes": [scope] if scope is not None else [], "count": count,
                 "digest": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}
        atomic_json(self.state_path, state)
        atomic_json(self.render_path, rendered)
        return rendered

    def select(self, boundary: Any, *, rendered: str, count: int,
               can_capture: bool) -> str:
        state = self.state()
        if state is not None:
            same_uid = (state["session"] == boundary.session_id
                        and state["uid"] == boundary.incoming_uid)
            same_turn = (state["session"] == boundary.session_id
                         and state["turn"] == boundary.turn_id)
            if state["phase"] == "complete" and (same_turn or same_uid):
                raise TurnViewUnavailable("this turn's snapshot has already been retired")
            if state["phase"] == "selected" and same_turn and not same_uid:
                raise TurnViewUnavailable("admitted UID changed within the logical turn")
            if state["phase"] != "complete" and (same_turn or same_uid):
                if state["turn"] is not None and not same_turn:
                    raise TurnViewUnavailable("admitted UID was rebound to another logical turn")
                saved = self.render(state)
                state["phase"], state["turn"] = "selected", boundary.turn_id
                if boundary.reading_scope_id not in state["scopes"]:
                    state["scopes"].append(boundary.reading_scope_id)
                atomic_json(self.state_path, state)
                return saved
            # Validate an existing render even when switching turns: never hide
            # missing recovery evidence by overwriting it with the current view.
            if state["phase"] != "complete":
                self.render(state)
        if not can_capture:
            raise TurnViewUnavailable("no saved render predates the admitted occurrence")
        return self.capture(rendered, session=boundary.session_id,
                            uid=boundary.incoming_uid, count=count,
                            turn=boundary.turn_id, scope=boundary.reading_scope_id)

    def before_ingest(self, rendered: str, *, session: str, uid: str | None,
                      count: int) -> None:
        state = self.state()
        if state is not None and state["phase"] != "complete":
            self.render(state)
            return
        self.capture(rendered, session=session, uid=uid, count=count)

    def complete(self, *, count: int) -> None:
        state = self.state()
        if state is None:
            return
        state.update(phase="complete", scopes=[], count=count)
        state.pop("digest", None)
        atomic_json(self.state_path, state)
        # Retain only the most recent association as a tombstone, no view bytes.
        atomic_json(self.render_path, None)
