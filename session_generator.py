"""
session_generator.py

Generates fully deterministic, seeded synthetic agent sessions to drive
the memory engines. A session is a stream of events:

    (turn, event_type, mem_id, is_foundational)

event_type is either "register" (a new fact enters memory) or
"recall" (an existing fact is queried/used again).

Design of a session:
  - `num_foundational` facts are registered at turn 1. These represent
    things like a core tech-stack constraint stated once at the start
    of a long session.
  - Each foundational fact is recalled at a handful of scheduled turns
    later in the session, simulating the agent actually using that fact
    again during ongoing work.
  - `noise_per_turn` non-foundational items are registered every turn.
    These are never recalled -- they represent one-off intermediate
    chatter (log lines, transient tool output) that should be allowed
    to decay away.

Every value here is driven by a single `random.Random(seed)` instance,
so the exact same seed always produces the exact same session.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Event = Tuple[int, str, str, bool]  # (turn, event_type, mem_id, is_foundational)


@dataclass
class SessionConfig:
    seed: int
    session_length: int = 100
    num_foundational: int = 1
    recalls_per_foundational: int = 3
    noise_per_turn: int = 1
    # base offsets (in turns after registration) for recall scheduling,
    # mimicking classic spaced-repetition intervals: a short first gap,
    # then widening gaps. Actual offsets get a small random jitter per
    # seed so sessions aren't identical in shape. Using purely uniform-
    # random recall placement across the whole session was tried first
    # and produced an unrealistic scenario where the first recall often
    # landed 30+ turns after registration -- long past any plausible
    # short-term grace period, which made every policy fail trivially.
    recall_base_offsets: Tuple[int, ...] = (3, 8, 20, 45, 90, 150)
    # multiplicative jitter range applied to each offset (e.g. 0.7-1.3
    # means an offset can land anywhere from 70% to 130% of its base
    # value). This is deliberately a wider source of randomness than a
    # small additive jitter -- a fixed offset schedule with only a tiny
    # additive jitter produces near-zero seed-to-seed variance, which
    # looks suspicious even when it's an honest result of the schedule's
    # structure. Multiplicative jitter keeps the widening-interval shape
    # while giving every seed a genuinely different session.
    recall_jitter_range: Tuple[float, float] = (0.7, 1.3)


@dataclass
class Session:
    config: SessionConfig
    events: List[Event]
    foundational_ids: List[str]


def generate_session(config: SessionConfig) -> Session:
    rng = random.Random(config.seed)
    events: List[Event] = []

    foundational_ids = [f"foundational_{i}" for i in range(config.num_foundational)]
    for fid in foundational_ids:
        events.append((1, "register", fid, True))

    # schedule recalls using widening, spaced-repetition-style offsets
    # from registration (turn 1), with small per-seed jitter so sessions
    # aren't all identically shaped. Each offset is clamped into
    # [previous_turn + 1, session_length] to guarantee strictly
    # increasing, in-bounds recall turns.
    k = min(config.recalls_per_foundational, len(config.recall_base_offsets))
    lo, hi = config.recall_jitter_range
    for fid in foundational_ids:
        last_turn = 1
        for i in range(k):
            base_offset = config.recall_base_offsets[i]
            factor = rng.uniform(lo, hi)
            candidate_turn = 1 + round(base_offset * factor)
            turn = max(last_turn + 1, min(candidate_turn, config.session_length))
            if turn > config.session_length:
                break
            events.append((turn, "recall", fid, True))
            last_turn = turn

    # noise: registered every turn, never recalled
    noise_counter = 0
    for turn in range(1, config.session_length + 1):
        for _ in range(config.noise_per_turn):
            nid = f"noise_{turn}_{noise_counter}"
            noise_counter += 1
            events.append((turn, "register", nid, False))

    events.sort(key=lambda e: e[0])
    return Session(config=config, events=events, foundational_ids=foundational_ids)


def events_by_turn(session: Session) -> Dict[int, List[Event]]:
    """Group events by turn for fast lookup during simulation."""
    grouped: Dict[int, List[Event]] = {}
    for ev in session.events:
        grouped.setdefault(ev[0], []).append(ev)
    return grouped
