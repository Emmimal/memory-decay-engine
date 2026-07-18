"""
simulate.py

Drives a single engine (EbbinghausMemoryEngine or RecencyOnlyBaseline)
through a single generated Session, turn by turn, and records a RunTrace
that the metrics module can compute FRR / ST / NR from afterward.

The simulation loop itself contains no policy logic -- it just replays
register/recall events in turn order and calls engine.step() once per
turn. Whatever differs between the two engines' outputs is entirely a
product of their eviction policy, not the simulation harness.
"""

from __future__ import annotations
from typing import List

from metrics import RunTrace, compute_frr
from session_generator import Session, events_by_turn


def run_simulation(engine, session: Session, checkpoint_turns: List[int] = None) -> RunTrace:
    grouped = events_by_turn(session)
    all_ids: List[str] = []
    trace = RunTrace(
        foundational_ids=list(session.foundational_ids),
        all_registered_ids=all_ids,
    )

    if checkpoint_turns is None:
        # default: checkpoint every 10 turns plus the final turn
        checkpoint_turns = list(range(10, session.config.session_length + 1, 10))
        if session.config.session_length not in checkpoint_turns:
            checkpoint_turns.append(session.config.session_length)

    for turn in range(1, session.config.session_length + 1):
        for (t, etype, mem_id, is_foundational) in grouped.get(turn, []):
            if etype == "register":
                engine.register(mem_id, f"content:{mem_id}", turn, is_foundational=is_foundational)
                trace.created_turn[mem_id] = turn
                all_ids.append(mem_id)
            elif etype == "recall":
                engine.recall(mem_id, turn)
            else:
                raise ValueError(f"unknown event type: {etype}")

        engine.step(turn)
        trace.working_set_sizes.append(engine.working_set_size())

        if turn in checkpoint_turns:
            trace.frr_checkpoints[turn] = compute_frr(engine, session.foundational_ids)

    # merge eviction log from the engine (mem_id -> turn evicted)
    trace.eviction_log = dict(engine.eviction_log)

    # record final presence for every item ever created
    for mem_id in all_ids:
        trace.final_present[mem_id] = engine.is_present(mem_id)

    return trace
