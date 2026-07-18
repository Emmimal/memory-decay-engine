"""
sanity_check.py

Checks run BEFORE any benchmark number is trusted. Each one asserts an
invariant that must hold if the engines are behaving correctly. If any
of these fail, the benchmark results downstream are not to be trusted
until the failure is understood and fixed.
"""

from __future__ import annotations
import math

from memory_engine import EbbinghausMemoryEngine, RecencyOnlyBaseline
from session_generator import SessionConfig, generate_session
from simulate import run_simulation
from metrics import compute_survival_time_stats


def check_determinism_same_seed():
    """Running the same seed twice must produce byte-identical eviction
    logs. This is the direct regression test for the wall-clock-time bug
    (time.time() vs turn counters) -- if this ever fails, non-determinism
    has crept back in somewhere."""
    cfg = SessionConfig(seed=42, session_length=100, num_foundational=2, noise_per_turn=3)
    results = []
    for _ in range(2):
        session = generate_session(cfg)
        engine = EbbinghausMemoryEngine(eviction_threshold=0.20)
        trace = run_simulation(engine, session)
        results.append(sorted(trace.eviction_log.items()))
    assert results[0] == results[1], "FAIL: same seed produced different eviction logs (non-determinism detected)"
    print("PASS: determinism check (same seed -> identical eviction log, 2/2 runs)")


def check_reinforced_item_outlives_baseline_stability():
    """An item recalled several times must have a strictly longer time-
    to-eviction than an otherwise-identical item that is never recalled,
    all else equal. This is the core mechanism claim -- if this fails,
    the reinforcement math is broken."""
    engine_reinforced = EbbinghausMemoryEngine(eviction_threshold=0.20)
    engine_plain = EbbinghausMemoryEngine(eviction_threshold=0.20)

    engine_reinforced.register("item", "x", current_turn=1)
    engine_plain.register("item", "x", current_turn=1)

    # reinforce the first one at turns 5, 10, 15
    for t in (5, 10, 15):
        engine_reinforced.recall("item", current_turn=t)

    evicted_plain = None
    evicted_reinforced = None
    for turn in range(1, 500):
        if evicted_plain is None:
            ev = engine_plain.step(turn)
            if "item" in ev:
                evicted_plain = turn
        if evicted_reinforced is None:
            ev = engine_reinforced.step(turn)
            if "item" in ev:
                evicted_reinforced = turn
        if evicted_plain is not None and evicted_reinforced is not None:
            break

    assert evicted_reinforced > evicted_plain, (
        f"FAIL: reinforced item evicted at turn {evicted_reinforced}, "
        f"plain item evicted at turn {evicted_plain} -- reinforcement should extend survival"
    )
    print(f"PASS: reinforcement check (plain evicted @ turn {evicted_plain}, "
          f"reinforced evicted @ turn {evicted_reinforced})")


def check_recency_baseline_ignores_recall_count():
    """The recency-only baseline must evict based purely on last-touched
    age, regardless of how many times an item was recalled. This
    confirms the baseline is a fair, uncontaminated comparison point."""
    engine = RecencyOnlyBaseline(window_size=10)
    engine.register("heavily_recalled", "x", current_turn=1)
    engine.register("never_recalled", "y", current_turn=1)

    # recall "heavily_recalled" many times, but let both go stale afterward
    for t in range(2, 8):
        engine.recall("heavily_recalled", current_turn=t)

    # neither touched again after turn 7; both should evict at the same
    # elapsed-age threshold relative to their own last_touched_turn
    evicted_at = {}
    for turn in range(8, 30):
        ev = engine.step(turn)
        for mem_id in ev:
            evicted_at[mem_id] = turn

    # heavily_recalled last touched at turn 7, never_recalled at turn 1
    # so heavily_recalled should survive LONGER in wall terms, but the
    # WINDOW length relative to last touch must be identical (10 turns)
    assert evicted_at["never_recalled"] - 1 == engine.window_size + 1 or \
           evicted_at["never_recalled"] - 1 == engine.window_size, \
           "FAIL: baseline window arithmetic off for never_recalled item"
    assert evicted_at["heavily_recalled"] - 7 == engine.window_size + 1 or \
           evicted_at["heavily_recalled"] - 7 == engine.window_size, \
           "FAIL: baseline window arithmetic off for heavily_recalled item"
    print("PASS: recency baseline evicts purely on last-touched age "
          "(recall count does not extend survival beyond the window)")


def check_no_false_evictions_of_actively_recalled_items():
    """An item that is recalled every single turn must never be evicted
    by either engine -- this would indicate an off-by-one or ordering
    bug in the step() logic."""
    for EngineCls, kwargs in [
        (EbbinghausMemoryEngine, {"eviction_threshold": 0.20}),
        (RecencyOnlyBaseline, {"window_size": 5}),
    ]:
        engine = EngineCls(**kwargs)
        engine.register("always_used", "x", current_turn=1)
        for turn in range(2, 200):
            engine.recall("always_used", current_turn=turn)
            engine.step(turn)
        assert engine.is_present("always_used"), (
            f"FAIL: {EngineCls.__name__} evicted an item recalled every single turn"
        )
    print("PASS: continuously-recalled items are never evicted by either engine")


def check_frr_bounds():
    """FRR must always be in [0, 1] and must equal exactly 1.0 at the
    checkpoint immediately after registration (turn 1), before any
    decay has had a chance to occur."""
    cfg = SessionConfig(seed=7, session_length=50, num_foundational=3, noise_per_turn=2)
    session = generate_session(cfg)
    engine = EbbinghausMemoryEngine(eviction_threshold=0.20)
    trace = run_simulation(engine, session, checkpoint_turns=[1, 25, 50])
    for turn, frr in trace.frr_checkpoints.items():
        assert 0.0 <= frr <= 1.0, f"FAIL: FRR out of bounds at turn {turn}: {frr}"
    assert trace.frr_checkpoints[1] == 1.0, (
        f"FAIL: FRR at turn 1 should be 1.0 (no decay possible yet), got {trace.frr_checkpoints[1]}"
    )
    print("PASS: FRR stays within [0, 1] and equals 1.0 immediately after registration")


def check_survival_time_undefined_for_survivors():
    """Items that survive to the end of the session must NOT contribute
    a survival-time value -- only Terminal Survival Rate should reflect
    them. This is the direct regression test for the 'undefined ceiling'
    issue flagged before any code was written."""
    cfg = SessionConfig(seed=3, session_length=20, num_foundational=2,
                         recalls_per_foundational=5, noise_per_turn=1)
    session = generate_session(cfg)
    engine = EbbinghausMemoryEngine(eviction_threshold=0.05)  # very lenient, foundational items should survive
    trace = run_simulation(engine, session)
    stats = compute_survival_time_stats(trace, session.foundational_ids)
    # with a lenient threshold and active recall, foundational facts should
    # survive to the end -> n_evicted should be 0 and TSR should be 1.0
    assert stats["n_evicted"] == 0, f"FAIL: expected 0 evictions of foundational facts, got {stats['n_evicted']}"
    assert stats["terminal_survival_rate"] == 1.0, (
        f"FAIL: expected TSR=1.0 for reinforced foundational facts, got {stats['terminal_survival_rate']}"
    )
    assert stats["mean_survival_time"] is None, "FAIL: mean_survival_time should be None when nothing was evicted"
    print("PASS: survival-time stats correctly separate survivors from evicted items")


if __name__ == "__main__":
    print("Running sanity checks before trusting any benchmark number...\n")
    check_determinism_same_seed()
    check_reinforced_item_outlives_baseline_stability()
    check_recency_baseline_ignores_recall_count()
    check_no_false_evictions_of_actively_recalled_items()
    check_frr_bounds()
    check_survival_time_undefined_for_survivors()
    print("\nAll sanity checks passed.")
