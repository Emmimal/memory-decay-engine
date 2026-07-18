"""
demo.py

A quick, single-scenario illustration -- no statistics, no sweeps, just
one concrete session run through both engines so you can see exactly
what happens turn by turn. For the full statistical benchmark (many
seeds, sensitivity sweeps, the failure case), run benchmark.py instead.

Usage:
    python3 demo.py
"""

from __future__ import annotations

from memory_engine import EbbinghausMemoryEngine, RecencyOnlyBaseline
from session_generator import SessionConfig, generate_session, events_by_turn
from metrics import compute_frr


def run_and_narrate(engine, engine_label: str, session, checkpoints):
    grouped = events_by_turn(session)
    print(f"\n--- {engine_label} ---")
    for turn in range(1, session.config.session_length + 1):
        for (t, etype, mem_id, is_foundational) in grouped.get(turn, []):
            if etype == "register":
                engine.register(mem_id, f"content:{mem_id}", turn, is_foundational=is_foundational)
            elif etype == "recall":
                success = engine.recall(mem_id, turn)
                if is_foundational:
                    if success:
                        print(f"  turn {turn:>3}: RECALL  {mem_id}")
                    else:
                        print(f"  turn {turn:>3}: RECALL ATTEMPTED on {mem_id} -- but it's already gone, no-op")
        evicted = engine.step(turn)
        for mem_id in evicted:
            if mem_id.startswith("foundational"):
                print(f"  turn {turn:>3}: EVICTED {mem_id}  <-- foundational fact lost")
        if turn in checkpoints:
            frr = compute_frr(engine, session.foundational_ids)
            print(f"  turn {turn:>3}: [checkpoint] Foundational Recall Rate = {frr:.2f}")


def main():
    print("=" * 70)
    print("DEMO: one foundational fact, stated once at turn 1, referenced")
    print("a handful of times early on, then not touched again for the")
    print("rest of a 150-turn session -- against a background of ongoing")
    print("noise (3 new unrelated items registered every turn).")
    print("=" * 70)

    cfg = SessionConfig(
        seed=0,
        session_length=150,
        num_foundational=1,
        recalls_per_foundational=4,
        noise_per_turn=3,
    )
    session = generate_session(cfg)
    checkpoints = [30, 60, 90, 120, 150]

    baseline = RecencyOnlyBaseline(window_size=15)
    run_and_narrate(baseline, "Recency-Only Baseline (window=15)", session, checkpoints)

    # regenerate the identical session for a clean second run
    session2 = generate_session(cfg)
    engine = EbbinghausMemoryEngine(eviction_threshold=0.20)
    run_and_narrate(engine, "Ebbinghaus Decay Engine (threshold=0.20)", session2, checkpoints)

    print("\n" + "=" * 70)
    print("Same session, same recall pattern, two different outcomes.")
    print("Run `python3 benchmark.py` for the full statistical picture")
    print("across 50 seeds, a noise-resistance sweep, a sensitivity sweep,")
    print("and an honest failure case where the decay engine's advantage")
    print("disappears (single-use facts with no reinforcement).")
    print("=" * 70)


if __name__ == "__main__":
    main()
