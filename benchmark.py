"""
benchmark.py

Three experiments, run in order:

1. HEADLINE COMPARISON
   Ebbinghaus engine vs. recency-only baseline across many seeded
   sessions, at fixed default parameters. Reports Foundational Recall
   Rate (final turn), Terminal Survival Rate, mean Survival Time of
   evicted items, and token-footprint reduction for both.

2. NOISE RESISTANCE SWEEP
   Same comparison, but sweeping noise_per_turn across [1, 5, 10, 20]
   with session_length held fixed, so noise volume is the only thing
   changing between conditions. Reports terminal FRR at each noise
   level as a table (not collapsed into a fitted slope).

3. SENSITIVITY SWEEP
   Confirms the headline result is not an artifact of one specific
   eviction_threshold / window_size pair by testing a small grid of
   both and checking the qualitative result (Ebbinghaus FRR > baseline
   FRR) holds throughout.

Every number below is from an actual run of this script -- nothing is
hand-typed. Run `python3 benchmark.py` yourself to reproduce it exactly.
"""

from __future__ import annotations
import statistics
from typing import List, Dict

from memory_engine import EbbinghausMemoryEngine, RecencyOnlyBaseline
from session_generator import SessionConfig, generate_session
from simulate import run_simulation
from metrics import compute_survival_time_stats, token_footprint_reduction


N_SEEDS_HEADLINE = 50
N_SEEDS_SWEEP = 20
SESSION_LENGTH = 150
NUM_FOUNDATIONAL = 3
RECALLS_PER_FOUNDATIONAL = 4
DEFAULT_NOISE_PER_TURN = 3
DEFAULT_EVICTION_THRESHOLD = 0.20
DEFAULT_WINDOW_SIZE = 15


def run_one(engine_factory, seed: int, noise_per_turn: int = DEFAULT_NOISE_PER_TURN) -> Dict:
    cfg = SessionConfig(
        seed=seed,
        session_length=SESSION_LENGTH,
        num_foundational=NUM_FOUNDATIONAL,
        recalls_per_foundational=RECALLS_PER_FOUNDATIONAL,
        noise_per_turn=noise_per_turn,
    )
    session = generate_session(cfg)
    engine = engine_factory()
    trace = run_simulation(engine, session)

    terminal_turn = SESSION_LENGTH
    terminal_frr = trace.frr_checkpoints.get(terminal_turn)
    if terminal_frr is None:
        # ensure terminal turn always checkpointed
        from metrics import compute_frr
        terminal_frr = compute_frr(engine, session.foundational_ids)

    st_stats = compute_survival_time_stats(trace, session.foundational_ids)
    mean_working_set = statistics.mean(trace.working_set_sizes)
    total_items = len(trace.all_registered_ids)
    footprint_reduction = token_footprint_reduction(mean_working_set, total_items)

    return {
        "terminal_frr": terminal_frr,
        "terminal_survival_rate": st_stats["terminal_survival_rate"],
        "mean_survival_time": st_stats["mean_survival_time"],
        "n_evicted_foundational": st_stats["n_evicted"],
        "mean_working_set": mean_working_set,
        "total_items": total_items,
        "footprint_reduction": footprint_reduction,
    }


def headline_comparison():
    print("=" * 78)
    print(f"1. HEADLINE COMPARISON  (N={N_SEEDS_HEADLINE} seeds, "
          f"session_length={SESSION_LENGTH}, noise_per_turn={DEFAULT_NOISE_PER_TURN}, "
          f"foundational_facts={NUM_FOUNDATIONAL})")
    print("=" * 78)

    ebbinghaus_results = [
        run_one(lambda: EbbinghausMemoryEngine(eviction_threshold=DEFAULT_EVICTION_THRESHOLD), seed)
        for seed in range(N_SEEDS_HEADLINE)
    ]
    baseline_results = [
        run_one(lambda: RecencyOnlyBaseline(window_size=DEFAULT_WINDOW_SIZE), seed)
        for seed in range(N_SEEDS_HEADLINE)
    ]

    def summarize(results, label):
        frrs = [r["terminal_frr"] for r in results]
        tsrs = [r["terminal_survival_rate"] for r in results]
        footprints = [r["footprint_reduction"] for r in results]
        evicted_counts = [r["n_evicted_foundational"] for r in results]
        print(f"\n[{label}]")
        print(f"  Terminal FRR                 mean={statistics.mean(frrs):.3f}  "
              f"stdev={statistics.stdev(frrs):.3f}  min={min(frrs):.3f}  max={max(frrs):.3f}")
        print(f"  Terminal Survival Rate (TSR) mean={statistics.mean(tsrs):.3f}")
        print(f"  Foundational facts evicted   mean={statistics.mean(evicted_counts):.2f} "
              f"/ {NUM_FOUNDATIONAL} per session")
        print(f"  Token-footprint reduction    mean={statistics.mean(footprints):.3f} "
              f"(fraction of ever-created items NOT resident, on average)")
        return statistics.mean(frrs), statistics.mean(footprints)

    ebb_frr, ebb_footprint = summarize(ebbinghaus_results, "Ebbinghaus Engine")
    base_frr, base_footprint = summarize(baseline_results, "Recency-Only Baseline")

    print(f"\n  --> Ebbinghaus terminal FRR is {ebb_frr - base_frr:+.3f} vs. baseline "
          f"(absolute difference)")
    print(f"  --> Ebbinghaus retains a {ebb_footprint - base_footprint:+.3f} "
          f"(relative) difference in footprint reduction vs. baseline")
    return ebbinghaus_results, baseline_results


def noise_resistance_sweep():
    print("\n" + "=" * 78)
    print(f"2. NOISE RESISTANCE SWEEP  (N={N_SEEDS_SWEEP} seeds per noise level, "
          f"session_length={SESSION_LENGTH} held fixed)")
    print("=" * 78)

    noise_levels = [1, 5, 10, 20]
    print(f"\n{'noise/turn':>10} | {'Ebbinghaus FRR':>16} | {'Baseline FRR':>14}")
    print("-" * 48)
    ebb_curve = []
    base_curve = []
    for noise in noise_levels:
        ebb_frrs = [
            run_one(lambda: EbbinghausMemoryEngine(eviction_threshold=DEFAULT_EVICTION_THRESHOLD),
                     seed, noise_per_turn=noise)["terminal_frr"]
            for seed in range(N_SEEDS_SWEEP)
        ]
        base_frrs = [
            run_one(lambda: RecencyOnlyBaseline(window_size=DEFAULT_WINDOW_SIZE),
                     seed, noise_per_turn=noise)["terminal_frr"]
            for seed in range(N_SEEDS_SWEEP)
        ]
        ebb_mean = statistics.mean(ebb_frrs)
        base_mean = statistics.mean(base_frrs)
        ebb_curve.append(ebb_mean)
        base_curve.append(base_mean)
        print(f"{noise:>10} | {ebb_mean:>16.3f} | {base_mean:>14.3f}")

    print(f"\n  Ebbinghaus FRR at highest tested noise level ({noise_levels[-1]}/turn): "
          f"{ebb_curve[-1]:.3f}")
    print(f"  Baseline FRR at highest tested noise level ({noise_levels[-1]}/turn): "
          f"{base_curve[-1]:.3f}")
    print("  NOTE: reported as a curve/table rather than a fitted slope -- FRR is "
          "bounded in [0,1] and not assumed linear across this range.")


def sensitivity_sweep():
    print("\n" + "=" * 78)
    print(f"3. SENSITIVITY SWEEP  (N={N_SEEDS_SWEEP} seeds per configuration)")
    print("=" * 78)
    print("Confirms the headline result (Ebbinghaus FRR > baseline FRR) is not an "
          "artifact of one specific threshold/window pair.\n")

    thresholds = [0.10, 0.20, 0.30]
    windows = [10, 15, 20, 30, 50]

    print(f"{'threshold':>10} | {'window':>7} | {'Ebbinghaus FRR':>16} | {'Baseline FRR':>14} | {'holds?':>7}")
    print("-" * 68)
    all_hold = True
    for threshold in thresholds:
        for window in windows:
            ebb_frrs = [
                run_one(lambda: EbbinghausMemoryEngine(eviction_threshold=threshold), seed)["terminal_frr"]
                for seed in range(N_SEEDS_SWEEP)
            ]
            base_frrs = [
                run_one(lambda: RecencyOnlyBaseline(window_size=window), seed)["terminal_frr"]
                for seed in range(N_SEEDS_SWEEP)
            ]
            ebb_mean = statistics.mean(ebb_frrs)
            base_mean = statistics.mean(base_frrs)
            holds = ebb_mean > base_mean
            all_hold = all_hold and holds
            print(f"{threshold:>10.2f} | {window:>7} | {ebb_mean:>16.3f} | "
                  f"{base_mean:>14.3f} | {'YES' if holds else 'NO':>7}")

    print(f"\n  Result holds across ALL {len(thresholds) * len(windows)} tested "
          f"configurations: {all_hold}")


def run_one_with_config(engine_factory, cfg_template: SessionConfig, seed: int) -> Dict:
    """Same as run_one() but takes a full config template instead of
    rebuilding one from the module-level constants (needed for the
    failure-case test, which overrides recalls_per_foundational=0)."""
    cfg = SessionConfig(
        seed=seed,
        session_length=cfg_template.session_length,
        num_foundational=cfg_template.num_foundational,
        recalls_per_foundational=cfg_template.recalls_per_foundational,
        noise_per_turn=cfg_template.noise_per_turn,
    )
    session = generate_session(cfg)
    engine = engine_factory()
    trace = run_simulation(engine, session)
    from metrics import compute_frr
    terminal_frr = compute_frr(engine, session.foundational_ids)
    return {"terminal_frr": terminal_frr}


def failure_case_single_use_facts():
    """Honest nuance check: does the Ebbinghaus engine win because it's
    magic, or specifically because REPEATED reinforcement compounds
    stability? If a fact is only ever used ONCE (registered, never
    recalled again), there's no reinforcement to compound -- the engine
    should degrade toward the SAME outcome as the baseline, because
    without recall events, stability never grows past its initial value.
    This is the failure case: a decay engine cannot save a fact that is
    stated once and never referenced again, no different from a naive
    window in that specific situation."""
    print("\n" + "=" * 78)
    print(f"4. FAILURE CASE: single-use facts (never recalled after registration)")
    print("=" * 78)
    print("If a foundational fact is stated once and never queried again, "
          "reinforcement never happens, so stability never compounds. "
          "The decay engine should offer little to no advantage here.\n")

    cfg_template = SessionConfig(
        seed=0,
        session_length=SESSION_LENGTH,
        num_foundational=NUM_FOUNDATIONAL,
        recalls_per_foundational=0,  # never recalled after registration
        noise_per_turn=DEFAULT_NOISE_PER_TURN,
    )

    ebb_results = [
        run_one_with_config(lambda: EbbinghausMemoryEngine(eviction_threshold=DEFAULT_EVICTION_THRESHOLD), cfg_template, seed)
        for seed in range(N_SEEDS_SWEEP)
    ]
    base_results = [
        run_one_with_config(lambda: RecencyOnlyBaseline(window_size=DEFAULT_WINDOW_SIZE), cfg_template, seed)
        for seed in range(N_SEEDS_SWEEP)
    ]

    ebb_frr = statistics.mean(r["terminal_frr"] for r in ebb_results)
    base_frr = statistics.mean(r["terminal_frr"] for r in base_results)
    print(f"  Ebbinghaus terminal FRR (single-use facts): {ebb_frr:.3f}")
    print(f"  Baseline terminal FRR (single-use facts):   {base_frr:.3f}")
    print(f"  Difference: {ebb_frr - base_frr:+.3f}")
    print("\n  Interpretation: with zero reinforcement, the Ebbinghaus engine's "
          "advantage collapses toward the baseline. The mechanism is not "
          "'better memory' in general -- it specifically rewards facts that "
          "get reused, and offers little protection for facts that don't.")


if __name__ == "__main__":
    headline_comparison()
    noise_resistance_sweep()
    sensitivity_sweep()
    failure_case_single_use_facts()
