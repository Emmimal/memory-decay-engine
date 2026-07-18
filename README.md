memory-decay-engine
====================

A pure-Python, zero-dependency memory decay engine for AI agent context — implements Ebbinghaus forgetting-curve retention with usage-based reinforcement, benchmarked against naive recency-window pruning.

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)

Most long-running agent memory systems evict old context with a sliding window: if something hasn't been touched in N turns, it's gone. This treats a foundational fact stated once on turn 1 and a throwaway debug log from turn 40 as identical — whichever is older loses, no matter how many times either one has actually been used. This library scores retention using the Ebbinghaus forgetting curve instead, where every recall reinforces an item's stability and pushes its eviction horizon out non-linearly.

Read the full write-up on Towards Data Science → *Context Windows Forget What Matters — I Used a 140-Year-Old Psychology Paper to Fix AI Memory* **[add live URL here once published]**

## What It Does

```
Session events (register / recall) → Engine.step() (eviction decision) → RunTrace → Metrics
                                            ↑
                              EbbinghausMemoryEngine  or  RecencyOnlyBaseline
                              (same interface, different eviction policy)
```

Seven files, one comparison:

| Component | Job |
|---|---|
| `session_generator.py` | Seeded synthetic session generator — foundational facts + background noise, spaced-repetition-style recall scheduling |
| `memory_engine.py` | Two interchangeable eviction policies: `EbbinghausMemoryEngine` (usage-reinforced decay) and `RecencyOnlyBaseline` (fixed window) |
| `simulate.py` | Drives either engine through a session, records a full metrics trace |
| `metrics.py` | Foundational Recall Rate (FRR), Survival Time (ST), and Noise Resistance (NR) — precise definitions in the module docstring |
| `sanity_check.py` | Six invariant checks, run before trusting any benchmark number |
| `demo.py` | One narrated session, turn by turn — start here |
| `benchmark.py` | Full statistical suite: headline comparison, noise-resistance sweep, sensitivity sweep, and an honest failure case |

## Installation

```bash
git clone https://github.com/Emmimal/memory-decay-engine.git
cd memory-decay-engine
```

No `pip install` required. Everything runs on the Python standard library only (tested on Python 3.12). `requirements.txt` is included and intentionally empty, so `pip install -r requirements.txt` is a documented no-op rather than an assumption.

## Quick Start

```python
from memory_engine import EbbinghausMemoryEngine, RecencyOnlyBaseline

engine = EbbinghausMemoryEngine(eviction_threshold=0.20, baseline_stability=8.0)

# turn 1: a foundational fact enters memory
engine.register("core_rule", "We only use Python 3.11", current_turn=1)

# turn 5, 10, 20: the fact gets referenced again — each recall
# reinforces its stability and pushes its eviction horizon out
for turn in (5, 10, 20):
    engine.recall("core_rule", current_turn=turn)

# turn 39: still present, because reinforcement outpaced decay
engine.step(current_turn=39)
print(engine.is_present("core_rule"))  # True

# compare against the naive baseline on the exact same touch pattern
baseline = RecencyOnlyBaseline(window_size=15)
baseline.register("core_rule", "We only use Python 3.11", current_turn=1)
for turn in (5, 10, 20):
    baseline.recall("core_rule", current_turn=turn)
baseline.step(current_turn=39)
print(baseline.is_present("core_rule"))  # False — window=15 already elapsed since the last touch
```

## Running the Scripts

Three entry points, run in this order:

| Script | What It Shows |
|---|---|
| `sanity_check.py` | Six invariant checks that must pass before any benchmark number is trusted (determinism, reinforcement direction, FRR bounds, and more) |
| `demo.py` | One concrete 150-turn session, both engines, turn-by-turn narration — no statistics, just what actually happens |
| `benchmark.py` | Headline comparison (N=50 seeds), noise-resistance sweep, sensitivity sweep (15 threshold × window configurations), and the honest failure case where the engine's advantage disappears |

```bash
python3 sanity_check.py
python3 demo.py
python3 benchmark.py
```

## Configuration Reference

```python
EbbinghausMemoryEngine(
    eviction_threshold=0.20,   # retention score below this -> evicted
    baseline_stability=8.0,    # initial stability for a freshly registered item
)

RecencyOnlyBaseline(
    window_size=15,            # turns since last touch before eviction
)

SessionConfig(
    seed=0,
    session_length=100,
    num_foundational=1,
    recalls_per_foundational=3,
    noise_per_turn=1,
    recall_base_offsets=(3, 8, 20, 45, 90, 150),   # spaced-repetition-style intervals
    recall_jitter_range=(0.7, 1.3),                # multiplicative jitter per seed
)
```

Tuning `baseline_stability` against `eviction_threshold` (naive, no-reinforcement eviction turn):

| baseline_stability | Naive eviction turn (threshold=0.20) |
|---|---|
| 2.0 | ~3.2 |
| 4.0 | ~6.4 |
| 6.0 | ~9.7 |
| 8.0 (default) | ~12.9 |

The default (8.0) was chosen specifically to give a freshly registered item enough of a grace period to survive to a plausible first reinforcement — see "Two Bugs That Almost Lied to Me" in the write-up for why this matters.

## Project Structure

```
memory-decay-engine/
├── memory_engine.py      # EbbinghausMemoryEngine + RecencyOnlyBaseline
├── session_generator.py  # Seeded synthetic session generator
├── simulate.py           # Drives an engine through a session
├── metrics.py            # FRR / Survival Time / Noise Resistance definitions
├── sanity_check.py       # 6 invariant checks — run first, always
├── demo.py               # One narrated session, readable walkthrough
├── benchmark.py          # Full statistical suite
├── requirements.txt      # Empty — stdlib only
├── LICENSE
└── README.md
```

## Benchmark Results

From actual runs of `benchmark.py`, reproducible by cloning this repo (N=50 seeds, 150-turn sessions, 3 foundational facts, background noise of 3 items/turn):

| Metric | Ebbinghaus Engine | Recency-Only Baseline |
|---|---|---|
| Terminal Foundational Recall Rate | 1.000 (stdev 0.000) | 0.000 (stdev 0.000) |
| Terminal Survival Rate | 1.000 | 0.000 |
| Foundational facts evicted | 0.00 / 3 per session | 3.00 / 3 per session |
| Token-footprint reduction | 0.911 | 0.898 |

The result held across all 15 tested threshold/window combinations, and was unaffected by a 20x increase in background noise. It is **not** a general "better memory" claim — a dedicated failure-case test shows the advantage collapses to 0.000 vs. 0.000 when a fact is never recalled after registration, meaning reinforcement, not decay itself, is what drives the result. Full methodology, the two bugs that almost invalidated these numbers, and the failure case are in the write-up linked above.

## When to Use This

Worth building for:
- Long-running, multi-turn agent sessions where something important is established early, goes quiet for a long stretch, and needs to still be correct when it resurfaces
- Coding agents working multi-day tasks, support bots handling extended troubleshooting threads, or any pipeline where "memory" needs to mean more than the last few messages

Skip it if:
- Your session is short enough that everything fits in context anyway
- Every fact in your session has roughly equal, constant relevance throughout (a simple window is fine here)

## Known Limitations

- **Item count is a token-count proxy, not the real thing.** Every synthetic item is treated as roughly equal weight for the footprint-reduction metric. A production version should weight footprint by actual token count.
- **Recall is triggered explicitly by the simulation, not inferred.** A real system needs to define what counts as a genuine recall (was this item actually used to produce an answer, not merely present in context) — that detection logic is out of scope here.
- **The defaults are tuned for one session shape.** `eviction_threshold=0.20` and `baseline_stability=8.0` came out of a sensitivity sweep for a specific rhythm (early establishment, quiet middle, late resurfacing). A session with continuous light usage throughout deserves its own sweep.
- **Conflicting memories are not resolved.** If two contradictory facts both stay reinforced, this engine keeps both. Deciding which one is currently true is a different problem than deciding what stays in the working set.

## Related Articles

- *Context Windows Forget What Matters — I Used a 140-Year-Old Psychology Paper to Fix AI Memory* (Towards Data Science) — **https://towardsdatascience.com/author/emmimalp-alexander/**
- *RAG Is Blind to Time — I Built a Temporal Layer to Fix It in Production* (Towards Data Science) — https://towardsdatascience.com/rag-is-blind-to-time-i-built-a-temporal-layer-to-fix-it-in-production/

## License

MIT — **no LICENSE file exists in this project yet; add one before making the repo public if you intend to license it this way.**
