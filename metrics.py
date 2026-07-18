"""
metrics.py

Precise definitions for the three headline metrics used in this project.
Each one is deliberately simple to compute so results are easy to verify
by hand against the raw simulation trace.

1. Foundational Recall Rate (FRR)
   FRR at turn T = (# foundational facts present in the engine at turn T)
                    / (total # foundational facts)
   Binary presence only -- this does not check semantic correctness,
   only whether the fact is still in the working memory pool at all.

2. Survival Time (ST)
   For an item that WAS evicted: ST = turn_evicted - created_turn.
   For an item that was NEVER evicted during the session: it does not
   get an ST value. Instead it counts toward Terminal Survival Rate
   (TSR) = fraction of items still present at the final turn.
   Reporting these two numbers separately avoids the undefined-ceiling
   problem of trying to average in items that never got evicted.

3. Noise Resistance (NR)
   NR is reported as the terminal FRR (FRR at the final turn of the
   session) measured across a sweep of noise_per_turn values, holding
   session_length fixed. This is reported as a curve/table, not
   collapsed into a single fitted slope -- FRR is bounded in [0, 1] and
   there's no guarantee the degradation is linear across the tested
   range, so a linear-regression slope would be a misleading summary.
   If a single number is needed, use "FRR at the highest tested noise
   level" rather than a fitted slope.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunTrace:
    """Everything recorded while simulating one session against one engine."""
    foundational_ids: List[str]
    all_registered_ids: List[str]
    frr_checkpoints: Dict[int, float] = field(default_factory=dict)  # turn -> FRR
    eviction_log: Dict[str, int] = field(default_factory=dict)       # mem_id -> turn evicted
    created_turn: Dict[str, int] = field(default_factory=dict)       # mem_id -> turn created
    final_present: Dict[str, bool] = field(default_factory=dict)     # mem_id -> present at end
    working_set_sizes: List[int] = field(default_factory=list)       # per-turn snapshot


def compute_frr(engine, foundational_ids: List[str]) -> float:
    if not foundational_ids:
        return 1.0
    present = sum(1 for fid in foundational_ids if engine.is_present(fid))
    return present / len(foundational_ids)


def compute_survival_time_stats(trace: RunTrace, ids: List[str]) -> Dict[str, Optional[float]]:
    """Returns mean/min/max survival time for evicted items in `ids`,
    plus terminal survival rate. Items that survived to the end are
    excluded from the ST statistics and instead reflected in TSR."""
    survival_times = []
    for mem_id in ids:
        if mem_id in trace.eviction_log:
            st = trace.eviction_log[mem_id] - trace.created_turn[mem_id]
            survival_times.append(st)

    n_survived_to_end = sum(1 for mem_id in ids if trace.final_present.get(mem_id, False))
    tsr = n_survived_to_end / len(ids) if ids else 0.0

    if not survival_times:
        return {
            "mean_survival_time": None,
            "min_survival_time": None,
            "max_survival_time": None,
            "n_evicted": 0,
            "terminal_survival_rate": tsr,
        }

    return {
        "mean_survival_time": sum(survival_times) / len(survival_times),
        "min_survival_time": min(survival_times),
        "max_survival_time": max(survival_times),
        "n_evicted": len(survival_times),
        "terminal_survival_rate": tsr,
    }


def token_footprint_reduction(mean_working_set: float, total_items_ever_created: int) -> float:
    """Proxy for token savings: fraction of ever-created items NOT
    resident in the working set on average. Using item count as a proxy
    for token count (each synthetic item is treated as roughly equal
    weight -- stated explicitly here since it's a simplification)."""
    if total_items_ever_created == 0:
        return 0.0
    return 1.0 - (mean_working_set / total_items_ever_created)
