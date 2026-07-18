"""
memory_engine.py

Two memory-management strategies for long-running agent sessions, both
driven entirely by an explicit logical turn counter (never wall-clock time).
This is what makes every result in this project reproducible: run the same
seed twice, get the exact same eviction decisions both times.

1. EbbinghausMemoryEngine
   Retention is a function of elapsed turns AND recall frequency. Every
   time an item is recalled, its "stability" increases non-linearly, which
   flattens its future decay curve (spaced-repetition style reinforcement).
   Items that are never recalled decay on their base stability alone.

2. RecencyOnlyBaseline
   The naive comparison point. Keeps anything touched (registered or
   recalled) within the last N turns. Older items are evicted regardless
   of how many times they were ever recalled. This isolates recency as
   the *only* signal, which is exactly the failure mode this project
   is measuring.

Both engines expose the same interface (register / recall / step /
is_present / working_set_size) so they can be driven by identical
simulated sessions in session_generator.py, with only the eviction
policy differing between them.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemoryItem:
    content: str
    created_turn: int
    last_touched_turn: int
    is_foundational: bool
    stability: float = 2.0
    recall_count: int = 1


class EbbinghausMemoryEngine:
    """
    Retention score at elapsed time t (in turns) with stability S:

        Ret = e^(-t / S)

    Each recall reinforces stability non-linearly (diminishing returns,
    matching spaced-repetition literature):

        S_new = S_old * (1 + ln(1 + recall_count))

    An item is evicted the first time its retention score drops below
    `eviction_threshold`.
    """

    def __init__(self, eviction_threshold: float = 0.20, baseline_stability: float = 8.0):
        if not (0.0 < eviction_threshold < 1.0):
            raise ValueError("eviction_threshold must be in (0, 1)")
        if baseline_stability <= 0:
            raise ValueError("baseline_stability must be positive")
        self.eviction_threshold = eviction_threshold
        self.baseline_stability = baseline_stability
        self.store: Dict[str, MemoryItem] = {}
        # bookkeeping for metrics, not used by the eviction logic itself
        self.eviction_log: Dict[str, int] = {}  # mem_id -> turn evicted

    def register(self, mem_id: str, content: str, current_turn: int, is_foundational: bool = False) -> None:
        self.store[mem_id] = MemoryItem(
            content=content,
            created_turn=current_turn,
            last_touched_turn=current_turn,
            is_foundational=is_foundational,
            stability=self.baseline_stability,
            recall_count=1,
        )

    def recall(self, mem_id: str, current_turn: int) -> bool:
        item = self.store.get(mem_id)
        if item is None:
            return False
        item.recall_count += 1
        item.stability *= (1.0 + math.log(1.0 + item.recall_count))
        item.last_touched_turn = current_turn
        return True

    def _retention_score(self, item: MemoryItem, current_turn: int) -> float:
        elapsed = current_turn - item.last_touched_turn
        if elapsed <= 0:
            return 1.0
        return math.exp(-elapsed / item.stability)

    def step(self, current_turn: int) -> List[str]:
        """Evaluate every stored item against the current turn and evict
        anything whose retention score has dropped below threshold.
        Returns the list of mem_ids evicted this step."""
        evicted = []
        for mem_id, item in list(self.store.items()):
            score = self._retention_score(item, current_turn)
            if score < self.eviction_threshold:
                evicted.append(mem_id)
                self.eviction_log[mem_id] = current_turn
                del self.store[mem_id]
        return evicted

    def is_present(self, mem_id: str) -> bool:
        return mem_id in self.store

    def working_set_size(self) -> int:
        return len(self.store)


class RecencyOnlyBaseline:
    """
    Naive sliding-window policy: an item survives only if it has been
    touched (registered or recalled) within the last `window_size` turns.
    Recall count is completely irrelevant to this policy -- that's the
    point of using it as a baseline.
    """

    def __init__(self, window_size: int = 15):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.store: Dict[str, MemoryItem] = {}
        self.eviction_log: Dict[str, int] = {}

    def register(self, mem_id: str, content: str, current_turn: int, is_foundational: bool = False) -> None:
        self.store[mem_id] = MemoryItem(
            content=content,
            created_turn=current_turn,
            last_touched_turn=current_turn,
            is_foundational=is_foundational,
        )

    def recall(self, mem_id: str, current_turn: int) -> bool:
        item = self.store.get(mem_id)
        if item is None:
            return False
        item.recall_count += 1
        item.last_touched_turn = current_turn
        return True

    def step(self, current_turn: int) -> List[str]:
        evicted = []
        for mem_id, item in list(self.store.items()):
            age = current_turn - item.last_touched_turn
            if age > self.window_size:
                evicted.append(mem_id)
                self.eviction_log[mem_id] = current_turn
                del self.store[mem_id]
        return evicted

    def is_present(self, mem_id: str) -> bool:
        return mem_id in self.store

    def working_set_size(self) -> int:
        return len(self.store)
