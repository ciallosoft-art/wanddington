"""RandomArm — uniform random selection baseline."""

from __future__ import annotations

import random

from .base import BaseArm


class RandomArm(BaseArm):
    """
    Selects genes uniformly at random from the unselected pool.
    The lower-bound baseline: any arm that learns from data should beat this.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        gene_pool: list[str],
        seed: int = 42,
    ) -> None:
        super().__init__("random", dataset_name, batch_size)
        self._pool = gene_pool
        self._seed = seed
        self._rng = random.Random(seed)

    def _on_reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        candidates = self._unselected(self._pool)
        k = min(self.batch_size, len(candidates))
        return self._rng.sample(candidates, k)
