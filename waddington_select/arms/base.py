"""BaseArm — unified interface for all selection strategies (arms)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseArm(ABC):
    """
    All arms implement this interface so they can be swapped into
    SequentialRunner without any other changes.

    select() is called once per round BEFORE oracle.reveal().
    update() is called once per round AFTER oracle.reveal() with the
    newly obtained labels — static arms ignore it, adaptive arms use it.
    """

    def __init__(self, name: str, dataset_name: str, batch_size: int) -> None:
        self.name = name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self._selected: set[str] = set()

    def reset(self) -> None:
        """Reset arm state for a fresh run (called before each run)."""
        self._selected = set()
        self._on_reset()

    def _on_reset(self) -> None:
        """Override for arm-specific reset logic."""

    @abstractmethod
    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        """
        Select the next batch of genes to perturb.

        Args:
            round_idx: 0-based round number.
            revealed:  All gene labels revealed so far {gene: is_hit}.

        Returns:
            List of exactly batch_size gene names (not yet selected).
        """

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        """
        Receive the labels for this round's selection.
        Static arms ignore this; adaptive arms update their model/memory.
        """
        self._selected.update(revealed_new.keys())

    def _unselected(self, pool: list[str]) -> list[str]:
        """Filter out genes already selected in previous rounds."""
        return [g for g in pool if g not in self._selected]
