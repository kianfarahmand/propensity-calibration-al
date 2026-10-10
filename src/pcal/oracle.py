"""The label oracle.

The campaign may not see a label it has not paid for. That is enforced
structurally rather than by convention: the Oracle holds every label and the
only way a campaign obtains one is reveal(idx), which records the request.

Ground truth over the whole pool is needed by the *evaluator* - the estimand in
PREREGISTRATION.md Section 1 is defined over the unlabeled pool and is exactly
computable on a retrospective benchmark. That access goes through
evaluator_view(), whose name is deliberately awkward, and anything it returns
is written only to the evaluation half of the run record, never to the
campaign half. Validity tests 1 and 2 check that this separation held.
"""
from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    """Raised when campaign code reaches for a label it has not revealed."""


class Oracle:
    def __init__(self, y: np.ndarray, name: str = "pool"):
        self._y = np.array(y, dtype=float, copy=True)
        self._y.flags.writeable = False
        self.name = name
        self._revealed: dict[int, float] = {}
        self._order: list[int] = []
        self.n_calls = 0

    @property
    def n(self) -> int:
        return int(self._y.shape[0])

    @property
    def revealed_indices(self) -> list[int]:
        """Indices in the order they were revealed."""
        return list(self._order)

    @property
    def revealed_values(self) -> np.ndarray:
        return np.asarray([self._revealed[i] for i in self._order], dtype=float)

    def is_revealed(self, idx: int) -> bool:
        return int(idx) in self._revealed

    def reveal(self, idx: int) -> float:
        """Pay for one label. Re-revealing is allowed and is not double-counted."""
        idx = int(idx)
        if idx < 0 or idx >= self.n:
            raise IndexError(f"index {idx} outside pool of {self.n}")
        self.n_calls += 1
        if idx not in self._revealed:
            self._revealed[idx] = float(self._y[idx])
            self._order.append(idx)
        return self._revealed[idx]

    def reveal_many(self, indices) -> np.ndarray:
        return np.asarray([self.reveal(i) for i in indices], dtype=float)

    def unlabeled_indices(self) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if self._order:
            mask[np.asarray(self._order, dtype=int)] = False
        return np.flatnonzero(mask)

    def evaluator_view(self) -> np.ndarray:
        """All labels, for the evaluator only.

        Campaign code calling this is a bug. Nothing derived from it may enter
        the surrogate, the acquisition, the monitor or the controller.
        """
        return np.asarray(self._y)

    def __len__(self) -> int:
        return self.n
