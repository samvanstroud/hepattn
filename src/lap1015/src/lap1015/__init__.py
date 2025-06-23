from __future__ import annotations

from functools import partial

from ._core import linear_sum_assignment

lap_late = partial(linear_sum_assignment, omp=False, eps=True)
lap_early = partial(linear_sum_assignment, omp=True, eps=True)

__all__ = ["lap_early", "lap_late", "linear_sum_assignment"]
