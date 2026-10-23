"""Efficient exact top-k selection shared by image and video search.

Candidates at a score boundary are kept until the stable, caller-defined
secondary ordering is applied; this preserves the legacy ranking exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def select_top_k[K](
    scores: NDArray[np.floating],
    *,
    limit: int,
    start_row: int,
    sort_key: Callable[[tuple[float, int]], K],
    current: list[tuple[float, int]] | None = None,
) -> list[tuple[float, int]]:
    """Return exact top-k with caller-defined ordering and deterministic ties.

    Avoid per-row Python objects for low-scoring rows. Equal cutoff scores are
    all retained before the full tie-break, unlike an unqualified argpartition.
    The caller must validate scores are finite and limit is positive.
    """
    if scores.ndim != 1 or limit < 1 or start_row < 0:
        raise ValueError("scores must be 1-D, limit positive, and start_row nonnegative.")
    best = list(current) if current is not None else []
    if scores.size:
        count = min(limit, scores.size)
        if count < scores.size:
            cutoff = np.partition(scores, scores.size - count)[scores.size - count]
            local_rows = np.flatnonzero(scores >= cutoff)
        else:
            local_rows = np.arange(scores.size)
        best.extend((float(scores[row]), int(row) + start_row) for row in local_rows)
        best.sort(key=sort_key)
        del best[limit:]
    return best
