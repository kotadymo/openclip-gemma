"""Exact partial ranking parity with the original full stable sort."""

from __future__ import annotations

import numpy as np
import pytest

from openvlclip._ranking import select_top_k


@pytest.mark.parametrize("count", [1, 3, 10, 4096, 5001, 10013])
@pytest.mark.parametrize("top_k", [1, 3, 15, 4096])
def test_exact_partial_rank_matches_full_sort_with_score_ties(
    count: int, top_k: int
) -> None:
    rng = np.random.default_rng(817)
    scores = rng.integers(-5, 6, size=count).astype(np.float32) / 10
    paths = [f"folder/{(count - i) % 123:03d}_{i:06d}.png" for i in range(count)]
    key = lambda pair: (-pair[0], paths[pair[1]].casefold(), paths[pair[1]])
    expected = sorted(((float(v), i) for i, v in enumerate(scores)), key=key)[:top_k]
    found: list[tuple[float, int]] = []
    for start in range(0, count, 4096):
        found = select_top_k(
            scores[start : start + 4096],
            limit=top_k,
            start_row=start,
            sort_key=key,
            current=found,
        )
    assert found == expected


def test_ranking_handles_empty_and_invalid_shapes() -> None:
    key = lambda pair: (-pair[0], pair[1])
    assert select_top_k(np.array([], dtype=np.float32), limit=1, start_row=0, sort_key=key) == []
    with pytest.raises(ValueError):
        select_top_k(np.zeros((2, 3), dtype=np.float32), limit=1, start_row=0, sort_key=key)


def test_full_tie_keeps_secondary_ordering() -> None:
    scores = np.ones(10000, dtype=np.float32)
    key = lambda pair: (-pair[0], -pair[1])
    result = select_top_k(scores, limit=2, start_row=0, sort_key=key)
    assert [row for _, row in result] == [9999, 9998]
