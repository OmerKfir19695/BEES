"""
Concentration projection for conserved pools in ODE state vectors.

When stiff integrators produce small negative concentrations, naively clipping
to zero can inject phantom mass into conserved pools. This module redistributes
deficits onto configured donor species where possible.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ConservedPool:
    """One conserved pool with a donor species that absorbs deficits."""

    name: str
    donor_lc: str
    count: Callable[[str], int]


DEFAULT_CONSERVED_POOLS: Tuple[ConservedPool, ...] = (
    ConservedPool("[ACP]", "holo-[acp]", lambda lc: lc.count("[acp]")),
    ConservedPool(
        "CoA",
        "coenzyme a",
        lambda lc: (
            1
            if (lc == "coenzyme a" or lc.endswith("-coa") or lc.endswith(" coa"))
            else 0
        ),
    ),
    ConservedPool("NADP+/H", "nadp", lambda lc: 1 if lc in ("nadp", "nadph") else 0),
    ConservedPool("NAD+/H", "nad", lambda lc: 1 if lc in ("nad", "nadh") else 0),
)


@dataclass
class _ResolvedPool:
    """Precomputed indices for one pool at fixed species order."""

    pool_items: List[Tuple[int, int]]
    donor_idx: Optional[int]


class Conservator:
    """
    Project concentration vectors to nonnegative values while conserving
    configured pools where possible.

    Species not in any configured pool are simply clipped to zero if negative.
    """

    def __init__(
        self,
        species_labels: Sequence[str],
        pools: Sequence[ConservedPool] = DEFAULT_CONSERVED_POOLS,
    ):
        self._labels_lc = [lab.lower().strip() for lab in species_labels]
        self._n_sp = len(self._labels_lc)
        self._resolved: List[_ResolvedPool] = []
        pool_member_mask = np.zeros(self._n_sp, dtype=bool)

        for pool in pools:
            pool_items = [
                (i, pool.count(lc))
                for i, lc in enumerate(self._labels_lc)
                if pool.count(lc) > 0
            ]
            if not pool_items:
                continue
            for i, _ in pool_items:
                pool_member_mask[i] = True
            donor_idx = next(
                (i for i, lc in enumerate(self._labels_lc) if lc == pool.donor_lc),
                None,
            )
            self._resolved.append(
                _ResolvedPool(pool_items=pool_items, donor_idx=donor_idx)
            )

        self._pool_member_mask = pool_member_mask

    def clip(self, y: np.ndarray) -> np.ndarray:
        """Clip negatives; redistribute pool deficits to donors."""
        y_out = np.array(y, dtype=float)
        if y_out.shape[0] != self._n_sp:
            raise ValueError(
                f"y length {y_out.shape[0]} != species count {self._n_sp}"
            )

        for resolved in self._resolved:
            donor_idx = resolved.donor_idx
            deficit = 0.0
            for i, n in resolved.pool_items:
                if donor_idx is not None and i == donor_idx:
                    continue
                if y_out[i] < 0.0:
                    deficit += abs(y_out[i]) * n
                    y_out[i] = 0.0
            if donor_idx is not None:
                donor_base = max(0.0, float(y_out[donor_idx]))
                y_out[donor_idx] = max(0.0, donor_base - deficit)

        non_pool_neg = (~self._pool_member_mask) & (y_out < 0.0)
        y_out[non_pool_neg] = 0.0
        return y_out

    def clip_2d(self, y: np.ndarray) -> np.ndarray:
        """Apply :meth:`clip` to every column of ``y`` (shape n_species x n_time)."""
        if y.ndim == 1:
            return self.clip(y)

        if y.shape[0] != self._n_sp:
            raise ValueError(
                f"y rows {y.shape[0]} != species count {self._n_sp}"
            )

        y_out = np.maximum(y, 0.0)

        for resolved in self._resolved:
            donor_idx = resolved.donor_idx
            if donor_idx is None:
                continue
            total_deficits = np.zeros(y.shape[1], dtype=np.float64)
            for idx, n_count in resolved.pool_items:
                if idx == donor_idx:
                    continue
                total_deficits += np.where(y[idx] < 0.0, -y[idx] * n_count, 0.0)
            donor_row = np.maximum(0.0, y[donor_idx])
            y_out[donor_idx, :] = np.maximum(0.0, donor_row - total_deficits)

        return y_out
