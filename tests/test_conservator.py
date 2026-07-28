"""Tests for conserved-pool concentration projection."""

import numpy as np
import pytest

from bees.conservator import (
    DEFAULT_CONSERVED_POOLS,
    Conservator,
    ConservedPool,
)


class TestConservator:
    """Regression: donor species must stay nonnegative after pool redistribution."""

    def test_donor_negative_only_resets_donor(self):
        labels = ["holo-[ACP]", "malonyl-[ACP]"]
        y = np.array([-0.1, 0.05])
        out = Conservator(labels).clip(y)
        assert (out >= -1e-15).all()
        assert out[0] == 0.0
        assert out[1] == 0.05

    def test_acyl_negatives_do_not_make_donor_negative(self):
        labels = ["holo-[ACP]", "malonyl-[ACP]", "butanoyl-[ACP]"]
        y = np.array([0.01, -0.5, -0.5])
        out = Conservator(labels).clip(y)
        assert (out >= -1e-15).all()
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] == 0.0

    def test_clip_2d_donor_nonnegative(self):
        labels = ["holo-[ACP]", "malonyl-[ACP]"]
        y = np.array(
            [
                [-0.2, 0.02],
                [-0.1, -0.99],
            ]
        )
        out = Conservator(labels).clip_2d(y)
        assert (out >= -1e-15).all()
        assert out[0, 0] == 0.0
        assert out[0, 1] == 0.0

    def test_non_pool_negative_clips_to_zero(self):
        labels = ["substrate", "product"]
        y = np.array([-0.5, 0.1])
        out = Conservator(labels).clip(y)
        assert out[0] == 0.0
        assert out[1] == 0.1

    def test_unsupported_cofactor_no_redistribution(self):
        labels = ["FAD", "FADH2"]
        y = np.array([0.5, -0.2])
        out = Conservator(labels, pools=()).clip(y)
        assert out[0] == 0.5
        assert out[1] == 0.0

    def test_custom_fad_pool_redistributes(self):
        labels = ["FAD", "FADH2"]
        y = np.array([0.5, -0.2])
        fad_pool = ConservedPool(
            "FAD/FADH2",
            "fad",
            lambda lc: 1 if lc in ("fad", "fadh2") else 0,
        )
        out = Conservator(labels, pools=(fad_pool,)).clip(y)
        assert (out >= -1e-15).all()
        assert out[1] == 0.0
        assert out[0] == pytest.approx(0.3)

    def test_default_pools_match_legacy_acp_behavior(self):
        labels = ["holo-[ACP]", "malonyl-[ACP]"]
        y = np.array([0.01, -0.5])
        out = Conservator(labels, pools=DEFAULT_CONSERVED_POOLS).clip(y)
        assert out[0] == 0.0
        assert out[1] == 0.0
