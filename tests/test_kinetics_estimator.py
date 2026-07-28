"""
Tests for the kinetics_estimator module.

To run: pytest -v tests/test_kinetics_estimator.py
"""

import pytest

from bees.kinetics_estimator import (
    EstimatedKinetics,
    BaseKineticsEstimator,
    build_estimator,
)


class TestEstimatedKinetics:
    def test_defaults(self):
        k = EstimatedKinetics()
        assert k.km is None
        assert k.kcat is None
        assert k.source == "estimator"

    def test_with_values(self):
        k = EstimatedKinetics(km=0.1, kcat=100.0)
        assert k.km == 0.1
        assert k.kcat == 100.0

    def test_frozen(self):
        k = EstimatedKinetics(km=0.1)
        with pytest.raises(AttributeError):
            k.km = 0.2


class TestBuildEstimator:
    def test_none_returns_none(self):
        assert build_estimator(None) is None

    def test_catpred_returns_estimator(self):
        est = build_estimator("catpred")
        assert est is not None
        assert est.name == "catpred"

    def test_catpred_with_include_sd(self):
        est = build_estimator("catpred", include_sd=True)
        assert est.include_sd is True

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetics_estimator"):
            build_estimator("unknown_backend")


class TestBaseKineticsEstimator:
    def test_estimate_not_implemented(self):
        base = BaseKineticsEstimator()
        with pytest.raises(NotImplementedError):
            base.estimate(
                enzyme_sequence="MKTAY",
                reactant_smiles={"S": "C"},
            )


class TestPersistentCache:
    """Persistent prediction cache (default ON; CatPred is the slow step)."""

    def test_on_by_default_global_path(self, monkeypatch, tmp_path):
        from bees.kinetics_estimator import CatPredEstimator
        monkeypatch.delenv("BEES_CATPRED_CACHE", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        est = CatPredEstimator()
        assert est._cache_path == str(tmp_path / "bees" / "catpred_predictions.pkl")

    def test_disabled_when_off(self, monkeypatch):
        from bees.kinetics_estimator import CatPredEstimator
        for val in ("off", "0", "none", "False"):
            monkeypatch.setenv("BEES_CATPRED_CACHE", val)
            est = CatPredEstimator()
            assert est._cache_path is None, f"{val!r} should disable the cache"
            est._save_persistent_cache()  # no-op, must not raise

    def test_roundtrip(self, tmp_path, monkeypatch):
        from bees.kinetics_estimator import CatPredEstimator
        cache = tmp_path / "catpred.pkl"
        monkeypatch.setenv("BEES_CATPRED_CACHE", str(cache))

        est = CatPredEstimator(include_sd=True)
        assert est._cache_path == str(cache)
        key = ("SEQ", (("S", "C"),), None, True)
        est._memo[key] = EstimatedKinetics(km=0.05, kcat=3.14, source="catpred")
        est._save_persistent_cache()
        assert cache.exists()

        est2 = CatPredEstimator(include_sd=True)
        assert key in est2._memo
        assert est2._memo[key].kcat == 3.14
        assert est2._memo[key].km == 0.05
