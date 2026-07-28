"""Regression tests for the physics rule layer.

Each rule is tested in isolation against hand-built fake reactions; the goal
is to assert each rule's `apply()` reproduces the input/output pairs that
the old inline logic in bees.thermodynamics and bees.reaction_generator
produced before the refactor.

The tests intentionally use plain dataclasses for `kinetics` and `thermo`,
not the BEES production classes, so a refactor of those classes does not
break the rule semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

import math

from bees.rules import RuleRegistry
from bees.rules.physics_rules import (
    DecarboxylationIrreversible,
    DgrIrreversibility,
    HaldaneReverseKcat,
    HydrophobicChainLengthKm,
    KCAT_REV_MAX,
    ReverseKcatCeiling,
    _DG_PER_CH2_KJMOL,
    _HYDROPHOBIC_REFERENCE_CHAIN,
    _HYDROPHOBIC_TEMPERATURE_K,
    _R_KJ,
    compute_haldane_reverse_kcat,
)
from bees.rules.calibrations import (
    ElongationChainCliff,
    TesALongChainPreference,
    _CLIFF_F_MIN,
    _CLIFF_N_HALF,
    _CLIFF_SHARPNESS,
    _ELONGATION_SYNTHASE_ECS,
    _TESA_PREF_G_MIN,
    _TESA_PREF_N_HALF,
    _TESA_PREF_SHARPNESS,
    _TESA_THIOESTERASE_ECS,
)


@dataclass
class FakeKinetics:
    kcat: Optional[float] = None
    km: Optional[float] = None
    km_per_substrate: Optional[dict] = None
    compound_smiles: Optional[dict] = None


@dataclass
class FakeThermo:
    dgr_prime_kJmol: float = 0.0
    sigma_kJmol: float = 1.0
    keq: float = 1.0
    kcat_rev: Optional[float] = None
    irreversible: bool = False
    source: str = "equilibrator"


@dataclass
class FakeReaction:
    enzyme_label: str = "E1"
    ec_number: Optional[str] = None
    reactant_labels: list = field(default_factory=list)
    product_labels: list = field(default_factory=list)
    stoichiometry: dict = field(default_factory=dict)
    kinetics: Optional[FakeKinetics] = None
    thermo: Optional[FakeThermo] = None


def _rule(cls, **kwargs):
    """Build a rule instance with minimal metadata for tests."""
    from bees.rules import Reference
    return cls(
        name=kwargs.pop("name", cls.__name__.lower()),
        description="test",
        reference=Reference(authors=("Test",), title="t", year="2024"),
        reference_type="theoretical",
        params=kwargs.pop("params", {}),
    )


# ---------------------------------------------------------------------------
# compute_haldane_reverse_kcat — pure math helper
# ---------------------------------------------------------------------------

class TestComputeHaldane:
    def test_basic(self):
        # kcat_rev = kcat_fwd * Km_p / (Keq * Km_s) = 100 * 2 / (10 * 0.5) = 40
        result = compute_haldane_reverse_kcat(100.0, 10.0, [0.5], [2.0])
        assert result == pytest.approx(40.0)

    def test_returns_none_on_nonfinite_keq(self):
        assert compute_haldane_reverse_kcat(100.0, float("inf"), [0.5], [2.0]) is None
        assert compute_haldane_reverse_kcat(100.0, 0.0, [0.5], [2.0]) is None
        assert compute_haldane_reverse_kcat(100.0, -1.0, [0.5], [2.0]) is None

    def test_returns_none_on_invalid_kcat(self):
        assert compute_haldane_reverse_kcat(0.0, 10.0, [0.5], [2.0]) is None
        assert compute_haldane_reverse_kcat(-1.0, 10.0, [0.5], [2.0]) is None

    def test_returns_none_on_missing_km(self):
        assert compute_haldane_reverse_kcat(100.0, 10.0, [None], [2.0]) is None
        assert compute_haldane_reverse_kcat(100.0, 10.0, [0.0], [2.0]) is None
        assert compute_haldane_reverse_kcat(100.0, 10.0, [0.5], [-2.0]) is None

    def test_does_not_apply_ceiling(self):
        # In contrast to bees.thermodynamics.haldane_kcat_rev, this helper
        # returns even very large values; clamping is ReverseKcatCeiling's job.
        result = compute_haldane_reverse_kcat(1e6, 1e-6, [1.0], [1.0])
        assert result == pytest.approx(1e12)


# ---------------------------------------------------------------------------
# DgrIrreversibility
# ---------------------------------------------------------------------------

class TestDgrIrreversibility:
    @staticmethod
    def _rule():
        return _rule(DgrIrreversibility, params={"dgr_kjmol_cutoff": 30.0})

    def test_flag_set_for_strongly_negative_dgr(self):
        rxn = FakeReaction(thermo=FakeThermo(dgr_prime_kJmol=-50.0, keq=1e6))
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is True
        assert rxn.thermo.kcat_rev is None

    def test_flag_set_for_strongly_positive_dgr(self):
        rxn = FakeReaction(thermo=FakeThermo(dgr_prime_kJmol=45.0, keq=1e-6))
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is True

    def test_flag_left_alone_for_moderate_dgr(self):
        rxn = FakeReaction(thermo=FakeThermo(dgr_prime_kJmol=-10.0, keq=1.0))
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is False

    def test_flag_set_for_nonfinite_keq(self):
        rxn = FakeReaction(thermo=FakeThermo(dgr_prime_kJmol=-5.0, keq=float("inf")))
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is True

    def test_flag_set_for_nonpositive_keq(self):
        rxn = FakeReaction(thermo=FakeThermo(dgr_prime_kJmol=-5.0, keq=0.0))
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is True

    def test_does_not_apply_when_thermo_missing(self):
        rule = self._rule()
        rxn = FakeReaction(thermo=None)
        assert rule.applies_to(rxn) is False


# ---------------------------------------------------------------------------
# DecarboxylationIrreversible
# ---------------------------------------------------------------------------

class TestDecarboxylationIrreversible:
    @staticmethod
    def _rule():
        return _rule(DecarboxylationIrreversible)

    def test_flags_irreversible_on_co2_label_product(self):
        # CO2 as a product (positive coeff) → irreversible, kcat_rev cleared.
        rxn = FakeReaction(
            product_labels=["3-oxoacyl-[ACP]", "Carbon dioxide"],
            stoichiometry={"acyl-[ACP]": -1, "malonyl-[ACP]": -1,
                           "3-oxoacyl-[ACP]": 1, "Carbon dioxide": 1},
            kinetics=FakeKinetics(kcat=50.0),
            thermo=FakeThermo(kcat_rev=12.0),
        )
        assert self._rule().applies_to(rxn) is True
        self._rule().apply(rxn)
        assert rxn.thermo.irreversible is True
        assert rxn.thermo.kcat_rev is None

    def test_detects_co2_by_smiles(self):
        # Label isn't "carbon dioxide" but SMILES canonicalizes to O=C=O.
        rxn = FakeReaction(
            product_labels=["P", "co2_species"],
            stoichiometry={"S": -1, "P": 1, "co2_species": 1},
            kinetics=FakeKinetics(
                kcat=50.0,
                km_per_substrate={},
            ),
            thermo=FakeThermo(),
        )
        rxn.kinetics.compound_smiles = {"co2_species": "O=C=O", "P": "CCO"}
        assert self._rule().applies_to(rxn) is True

    def test_does_not_flag_when_co2_is_reactant(self):
        # Carboxylation (CO2 consumed, negative coeff) must NOT be flagged.
        rxn = FakeReaction(
            product_labels=["malonyl-CoA"],
            stoichiometry={"acetyl-CoA": -1, "Carbon dioxide": -1, "malonyl-CoA": 1},
            kinetics=FakeKinetics(kcat=50.0),
            thermo=FakeThermo(),
        )
        assert self._rule().applies_to(rxn) is False

    def test_skips_when_already_irreversible(self):
        rxn = FakeReaction(
            product_labels=["P", "Carbon dioxide"],
            stoichiometry={"S": -1, "P": 1, "Carbon dioxide": 1},
            kinetics=FakeKinetics(kcat=50.0),
            thermo=FakeThermo(irreversible=True),
        )
        assert self._rule().applies_to(rxn) is False

    def test_no_co2_does_not_apply(self):
        rxn = FakeReaction(
            product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=50.0),
            thermo=FakeThermo(),
        )
        assert self._rule().applies_to(rxn) is False


# ---------------------------------------------------------------------------
# HydrophobicChainLengthKm
# ---------------------------------------------------------------------------

class TestHydrophobicChainLengthKm:
    @staticmethod
    def _rule():
        return _rule(HydrophobicChainLengthKm, params={
            "dG_per_CH2_kJmol": _DG_PER_CH2_KJMOL,
            "reference_chain_length": _HYDROPHOBIC_REFERENCE_CHAIN,
            "temperature_K": _HYDROPHOBIC_TEMPERATURE_K,
        })

    @staticmethod
    def _expected_factor(n, n_ref=_HYDROPHOBIC_REFERENCE_CHAIN):
        rt = _R_KJ * _HYDROPHOBIC_TEMPERATURE_K
        return math.exp(_DG_PER_CH2_KJMOL * (n - n_ref) / rt)

    def _tesa_rxn(self, km0):
        # TesA on tetradecanoyl (C14): substrate negative stoich, product + cofactors.
        return FakeReaction(
            enzyme_label="TesA",
            stoichiometry={"tetradecanoyl-[ACP]": -1, "h2o": -1,
                           "tetradecanoate": 1, "holo-[ACP]": 1},
            kinetics=FakeKinetics(kcat=3.0, km_per_substrate={"tetradecanoyl-[ACP]": km0}),
            thermo=FakeThermo(),
        )

    # -- applies_to --------------------------------------------------------
    def test_applies_to_acyl_substrate(self):
        assert self._rule().applies_to(self._tesa_rxn(0.005)) is True

    def test_applies_to_any_enzyme_not_just_tesa(self):
        # Universal scope: fires for an elongation enzyme too.
        rxn = self._tesa_rxn(0.005)
        rxn.enzyme_label = "FabB"
        assert self._rule().applies_to(rxn) is True

    def test_does_not_apply_without_acyl_substrate(self):
        rxn = FakeReaction(
            enzyme_label="SomeEnzyme",
            stoichiometry={"glucose": -1, "atp": -1, "g6p": 1, "adp": 1},
            kinetics=FakeKinetics(kcat=10.0, km_per_substrate={"glucose": 0.2, "atp": 0.5}),
            thermo=FakeThermo(),
        )
        assert self._rule().applies_to(rxn) is False

    def test_does_not_apply_empty_km(self):
        rxn = FakeReaction(
            enzyme_label="TesA",
            stoichiometry={"tetradecanoyl-[ACP]": -1},
            kinetics=FakeKinetics(kcat=3.0, km_per_substrate={}),
            thermo=FakeThermo(),
        )
        assert self._rule().applies_to(rxn) is False

    def test_does_not_apply_below_reference_chain(self):
        # With n_ref=2, an acetyl (C2) substrate is exactly at ref → no length
        # strictly greater is required for applies; n>=n_ref qualifies but the
        # factor is 1.0. A sub-ref species can't exist (acetyl is the shortest),
        # so test a non-acyl-only reaction instead is covered above. Here verify
        # acetyl qualifies (n==n_ref) but produces no change (see apply test).
        rxn = FakeReaction(
            enzyme_label="FabH",
            stoichiometry={"acetyl-[ACP]": -1, "malonyl-[ACP]": -1},
            kinetics=FakeKinetics(kcat=5.0, km_per_substrate={"acetyl-[ACP]": 0.01}),
            thermo=FakeThermo(),
        )
        assert self._rule().applies_to(rxn) is True  # n==n_ref still "applies"

    # -- apply -------------------------------------------------------------
    def test_scaling_is_baseline_relative_and_tighter(self):
        km0 = 0.005
        rxn = self._tesa_rxn(km0)
        self._rule().apply(rxn)
        got = rxn.kinetics.km_per_substrate["tetradecanoyl-[ACP]"]
        assert got == pytest.approx(km0 * self._expected_factor(14))
        assert got < km0  # longer chain → tighter (smaller) Km

    def test_reference_length_unchanged(self):
        # acetyl C2 == n_ref → factor exactly 1.0, Km unchanged.
        km0 = 0.01
        rxn = FakeReaction(
            enzyme_label="FabH",
            stoichiometry={"acetyl-[ACP]": -1},
            kinetics=FakeKinetics(kcat=5.0, km_per_substrate={"acetyl-[ACP]": km0}),
            thermo=FakeThermo(),
        )
        self._rule().apply(rxn)
        assert rxn.kinetics.km_per_substrate["acetyl-[ACP]"] == pytest.approx(km0)

    def test_cofactor_and_product_km_untouched(self):
        km0 = 0.005
        rxn = FakeReaction(
            enzyme_label="TesA",
            stoichiometry={"tetradecanoyl-[ACP]": -1, "h2o": -1,
                           "tetradecanoate": 1},
            kinetics=FakeKinetics(
                kcat=3.0,
                km_per_substrate={"tetradecanoyl-[ACP]": km0, "h2o": 55.0,
                                  "tetradecanoate": 0.7},
            ),
            thermo=FakeThermo(),
        )
        self._rule().apply(rxn)
        km = rxn.kinetics.km_per_substrate
        assert km["tetradecanoyl-[ACP]"] == pytest.approx(km0 * self._expected_factor(14))
        assert km["h2o"] == 55.0          # cofactor untouched
        assert km["tetradecanoate"] == 0.7  # product (positive coeff) untouched

    def test_multiple_acyl_substrates_each_own_factor(self):
        rxn = FakeReaction(
            enzyme_label="FabB",
            stoichiometry={"octanoyl-[ACP]": -1, "malonyl-[ACP]": -1,
                           "3-oxodecanoyl-[ACP]": 1},
            kinetics=FakeKinetics(
                kcat=5.0,
                km_per_substrate={"octanoyl-[ACP]": 0.01, "malonyl-[ACP]": 0.02},
            ),
            thermo=FakeThermo(),
        )
        self._rule().apply(rxn)
        km = rxn.kinetics.km_per_substrate
        assert km["octanoyl-[ACP]"] == pytest.approx(0.01 * self._expected_factor(8))
        # malonyl is C3 (detected), so it also gets a (small) factor for n=3
        assert km["malonyl-[ACP]"] == pytest.approx(0.02 * self._expected_factor(3))

    def test_forward_only_does_not_touch_haldane_snapshot(self):
        rxn = self._tesa_rxn(0.005)
        rxn.kinetics._substrate_kms_for_haldane = {"tetradecanoyl-[ACP]": 0.005}
        self._rule().apply(rxn)
        # The Haldane snapshot is left exactly as set (forward-only).
        assert rxn.kinetics._substrate_kms_for_haldane == {"tetradecanoyl-[ACP]": 0.005}

    def test_idempotent_across_repeated_apply_all(self):
        km0 = 0.005
        rxn = self._tesa_rxn(km0)
        reg = RuleRegistry()
        reg.register(self._rule())
        for _ in range(5):
            reg.apply_all([rxn])
        # Not compounded: factor applied once relative to baseline each pass.
        assert rxn.kinetics.km_per_substrate["tetradecanoyl-[ACP]"] == pytest.approx(
            km0 * self._expected_factor(14)
        )


# ---------------------------------------------------------------------------
# ElongationChainCliff
# ---------------------------------------------------------------------------

class TestElongationChainCliff:
    @staticmethod
    def _rule():
        return _rule(ElongationChainCliff, params={
            "ec_numbers": _ELONGATION_SYNTHASE_ECS,
            "f_min": _CLIFF_F_MIN,
            "n_cliff": _CLIFF_N_HALF,
            "sharpness_k": _CLIFF_SHARPNESS,
        })

    @staticmethod
    def _expected_factor(n):
        return _CLIFF_F_MIN + (1.0 - _CLIFF_F_MIN) / (
            1.0 + math.exp(_CLIFF_SHARPNESS * (n - _CLIFF_N_HALF))
        )

    def _synthase_rxn(self, n_acyl, kcat0=5.0, ec="EC 2.3.1.179", enzyme="FabF"):
        # Condensation: acyl-[ACP](n) + malonyl-[ACP] -> 3-oxo(n+2)-[ACP] + CO2.
        acyl = {12: "dodecanoyl-[ACP]", 14: "tetradecanoyl-[ACP]",
                16: "hexadecanoyl-[ACP]"}[n_acyl]
        return FakeReaction(
            enzyme_label=enzyme, ec_number=ec,
            stoichiometry={acyl: -1, "malonyl-[ACP]": -1, "co2": 1},
            kinetics=FakeKinetics(kcat=kcat0,
                                  km_per_substrate={acyl: 0.04, "malonyl-[ACP]": 0.0082}),
            thermo=FakeThermo(),
        )

    # -- applies_to --------------------------------------------------------
    def test_applies_to_synthase(self):
        assert self._rule().applies_to(self._synthase_rxn(16)) is True
        assert self._rule().applies_to(
            self._synthase_rxn(12, ec="EC 2.3.1.41", enzyme="FabB")) is True

    def test_does_not_apply_to_wrong_ec(self):
        # TesA hydrolysis EC must not trigger the synthase cliff.
        rxn = self._synthase_rxn(16, ec="EC 3.1.2.14", enzyme="TesA")
        assert self._rule().applies_to(rxn) is False

    def test_does_not_apply_without_ec(self):
        rxn = self._synthase_rxn(16, ec=None)
        assert self._rule().applies_to(rxn) is False

    # -- apply -------------------------------------------------------------
    def test_long_chain_floored(self):
        rxn = self._synthase_rxn(16, kcat0=5.0)
        self._rule().apply(rxn)
        assert rxn.kinetics.kcat == pytest.approx(5.0 * self._expected_factor(16))
        assert rxn.kinetics.kcat < 5.0 * 0.05  # C16 strongly cut (~0.02x)

    def test_mid_chain_barely_touched(self):
        rxn = self._synthase_rxn(12, kcat0=5.0)
        self._rule().apply(rxn)
        assert rxn.kinetics.kcat == pytest.approx(5.0 * self._expected_factor(12))
        assert rxn.kinetics.kcat > 5.0 * 0.9  # C12 ~unchanged

    def test_monotonic_decreasing_with_chain(self):
        f = self._expected_factor
        assert f(12) > f(14) > f(16)

    def test_uses_longest_acyl_substrate_not_malonyl(self):
        # Factor must key on the C16 acyl chain, not the C3 malonyl extender.
        rxn = self._synthase_rxn(16, kcat0=5.0)
        self._rule().apply(rxn)
        assert rxn.kinetics.kcat == pytest.approx(5.0 * self._expected_factor(16))

    def test_idempotent_across_repeated_apply_all(self):
        rxn = self._synthase_rxn(16, kcat0=5.0)
        reg = RuleRegistry()
        reg.register(self._rule())
        for _ in range(5):
            reg.apply_all([rxn])
        assert rxn.kinetics.kcat == pytest.approx(5.0 * self._expected_factor(16))


# ---------------------------------------------------------------------------
# TesALongChainPreference
# ---------------------------------------------------------------------------

class TestTesALongChainPreference:
    @staticmethod
    def _rule():
        return _rule(TesALongChainPreference, params={
            "ec_numbers": _TESA_THIOESTERASE_ECS,
            "g_min": _TESA_PREF_G_MIN,
            "n_half": _TESA_PREF_N_HALF,
            "sharpness_k": _TESA_PREF_SHARPNESS,
        })

    @staticmethod
    def _expected_factor(n):
        return _TESA_PREF_G_MIN + (1.0 - _TESA_PREF_G_MIN) / (
            1.0 + math.exp(-_TESA_PREF_SHARPNESS * (n - _TESA_PREF_N_HALF))
        )

    def _tesa_rxn(self, n_acyl, kcat0=10.0, km0=0.01):
        acyl = {4: "butanoyl-[ACP]", 8: "octanoyl-[ACP]",
                12: "dodecanoyl-[ACP]", 16: "hexadecanoyl-[ACP]"}[n_acyl]
        acid = {4: "butanoate", 8: "octanoate",
                12: "dodecanoate", 16: "hexadecanoate"}[n_acyl]
        return FakeReaction(
            enzyme_label="TesA", ec_number="EC 3.1.2.14",
            stoichiometry={acyl: -1, "h2o": -1, acid: 1, "holo-[ACP]": 1},
            kinetics=FakeKinetics(kcat=kcat0, km_per_substrate={acyl: km0}),
            thermo=FakeThermo(),
        )

    # -- applies_to --------------------------------------------------------
    def test_applies_to_tesa(self):
        assert self._rule().applies_to(self._tesa_rxn(16)) is True

    def test_does_not_apply_to_wrong_ec(self):
        rxn = self._tesa_rxn(16)
        rxn.ec_number = "EC 2.3.1.179"  # FabF
        assert self._rule().applies_to(rxn) is False

    # -- apply -------------------------------------------------------------
    def test_short_chain_suppressed_to_floor(self):
        rxn = self._tesa_rxn(4, kcat0=10.0)
        self._rule().apply(rxn)
        assert rxn.kinetics.kcat == pytest.approx(10.0 * self._expected_factor(4))
        assert rxn.kinetics.kcat < 10.0 * 0.02  # C4 strongly suppressed

    def test_long_chain_mostly_kept(self):
        rxn = self._tesa_rxn(16, kcat0=10.0)
        self._rule().apply(rxn)
        assert rxn.kinetics.kcat == pytest.approx(10.0 * self._expected_factor(16))
        assert rxn.kinetics.kcat > 10.0 * 0.5  # C16 mostly retained

    def test_monotonic_increasing_with_chain(self):
        f = self._expected_factor
        assert f(4) < f(8) < f(12) < f(16)

    def test_does_not_touch_km(self):
        # Orthogonal to HydrophobicChainLengthKm: this rule only scales kcat.
        rxn = self._tesa_rxn(16, kcat0=10.0, km0=0.0123)
        self._rule().apply(rxn)
        assert rxn.kinetics.km_per_substrate["hexadecanoyl-[ACP]"] == 0.0123

    def test_idempotent_across_repeated_apply_all(self):
        rxn = self._tesa_rxn(4, kcat0=10.0)
        reg = RuleRegistry()
        reg.register(self._rule())
        for _ in range(5):
            reg.apply_all([rxn])
        assert rxn.kinetics.kcat == pytest.approx(10.0 * self._expected_factor(4))


# ---------------------------------------------------------------------------
# HaldaneReverseKcat
# ---------------------------------------------------------------------------

class TestHaldaneReverseKcat:
    @staticmethod
    def _rule():
        return _rule(HaldaneReverseKcat)

    def test_skips_when_already_irreversible(self):
        rule = self._rule()
        rxn = FakeReaction(
            kinetics=FakeKinetics(kcat=100.0),
            thermo=FakeThermo(irreversible=True),
        )
        assert rule.applies_to(rxn) is False

    def test_skips_when_kcat_missing(self):
        rule = self._rule()
        rxn = FakeReaction(
            kinetics=FakeKinetics(kcat=None),
            thermo=FakeThermo(),
        )
        assert rule.applies_to(rxn) is False

    def test_computes_kcat_rev_when_inputs_valid(self):
        # Single substrate "S" (-1), single product "P" (+1), neither cofactor.
        # kcat_rev = 100 * 2 / (10 * 0.5) = 40
        # The rule uses _substrate_kms_for_haldane (pre-absorption snapshot) and
        # _product_kms_for_haldane (reverse-query only), not merged km_per_substrate.
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=100.0, km_per_substrate={"S": 0.5, "P": 2.0}),
            thermo=FakeThermo(keq=10.0),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"S": 0.5}
        rxn.kinetics._product_kms_for_haldane = {"P": 2.0}
        self._rule().apply(rxn)
        assert rxn.thermo.kcat_rev == pytest.approx(40.0)
        assert rxn.thermo.irreversible is False

    def test_flags_irreversible_when_haldane_returns_none(self):
        # Non-positive Keq → compute_haldane_reverse_kcat returns None → flag set.
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=100.0, km_per_substrate={"S": 0.5, "P": 2.0}),
            thermo=FakeThermo(keq=0.0),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"S": 0.5}
        rxn.kinetics._product_kms_for_haldane = {"P": 2.0}
        self._rule().apply(rxn)
        assert rxn.thermo.kcat_rev is None
        assert rxn.thermo.irreversible is True

    def test_cofactors_filtered_from_haldane(self):
        # H+ and H2O are buffered cofactors and must be excluded from ∏Km_p and ∏Km_s.
        # The substrate snapshot (_substrate_kms_for_haldane) excludes h2o (cofactor).
        # The product dict (_product_kms_for_haldane) excludes h+ (cofactor).
        # Result: kcat_rev = 100 * 2 / (10 * 0.5) = 40.
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1, "h+": 1, "h2o": -1},
            kinetics=FakeKinetics(
                kcat=100.0,
                km_per_substrate={"S": 0.5, "P": 2.0, "h+": 1e-6, "h2o": 55.0},
            ),
            thermo=FakeThermo(keq=10.0),
        )
        # Substrate snapshot: h2o excluded (cofactor), even though stoich < 0.
        rxn.kinetics._substrate_kms_for_haldane = {"S": 0.5}
        # Product Kms from reverse query: h+ excluded by rule's cofactor filter.
        rxn.kinetics._product_kms_for_haldane = {"P": 2.0, "h+": 1e-6}
        self._rule().apply(rxn)
        assert rxn.thermo.kcat_rev == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# ReverseKcatCeiling
# ---------------------------------------------------------------------------

class TestReverseKcatCeiling:
    @staticmethod
    def _rule():
        return _rule(ReverseKcatCeiling, params={"kcat_rev_max": KCAT_REV_MAX})

    def test_does_not_fire_below_cap(self):
        rule = self._rule()
        rxn = FakeReaction(thermo=FakeThermo(kcat_rev=1e3))
        assert rule.applies_to(rxn) is False

    def test_clamps_and_flags_above_cap(self):
        rxn = FakeReaction(thermo=FakeThermo(kcat_rev=1e6))
        self._rule().apply(rxn)
        assert rxn.thermo.kcat_rev is None
        assert rxn.thermo.irreversible is True

    def test_ignores_none_kcat_rev(self):
        rule = self._rule()
        rxn = FakeReaction(thermo=FakeThermo(kcat_rev=None))
        assert rule.applies_to(rxn) is False


# ---------------------------------------------------------------------------
# Registry integration — order matters
# ---------------------------------------------------------------------------

class TestRegistryOrderingAndIdempotence:
    def test_haldane_runs_after_dgr_flag(self):
        # ΔG°' = -50 → DgrIrreversibility flags it; HaldaneReverseKcat then
        # skips because thermo.irreversible is True.
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=100.0, km_per_substrate={"S": 0.5, "P": 2.0}),
            thermo=FakeThermo(dgr_prime_kJmol=-50.0, keq=1e6),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"S": 0.5}
        rxn.kinetics._product_kms_for_haldane = {"P": 2.0}
        reg = RuleRegistry()
        reg.register(_rule(DgrIrreversibility, params={"dgr_kjmol_cutoff": 30.0}))
        reg.register(_rule(HaldaneReverseKcat))
        reg.apply_all([rxn])
        assert rxn.thermo.irreversible is True
        assert rxn.thermo.kcat_rev is None

    def test_decarboxylation_makes_haldane_skip(self):
        # A CO2-releasing condensation with otherwise-valid Haldane inputs.
        # DecarboxylationIrreversible flags it first; Haldane then skips, so
        # kcat_rev stays None rather than the spurious large reverse rate.
        rxn = FakeReaction(
            product_labels=["3-oxoacyl-[ACP]", "Carbon dioxide"],
            stoichiometry={"acyl": -1, "malonyl": -1,
                           "3-oxoacyl-[ACP]": 1, "Carbon dioxide": 1},
            kinetics=FakeKinetics(kcat=100.0, km_per_substrate={"acyl": 0.5}),
            thermo=FakeThermo(dgr_prime_kJmol=-5.0, keq=1e-3),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"acyl": 0.5}
        rxn.kinetics._product_kms_for_haldane = {"3-oxoacyl-[ACP]": 1.0}
        reg = RuleRegistry()
        reg.register(_rule(DgrIrreversibility, params={"dgr_kjmol_cutoff": 30.0}))
        reg.register(_rule(DecarboxylationIrreversible))
        reg.register(_rule(HaldaneReverseKcat))
        reg.apply_all([rxn])
        assert rxn.thermo.irreversible is True
        assert rxn.thermo.kcat_rev is None

    def test_ceiling_runs_after_haldane(self):
        # Huge Haldane kcat_rev (1e6 * 1 / (1e-6 * 1) = 1e12) → ceiling clamps.
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=1e6, km_per_substrate={"S": 1.0, "P": 1.0}),
            thermo=FakeThermo(dgr_prime_kJmol=-5.0, keq=1e-6),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"S": 1.0}
        rxn.kinetics._product_kms_for_haldane = {"P": 1.0}
        reg = RuleRegistry()
        reg.register(_rule(DgrIrreversibility, params={"dgr_kjmol_cutoff": 30.0}))
        reg.register(_rule(HaldaneReverseKcat))
        reg.register(_rule(ReverseKcatCeiling, params={"kcat_rev_max": KCAT_REV_MAX}))
        reg.apply_all([rxn])
        assert rxn.thermo.kcat_rev is None
        assert rxn.thermo.irreversible is True

    def test_idempotent_across_multiple_calls(self):
        rxn = FakeReaction(
            stoichiometry={"S": -1, "P": 1},
            kinetics=FakeKinetics(kcat=100.0, km_per_substrate={"S": 0.5, "P": 2.0}),
            thermo=FakeThermo(dgr_prime_kJmol=-10.0, keq=10.0),
        )
        rxn.kinetics._substrate_kms_for_haldane = {"S": 0.5}
        rxn.kinetics._product_kms_for_haldane = {"P": 2.0}
        reg = RuleRegistry()
        reg.register(_rule(DgrIrreversibility, params={"dgr_kjmol_cutoff": 30.0}))
        reg.register(_rule(HaldaneReverseKcat))
        for _ in range(5):
            reg.apply_all([rxn])
        assert rxn.thermo.kcat_rev == pytest.approx(40.0)
        assert rxn.thermo.irreversible is False


# ---------------------------------------------------------------------------
# Production registry sanity
# ---------------------------------------------------------------------------

class TestProductionRegistry:
    def test_load_physics_rules_registers_all(self):
        from bees.rules import load_physics_rules
        rules = load_physics_rules()
        names = [r.name for r in rules]
        assert "dgr_irreversibility" in names
        assert "decarboxylation_irreversible" in names
        assert "hydrophobic_chain_length_km" in names
        assert "haldane_reverse_kcat" in names
        assert "reverse_kcat_ceiling" in names

    def test_rule_order_flags_and_km_before_haldane_before_ceiling(self):
        from bees.rules import load_physics_rules
        rules = list(load_physics_rules())
        names = [r.name for r in rules]
        i_dgr = names.index("dgr_irreversibility")
        i_decarb = names.index("decarboxylation_irreversible")
        i_hydro = names.index("hydrophobic_chain_length_km")
        i_haldane = names.index("haldane_reverse_kcat")
        i_ceiling = names.index("reverse_kcat_ceiling")
        # Flag rules and the Km-mutating rule all run before Haldane; the
        # Km rule must be before Haldane so Haldane would see corrected Km
        # (forward-only caveat aside); Haldane before Ceiling.
        assert i_dgr < i_haldane
        assert i_decarb < i_haldane
        assert i_hydro < i_haldane
        assert i_haldane < i_ceiling
