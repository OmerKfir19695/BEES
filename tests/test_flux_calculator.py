"""
Tests for the flux calculator module.

To run:  pytest -v tests/test_flux_calculator.py
"""


from unittest.mock import MagicMock

from bees.flux_calculator import (
    calculate_characteristic_rate,
    calculate_species_rates,
    compute_mm_rate,
    explicit_product_km,
    has_complete_explicit_product_kms,
    identify_insignificant_species_from_peak_ratios,
    identify_significant_species_at_interrupt,
)


# ---------------------------------------------------------------------------
# Helper to build a mock GeneratedReaction
# ---------------------------------------------------------------------------

def _make_reaction(
    enzyme_label="Hexokinase",
    substrate_label="Glucose",
    reactant_labels=None,
    product_labels=None,
    stoichiometry=None,
    rate_law="Michaelis-Menten",
    km=0.1,
    kcat=100.0,
    vmax=None,
    km_per_substrate=None,
):
    """Create a minimal mock GeneratedReaction for testing."""
    if reactant_labels is None:
        reactant_labels = ["Glucose", "ATP"]
    if product_labels is None:
        product_labels = ["Glucose-6P", "ADP"]
    if stoichiometry is None:
        stoichiometry = {"Glucose": -1, "ATP": -1, "Glucose-6P": 1, "ADP": 1}

    kinetics = MagicMock()
    kinetics.km = km
    kinetics.kcat = kcat
    kinetics.vmax = vmax
    kinetics.km_per_substrate = km_per_substrate

    rxn = MagicMock()
    rxn.enzyme_label = enzyme_label
    rxn.substrate_label = substrate_label
    rxn.reactant_labels = reactant_labels
    rxn.product_labels = product_labels
    rxn.stoichiometry = stoichiometry
    rxn.rate_law = rate_law
    rxn.kinetics = kinetics
    return rxn


# ---------------------------------------------------------------------------
# compute_mm_rate
# ---------------------------------------------------------------------------

class TestComputeMmRate:
    def test_basic_mm_rate(self):
        rxn = _make_reaction(km=0.1, kcat=100.0)
        conc = {"glucose": 1.0, "atp": 2.0}
        enz_conc = {"hexokinase": 0.001}
        rate = compute_mm_rate(rxn, conc, enz_conc)
        # v = kcat * [E] * [S1]/(Km+[S1]) * [S2]/(Km+[S2])
        expected = 100.0 * 0.001 * (1.0 / (0.1 + 1.0)) * (2.0 / (0.1 + 2.0))
        assert abs(rate - expected) < 1e-10

    def test_vmax_fallback(self):
        rxn = _make_reaction(km=0.5, kcat=None, vmax=10.0)
        conc = {"glucose": 1.0, "atp": 1.0}
        enz_conc = {}  # no enzyme conc needed for vmax mode
        rate = compute_mm_rate(rxn, conc, enz_conc)
        expected = 10.0 * (1.0 / (0.5 + 1.0)) * (1.0 / (0.5 + 1.0))
        assert abs(rate - expected) < 1e-10

    def test_no_kinetics(self):
        rxn = _make_reaction()
        rxn.kinetics = None
        rate = compute_mm_rate(rxn, {}, {})
        assert rate == 0.0

    def test_no_rate_law(self):
        rxn = _make_reaction()
        rxn.rate_law = None
        rate = compute_mm_rate(rxn, {}, {})
        assert rate == 0.0

    def test_zero_substrate(self):
        rxn = _make_reaction(km=0.1, kcat=100.0)
        conc = {"glucose": 0.0, "atp": 1.0}
        enz_conc = {"hexokinase": 0.001}
        rate = compute_mm_rate(rxn, conc, enz_conc)
        assert rate == 0.0

    def test_per_substrate_km(self):
        rxn = _make_reaction(
            km=None,
            kcat=50.0,
            km_per_substrate={"Glucose": 0.2, "ATP": 0.5},
        )
        conc = {"glucose": 1.0, "atp": 2.0}
        enz_conc = {"hexokinase": 0.01}
        rate = compute_mm_rate(rxn, conc, enz_conc)
        expected = 50.0 * 0.01 * (1.0 / (0.2 + 1.0)) * (2.0 / (0.5 + 2.0))
        assert abs(rate - expected) < 1e-10

    def test_stoichiometric_exponents(self):
        # 2 A + B → P : νA = 2, νB = 1.
        rxn = _make_reaction(
            reactant_labels=["A", "B"],
            product_labels=["P"],
            stoichiometry={"A": -2, "B": -1, "P": 1},
            km=None,
            kcat=10.0,
            km_per_substrate={"A": 0.5, "B": 0.5},
        )
        conc = {"a": 1.0, "b": 1.0}
        enz = {"hexokinase": 1.0}
        v = compute_mm_rate(rxn, conc, enz)
        # Vmax = kcat·[E] = 10
        # sat = (1/(0.5+1))**2 · (1/(0.5+1))**1 = (2/3)**3 = 8/27
        expected = 10.0 * (2.0 / 3.0) ** 3
        assert abs(v - expected) < 1e-12

    def test_negative_concentration_guarded(self):
        rxn = _make_reaction(km=0.1, kcat=100.0)
        conc = {"glucose": -0.5, "atp": 1.0}
        enz_conc = {"hexokinase": 0.001}
        rate = compute_mm_rate(rxn, conc, enz_conc)
        assert rate == 0.0

    def test_product_inhibition_zero_product_matches_legacy(self):
        # [P]=0 → Liebermeister collapses to (S/(Km+S)) form exactly.
        rxn = _make_reaction(
            reactant_labels=["S"], product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=None, kcat=10.0,
            km_per_substrate={"S": 1.0, "P": 2.0},
        )
        conc = {"s": 5.0, "p": 0.0}
        enz = {"hexokinase": 1.0}
        v = compute_mm_rate(rxn, conc, enz)
        expected_legacy = 10.0 * (5.0 / (1.0 + 5.0))
        assert abs(v - expected_legacy) < 1e-12

    def test_product_inhibition_reduces_rate(self):
        rxn = _make_reaction(
            reactant_labels=["S"], product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=None, kcat=10.0,
            km_per_substrate={"S": 1.0, "P": 2.0},
        )
        enz = {"hexokinase": 1.0}
        v_no_p  = compute_mm_rate(rxn, {"s": 5.0, "p": 0.0},   enz)
        v_high_p = compute_mm_rate(rxn, {"s": 5.0, "p": 100.0}, enz)
        assert v_high_p < v_no_p
        assert v_high_p > 0.0  # forward-only, never zero for positive S

    def test_product_inhibition_falls_back_when_product_km_missing(self):
        # No explicit Km for P → fall back to legacy, products ignored.
        rxn = _make_reaction(
            reactant_labels=["S"], product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=None, kcat=10.0,
            km_per_substrate={"S": 1.0},  # P absent
        )
        enz = {"hexokinase": 1.0}
        v_no_p   = compute_mm_rate(rxn, {"s": 5.0, "p": 0.0},   enz)
        v_high_p = compute_mm_rate(rxn, {"s": 5.0, "p": 100.0}, enz)
        assert v_no_p == v_high_p  # P ignored

    def test_product_inhibition_preserves_cofactor_skip(self):
        # H+ not in km_per_substrate → skipped as saturated.
        # Product Km present → product inhibition still active.
        rxn = _make_reaction(
            reactant_labels=["S", "H+"], product_labels=["P"],
            stoichiometry={"S": -1, "H+": -1, "P": 1},
            km=None, kcat=10.0,
            km_per_substrate={"S": 1.0, "P": 2.0},  # no H+ Km
        )
        enz = {"hexokinase": 1.0}
        v = compute_mm_rate(rxn, {"s": 5.0, "h+": 0.0, "p": 0.0}, enz)
        expected = 10.0 * (5.0 / (1.0 + 5.0))
        assert abs(v - expected) < 1e-12

    def test_buffered_cofactor_product_does_not_gate_completeness(self):
        # H2O product without Km must NOT force legacy fallback when the
        # organic product has an explicit Km (FabA/Z dehydrations).
        rxn = _make_reaction(
            reactant_labels=["OHACP"],
            product_labels=["enoylACP", "H2O"],
            stoichiometry={"OHACP": -1, "enoylACP": 1, "H2O": 1},
            km=None, kcat=10.0,
            km_per_substrate={"OHACP": 0.01, "enoylACP": 0.05},
        )
        assert has_complete_explicit_product_kms(rxn)
        enz = {"hexokinase": 1.0}
        v_low = compute_mm_rate(rxn, {"ohacp": 0.1, "enoylacp": 0.0, "h2o": 0.0}, enz)
        v_high_h2o = compute_mm_rate(rxn, {"ohacp": 0.1, "enoylacp": 0.0, "h2o": 55.0}, enz)
        v_high_enoyl = compute_mm_rate(rxn, {"ohacp": 0.1, "enoylacp": 5.0, "h2o": 0.0}, enz)
        assert abs(v_low - v_high_h2o) < 1e-12  # H2O ignored in denom
        assert v_high_enoyl < v_low            # organic product inhibits

    def test_coa_product_still_required(self):
        # CoA is regulatory — missing Km must keep the reaction incomplete.
        rxn = _make_reaction(
            reactant_labels=["holo", "Malonyl-CoA"],
            product_labels=["malonylACP", "Coenzyme A"],
            stoichiometry={"holo": -1, "Malonyl-CoA": -1, "malonylACP": 1, "Coenzyme A": 1},
            km=None, kcat=10.0,
            km_per_substrate={"holo": 0.04, "Malonyl-CoA": 0.03, "malonylACP": 0.04},
        )
        assert not has_complete_explicit_product_kms(rxn)


# ---------------------------------------------------------------------------
# calculate_species_rates
# ---------------------------------------------------------------------------

class TestCalculateSpeciesRates:
    def test_single_reaction(self):
        rxn = _make_reaction(km=0.1, kcat=100.0)
        conc = {"glucose": 1.0, "atp": 2.0}
        enz_conc = {"hexokinase": 0.001}
        rates = calculate_species_rates([rxn], conc, enz_conc)
        v = compute_mm_rate(rxn, conc, enz_conc)
        assert abs(rates.get("glucose", 0) - (-v)) < 1e-10
        assert abs(rates.get("atp", 0) - (-v)) < 1e-10
        assert abs(rates.get("glucose-6p", 0) - v) < 1e-10
        assert abs(rates.get("adp", 0) - v) < 1e-10


# ---------------------------------------------------------------------------
# calculate_characteristic_rate
# ---------------------------------------------------------------------------

class TestCharacteristicRate:
    def test_basic(self):
        rates = {"a": 3.0, "b": 4.0}
        r_char = calculate_characteristic_rate(rates)
        assert abs(r_char - 5.0) < 1e-10  # sqrt(9 + 16)

    def test_empty(self):
        assert calculate_characteristic_rate({}) == 0.0


# ---------------------------------------------------------------------------
# identify_significant_species_at_interrupt
# ---------------------------------------------------------------------------


class TestIdentifySignificantAtInterrupt:
    def test_normal_promotion(self):
        """Species exceeding tol are returned, sorted by rr desc."""
        edge = {"a": 1.0, "b": 0.5, "c": 0.001}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=1.0, tol_move_to_core=0.1
        )
        labels = [sf.label for sf in result]
        assert labels == ["a", "b"]
        assert result[0].normalized_rate > result[1].normalized_rate

    def test_max_objects_truncation(self):
        """Only top max_objects candidates are returned."""
        edge = {"a": 1.0, "b": 0.8, "c": 0.6}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=1.0, tol_move_to_core=0.1, max_objects=2
        )
        assert len(result) == 2
        assert result[0].label == "a"

    def test_flat_core_abs_flux_floor(self):
        """When char_rate == 0, use abs_flux_floor instead of ratio."""
        edge = {"big": 0.5, "tiny": 1e-15, "medium": 0.01}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=0.0, tol_move_to_core=0.1,
            abs_flux_floor=1e-3,
        )
        labels = [sf.label for sf in result]
        assert "big" in labels
        assert "medium" in labels
        assert "tiny" not in labels
        assert result[0].normalized_rate == float("inf")

    def test_flat_core_truncation(self):
        """In flat-core mode, max_objects still caps output."""
        edge = {"a": 1.0, "b": 0.5, "c": 0.3}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=0.0, tol_move_to_core=0.1,
            abs_flux_floor=1e-12, max_objects=1,
        )
        assert len(result) == 1
        assert result[0].label == "a"

    def test_empty_edge(self):
        result = identify_significant_species_at_interrupt(
            {}, char_rate=1.0, tol_move_to_core=0.1
        )
        assert result == []

    def test_none_above_threshold(self):
        edge = {"a": 0.001}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=1.0, tol_move_to_core=0.1
        )
        assert result == []

    def test_flat_core_all_below_floor(self):
        """When char_rate == 0 and all |rates| < floor, nothing promoted."""
        edge = {"a": 1e-20, "b": 1e-18}
        result = identify_significant_species_at_interrupt(
            edge, char_rate=0.0, tol_move_to_core=0.1,
            abs_flux_floor=1e-12,
        )
        assert result == []


# ---------------------------------------------------------------------------
# identify_insignificant_species_from_peak_ratios
# ---------------------------------------------------------------------------


class TestIdentifyInsignificantFromPeakRatios:
    def test_prune_low_peak_ratio(self):
        max_rr = {"a": 0.5, "b": 0.001}
        out = identify_insignificant_species_from_peak_ratios(
            max_rr, max_char_rate=1.0, tol_keep_in_edge=0.01
        )
        assert "b" in out
        assert "a" not in out

    def test_ineligible_skipped(self):
        max_rr = {"a": 0.001}
        out = identify_insignificant_species_from_peak_ratios(
            max_rr,
            max_char_rate=1.0,
            tol_keep_in_edge=0.01,
            ineligible_for_prune={"a"},
        )
        assert out == set()

    def test_zero_max_char(self):
        assert identify_insignificant_species_from_peak_ratios(
            {"a": 0.5}, 0.0, 0.01
        ) == set()


# ---------------------------------------------------------------------------
# Reversible Michaelis-Menten (Haldane-derived)
# ---------------------------------------------------------------------------

from bees.flux_calculator import compute_reversible_mm_rate
from bees.thermodynamics import ThermoData


def _make_reversible_reaction(
    keq=10.0,
    kcat_fwd=100.0,
    km_s=1.0,
    km_p=2.0,
    template_reversible=True,
    irreversible_thermo=False,
):
    """Build a single-substrate / single-product reversible reaction.

    Stoichiometry: S -> P, νs=1, νp=1.
    """
    rxn = _make_reaction(
        reactant_labels=["S"],
        product_labels=["P"],
        stoichiometry={"S": -1, "P": 1},
        km_per_substrate={"S": km_s, "P": km_p},
        kcat=kcat_fwd,
        km=None,
    )
    rxn.template = MagicMock(reversible=template_reversible)
    rxn.thermo = ThermoData(
        dgr_prime_kJmol=-5.0,
        sigma_kJmol=0.5,
        keq=keq,
        kcat_rev=kcat_fwd * km_p / (keq * km_s),
        irreversible=irreversible_thermo,
        source="equilibrator",
    )
    return rxn


class TestComputeReversibleMmRate:
    def test_detailed_balance_zero_at_q_equals_keq(self):
        # Choose [S], [P] so that Q = [P]/[S] = Keq → net rate must be ~0.
        keq = 10.0
        rxn = _make_reversible_reaction(keq=keq, km_s=1.0, km_p=2.0)
        conc = {"s": 1.0, "p": keq * 1.0}  # Q = 10 = Keq
        enz = {"hexokinase": 0.001}
        v = compute_reversible_mm_rate(rxn, conc, enz)
        assert abs(v) < 1e-12

    def test_forward_when_q_below_keq(self):
        rxn = _make_reversible_reaction(keq=10.0)
        conc = {"s": 1.0, "p": 0.0}  # Q = 0 ≪ Keq
        v = compute_reversible_mm_rate(rxn, conc, {"hexokinase": 0.001})
        assert v > 0.0

    def test_reverse_when_q_above_keq(self):
        # Q = [P]/[S] = 100 ≫ Keq=10 → net rate must be negative.
        rxn = _make_reversible_reaction(keq=10.0)
        conc = {"s": 0.1, "p": 10.0}
        v = compute_reversible_mm_rate(rxn, conc, {"hexokinase": 0.001})
        assert v < 0.0

    def test_far_from_eq_matches_irreversible_within_pct(self):
        # When [P] ≈ 0 and the disequilibrium term ≈ 1, the reversible form
        # collapses to ~ Vmax_f * [S]/Km_s / (1 + [S]/Km_s + 1 - 1)
        #           = Vmax_f * [S]/(Km_s + [S]),
        # which matches the irreversible MM exactly for a 1S→1P case.
        rxn_rev = _make_reversible_reaction(keq=1e6, km_s=1.0, km_p=2.0)
        rxn_irr = _make_reaction(
            reactant_labels=["S"],
            product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km_per_substrate={"S": 1.0},
            kcat=100.0,
            km=None,
        )
        conc = {"s": 5.0, "p": 0.0}
        enz = {"hexokinase": 0.001}
        v_rev = compute_reversible_mm_rate(rxn_rev, conc, enz)
        v_irr = compute_mm_rate(rxn_irr, conc, enz)
        rel = abs(v_rev - v_irr) / max(v_irr, 1e-30)
        assert rel < 0.01, f"v_rev={v_rev}, v_irr={v_irr}, rel={rel}"

    def test_falls_back_to_irreversible_when_thermo_flagged(self):
        # When thermo.irreversible is True we should bypass reversible MM
        # entirely and return the legacy compute_mm_rate value.
        rxn = _make_reversible_reaction(irreversible_thermo=True)
        conc = {"s": 1.0, "p": 5.0}
        enz = {"hexokinase": 0.001}
        v = compute_reversible_mm_rate(rxn, conc, enz)
        # Same call against compute_mm_rate should give the same value.
        v_irr = compute_mm_rate(rxn, conc, enz)
        assert v == v_irr
        assert v > 0.0  # forward MM ignores [P]

    def test_falls_back_when_no_thermo(self):
        rxn = _make_reversible_reaction()
        rxn.thermo = None
        v = compute_reversible_mm_rate(rxn, {"s": 1.0, "p": 5.0}, {"hexokinase": 0.001})
        assert v > 0.0  # falls back to forward-only

    def test_falls_back_when_product_km_missing(self):
        # No Km_p ⇒ Haldane wasn't computable; defer to forward-only.
        rxn = _make_reversible_reaction()
        rxn.kinetics.km_per_substrate = {"S": 1.0}  # drop Km for P
        v = compute_reversible_mm_rate(rxn, {"s": 1.0, "p": 5.0}, {"hexokinase": 0.001})
        v_fwd = compute_mm_rate(rxn, {"s": 1.0, "p": 5.0}, {"hexokinase": 0.001})
        assert v == v_fwd

    def test_skips_buffered_h_plus_substrate_without_fallback(self):
        # FabG-like: H+ reactant without Km must not collapse to legacy when
        # all non-exempt substrate/product Kms are present.
        rxn = _make_reaction(
            reactant_labels=["oxoACP", "NADPH", "H+"],
            product_labels=["OHACP", "NADP"],
            stoichiometry={"oxoACP": -1, "NADPH": -1, "H+": -1, "OHACP": 1, "NADP": 1},
            km=None, kcat=10.0,
            km_per_substrate={
                "oxoACP": 0.08, "NADPH": 0.12, "OHACP": 0.2, "NADP": 0.15,
            },
        )
        rxn.template = MagicMock(reversible=True)
        rxn.thermo = ThermoData(
            dgr_prime_kJmol=-14.0, sigma_kJmol=1.0, keq=340.0,
            kcat_rev=0.4, irreversible=False, source="equilibrator",
        )
        enz = {"hexokinase": 0.001}
        conc_low = {"oxoacp": 0.1, "nadph": 0.1, "h+": 1e-7, "ohacp": 0.01, "nadp": 0.0}
        conc_high = dict(conc_low)
        conc_high["nadp"] = 0.5
        v_low = compute_reversible_mm_rate(rxn, conc_low, enz)
        v_high = compute_reversible_mm_rate(rxn, conc_high, enz)
        assert v_low > 0.0
        assert v_high < v_low  # NADP product term engages

    def test_h2o_product_missing_km_still_reversible(self):
        # FabA-like: missing H2O Km must not block reversible MM.
        rxn = _make_reaction(
            reactant_labels=["OHACP"],
            product_labels=["enoylACP", "H2O"],
            stoichiometry={"OHACP": -1, "enoylACP": 1, "H2O": 1},
            km=None, kcat=10.0,
            km_per_substrate={"OHACP": 0.01, "enoylACP": 0.05},
        )
        rxn.template = MagicMock(reversible=True)
        rxn.thermo = ThermoData(
            dgr_prime_kJmol=3.0, sigma_kJmol=1.0, keq=0.27,
            kcat_rev=20.0, irreversible=False, source="equilibrator",
        )
        enz = {"hexokinase": 0.001}
        v0 = compute_reversible_mm_rate(
            rxn, {"ohacp": 0.1, "enoylacp": 0.0, "h2o": 55.0}, enz
        )
        v_rev = compute_reversible_mm_rate(
            rxn, {"ohacp": 0.01, "enoylacp": 1.0, "h2o": 55.0}, enz
        )
        assert v0 > 0.0
        assert v_rev < 0.0  # reverse flux when Q > Keq

    def test_h2o_omitted_from_q(self):
        # Biochemical K'eq already folds in water activity; [H2O] must not
        # enter Q or dehydrations reverse spuriously at the 55 mM pool.
        rxn = _make_reaction(
            reactant_labels=["OHACP"],
            product_labels=["enoylACP", "H2O"],
            stoichiometry={"OHACP": -1, "enoylACP": 1, "H2O": 1},
            km=None, kcat=10.0,
            km_per_substrate={"OHACP": 0.01, "enoylACP": 0.05},
        )
        rxn.template = MagicMock(reversible=True)
        rxn.thermo = ThermoData(
            dgr_prime_kJmol=3.0, sigma_kJmol=1.0, keq=0.27,
            kcat_rev=20.0, irreversible=False, source="equilibrator",
        )
        enz = {"hexokinase": 0.001}
        conc = {"ohacp": 0.1, "enoylacp": 0.0, "h2o": 55.0}
        v_hi = compute_reversible_mm_rate(rxn, conc, enz)
        conc_lo = dict(conc)
        conc_lo["h2o"] = 1.0
        v_lo = compute_reversible_mm_rate(rxn, conc_lo, enz)
        assert v_hi > 0.0
        assert abs(v_hi - v_lo) < 1e-12


class TestDispatch:
    """calculate_species_rates picks reversible MM only when both the
    template flag AND a non-irreversible ThermoData are present."""

    def test_reversible_dispatch(self):
        rxn = _make_reversible_reaction(keq=10.0)
        # Q far above Keq ⇒ negative flux on the reversible path; the
        # irreversible path would give a positive flux instead.
        conc = {"s": 0.1, "p": 10.0}
        enz = {"hexokinase": 0.001}
        rates = calculate_species_rates([rxn], conc, enz)
        assert rates["s"] > 0.0  # S produced (reverse)
        assert rates["p"] < 0.0  # P consumed (reverse)

    def test_irreversible_template_skips_reversible_path(self):
        # template.reversible = False → reversible MM is not used even if
        # ThermoData says otherwise.
        rxn = _make_reversible_reaction(keq=10.0, template_reversible=False)
        conc = {"s": 0.1, "p": 10.0}
        rates = calculate_species_rates([rxn], conc, {"hexokinase": 0.001})
        # Forward MM with [S]=0.1, ignoring [P] → positive consumption of S.
        assert rates["s"] < 0.0

    def test_irreversible_thermo_flag_skips_reversible_path(self):
        rxn = _make_reversible_reaction(keq=10.0, irreversible_thermo=True)
        conc = {"s": 0.1, "p": 10.0}
        rates = calculate_species_rates([rxn], conc, {"hexokinase": 0.001})
        assert rates["s"] < 0.0  # forward MM, P ignored
