"""
Tests for the ODE simulator module.

To run:  pytest -v tests/test_simulator.py
"""


import pytest
import numpy as np
from itertools import chain, repeat
from unittest.mock import MagicMock, patch

from bees.core_edge_model import CoreEdgeModel, SpeciesData
from bees.simulator import ODESimulator, _VectorizedRHS


# ---------------------------------------------------------------------------
# Helper to build a mock reaction
# ---------------------------------------------------------------------------

def _make_reaction(
    enzyme_label="Enzyme",
    substrate_label="S",
    reactant_labels=None,
    product_labels=None,
    stoichiometry=None,
    rate_law="Michaelis-Menten",
    km=1.0,
    kcat=10.0,
    vmax=None,
    km_per_substrate=None,
):
    kinetics = MagicMock()
    kinetics.km = km
    kinetics.kcat = kcat
    kinetics.vmax = vmax
    kinetics.km_per_substrate = km_per_substrate

    rxn = MagicMock()
    rxn.enzyme_label = enzyme_label
    rxn.substrate_label = substrate_label
    rxn.reactant_labels = reactant_labels or ["S"]
    rxn.product_labels = product_labels or ["P"]
    rxn.stoichiometry = stoichiometry or {"S": -1, "P": 1}
    rxn.rate_law = rate_law
    rxn.kinetics = kinetics
    return rxn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """
    Minimal model: S -> P catalysed by E.
    Uses Michaelis-Menten kinetics with Km=1.0 mM, kcat=10.0 s^-1.
    Initial [S] = 10.0 mM, [E] = 0.01 mM.
    """
    model = CoreEdgeModel()

    model.add_core_species(SpeciesData(
        label="S", concentration=10.0, initial_concentration=10.0,
    ))
    model.add_core_species(SpeciesData(
        label="P", concentration=0.0, initial_concentration=0.0,
    ))
    model.add_core_species(SpeciesData(
        label="Enzyme", concentration=0.01, initial_concentration=0.01,
        is_enzyme=True, constant=True,
    ))

    rxn = _make_reaction(
        enzyme_label="Enzyme",
        substrate_label="S",
        reactant_labels=["S"],
        product_labels=["P"],
        stoichiometry={"S": -1, "P": 1},
        km=1.0,
        kcat=10.0,
    )
    model.core_reactions.append(rxn)
    return model


# ---------------------------------------------------------------------------
# Test ODESimulator
# ---------------------------------------------------------------------------

class TestODESimulator:
    def test_basic_simulation(self, simple_model):
        """S -> P should show S decreasing and P increasing."""
        sim = ODESimulator(simple_model)
        result = sim.simulate(end_time=100.0, time_step=10.0)

        assert result.success
        assert result.t.shape[0] > 1
        assert result.species_labels == ["S", "P", "Enzyme"]
        assert np.all(result.y >= 0.0)

        # S should decrease
        s_idx = 0
        assert result.y[s_idx, -1] < result.y[s_idx, 0]

        # P should increase
        p_idx = 1
        assert result.y[p_idx, -1] > result.y[p_idx, 0]

    def test_mass_conservation(self, simple_model):
        """Total [S] + [P] should be conserved (=10 mM)."""
        sim = ODESimulator(simple_model)
        result = sim.simulate(end_time=100.0, time_step=1.0)

        s_idx = 0
        p_idx = 1
        total = result.y[s_idx, :] + result.y[p_idx, :]
        np.testing.assert_allclose(total, 10.0, atol=0.01)

    def test_enzyme_constant(self, simple_model):
        """Enzyme concentration should remain constant."""
        sim = ODESimulator(simple_model)
        result = sim.simulate(end_time=50.0, time_step=5.0)
        e_idx = 2
        np.testing.assert_allclose(result.y[e_idx, :], 0.01, atol=1e-12)

    def test_empty_model(self):
        """An empty model should simulate without errors."""
        model = CoreEdgeModel()
        sim = ODESimulator(model)
        result = sim.simulate(end_time=10.0)
        assert result.success
        assert result.species_labels == []

    def test_flux_trajectory_metrics_attached(self, simple_model):
        """simulate() should attach max/final R_char and edge rate ratios."""
        sim = ODESimulator(simple_model)
        result = sim.simulate(end_time=10.0, time_step=1.0)
        assert result.max_char_rate >= 0.0
        assert result.final_char_rate >= 0.0
        assert isinstance(result.max_edge_rate_ratio, dict)
        assert result.max_char_rate >= result.final_char_rate or result.max_char_rate > 0

    def test_model_concentrations_updated(self, simple_model):
        """After simulation, model concentrations should reflect the final state."""
        sim = ODESimulator(simple_model)
        sim.simulate(end_time=100.0)
        # S should have decreased from 10.0
        s_sp = simple_model.get_core_species_by_label("S")
        assert s_sp.concentration < 10.0
        assert s_sp.concentration >= 0.0


# ---------------------------------------------------------------------------
# Stepwise interrupt tests
# ---------------------------------------------------------------------------

def _make_core_edge_model_with_immediate_flux():
    """
    Build a model where a core reaction S -> P (core) immediately
    produces edge flux for species E_prod via P -> E_prod (edge).
    Both reactions run from t=0, so edge rr should exceed any
    reasonable tolerance almost immediately.
    """
    model = CoreEdgeModel()

    model.add_core_species(SpeciesData(
        label="S", concentration=10.0, initial_concentration=10.0,
    ))
    model.add_core_species(SpeciesData(
        label="P", concentration=5.0, initial_concentration=5.0,
    ))
    model.add_core_species(SpeciesData(
        label="Enz", concentration=0.01, initial_concentration=0.01,
        is_enzyme=True, constant=True,
    ))

    model.add_edge_species(SpeciesData(
        label="E_prod", concentration=0.0, initial_concentration=0.0,
    ))

    rxn_core = _make_reaction(
        enzyme_label="Enz", substrate_label="S",
        reactant_labels=["S"], product_labels=["P"],
        stoichiometry={"S": -1, "P": 1},
        km=1.0, kcat=10.0,
    )
    model.core_reactions.append(rxn_core)

    rxn_edge = _make_reaction(
        enzyme_label="Enz", substrate_label="P",
        reactant_labels=["P"], product_labels=["E_prod"],
        stoichiometry={"P": -1, "E_prod": 1},
        km=1.0, kcat=10.0,
    )
    model.edge_reactions.append(rxn_edge)
    return model


class TestStepwiseInterrupt:
    """Tests for the stepwise interrupt path."""

    def test_interrupt_occurs_at_positive_time(self):
        """With immediate edge flux and tiny tolerance, interrupt must be at t > 0."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=100.0,
            interrupt_simulation_tol=1e-8,
        )
        assert result.simulation_interrupted
        assert result.t[-1] > 0.0, "Interrupt must occur after at least one step (t > 0)"

    def test_interrupt_rates_populated(self):
        """When interrupted, interrupt_char_rate and interrupt_edge_rates must be set."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=100.0,
            interrupt_simulation_tol=1e-8,
        )
        assert result.simulation_interrupted
        assert result.interrupt_char_rate > 0.0
        assert len(result.interrupt_edge_rates) > 0
        assert "e_prod" in result.interrupt_edge_rates

    def test_no_interrupt_without_tolerance(self):
        """Without interrupt_simulation_tol, simulation should run to end_time."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(end_time=1.0, time_step=0.1)
        assert not result.simulation_interrupted
        assert result.t[-1] >= 0.99

    def test_no_interrupt_with_high_tolerance(self):
        """With a very high tolerance, no interrupt should fire."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=1.0,
            interrupt_simulation_tol=1e6,
        )
        assert not result.simulation_interrupted

    def test_continuous_path_still_works(self):
        """When no edge species exist, the continuous path should be used."""
        model = CoreEdgeModel()
        model.add_core_species(SpeciesData(
            label="A", concentration=5.0, initial_concentration=5.0,
        ))
        model.add_core_species(SpeciesData(
            label="B", concentration=0.0, initial_concentration=0.0,
        ))
        model.add_core_species(SpeciesData(
            label="Enz", concentration=0.01, initial_concentration=0.01,
            is_enzyme=True, constant=True,
        ))
        rxn = _make_reaction(
            enzyme_label="Enz", substrate_label="A",
            reactant_labels=["A"], product_labels=["B"],
            stoichiometry={"A": -1, "B": 1},
            km=1.0, kcat=5.0,
        )
        model.core_reactions.append(rxn)

        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=10.0, time_step=1.0,
            interrupt_simulation_tol=1e-5,
        )
        assert not result.simulation_interrupted
        assert result.t[-1] >= 9.99

    def test_history_has_multiple_points(self):
        """Stepwise result should contain more than one time point."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=100.0,
            interrupt_simulation_tol=1e-8,
        )
        assert len(result.t) >= 2
        assert result.y.shape[1] == len(result.t)

    def test_stepwise_dt_is_capped_at_5s(self):
        """
        The outer-step dt in stepwise mode should never exceed 5 s.

        We check this indirectly via the returned time history, which stores
        the per-step endpoints produced by the outer step loop.
        """
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=50.0,
            interrupt_simulation_tol=1e9,
        )
        assert not result.simulation_interrupted
        assert len(result.t) >= 2
        dts = np.diff(result.t)
        assert np.all(dts <= 5.0 + 1e-12)

    def test_stepwise_wall_clock_limit_stops_pass(self):
        """
        When max_wall_time_s is exceeded after an outer step, the pass should end
        with success=False (enlarger treats this as ODE failure).
        """
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        mono_vals = chain([0.0, 0.0, 100.0], repeat(100.0))
        with patch("bees.simulator.time.monotonic", side_effect=lambda: next(mono_vals)):
            result = sim.simulate(
                end_time=50.0,
                interrupt_simulation_tol=1e9,
                max_wall_time_s=1.0,
                stepwise_heartbeat_interval_s=0.0,
            )
        assert not result.success
        assert "max_wall_time_per_iteration" in result.message
        assert not result.simulation_interrupted

    def test_simulate_accepts_custom_method_and_tolerances(self):
        """Custom method/rtol/atol should not break the stepwise path."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(
            end_time=1.0,
            interrupt_simulation_tol=1e9,
            method="BDF",
            rtol=1e-5,
            atol=1e-7,
            stepwise_heartbeat_interval_s=0.0,
        )
        assert result.success
        assert not result.simulation_interrupted


# ---------------------------------------------------------------------------
# Test product-inhibition overlay in _VectorizedRHS
# ---------------------------------------------------------------------------

class TestProductInhibitionOverlay:
    """
    Tests that _VectorizedRHS routes a product-Km-complete irreversible
    reaction to the product-inhibition overlay and that high product
    concentrations reduce compute_v output.
    """

    def _make_product_inhibited_rxn(self):
        """One irreversible S -> P reaction with explicit Km for both S and P."""
        rxn = _make_reaction(
            enzyme_label="Enzyme",
            substrate_label="S",
            reactant_labels=["S"],
            product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=None,
            kcat=10.0,
            km_per_substrate={"S": 1.0, "P": 2.0},
        )
        rxn.template = MagicMock()
        rxn.template.reversible = False
        rxn.thermo = None
        return rxn

    def test_product_accumulation_reduces_compute_v(self):
        rxn = self._make_product_inhibited_rxn()
        species_labels = ["S", "P"]
        rhs = _VectorizedRHS(
            species_labels=species_labels,
            reactions=[rxn],
            alias_to_model_label={"s": "s", "p": "p"},
            enzyme_conc_map={"enzyme": 1.0},
            constant_mask=np.array([False, False]),
        )

        v_no_p   = rhs.compute_v(np.array([5.0, 0.0]))
        v_high_p = rhs.compute_v(np.array([5.0, 100.0]))

        assert v_high_p[0] < v_no_p[0], "product accumulation must reduce forward rate"
        assert v_high_p[0] > 0.0,       "rate must remain forward-only (>= 0)"

    def test_reaction_routed_to_overlay_not_fast_path(self):
        """vmax for a product-Km-complete reaction must be 0 (overlay, not fast path)."""
        rxn = self._make_product_inhibited_rxn()
        rhs = _VectorizedRHS(
            species_labels=["S", "P"],
            reactions=[rxn],
            alias_to_model_label={"s": "s", "p": "p"},
            enzyme_conc_map={"enzyme": 1.0},
            constant_mask=np.array([False, False]),
        )
        assert rhs._vmax[0] == 0.0
        assert len(rhs._product_inhibited_indices) == 1
        assert rhs._product_inhibited_indices[0] == 0


# ---------------------------------------------------------------------------
# ODE-equations text emission: buffered species  must not
# appear in printed rate-law terms — they're already absorbed in K'eq and
# the SBML kinetic law omits them, so the text dump should match.
# ---------------------------------------------------------------------------

class TestExportOdeEquationsBufferedFilter:
    def _make_thermo(self, kcat_rev=5.0):
        td = MagicMock()
        td.irreversible = False
        td.dgr_prime_kJmol = -10.0
        td.sigma_kJmol = None  # formatter reads this; pin so it isn't a MagicMock
        td.keq = 100.0
        td.kcat_rev = kcat_rev
        td.source = "equilibrator"
        return td

    def test_buffered_species_omitted_from_reversible_rate_law(self, tmp_path):
        model = CoreEdgeModel()
        # Variable species
        model.add_core_species(SpeciesData(label="S", concentration=1.0, initial_concentration=1.0))
        model.add_core_species(SpeciesData(label="P", concentration=0.0, initial_concentration=0.0))
        # Buffered cofactors
        model.add_core_species(SpeciesData(
            label="H+", concentration=4e-5, initial_concentration=4e-5, constant=True,
        ))
        model.add_core_species(SpeciesData(
            label="H2O", concentration=55.0, initial_concentration=55.0, constant=True,
        ))
        # Enzyme
        model.add_core_species(SpeciesData(
            label="Enz", concentration=0.01, initial_concentration=0.01,
            is_enzyme=True, constant=True,
        ))

        rxn = _make_reaction(
            enzyme_label="Enz",
            substrate_label="S",
            reactant_labels=["S", "H+"],
            product_labels=["P", "H2O"],
            stoichiometry={"S": -1, "H+": -1, "P": 1, "H2O": 1},
            km=None,
            kcat=10.0,
            km_per_substrate={"S": 0.1, "P": 0.2},
        )
        # MagicMock auto-creates attribute access; pin formatter-touched fields to None.
        rxn.kinetics.kcat_sd = None
        rxn.kinetics.km_sd = None
        rxn.kinetics.km_sd_per_substrate = None
        rxn.ec_number = None
        rxn.thermo = self._make_thermo()
        model.core_reactions.append(rxn)

        sim = ODESimulator(model)
        out = tmp_path / "ode_eq.txt"
        sim.export_ode_equations(str(out), iteration=1, end_time=10.0)
        text = out.read_text()

        # Rate-law line should not carry [H+] or [H2O] as concentration factors.
        rate_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("v1 =")]
        assert rate_lines, f"expected v1 line in:\n{text}"
        rate_line = rate_lines[0]
        assert "[H+]" not in rate_line, rate_line
        assert "[H2O]" not in rate_line, rate_line
        assert "Km_H+" not in rate_line, rate_line
        assert "Km_H2O" not in rate_line, rate_line
        # Variable species must still be there
        assert "[S]/Km_S" in rate_line
        assert "[P]/Km_P" in rate_line

    def test_buffered_species_omitted_from_irreversible_rate_law(self, tmp_path):
        model = CoreEdgeModel()
        model.add_core_species(SpeciesData(label="S", concentration=1.0, initial_concentration=1.0))
        model.add_core_species(SpeciesData(label="P", concentration=0.0, initial_concentration=0.0))
        model.add_core_species(SpeciesData(
            label="H2O", concentration=55.0, initial_concentration=55.0, constant=True,
        ))
        model.add_core_species(SpeciesData(
            label="Enz", concentration=0.01, initial_concentration=0.01,
            is_enzyme=True, constant=True,
        ))

        rxn = _make_reaction(
            enzyme_label="Enz",
            substrate_label="S",
            reactant_labels=["S", "H2O"],
            product_labels=["P"],
            stoichiometry={"S": -1, "H2O": -1, "P": 1},
            km=None,
            kcat=10.0,
            km_per_substrate={"S": 0.1},
        )
        rxn.kinetics.kcat_sd = None
        rxn.kinetics.km_sd = None
        rxn.kinetics.km_sd_per_substrate = None
        rxn.ec_number = None
        rxn.thermo = None  # forces irreversible branch
        model.core_reactions.append(rxn)

        sim = ODESimulator(model)
        out = tmp_path / "ode_eq.txt"
        sim.export_ode_equations(str(out), iteration=1, end_time=10.0)
        text = out.read_text()
        rate_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("v1 =")]
        assert rate_lines, f"expected v1 line in:\n{text}"
        rate_line = rate_lines[0]
        assert "[H2O]" not in rate_line, rate_line
        assert "Km_H2O" not in rate_line, rate_line
        assert "[S]" in rate_line


class TestFeedbackInhibition:
    """Opt-in end-product / feedback inhibition overlay in _VectorizedRHS.

    A reaction carrying ``feedback_inhibitors = {label: (Ki, hill)}`` has its
    rate multiplied by ∏ 1/(1+([I]/Ki)^hill). Reactions without the attribute
    (None, or a MagicMock as built by _make_reaction) are unaffected.
    """

    def _build_rhs(self, feedback):
        # species: S, P, I (inhibitor), Enz ; reaction S -> P by Enz.
        species = ["S", "P", "I", "Enz"]
        rxn = _make_reaction(
            enzyme_label="Enz", substrate_label="S",
            reactant_labels=["S"], product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=None, kcat=10.0, km_per_substrate={"S": 1.0},
        )
        rxn.feedback_inhibitors = feedback  # dict or None
        return _VectorizedRHS(
            species_labels=species,
            reactions=[rxn],
            alias_to_model_label={s.lower(): s.lower() for s in species},
            enzyme_conc_map={"enz": 0.01},
            constant_mask=np.array([False, False, False, True]),
            n_core_species=4, n_core_reactions=1,
        )

    def _base_rate(self):
        # Vmax * S/(Km+S) = (10*0.01) * 10/(1+10)
        return 10.0 * 0.01 * (10.0 / (1.0 + 10.0))

    def test_no_feedback_attr_is_neutral(self):
        """A MagicMock feedback_inhibitors (auto-attr) must be ignored, not crash."""
        rhs = self._build_rhs(feedback=MagicMock())  # not a dict
        assert rhs._has_feedback is False
        v = rhs.compute_v(np.array([10.0, 0.0, 2.0, 0.01]))
        np.testing.assert_allclose(v[0], self._base_rate(), rtol=1e-9)

    def test_inhibitor_throttles_rate(self):
        """Ki=2 mM, [I]=2 mM, hill=1 -> factor 1/(1+1) = 0.5."""
        rhs = self._build_rhs(feedback={"i": (2.0, 1.0)})
        assert rhs._has_feedback is True
        v = rhs.compute_v(np.array([10.0, 0.0, 2.0, 0.01]))
        np.testing.assert_allclose(v[0], self._base_rate() * 0.5, rtol=1e-9)

    def test_zero_inhibitor_is_neutral(self):
        """No product yet ([I]=0) -> factor 1, rate equals baseline."""
        rhs = self._build_rhs(feedback={"i": (2.0, 1.0)})
        v = rhs.compute_v(np.array([10.0, 0.0, 0.0, 0.01]))
        np.testing.assert_allclose(v[0], self._base_rate(), rtol=1e-9)

    def test_hill_exponent(self):
        """Ki=2, [I]=4, hill=2 -> factor 1/(1+(4/2)^2) = 1/5."""
        rhs = self._build_rhs(feedback={"i": (2.0, 2.0)})
        v = rhs.compute_v(np.array([10.0, 0.0, 4.0, 0.01]))
        np.testing.assert_allclose(v[0], self._base_rate() * (1.0 / 5.0), rtol=1e-9)


# ---------------------------------------------------------------------------
# RMG-style edge isolation: edge species are not integrated, edge reactions
# don't drain core species, but edge rates are still scored for promotion.
# ---------------------------------------------------------------------------

class TestEdgeIsolation:
    def test_edge_species_concentration_stays_at_initial(self):
        """A core reaction with an edge product must not grow the edge species."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(end_time=10.0, time_step=1.0)
        assert result.success

        e_prod_idx = result.species_labels.index("E_prod")
        e_prod_initial = float(result.y[e_prod_idx, 0])
        e_prod_final = float(result.y[e_prod_idx, -1])
        assert e_prod_initial == pytest.approx(0.0)
        # With edge isolation the edge row is forced to dC/dt = 0 throughout.
        assert e_prod_final == pytest.approx(e_prod_initial, abs=1e-12), (
            f"edge species drifted from {e_prod_initial} to {e_prod_final}"
        )

    def test_edge_reaction_does_not_drain_core(self):
        """An edge reaction consuming a core species must not affect that core species."""
        model = CoreEdgeModel()
        model.add_core_species(SpeciesData(
            label="S", concentration=10.0, initial_concentration=10.0,
        ))
        model.add_core_species(SpeciesData(
            label="Enz", concentration=0.01, initial_concentration=0.01,
            is_enzyme=True, constant=True,
        ))
        model.add_edge_species(SpeciesData(
            label="E_prod", concentration=0.0, initial_concentration=0.0,
        ))
        # Only an EDGE reaction exists (S -> E_prod). Without isolation, [S]
        # would drain to ~0 via the edge reaction. With isolation, [S] stays.
        rxn_edge = _make_reaction(
            enzyme_label="Enz", substrate_label="S",
            reactant_labels=["S"], product_labels=["E_prod"],
            stoichiometry={"S": -1, "E_prod": 1},
            km=1.0, kcat=10.0,
        )
        model.edge_reactions.append(rxn_edge)

        sim = ODESimulator(model)
        result = sim.simulate(end_time=10.0, time_step=1.0)
        assert result.success
        s_idx = result.species_labels.index("S")
        s_initial = float(result.y[s_idx, 0])
        s_final = float(result.y[s_idx, -1])
        assert s_final == pytest.approx(s_initial, rel=1e-9), (
            f"core species [S] drained via edge reaction: {s_initial} -> {s_final}"
        )

    def test_edge_rate_ratio_still_computed(self):
        """Edge isolation must not silence the promotion-scoring signal."""
        model = _make_core_edge_model_with_immediate_flux()
        sim = ODESimulator(model)
        result = sim.simulate(end_time=10.0, time_step=1.0)
        assert result.success
        # max_edge_rate_ratio must be populated and the E_prod entry must be
        # > 0 — the edge IS receiving flux from a core reaction, even though
        # the integrator holds its concentration flat. The enlarger needs
        # this ratio to decide whether to promote E_prod into core.
        assert "e_prod" in result.max_edge_rate_ratio
        assert result.max_edge_rate_ratio["e_prod"] > 0.0, (
            f"edge rate ratio went to zero (got "
            f"{result.max_edge_rate_ratio['e_prod']}) — promotion signal lost"
        )

    def test_bootstrap_with_zero_core_reactions_still_scores_edges(self):
        """
        BEES enlarger bootstrap: the model starts with zero core reactions
        and every discovered reaction enters as an edge. Promotion to core
        depends on the rate-ratio interrupt firing — which needs both
        ``max_char_rate`` and ``max_edge_rate_ratio`` to be non-zero.

        Regression: with isolated R_char this entire scoring channel was
        silent in bootstrap (the FAS run converged trivially on iter 1 with
        0 reactions). The metrics path now reads from the unmasked residual
        so the enlarger can still see what's happening.
        """
        model = CoreEdgeModel()
        model.add_core_species(SpeciesData(
            label="S", concentration=10.0, initial_concentration=10.0,
        ))
        model.add_core_species(SpeciesData(
            label="Enz", concentration=0.01, initial_concentration=0.01,
            is_enzyme=True, constant=True,
        ))
        model.add_edge_species(SpeciesData(
            label="P_edge", concentration=0.0, initial_concentration=0.0,
        ))
        rxn_edge = _make_reaction(
            enzyme_label="Enz", substrate_label="S",
            reactant_labels=["S"], product_labels=["P_edge"],
            stoichiometry={"S": -1, "P_edge": 1},
            km=1.0, kcat=10.0,
        )
        model.edge_reactions.append(rxn_edge)
        assert len(model.core_reactions) == 0  # bootstrap precondition

        sim = ODESimulator(model)
        result = sim.simulate(end_time=10.0, time_step=1.0)
        assert result.success
        assert result.max_char_rate > 0.0, (
            "max_char_rate was zero in bootstrap — the enlarger's promotion "
            "signal is silent, and convergence will fire on iteration 1."
        )
        assert result.max_edge_rate_ratio.get("p_edge", 0.0) > 0.0, (
            "edge rate ratio went to zero in bootstrap — promotion impossible."
        )
        # The integrator itself still kept [S] flat (edge isolation works).
        s_idx = result.species_labels.index("S")
        assert float(result.y[s_idx, -1]) == pytest.approx(10.0, rel=1e-9)
