"""
Tests for the IterativeEnlarger module.

To run:  pytest -v tests/test_enlarger.py
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from bees.enlarger import IterativeEnlarger, EnlargerResult
from bees.simulator import SimulationResult


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
):
    kinetics = MagicMock()
    kinetics.km = km
    kinetics.kcat = kcat
    kinetics.vmax = None
    kinetics.km_per_substrate = None

    rxn = MagicMock()
    rxn.enzyme_label = enzyme_label
    rxn.substrate_label = substrate_label
    rxn.reactant_labels = reactant_labels or ["S"]
    rxn.product_labels = product_labels or ["P"]
    rxn.stoichiometry = stoichiometry or {"S": -1, "P": 1}
    rxn.rate_law = rate_law
    rxn.kinetics = kinetics
    # GeneratedReaction-like attributes
    rxn.ec_number = "EC 1.1.1.1"
    rxn.template = MagicMock()
    return rxn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bees_object():
    """Minimal bees_object with settings for iterative mode."""
    settings = SimpleNamespace(
        end_time=100.0,
        time_step=10.0,
        toleranceMoveToCore=1e-5,
        toleranceKeepInEdge=0,
        max_iterations=3,
        max_edge_species=None,
        termination_conversion=None,
        termination_rate_ratio=None,
        save_simulation_profiles=False,
        saveEdgeSpecies=True,
    )

    species = [
        SimpleNamespace(label="S", concentration=10.0, reactive=True, solvent=False, constant=False),
    ]
    enzymes = [
        SimpleNamespace(label="Enzyme", concentration=0.01, reactive=True, solvent=False,
                        ecnumber="EC 1.1.1.1", constant=True),
    ]

    bees_obj = SimpleNamespace(
        settings=settings,
        species=species,
        enzymes=enzymes,
        project="TestProject",
        database=SimpleNamespace(name="db"),
        environment=SimpleNamespace(temperature=298.15, pH=7.0),
    )
    return bees_obj


@pytest.fixture
def mock_reaction_generator():
    """ReactionGenerator that returns a canned reaction on generate_reactions()."""
    mg = MagicMock()
    mg.generate_reactions.return_value = [
        _make_reaction(
            enzyme_label="Enzyme",
            substrate_label="S",
            reactant_labels=["S"],
            product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            km=1.0,
            kcat=10.0,
        ),
    ]
    mg._generate_reactions.return_value = []
    return mg


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnlargerResult:
    def test_default_values(self):
        r = EnlargerResult()
        assert r.iterations == 0
        assert not r.converged
        assert r.simulation_profiles == []


class TestIterativeEnlarger:
    def test_rate_ratio_termination_rmg_style(self, mock_bees_object, mock_reaction_generator, output_dir):
        """termination_rate_ratio compares final_char_rate / max_char_rate."""
        mock_bees_object.settings.termination_rate_ratio = 0.5
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        sr = SimulationResult(
            t=np.array([0.0, 1.0]),
            y=np.zeros((1, 2)),
            species_labels=["x"],
            success=True,
            max_char_rate=10.0,
            final_char_rate=2.0,
        )
        assert enlarger._check_rate_ratio_termination(sr) is True

        sr2 = SimulationResult(
            t=np.array([0.0, 1.0]),
            y=np.zeros((1, 2)),
            species_labels=["x"],
            success=True,
            max_char_rate=10.0,
            final_char_rate=6.0,
        )
        assert enlarger._check_rate_ratio_termination(sr2) is False

    def test_initialise_model(self, mock_bees_object, mock_reaction_generator, output_dir):
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger._initialise_model()
        # Should have the substrate and the enzyme in core
        assert enlarger.model.is_core_species("S")
        assert enlarger.model.is_core_species("Enzyme")
        assert len(enlarger.model.core_species) == 2

    def test_ingest_reactions(self, mock_bees_object, mock_reaction_generator, output_dir):
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger._initialise_model()

        rxn = _make_reaction()
        enlarger._ingest_reactions([rxn])

        # P should have been added to edge (not in core initially)
        assert enlarger.model.is_edge_species("P")
        # S should still be in core
        assert enlarger.model.is_core_species("S")

    def test_run_converges(self, mock_bees_object, mock_reaction_generator, output_dir):
        """The enlarger should run and converge (no new significant species)."""
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        result = enlarger.run()
        assert isinstance(result, EnlargerResult)
        assert result.iterations >= 1
        assert result.final_core_species >= 1

    def test_iterations_count_simulation_passes(
        self, mock_bees_object, mock_reaction_generator, output_dir
    ):
        """
        Iteration count should reflect simulation passes (interrupt cycles),
        not an outer loop with hidden sub-iterations.
        """
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )

        # Ensure there is an edge species "p" that can be promoted when the
        # mocked simulator reports interrupt_edge_rates={"p": ...}.
        seed_rxn_p = _make_reaction(
            enzyme_label="Enzyme",
            substrate_label="S",
            reactant_labels=["S"],
            product_labels=["p"],
            stoichiometry={"S": -1, "p": 1},
            km=1.0,
            kcat=10.0,
        )
        seed_rxn_q = _make_reaction(
            enzyme_label="Enzyme",
            substrate_label="S",
            reactant_labels=["S"],
            product_labels=["q"],
            stoichiometry={"S": -1, "q": 1},
            km=1.0,
            kcat=10.0,
        )

        def _gen_rxns(labels, r_char_ref=None):
            # First call (initial discovery) seeds "p" into the edge; subsequent
            # calls (for promoted species) are no-ops for this test.
            return [seed_rxn_p, seed_rxn_q] if r_char_ref is None else []

        enlarger._generate_reactions_for_species = MagicMock(side_effect=_gen_rxns)

        def _sr(interrupted: bool, edge_label: str = "p") -> SimulationResult:
            # Minimal SimulationResult satisfying enlarger fields.
            sr = SimulationResult(
                t=np.array([0.0, 1.0]),
                y=np.zeros((1, 2)),
                species_labels=["x"],
                success=True,
                max_char_rate=1.0,
                final_char_rate=1.0,
                simulation_interrupted=interrupted,
                interrupt_char_rate=1.0 if interrupted else 0.0,
                interrupt_edge_rates={edge_label: 1.0} if interrupted else {},
            )
            return sr

        # Two interrupted passes then one non-interrupted (converged).
        sims = [_sr(True, "p"), _sr(True, "q"), _sr(False)]

        with patch("bees.enlarger.ODESimulator") as SimCls:
            inst = SimCls.return_value
            inst.simulate.side_effect = sims
            inst.export_ode_equations = MagicMock()
            result = enlarger.run()

        assert result.converged
        assert result.iterations == 3
        assert len(result.simulation_profiles) == 3
        # Iteration summaries include iteration 0 + one row per pass.
        assert len(enlarger._iteration_summaries) == 1 + 3

    def test_max_iterations_termination(self, mock_bees_object, mock_reaction_generator, output_dir):
        """If we set max_iterations=1, it should stop after 1 iteration."""
        mock_bees_object.settings.max_iterations = 1
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        result = enlarger.run()
        assert result.iterations <= 1

    @patch("bees.enlarger.ODESimulator")
    def test_sub_iteration_restart_calls_simulate_twice(
        self,
        ode_cls,
        mock_bees_object,
        mock_reaction_generator,
        output_dir,
    ):
        """Interrupt at toleranceInterruptSimulation triggers promote+reset; second run completes."""
        from bees.core_edge_model import SpeciesData

        mock_reaction_generator.ensure_estimator_initialized = MagicMock()
        mock_reaction_generator._generate_reactions.return_value = []

        interrupted = SimulationResult(
            t=np.array([0.0, 0.5]),
            y=np.ones((3, 2)),
            species_labels=["S", "Enzyme", "prodx"],
            success=True,
            simulation_interrupted=True,
            interrupt_char_rate=1.0,
            interrupt_edge_rates={"prodx": 0.1},
            max_char_rate=1.0,
            final_char_rate=0.5,
            max_edge_rate_ratio={"prodx": 0.5},
        )
        complete = SimulationResult(
            t=np.array([0.0, 1.0]),
            y=np.ones((3, 2)),
            species_labels=["S", "Enzyme", "prodx"],
            success=True,
            simulation_interrupted=False,
            max_char_rate=1.0,
            final_char_rate=0.5,
            max_edge_rate_ratio={},
        )
        sim_inst = MagicMock()
        sim_inst.simulate.side_effect = [interrupted, complete]
        ode_cls.return_value = sim_inst

        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger._initialise_model()
        enlarger.model.add_edge_species(
            SpeciesData(label="prodx", concentration=0.0, initial_concentration=0.0)
        )
        enlarger.run()
        assert sim_inst.simulate.call_count == 2

    @patch("bees.enlarger.ODESimulator")
    def test_no_prune_when_simulation_interrupted(
        self,
        ode_cls,
        mock_bees_object,
        mock_reaction_generator,
        output_dir,
    ):
        """If interrupt cannot promote (no matching edge species), only one simulate call."""
        mock_bees_object.settings.toleranceKeepInEdge = 0.01
        mock_bees_object.settings.minEdgeIterationsForPrune = 0
        mock_reaction_generator.ensure_estimator_initialized = MagicMock()
        mock_reaction_generator._generate_reactions.return_value = [
            _make_reaction(
                enzyme_label="Enzyme",
                substrate_label="S",
                reactant_labels=["S"],
                product_labels=["prodx"],
                stoichiometry={"S": -1, "prodx": 1},
                kcat=10.0,
            ),
        ]
        sim_inst = MagicMock()
        sim_inst.simulate.return_value = SimulationResult(
            t=np.array([0.0, 1.0]),
            y=np.ones((2, 2)),
            species_labels=["S", "Enzyme"],
            success=True,
            simulation_interrupted=True,
            max_char_rate=1.0,
            final_char_rate=0.5,
            max_edge_rate_ratio={"prodx": 0.5},
        )
        ode_cls.return_value = sim_inst

        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger.model.prune_edge = MagicMock()
        enlarger.run()
        enlarger.model.prune_edge.assert_not_called()


class TestResetToInitialConsistency:
    """Every outer enlargement iteration must start from initial concentrations."""

    def test_reset_called_before_each_outer_iteration(
        self, mock_bees_object, mock_reaction_generator, output_dir
    ):
        """Ensure reset_concentrations_to_initial is called at the start of
        each enlargement iteration, even when the previous run completed
        without interrupt (and thus simulator left final-time concentrations
        in the model)."""
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger._initialise_model()
        enlarger.reaction_generator.ensure_estimator_initialized = MagicMock()

        no_interrupt_result = SimulationResult(
            t=np.array([0.0, 50.0, 100.0]),
            y=np.array([[10.0, 5.0, 2.0]]),
            species_labels=["s"],
            success=True,
            simulation_interrupted=False,
            max_char_rate=1.0,
            final_char_rate=0.5,
        )

        with patch("bees.enlarger.ODESimulator") as MockSim:
            instance = MockSim.return_value
            instance.simulate.return_value = no_interrupt_result
            enlarger.run()

        for sp in enlarger.model.core_species:
            if not sp.constant:
                assert sp.concentration == sp.initial_concentration, (
                    f"{sp.label}: concentration should have been reset "
                    f"to initial ({sp.initial_concentration}) at the top "
                    f"of each iteration, got {sp.concentration}"
                )
