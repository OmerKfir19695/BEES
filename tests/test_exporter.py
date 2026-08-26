"""
Tests for the Exporter module.

To run:  pytest -v tests/test_exporter.py
"""

import csv
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bees.enlarger import IterativeEnlarger
from bees.exporter import EnlargerExporter


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


@pytest.fixture
def mock_bees_object():
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
        filter_reactions=False,
    )

    species = [
        SimpleNamespace(
            label="S",
            concentration=10.0,
            reactive=True,
            solvent=False,
            constant=False,
        ),
    ]
    enzymes = [
        SimpleNamespace(
            label="Enzyme",
            concentration=0.01,
            reactive=True,
            solvent=False,
            ecnumber="EC 1.1.1.1",
            constant=True,
        ),
    ]

    return SimpleNamespace(
        settings=settings,
        species=species,
        enzymes=enzymes,
        project="TestProject",
        database=SimpleNamespace(name="db"),
        environment=SimpleNamespace(temperature=298.15, pH=7.0),
    )


@pytest.fixture
def mock_reaction_generator():
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


class TestEnlargerExporter:
    def _make_exporter(self, enlarger: IterativeEnlarger, mock_bees_object, output_dir, logger):
        return EnlargerExporter(
            model=enlarger.model,
            profiles=enlarger._profiles,
            output_directory=output_dir,
            logger=logger,
            reaction_id_by_sig=enlarger._reaction_id_by_sig,
            reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
            reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
            reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
            iteration_summaries=enlarger._iteration_summaries,
            save_simulation_plots=getattr(
                mock_bees_object.settings, "save_simulation_plots", False
            ),
            plot_max_species=getattr(mock_bees_object.settings, "plot_max_species", None),
            plot_exclude_enzymes=getattr(
                mock_bees_object.settings, "plot_exclude_enzymes", True
            ),
            plot_exclude_cofactors=getattr(
                mock_bees_object.settings, "plot_exclude_cofactors", True
            ),
            bees_object=mock_bees_object,
        )

    def test_export_profiles(self, mock_bees_object, mock_reaction_generator, output_dir):
        """Ensure export_simulation_profiles writes a file."""
        mock_bees_object.settings.save_simulation_profiles = True
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger.run()
        exporter = self._make_exporter(enlarger, mock_bees_object, output_dir, logger)
        path = exporter.export_simulation_profiles()
        if path is not None:
            assert os.path.exists(path)

    def test_export_flux_analysis(self, mock_bees_object, mock_reaction_generator, output_dir):
        """Ensure export_flux_analysis writes a file."""
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger.run()
        exporter = self._make_exporter(enlarger, mock_bees_object, output_dir, logger)
        path = exporter.export_flux_analysis()
        assert path is not None
        assert os.path.exists(path)

    def test_export_core_edge_reaction_species_csvs(
        self, mock_bees_object, mock_reaction_generator, output_dir
    ):
        """Ensure core/edge CSV exports are written and include section headers."""
        logger = MagicMock()
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mock_reaction_generator,
            logger=logger,
            output_directory=output_dir,
        )
        enlarger.run()
        exporter = self._make_exporter(enlarger, mock_bees_object, output_dir, logger)
        paths = exporter.export_core_edge_reaction_species_csvs()
        assert "core" in paths and "edge" in paths
        assert os.path.exists(paths["core"])
        assert os.path.exists(paths["edge"])

        with open(paths["core"], "r", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][:5] == ["index", "reaction_id", "template", "ec_number", "family"]
        assert any(row and row[0] == "core species" for row in rows)

        with open(paths["edge"], "r", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][:5] == ["index", "reaction_id", "template", "ec_number", "family"]
        assert any(row and row[0] == "edge species" for row in rows)

# ---------------------------------------------------------------------------
# Phase 1 COPASI-compatibility tests:
#   - kinetic laws emit max(0, S) guards (prevents runaway negative rates
#     when a BDF corrector momentarily pushes a substrate below zero)
#   - SBML `reversible` attribute matches the kinetic law actually emitted
#     (fallback-thermo reactions must be reversible="false")
# ---------------------------------------------------------------------------

def _make_reaction_with_thermo(
    *,
    enzyme_label="EnzymeR",
    reactant_labels=("A",),
    product_labels=("B",),
    stoichiometry=None,
    km=1.0,
    kcat=10.0,
    thermo_source="equilibrator",
    irreversible=False,
    keq=100.0,
    kcat_rev=0.1,
    dgr=-10.0,
    template_reversible=True,
):
    """Build a reaction with a real-enough .thermo attribute for SBML export."""
    rxn = _make_reaction(
        enzyme_label=enzyme_label,
        reactant_labels=list(reactant_labels),
        product_labels=list(product_labels),
        stoichiometry=stoichiometry
        or {**{r: -1 for r in reactant_labels}, **{p: 1 for p in product_labels}},
        km=km,
        kcat=kcat,
    )
    thermo = SimpleNamespace(
        source=thermo_source,
        irreversible=irreversible,
        keq=keq,
        kcat_rev=kcat_rev,
        dgr_prime_kJmol=dgr,
        sigma_kJmol=1.0,
    )
    rxn.thermo = thermo
    rxn.template.reversible = template_reversible
    return rxn


def _export_sbml_with_reactions(
    rxns, output_dir, mock_bees_object, *, extra_species=(), strict_invariant=True,
    exporter_logger=None,
):
    """Build an enlarger, attach reactions and species, export SBML, return path.

    ``exporter_logger`` overrides the MagicMock logger on the exporter — pass a
    strict stub to assert the export uses single-arg log calls (BEES's real
    ``Logger`` signature), which a MagicMock would silently swallow.
    """
    from bees.core_edge_model import SpeciesData

    mg = MagicMock()
    mg.generate_reactions.return_value = []
    mg._generate_reactions.return_value = []

    enlarger = IterativeEnlarger(
        bees_object=mock_bees_object,
        reaction_generator=mg,
        logger=MagicMock(),
        output_directory=output_dir,
    )

    # Add every species referenced by any reaction (plus the enzyme).
    species_seen = set()
    for rxn in rxns:
        for lab in list(rxn.reactant_labels) + list(rxn.product_labels):
            if lab in species_seen:
                continue
            species_seen.add(lab)
            enlarger.model.add_core_species(SpeciesData(label=lab, concentration=1.0))
        enz_lab = rxn.enzyme_label
        if enz_lab not in species_seen:
            species_seen.add(enz_lab)
            enlarger.model.add_core_species(
                SpeciesData(label=enz_lab, concentration=0.01, is_enzyme=True)
            )
        enlarger.model.core_reactions.append(rxn)
    for sd in extra_species:
        enlarger.model.add_core_species(sd)


    exporter = EnlargerExporter(
        model=enlarger.model,
        profiles=enlarger._profiles,
        output_directory=output_dir,
        logger=exporter_logger if exporter_logger is not None else MagicMock(),
        reaction_id_by_sig=enlarger._reaction_id_by_sig,
        reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
        reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
        reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
        iteration_summaries=enlarger._iteration_summaries,
        bees_object=mock_bees_object,
    )
    path = exporter.export_sbml(filename="model.xml", strict_invariant=strict_invariant)
    return path


def _walk_ast_types(ast_node):
    """Yield the type integer of every node in an libsbml ASTNode tree."""
    if ast_node is None:
        return
    yield ast_node.getType()
    for i in range(ast_node.getNumChildren()):
        yield from _walk_ast_types(ast_node.getChild(i))


def test_kinetic_law_has_nonneg_guards_after_disk_roundtrip(
    mock_bees_object, output_dir
):
    """
    Phase 1, Step 1 verification: write SBML to disk, read it back, and walk
    the MathML AST of every kineticLaw. Every kinetic law must contain at
    least one AST_FUNCTION_MAX node (the `max(0, S)` guard) so a BDF overshoot
    to negative S cannot blow up the formula.
    """
    libsbml = pytest.importorskip("libsbml")

    rxn = _make_reaction_with_thermo(
        enzyme_label="EnzymeR",
        reactant_labels=("A",),
        product_labels=("B",),
    )

    path = _export_sbml_with_reactions([rxn], output_dir, mock_bees_object)
    assert path is not None and os.path.exists(path)

    doc = libsbml.SBMLReader().readSBMLFromFile(path)
    # No XML parse errors (consistency warnings are OK; we only block on parse errors).
    n_parse_errors = sum(
        1
        for i in range(doc.getNumErrors())
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    )
    assert n_parse_errors == 0, "model.xml has XML parse errors after export"

    m = doc.getModel()
    assert m.getNumReactions() >= 1
    saw_max = False
    for i in range(m.getNumReactions()):
        kl = m.getReaction(i).getKineticLaw()
        assert kl is not None, "every reaction should have a kineticLaw"
        ast = kl.getMath()
        assert ast is not None
        types = list(_walk_ast_types(ast))
        if libsbml.AST_FUNCTION_MAX in types:
            saw_max = True
    assert saw_max, (
        "No AST_FUNCTION_MAX in any kineticLaw AST — the max(0, S) guards "
        "are missing from the exported SBML. A BDF overshoot will runaway."
    )


def test_fallback_thermo_reaction_is_irreversible_in_sbml(
    mock_bees_object, output_dir
):
    """
    Phase 1, Step 2 verification: a reaction whose thermo is `source="fallback"`
    (eQuilibrator could not resolve it) must be exported with
    `reversible="false"`, because the kinetic law emitted is forward-only MM.
    SBML attribute and kinetic-law form must agree, or COPASI raises
    "never negative — unexpected for reversible reaction".
    """
    libsbml = pytest.importorskip("libsbml")

    rxn_fallback = _make_reaction_with_thermo(
        enzyme_label="EnzymeF",
        reactant_labels=("X",),
        product_labels=("Y",),
        thermo_source="fallback",
        irreversible=True,
        keq=float("nan"),
        kcat_rev=None,
        dgr=float("nan"),
    )

    # Toy 1-reaction fixture: Y has no consumer. The export invariant is
    # designed to catch this in real models; relax it here so the test can
    # focus on its actual concern (reversibility flag on fallback thermo).
    path = _export_sbml_with_reactions(
        [rxn_fallback], output_dir, mock_bees_object, strict_invariant=False,
    )
    assert path is not None and os.path.exists(path)

    doc = libsbml.SBMLReader().readSBMLFromFile(path)
    m = doc.getModel()
    assert m.getNumReactions() == 1
    sbml_rxn = m.getReaction(0)
    assert sbml_rxn.getReversible() is False, (
        "Fallback-thermo reaction must be reversible=false in SBML to match "
        "its forward-only kinetic law"
    )


def test_kinetic_law_is_scaled_by_compartment_volume(mock_bees_object, output_dir):
    """
    SBML defines a kineticLaw as a *substance* rate (amount/time) and a solver
    computes dC/dt = kineticLaw / V. BEES builds *concentration*-rate formulas
    (mM/s), so each kineticLaw must be multiplied by the compartment volume to
    survive that division — mirroring RMG's liquid solver (`core_species_rates *
    V`). Without it the exported model integrates 1/V times too fast in COPASI.

    Guards against regressing to a bare concentration-rate formula or a
    compartment size that does not cancel.
    """
    libsbml = pytest.importorskip("libsbml")

    rxn = _make_reaction_with_thermo(
        enzyme_label="EnzymeV",
        reactant_labels=("A",),
        product_labels=("B",),
    )
    path = _export_sbml_with_reactions([rxn], output_dir, mock_bees_object)
    assert path is not None and os.path.exists(path)

    doc = libsbml.SBMLReader().readSBMLFromFile(path)
    m = doc.getModel()

    comp = m.getCompartment("compartment1")
    assert comp is not None and comp.getSize() == 1.0, (
        "compartment 'compartment1' must have size 1.0 so amount (mmol) == "
        "concentration (mM) and dC/dt equals the bare rate expression"
    )

    assert m.getNumReactions() >= 1
    for i in range(m.getNumReactions()):
        kl = m.getReaction(i).getKineticLaw()
        assert kl is not None, "every reaction should have a kineticLaw"
        ci_names = {n for n in _walk_ast_ci_names(kl.getMath())}
        assert "compartment1" in ci_names, (
            "kineticLaw must reference the compartment 'compartment1' as a volume "
            "factor; otherwise the exported model runs 1/V too fast in COPASI"
        )


def _walk_ast_ci_names(ast_node):
    """Yield the names of every <ci> (identifier) node in an AST subtree."""
    if ast_node is None:
        return
    if ast_node.isName() and ast_node.getName():
        yield ast_node.getName()
    for i in range(ast_node.getNumChildren()):
        yield from _walk_ast_ci_names(ast_node.getChild(i))


def test_feedback_inhibition_exported_to_kinetic_law(mock_bees_object, output_dir):
    """
    A reaction carrying `feedback_inhibitors` (Enzyme.feedback_inhibition, the
    simulator's Overlay 3) must export the multiplier ∏ 1/(1+([I]/Ki)^h) into its
    kineticLaw, declare each inhibitor as a modifier, and emit Ki/hill parameters.
    Without this the SBML shows the baseline shape, not the feedback shape.
    """
    libsbml = pytest.importorskip("libsbml")
    from bees.core_edge_model import SpeciesData

    rxn = _make_reaction_with_thermo(
        enzyme_label="FabH",
        reactant_labels=("A",),
        product_labels=("B",),
        thermo_source="fallback",
        irreversible=True,
        keq=float("nan"),
        kcat_rev=None,
        dgr=float("nan"),
    )
    rxn.feedback_inhibitors = {"inhibitor_x": (0.005, 1.0)}

    path = _export_sbml_with_reactions(
        [rxn], output_dir, mock_bees_object, strict_invariant=False,
        extra_species=(SpeciesData(label="inhibitor_x", concentration=0.0),),
    )
    doc = libsbml.SBMLReader().readSBMLFromFile(path)
    m = doc.getModel()
    rx = m.getReaction(0)

    mods = {rx.getModifier(i).getSpecies() for i in range(rx.getNumModifiers())}
    assert "inhibitor_x" in mods, "inhibitor must be a modifierSpeciesReference"

    ki = next(
        (m.getParameter(i) for i in range(m.getNumParameters())
         if m.getParameter(i).getId().startswith("Ki_fb_")),
        None,
    )
    assert ki is not None and ki.getValue() == 0.005, "Ki parameter must be emitted"

    ci_names = set(_walk_ast_ci_names(rx.getKineticLaw().getMath()))
    assert "inhibitor_x" in ci_names, "kineticLaw must reference the inhibitor"
    assert any(n.startswith("Ki_fb_") for n in ci_names), (
        "kineticLaw must reference the feedback Ki parameter"
    )


class _StrictLogger:
    """Mimics bees.logger.Logger: every level takes a single message string.

    A MagicMock accepts `logger.debug(msg, a, b)` silently; the real BEES logger
    raises TypeError. This stub reproduces that so tests catch %-style log calls.
    """

    def _check(self, message, *args):
        assert not args, (
            "BEES logger takes a single pre-formatted message; use an f-string, "
            f"not %-args. Got extra args: {args!r}"
        )

    debug = info = warning = error = _check


def test_export_uses_single_arg_log_calls_on_missing_inhibitor(
    mock_bees_object, output_dir
):
    """
    Regression: the feedback-export path logs when an inhibitor is not a model
    species. That log call (and all others on the export path) must use BEES's
    single-arg Logger signature, not %-style args. Exercised with a strict logger
    and a feedback inhibitor that is deliberately absent from the species set.
    """
    pytest.importorskip("libsbml")

    rxn = _make_reaction_with_thermo(
        enzyme_label="FabH",
        reactant_labels=("A",),
        product_labels=("B",),
        thermo_source="fallback",
        irreversible=True,
        keq=float("nan"),
        kcat_rev=None,
        dgr=float("nan"),
    )
    # Inhibitor "ghost_acid" is never added as a species -> triggers the debug log.
    rxn.feedback_inhibitors = {"ghost_acid": (0.005, 1.0)}

    # Must not raise (a %-style log call would raise via _StrictLogger).
    path = _export_sbml_with_reactions(
        [rxn], output_dir, mock_bees_object, strict_invariant=False,
        exporter_logger=_StrictLogger(),
    )
    assert path is not None and os.path.exists(path)


def test_free_fatty_acid_global_quantities_exported(mock_bees_object, output_dir):
    """
    The exporter must emit two global quantities (parameters + assignment rules)
    so any SBML tool reads the released free fatty acid directly:
      total_free_FA_uM        = 1000·Σ[acid]
      palmitic_equivalents_uM = 1000·Σ(carbons/16)·[acid]   (Yu-2011 metric)
    """
    libsbml = pytest.importorskip("libsbml")

    rxn_c8 = _make_reaction_with_thermo(
        enzyme_label="TesA8", reactant_labels=("octanoyl-[ACP]",),
        product_labels=("octanoate",), thermo_source="fallback",
        irreversible=True, keq=float("nan"), kcat_rev=None, dgr=float("nan"),
    )
    rxn_c16 = _make_reaction_with_thermo(
        enzyme_label="TesA16", reactant_labels=("hexadecanoyl-[ACP]",),
        product_labels=("hexadecanoate",), thermo_source="fallback",
        irreversible=True, keq=float("nan"), kcat_rev=None, dgr=float("nan"),
    )
    path = _export_sbml_with_reactions(
        [rxn_c8, rxn_c16], output_dir, mock_bees_object, strict_invariant=False,
    )
    doc = libsbml.SBMLReader().readSBMLFromFile(path)
    m = doc.getModel()

    for pid in ("total_free_FA_uM", "palmitic_equivalents_uM"):
        p = m.getParameter(pid)
        assert p is not None and p.getConstant() is False, (
            f"{pid} must be a non-constant global quantity"
        )

    rule_vars = {m.getRule(i).getVariable() for i in range(m.getNumRules())}
    assert {"total_free_FA_uM", "palmitic_equivalents_uM"} <= rule_vars

    palm_rule = next(
        m.getRule(i) for i in range(m.getNumRules())
        if m.getRule(i).getVariable() == "palmitic_equivalents_uM"
    )
    ci_names = set(_walk_ast_ci_names(palm_rule.getMath()))
    assert {"octanoate", "hexadecanoate"} <= ci_names, (
        "palmitic-equivalents rule must weight every free-acid species"
    )


# ---------------------------------------------------------------------------
# Production-only-species invariant (RMG-style edge isolation companion).
# A non-boundary, non-terminal species that appears as a product in some
# reaction but never as a reactant will accumulate without bound under
# integration. The export must refuse to write such a model by default and
# only warn when explicitly demoted.
# ---------------------------------------------------------------------------

class TestProductionOnlyInvariant:
    def _make_pair(self, *, product_label):
        """A -> product_label, plus a downstream consumer of A so A itself isn't a leak."""
        # consumer keeps A from being production-only
        seed = _make_reaction_with_thermo(
            reactant_labels=("Seed",),
            product_labels=("A",),
            template_reversible=False,
            irreversible=True,
        )
        leaker = _make_reaction_with_thermo(
            enzyme_label="EnzL",
            reactant_labels=("A",),
            product_labels=(product_label,),
            template_reversible=False,
            irreversible=True,
        )
        return [seed, leaker]

    def test_refuses_production_only_species(self, output_dir, mock_bees_object):
        # "Mystery" is produced (by leaker) but never consumed and is not on
        # any allowlist or suffix family → must raise.
        rxns = self._make_pair(product_label="MysterySink")
        with pytest.raises(ValueError, match="produced but never consumed"):
            _export_sbml_with_reactions(rxns, output_dir, mock_bees_object)

    def test_allows_free_fatty_acid_terminals(self, output_dir, mock_bees_object):
        # "hexanoate" matches the *ate suffix → terminal-OK.
        rxns = self._make_pair(product_label="hexanoate")
        path = _export_sbml_with_reactions(rxns, output_dir, mock_bees_object)
        assert path is not None and os.path.exists(path)

    def test_allows_general_cofactor_terminals(self, output_dir, mock_bees_object):
        # CoA is a general cofactor — production-only cofactors are not leaks.
        rxns = self._make_pair(product_label="Coenzyme A")
        path = _export_sbml_with_reactions(rxns, output_dir, mock_bees_object)
        assert path is not None and os.path.exists(path)

    def test_strict_invariant_false_demotes_to_warning(
        self, output_dir, mock_bees_object,
    ):
        # Same broken model as test_refuses…, but called with strict=False.
        # Build the exporter manually so we can pass strict_invariant=False.
        from bees.core_edge_model import SpeciesData
        from bees.enlarger import IterativeEnlarger

        rxns = self._make_pair(product_label="MysterySink")
        mg = MagicMock()
        mg.generate_reactions.return_value = []
        mg._generate_reactions.return_value = []
        enlarger = IterativeEnlarger(
            bees_object=mock_bees_object,
            reaction_generator=mg,
            logger=MagicMock(),
            output_directory=output_dir,
        )
        species_seen = set()
        for rxn in rxns:
            for lab in list(rxn.reactant_labels) + list(rxn.product_labels):
                if lab in species_seen:
                    continue
                species_seen.add(lab)
                enlarger.model.add_core_species(SpeciesData(label=lab, concentration=1.0))
            enz_lab = rxn.enzyme_label
            if enz_lab not in species_seen:
                species_seen.add(enz_lab)
                enlarger.model.add_core_species(
                    SpeciesData(label=enz_lab, concentration=0.01, is_enzyme=True)
                )
            enlarger.model.core_reactions.append(rxn)

        logger = MagicMock()
        exporter = EnlargerExporter(
            model=enlarger.model,
            profiles=enlarger._profiles,
            output_directory=output_dir,
            logger=logger,
            reaction_id_by_sig=enlarger._reaction_id_by_sig,
            reaction_first_seen_iter=enlarger._reaction_first_seen_iter,
            reaction_core_enter_iter=enlarger._reaction_core_enter_iter,
            reaction_obj_by_sig=enlarger._reaction_obj_by_sig,
            iteration_summaries=enlarger._iteration_summaries,
            bees_object=mock_bees_object,
        )
        path = exporter.export_sbml(filename="model.xml", strict_invariant=False)
        assert path is not None and os.path.exists(path)
        # The warning should have been logged.
        warning_msgs = [str(c) for c in logger.warning.call_args_list]
        assert any("produced but never consumed" in m for m in warning_msgs), (
            f"expected production-only warning, got: {warning_msgs}"
        )
