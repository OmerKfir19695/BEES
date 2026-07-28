"""Tests for reverse-direction product Km queries in ReactionGenerator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from bees.reaction_generator import ReactionGenerator


def _make_generator_with_estimator(captured: dict) -> ReactionGenerator:
    bees = MagicMock()
    bees.enzymes = []
    bees.species = []
    bees.settings = MagicMock()
    logger = MagicMock()
    gen = ReactionGenerator(bees, logger, output_directory="/tmp")

    def _estimate(**kwargs):
        captured["reactant_smiles"] = dict(kwargs.get("reactant_smiles") or {})
        est = MagicMock()
        est.km_per_substrate = {lab: 0.05 for lab in captured["reactant_smiles"]}
        est.compound_smiles = dict(captured["reactant_smiles"])
        return est

    gen.kinetics_estimator = MagicMock()
    gen.kinetics_estimator.estimate = _estimate
    return gen


def test_regulatory_cofactor_product_queried_buffered_skipped():
    """CoA gets a reverse Km query; H2O and CO2 do not."""
    captured: dict = {}
    gen = _make_generator_with_estimator(captured)

    rxn = SimpleNamespace(
        enzyme_label="FabD",
        substrate_label="Malonyl-CoA",
        product_labels=["malonyl-[ACP]", "Coenzyme A", "H2O", "Carbon dioxide"],
        stoichiometry={
            "holo-[ACP]": -1,
            "Malonyl-CoA": -1,
            "malonyl-[ACP]": 1,
            "Coenzyme A": 1,
            "H2O": 1,
            "Carbon dioxide": 1,
        },
        ec_number="EC 2.3.1.39",
    )
    smiles_map = {
        "malonyl-[ACP]": "CC(=O)CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)O",
        "Coenzyme A": (
            "CC(C)(COP(O)(=O)OP(O)(=O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(O)(O)=O)"
            "N1C=NC2=C1N=CN=C2N)C(O)C(=O)NCCC(=O)NCCS"
        ),
        "H2O": "O",
        "Carbon dioxide": "O=C=O",
    }

    product_kms, _ = gen._lookup_product_kms_via_reverse_query(
        reaction=rxn,
        stoich=rxn.stoichiometry,
        ec_numbers_to_try=["EC 2.3.1.39"],
        temp_range=None,
        ph_range=None,
        provided_species_labels_lc=set(),
        enzyme_sequence="MKT",
        smiles_map=smiles_map,
        substitutor=None,
    )

    queried = set(captured.get("reactant_smiles", {}))
    assert "Coenzyme A" in queried
    assert "malonyl-[ACP]" in queried
    assert "H2O" not in queried
    assert "Carbon dioxide" not in queried
    assert "Coenzyme A" in product_kms
    assert "H2O" not in product_kms


def test_nadp_product_queried_for_fabg_like():
    """NADP (regulatory) is queried; H+ (buffered) is not."""
    captured: dict = {}
    gen = _make_generator_with_estimator(captured)
    rxn = SimpleNamespace(
        enzyme_label="FabG",
        substrate_label="3-oxobutanoyl-[ACP]",
        product_labels=["(3R)-hydroxybutanoyl-[ACP]", "NADP"],
        stoichiometry={
            "3-oxobutanoyl-[ACP]": -1,
            "NADPH": -1,
            "H+": -1,
            "(3R)-hydroxybutanoyl-[ACP]": 1,
            "NADP": 1,
        },
        ec_number="EC 1.1.1.100",
    )
    smiles_map = {
        "(3R)-hydroxybutanoyl-[ACP]": "CC(O)CC(=O)S",
        "NADP": (
            "NC(=O)C1=C[N+](=CC=C1)C1OC(COP(=O)(O)OP(=O)(O)OCC2OC("
            "N3C=NC4=C(N)N=CN=C43)C(OP(=O)(O)O)C2O)C(O)C1O"
        ),
        "H+": "[H+]",
    }
    product_kms, _ = gen._lookup_product_kms_via_reverse_query(
        reaction=rxn,
        stoich=rxn.stoichiometry,
        ec_numbers_to_try=["EC 1.1.1.100"],
        temp_range=None,
        ph_range=None,
        provided_species_labels_lc=set(),
        enzyme_sequence="MNF",
        smiles_map=smiles_map,
        substitutor=None,
    )
    assert "NADP" in captured["reactant_smiles"]
    assert "H+" not in captured["reactant_smiles"]
    assert product_kms.get("NADP", 0) > 0
