"""
Tests for the model_generator module.

To run: pytest -v tests/test_model_generator.py
"""

from unittest.mock import MagicMock
import pytest
from bees.reaction_generator import GeneratedReaction, ReactionGenerator
from bees.reaction_template import ReactionTemplate
from bees.reaction_template import ECClass


def _make_template(ec_class=ECClass.TRANSFERASE):
    return ReactionTemplate(
        template_type="phosphorylation",
        ec_class=ec_class,
    )


class TestGeneratedReaction:
    def test_repr(self):
        rxn = GeneratedReaction(
            enzyme_label="Hexokinase",
            substrate_label="Glucose",
            ec_number="EC 2.7.1.1",
            template=_make_template(),
            kinetics=None,
            reactant_labels=["Glucose", "ATP"],
            product_labels=["Glucose-6P", "ADP"],
            stoichiometry={"Glucose": -1, "ATP": -1, "Glucose-6P": 1, "ADP": 1},
        )
        s = repr(rxn)
        assert "Glucose" in s
        assert "ATP" in s
        assert "Glucose-6P" in s
        assert "ADP" in s

    def test_with_rate_law(self):
        kin = MagicMock()
        rxn = GeneratedReaction(
            enzyme_label="E",
            substrate_label="S",
            ec_number="EC 1.1.1.1",
            template=_make_template(),
            kinetics=kin,
            reactant_labels=["S"],
            product_labels=["P"],
            stoichiometry={"S": -1, "P": 1},
            rate_law="Michaelis-Menten",
        )
        assert rxn.rate_law == "Michaelis-Menten"


class TestReactionGenerator:
    def test_init(self):
        bees_obj = MagicMock()
        logger = MagicMock()
        mg = ReactionGenerator(bees_obj, logger, "/tmp/out")
        assert mg.bees_object is bees_obj
        assert mg.logger is logger
        assert mg.output_directory == "/tmp/out"
        assert mg.reactions == []
        assert mg.kinetic_db is None
        assert mg.kinetics_estimator is None

    def test_check_heavy_atom_balance_balanced(self):
        """
        Stoichiometry is balanced if total heavy atoms on both sides match.
        """
        mg = ReactionGenerator(MagicMock(), MagicMock(), "/tmp/out")

        smiles_map = {
            "A": "CCO",   # 3 heavy atoms
            "B": "CCO",   # 3 heavy atoms
        }

        mg._resolve_smiles_for_compound = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda compound_label, kinetic_data, substrate_label: smiles_map.get(compound_label)
        )

        stoich = {"A": -1, "B": 1}
        assert mg._check_heavy_atom_balance(stoich, kinetic_data=None, substrate_label="A") is True

    def test_check_heavy_atom_balance_unbalanced(self):
        mg = ReactionGenerator(MagicMock(), MagicMock(), "/tmp/out")

        smiles_map = {
            "A": "CCO",  # 3
            "B": "CC",   # 2
        }
        mg._resolve_smiles_for_compound = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda compound_label, kinetic_data, substrate_label: smiles_map.get(compound_label)
        )

        stoich = {"A": -1, "B": 1}
        assert mg._check_heavy_atom_balance(stoich, kinetic_data=None, substrate_label="A") is False

    def test_check_heavy_atom_balance_missing_smiles_returns_none(self):
        mg = ReactionGenerator(MagicMock(), MagicMock(), "/tmp/out")
        mg._resolve_smiles_for_compound = MagicMock(return_value=None)  # type: ignore[method-assign]

        stoich = {"A": -1, "B": 1}
        assert mg._check_heavy_atom_balance(stoich, kinetic_data=None, substrate_label="A") is None

    def test_check_heavy_atom_balance_invalid_coeff_returns_none(self):
        mg = ReactionGenerator(MagicMock(), MagicMock(), "/tmp/out")
        mg._resolve_smiles_for_compound = MagicMock(return_value="CC")  # type: ignore[method-assign]

        stoich = {"A": "not_a_number", "B": 1}
        assert mg._check_heavy_atom_balance(stoich, kinetic_data=None, substrate_label="A") is None
