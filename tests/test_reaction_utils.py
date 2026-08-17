"""
Tests for the reaction_utils module.

To run: pytest -v tests/test_reaction_utils.py
"""

import pytest

from bees.reaction_utils import (
    get_ec_aliases,
    get_enzyme_domain_cofactors,
    check_reactant_availability,
    validate_reaction_reactants,
)


class TestGetEcAliases:
    def test_ec_with_alias(self):
        result = get_ec_aliases("EC 2.3.1.85")
        assert "EC 2.3.1.85" in result
        assert "EC 2.3.1.86" in result

    def test_ec_without_alias(self):
        result = get_ec_aliases("EC 2.7.1.1")
        assert result == ["EC 2.7.1.1"]


class TestGetEnzymeDomainCofactors:
    def test_synthase_returns_acp(self):
        result = get_enzyme_domain_cofactors("fatty acid synthase")
        assert "acp" in result
        assert "acyl carrier protein" in result

    def test_carboxylase_returns_biotin(self):
        result = get_enzyme_domain_cofactors("pyruvate carboxylase")
        assert "biotin" in result

    def test_no_match(self):
        result = get_enzyme_domain_cofactors("hexokinase")
        assert result == []

    def test_multiple_patterns(self):
        result = get_enzyme_domain_cofactors("acetyl-CoA carboxylase synthase")
        assert "biotin" in result
        assert "acp" in result


class TestCheckReactantAvailability:
    def test_high_energy_cofactor_requires_explicit_presence(self):
       
        avail, reason = check_reactant_availability("ATP", set())
        assert avail is False
        assert reason is None
    
    def test_direct_match(self):
        avail, reason = check_reactant_availability("Glucose", {"glucose"})
        assert avail is True
        assert reason == "direct_match"

    def test_domain_cofactor_carboxylase(self):
        avail, reason = check_reactant_availability(
            "biotin",
            set(),
            enzyme_label="pyruvate carboxylase",
        )
        assert avail is True
        assert reason == "domain_cofactor"

    def test_not_available(self):
        avail, reason = check_reactant_availability(
            "UnknownCompound",
            set(),
            enzyme_label=None,
        )
        assert avail is False

    def test_reactant_stripped_and_lowercased(self):
        avail, reason = check_reactant_availability("  GLUCOSE  ", {"glucose"})
        assert avail is True
        assert reason == "direct_match"

    def test_sibling_acyl_coa_not_available_via_shared_category(self):
        # Mimic enlarger expansion: Acetyl-CoA puts parent categories into available.
        from bees.common import get_ontology_equivalents

        available = set(get_ontology_equivalents("Acetyl-CoA"))
        assert "an acyl-coa" in {a.lower() for a in available} or "an acyl-CoA" in available

        avail, reason = check_reactant_availability(
            "propanoyl-CoA",
            available,
            enzyme_label="FabH",
        )
        assert avail is False
        assert reason is None

        avail_acetyl, reason_acetyl = check_reactant_availability(
            "Acetyl-CoA",
            available,
            enzyme_label="FabH",
        )
        assert avail_acetyl is True
        assert reason_acetyl in {"direct_match", "ontology"}

    def test_category_reactant_available_when_member_present(self):
        avail, reason = check_reactant_availability(
            "an acyl-CoA",
            {"acetyl-coa"},
            enzyme_label="FabH",
        )
        assert avail is True
        assert reason == "ontology"


class TestValidateReactionReactants:
    def test_all_available(self):
        ok, missing = validate_reaction_reactants(
            ["glucose", "atp"],
            {"glucose", "atp"},
        )
        assert ok is True
        assert missing == []

    def test_one_missing(self):
        ok, missing = validate_reaction_reactants(
            ["glucose", "unknown_x"],
            {"glucose"},
        )
        assert ok is False
        assert "unknown_x" in missing

    def test_multiple_missing(self):
        ok, missing = validate_reaction_reactants(
            ["a", "b", "c"],
            set(),
        )
        assert ok is False
        assert set(missing) == {"a", "b", "c"}

    def test_cofactors_implicitly_available(self):
        ok, missing = validate_reaction_reactants(
            ["glucose", "atp", "h2o"],
            {"glucose"},
        )
        # H2O is implicitly available, but ATP is not; reaction should be rejected.
        assert ok is False
        assert set(missing) == {"atp"}
