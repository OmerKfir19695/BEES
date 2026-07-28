"""
Test reaction template module

Tests for bees/reaction_template.py covering:
- EC number parsing
- EC class determination
- Template determination from EC numbers
- Product inference
- Complete reaction creation

To run: pytest -v tests/test_reaction_template.py
"""

import pytest
from bees.reaction_template import (
    ECClass,
    ReactionTemplate,
    parse_ec_number,
    get_ec_class,
    determine_template_from_ec,
    infer_products,
    create_reaction_from_database
)


class TestECNumberParsing:
    """Test EC number parsing functions."""
    
    def test_parse_ec_number_with_prefix(self):
        """Test parsing EC number with 'EC ' prefix."""
        result = parse_ec_number("EC 1.2.3.4")
        assert result == (1, 2, 3, 4)
    
    def test_parse_ec_number_without_prefix(self):
        """Test parsing EC number without prefix."""
        result = parse_ec_number("2.7.1.1")
        assert result == (2, 7, 1, 1)
    
    def test_parse_ec_number_with_whitespace(self):
        """Test parsing handles whitespace."""
        result = parse_ec_number("  EC 3.1.3.1  ")
        assert result == (3, 1, 3, 1)
    
    def test_parse_ec_number_case_insensitive(self):
        """Test parsing is case-insensitive."""
        result1 = parse_ec_number("ec 1.1.1.1")
        result2 = parse_ec_number("EC 1.1.1.1")
        result3 = parse_ec_number("Ec 1.1.1.1")
        assert result1 == result2 == result3 == (1, 1, 1, 1)
    
    def test_parse_ec_number_invalid_format(self):
        """Test parsing raises error for invalid format."""
        with pytest.raises(ValueError, match="Invalid EC number format"):
            parse_ec_number("invalid")
        
        with pytest.raises(ValueError):
            parse_ec_number("1.2.3")  # Missing component
        
        with pytest.raises(ValueError):
            parse_ec_number("1.2.3.4.5")  # Too many components


class TestECClass:
    """Test EC class determination."""
    
    def test_get_ec_class_oxidoreductase(self):
        """Test EC class 1 (oxidoreductases)."""
        ec_class = get_ec_class("EC 1.1.1.1")
        assert ec_class == ECClass.OXIDOREDUCTASE
    
    def test_get_ec_class_transferase(self):
        """Test EC class 2 (transferases)."""
        ec_class = get_ec_class("EC 2.7.1.1")
        assert ec_class == ECClass.TRANSFERASE
    
    def test_get_ec_class_hydrolase(self):
        """Test EC class 3 (hydrolases)."""
        ec_class = get_ec_class("EC 3.1.3.1")
        assert ec_class == ECClass.HYDROLASE
    
    def test_get_ec_class_lyase(self):
        """Test EC class 4 (lyases)."""
        ec_class = get_ec_class("EC 4.2.1.1")
        assert ec_class == ECClass.LYASE
    
    def test_get_ec_class_isomerase(self):
        """Test EC class 5 (isomerases)."""
        ec_class = get_ec_class("EC 5.3.1.1")
        assert ec_class == ECClass.ISOMERASE
    
    def test_get_ec_class_ligase(self):
        """Test EC class 6 (ligases)."""
        ec_class = get_ec_class("EC 6.1.1.1")
        assert ec_class == ECClass.LIGASE
    
    def test_get_ec_class_translocase(self):
        """Test EC class 7 (translocases)."""
        ec_class = get_ec_class("EC 7.1.1.1")
        assert ec_class == ECClass.TRANSLOCASE
    
    def test_get_ec_class_invalid(self):
        """Test invalid EC number returns UNKNOWN."""
        ec_class = get_ec_class("invalid")
        assert ec_class == ECClass.UNKNOWN


class TestReactionTemplate:
    """Test ReactionTemplate dataclass."""
    
    def test_reaction_template_creation(self):
        """Test creating a reaction template."""
        template = ReactionTemplate(
            template_type="phosphorylation",
            ec_class=ECClass.TRANSFERASE,
            reactants=["Glucose", "ATP"],
            products=["Glucose-6-phosphate", "ADP"],
            stoichiometry={"Glucose": -1, "ATP": -1, "Glucose-6-phosphate": 1, "ADP": 1}
        )
        assert template.template_type == "phosphorylation"
        assert template.ec_class == ECClass.TRANSFERASE
        assert len(template.reactants) == 2
        assert len(template.products) == 2
    
    def test_reaction_template_defaults(self):
        """Test default values for optional fields."""
        template = ReactionTemplate(
            template_type="generic",
            ec_class=ECClass.UNKNOWN
        )
        assert template.reactants == []
        assert template.products == []
        assert template.stoichiometry == {}
        assert template.cofactors == []
        assert template.description == ""
        assert template.reversible is True

    def test_reaction_template_repr(self):
        """Test string representation."""
        template = ReactionTemplate(
            template_type="hydrolysis",
            ec_class=ECClass.HYDROLASE,
            reactants=["ATP"],
            products=["ADP", "Pi"]
        )
        repr_str = repr(template)
        assert "hydrolysis" in repr_str
        assert "HYDROLASE" in repr_str


class TestDetermineTemplate:
    """Test template determination from EC numbers."""
    
    def test_determine_template_oxidoreductase(self):
        """Test oxidoreductase template."""
        template = determine_template_from_ec("EC 1.1.1.1")
        assert template.ec_class == ECClass.OXIDOREDUCTASE
        assert "oxidation" in template.template_type.lower()
        assert template.reversible is True
    
    def test_determine_template_transferase_kinase(self):
        """Test transferase template (kinase)."""
        template = determine_template_from_ec("EC 2.7.1.1")
        assert template.ec_class == ECClass.TRANSFERASE
        assert template.template_type == "phosphorylation"
        assert "ATP" in template.cofactors
        assert "Mg2+" in template.cofactors
        assert template.reversible is True

    def test_determine_template_hydrolase(self):
        """Test hydrolase template."""
        template = determine_template_from_ec("EC 3.1.3.1")
        assert template.ec_class == ECClass.HYDROLASE
        assert "hydrolysis" in template.template_type.lower()
        assert template.reversible is True
    
    def test_determine_template_lyase(self):
        """Test lyase template."""
        template = determine_template_from_ec("EC 4.2.1.1")
        assert template.ec_class == ECClass.LYASE
        assert template.template_type == "elimination"
        assert template.reversible is True
    
    def test_determine_template_isomerase(self):
        """Test isomerase template."""
        template = determine_template_from_ec("EC 5.3.1.1")
        assert template.ec_class == ECClass.ISOMERASE
        assert template.template_type == "isomerization"
        assert template.reversible is True
    
    def test_determine_template_ligase(self):
        """Test ligase template."""
        template = determine_template_from_ec("EC 6.1.1.1")
        assert template.ec_class == ECClass.LIGASE
        assert template.template_type == "ligation"
        assert "ATP" in template.cofactors
    
    def test_determine_template_invalid_ec(self):
        """Test template for invalid EC number."""
        template = determine_template_from_ec("invalid")
        assert template.ec_class == ECClass.UNKNOWN
        assert template.template_type == "generic"


class TestInferProducts:
    """Test product inference."""
    
    def test_infer_products_from_database(self):
        """Test using database products when available."""
        template = ReactionTemplate(template_type="generic", ec_class=ECClass.UNKNOWN)
        products = infer_products(
            substrate="Glucose",
            enzyme_label="Hexokinase",
            template=template,
            database_products="Glucose-6-phosphate + ADP"
        )
        assert "Glucose-6-phosphate" in products
        assert "ADP" in products
    
    def test_infer_products_phosphorylation(self):
        """Test phosphorylation product inference."""
        template = ReactionTemplate(
            template_type="phosphorylation",
            ec_class=ECClass.TRANSFERASE
        )
        products = infer_products(
            substrate="Glucose",
            enzyme_label="Hexokinase",
            template=template,
            cofactor="ATP"
        )
        assert any("phosphate" in p.lower() for p in products)
        assert "ADP" in products
    
    def test_infer_products_hydrolysis(self):
        """Test hydrolysis product inference."""
        template = ReactionTemplate(
            template_type="hydrolysis",
            ec_class=ECClass.HYDROLASE
        )
        products = infer_products(
            substrate="ATP",
            enzyme_label="ATPase",
            template=template
        )
        assert len(products) > 0
        assert any("hydrolyzed" in p.lower() or "ADP" in p or "Pi" in p for p in products)
    
    def test_infer_products_oxidation(self):
        """Test oxidation product inference."""
        template = ReactionTemplate(
            template_type="oxidation_CH-OH",
            ec_class=ECClass.OXIDOREDUCTASE
        )
        products = infer_products(
            substrate="Lactate",
            enzyme_label="LDH",
            template=template,
            cofactor="NAD+"
        )
        assert any("oxidized" in p.lower() for p in products)
        assert "NADH" in products
    
    def test_infer_products_isomerization(self):
        """Test isomerization product inference."""
        template = ReactionTemplate(
            template_type="isomerization",
            ec_class=ECClass.ISOMERASE
        )
        products = infer_products(
            substrate="Glucose-6-phosphate",
            enzyme_label="PGI",
            template=template
        )
        assert len(products) > 0
        assert any("isomer" in p.lower() for p in products)
    
    def test_infer_products_generic(self):
        """Test generic product inference."""
        template = ReactionTemplate(
            template_type="generic",
            ec_class=ECClass.UNKNOWN
        )
        products = infer_products(
            substrate="Substrate",
            enzyme_label="Enzyme",
            template=template
        )
        assert len(products) > 0
        assert any("product" in p.lower() for p in products)


class TestCreateReactionFromDatabase:
    """Test complete reaction creation."""
    
    def test_create_reaction_kinase(self):
        """Test creating kinase reaction."""
        reaction = create_reaction_from_database(
            substrate="Glucose",
            enzyme_label="Hexokinase",
            ec_number="EC 2.7.1.1",
            cofactor="ATP",
            database_products="Glucose-6-phosphate + ADP"
        )
        
        assert reaction.template_type == "phosphorylation"
        assert "Glucose" in reaction.reactants
        assert "ATP" in reaction.reactants
        assert "Glucose-6-phosphate" in reaction.products
        assert "ADP" in reaction.products
        assert reaction.stoichiometry["Glucose"] == -1
        assert reaction.stoichiometry["Glucose-6-phosphate"] == 1
    
    def test_create_reaction_hydrolase(self):
        """Test creating hydrolase reaction."""
        reaction = create_reaction_from_database(
            substrate="ATP",
            enzyme_label="ATPase",
            ec_number="EC 3.1.3.1",
            cofactor="Mg2+",
            database_products="ADP + Pi"
        )
        
        assert reaction.ec_class == ECClass.HYDROLASE
        assert "ATP" in reaction.reactants
        assert "ADP" in reaction.products
        assert "Pi" in reaction.products
    
    def test_create_reaction_oxidoreductase(self):
        """Test creating oxidoreductase reaction."""
        reaction = create_reaction_from_database(
            substrate="Lactate",
            enzyme_label="LDH",
            ec_number="EC 1.1.1.27",
            cofactor="NAD+",
            database_products="Pyruvate + NADH"
        )
        
        assert reaction.ec_class == ECClass.OXIDOREDUCTASE
        assert "Lactate" in reaction.reactants
        assert "NAD+" in reaction.reactants
        assert "Pyruvate" in reaction.products
        assert "NADH" in reaction.products
        assert reaction.reversible is True
    
    def test_create_reaction_no_cofactor(self):
        """Test creating reaction without cofactor."""
        reaction = create_reaction_from_database(
            substrate="Sucrose",
            enzyme_label="Invertase",
            ec_number="EC 4.2.1.1",
            cofactor=None,
            database_products="Glucose + Fructose"
        )
        
        assert "Sucrose" in reaction.reactants
        assert len([r for r in reaction.reactants if r != "Sucrose"]) == 0
        assert "Glucose" in reaction.products
        assert "Fructose" in reaction.products
    
    def test_create_reaction_inferred_products(self):
        """Test creating reaction with inferred products (no database)."""
        reaction = create_reaction_from_database(
            substrate="UnknownSubstrate",
            enzyme_label="UnknownEnzyme",
            ec_number="EC 3.1.3.1",
            cofactor=None,
            database_products=None  # No database info
        )
        
        assert "UnknownSubstrate" in reaction.reactants
        assert len(reaction.products) > 0  # Should have inferred products
        assert reaction.stoichiometry["UnknownSubstrate"] == -1


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow_hexokinase(self):
        """Test complete workflow for hexokinase reaction."""
        # Parse EC number
        ec_class = get_ec_class("EC 2.7.1.1")
        assert ec_class == ECClass.TRANSFERASE
        
        # Determine template
        template = determine_template_from_ec("EC 2.7.1.1")
        assert template.template_type == "phosphorylation"
        
        # Create complete reaction
        reaction = create_reaction_from_database(
            substrate="Glucose",
            enzyme_label="Hexokinase",
            ec_number="EC 2.7.1.1",
            cofactor="ATP",
            database_products="Glucose-6-phosphate + ADP"
        )
        
        # Verify complete reaction
        assert len(reaction.reactants) == 2
        assert len(reaction.products) == 2
        assert sum(reaction.stoichiometry.values()) == 0  # Balanced
    
    def test_multiple_ec_classes(self):
        """Test that different EC classes produce different templates."""
        ec_numbers = [
            "EC 1.1.1.1",
            "EC 2.7.1.1",
            "EC 3.1.3.1",
            "EC 4.2.1.1",
            "EC 5.3.1.1",
            "EC 6.1.1.1"
        ]
        
        templates = [determine_template_from_ec(ec) for ec in ec_numbers]
        
        # All should have different EC classes
        ec_classes = [t.ec_class for t in templates]
        assert len(set(ec_classes)) == len(ec_classes)  # All unique
        
        # All should have template types
        assert all(t.template_type for t in templates)




