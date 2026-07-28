"""
Tests for the CoreEdgeModel module.

To run:  pytest -v tests/test_core_edge_model.py
"""

import pytest

from bees.core_edge_model import CoreEdgeModel, SpeciesData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return CoreEdgeModel()


@pytest.fixture
def species_a():
    return SpeciesData(label="Glucose", concentration=5.0, initial_concentration=5.0)


@pytest.fixture
def species_b():
    return SpeciesData(label="Fructose-6P", concentration=0.0, initial_concentration=0.0)


@pytest.fixture
def species_c():
    return SpeciesData(label="ATP", concentration=2.0, initial_concentration=2.0, constant=True)


# ---------------------------------------------------------------------------
# Core species tests
# ---------------------------------------------------------------------------

class TestCoreSpecies:
    def test_add_core_species(self, model, species_a):
        model.add_core_species(species_a)
        assert model.is_core_species("Glucose")
        assert model.is_core_species("glucose")  # case-insensitive
        assert len(model.core_species) == 1

    def test_add_duplicate_core_species(self, model, species_a):
        model.add_core_species(species_a)
        model.add_core_species(species_a)
        assert len(model.core_species) == 1

    def test_get_core_species_by_label(self, model, species_a):
        model.add_core_species(species_a)
        sp = model.get_core_species_by_label("glucose")
        assert sp is not None
        assert sp.label == "Glucose"

    def test_get_core_species_by_label_missing(self, model):
        sp = model.get_core_species_by_label("water")
        assert sp is None


# ---------------------------------------------------------------------------
# Edge species tests
# ---------------------------------------------------------------------------

class TestEdgeSpecies:
    def test_add_edge_species(self, model, species_b):
        model.add_edge_species(species_b)
        assert model.is_edge_species("Fructose-6P")
        assert len(model.edge_species) == 1

    def test_add_edge_species_already_in_core(self, model, species_a):
        model.add_core_species(species_a)
        model.add_edge_species(species_a)  # should be no-op
        assert len(model.edge_species) == 0
        assert len(model.core_species) == 1

    def test_promote_species_to_core(self, model, species_b):
        model.add_edge_species(species_b)
        result = model.promote_species_to_core("Fructose-6P")
        assert result is not None
        assert result.label == "Fructose-6P"
        assert model.is_core_species("Fructose-6P")
        assert not model.is_edge_species("Fructose-6P")
        assert len(model.edge_species) == 0

    def test_promote_missing_species(self, model):
        result = model.promote_species_to_core("Nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Prune edge tests
# ---------------------------------------------------------------------------

class TestPruneEdge:
    def test_prune_edge(self, model, species_b):
        sp2 = SpeciesData(label="ADP", concentration=0.0)
        model.add_edge_species(species_b)
        model.add_edge_species(sp2)
        assert len(model.edge_species) == 2

        removed = model.prune_edge({"fructose-6p"})
        assert removed == 1
        assert len(model.edge_species) == 1
        assert model.is_edge_species("ADP")
        assert not model.is_edge_species("Fructose-6P")


# ---------------------------------------------------------------------------
# Concentration vector tests
# ---------------------------------------------------------------------------

class TestConcentrationVector:
    def test_get_and_set_concentrations(self, model, species_a, species_c):
        model.add_core_species(species_a)
        model.add_core_species(species_c)

        vec = model.get_core_concentration_vector()
        assert vec == [5.0, 2.0]

        model.set_core_concentrations([4.0, 1.5])
        # species_a is not constant -> updated
        assert model.core_species[0].concentration == 4.0
        # species_c is constant -> NOT updated
        assert model.core_species[1].concentration == 2.0

    def test_get_core_species_labels(self, model, species_a, species_b):
        model.add_core_species(species_a)
        model.add_core_species(species_b)
        labels = model.get_core_species_labels()
        assert labels == ["Glucose", "Fructose-6P"]

    def test_get_all_species_labels_and_concentrations(self, model, species_a, species_b):
        model.add_core_species(species_a)
        model.add_edge_species(species_b)
        labels = model.get_all_species_labels()
        assert labels == ["Glucose", "Fructose-6P"]
        vec = model.get_all_concentration_vector()
        assert vec == [5.0, 0.0]
        model.set_all_concentrations([3.0, 0.5])
        assert model.core_species[0].concentration == 3.0
        assert model.edge_species[0].concentration == 0.5

    def test_reset_concentrations_to_initial(self, model, species_a, species_b):
        model.add_core_species(species_a)
        model.add_edge_species(species_b)
        model.set_all_concentrations([9.0, 7.0])
        model.reset_concentrations_to_initial()
        assert model.core_species[0].concentration == 5.0
        assert model.edge_species[0].concentration == 0.0


# ---------------------------------------------------------------------------
# Summary test
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary(self, model, species_a, species_b):
        model.add_core_species(species_a)
        model.add_edge_species(species_b)
        s = model.summary()
        assert s["core_species"] == 1
        assert s["edge_species"] == 1
        assert s["core_reactions"] == 0
        assert s["edge_reactions"] == 0
