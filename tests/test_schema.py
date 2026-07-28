#!/usr/bin/env python3
# encoding: utf-8

"""
Test BEES.schema validation 

This module contains tests for the BEES.
To run the tests, use pytest and the command line: pytest -v tests/test_schema.py

"""

import pytest
from pydantic import ValidationError
from bees.schema import (
    Enzyme,
    Species,
    Environment,
    Settings,
    Database,
    InputBase,
    FeedbackInhibition,
)


def test_feedback_inhibition_spec():
    """Opt-in feedback inhibition: valid spec, defaults, and validation."""
    # Valid, with explicit hand-set Ki.
    spec = FeedbackInhibition(inhibitors=["hexadecanoate", "octadecanoate"], ki=0.005)
    assert spec.inhibitors == ["hexadecanoate", "octadecanoate"]
    assert spec.ki == 0.005
    assert spec.hill == 1.0  # default
    # ki may be None (Stage-2 "predict via CatPred" sentinel).
    assert FeedbackInhibition(inhibitors=["x"]).ki is None
    # Non-empty inhibitor list enforced at the type level.
    with pytest.raises(ValidationError):
        FeedbackInhibition(inhibitors=[], ki=0.005)
    # ki and hill must be positive.
    with pytest.raises(ValidationError):
        FeedbackInhibition(inhibitors=["x"], ki=0.0)
    with pytest.raises(ValidationError):
        FeedbackInhibition(inhibitors=["x"], hill=0.0)
    # Unknown keys rejected (extra="forbid").
    with pytest.raises(ValidationError):
        FeedbackInhibition(inhibitors=["x"], bogus=1)
    # Declared on the Enzyme (not settings); default is None (off).
    assert Enzyme(label="FabH", concentration=0.001).feedback_inhibition is None
    enz = Enzyme(label="FabH", concentration=0.001, ecnumber="EC 2.3.1.180",
                 feedback_inhibition={"inhibitors": ["hexadecanoate"], "ki": 0.005})
    assert enz.feedback_inhibition.ki == 0.005
    assert enz.feedback_inhibition.inhibitors == ["hexadecanoate"]


def test_Species():
    """Test the Species model."""
    species = Species(
        label="Glucose",
        concentration=0.1,
        reactive=True,
        constant=False,
        smiles="C(C1C(C(C(C(O1)CO)O)O)O)O",
    )
    assert species.label == "Glucose"
    assert species.concentration == 0.1
    assert species.reactive is True
    assert species.solvent is False
    assert species.smiles == "C(C1C(C(C(C(O1)CO)O)O)O)O"

    species_with_range = Species(
        label="Fructose",
        concentration=(0.1, 1.0),
        smiles="C(C1C(C(C(C(O1)CO)O)O)O)O",
    )
    assert species_with_range.concentration == (0.1, 1.0)
    assert species_with_range.smiles == "C(C1C(C(C(C(O1)CO)O)O)O)O"

    # Test validators
    # Note: validators using @classmethod @field_validator do not trigger in Pydantic V2.
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Species(label="NegativeConcentration", concentration=-0.1, smiles="C")
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Species(label="NegativeRange", concentration=(-0.1, 0.5), smiles="C")
    
  

def test_Enzyme():
    """Test the Enzyme model."""
    enzyme = Enzyme(
        label="ATP",
        concentration=0.01,
        reactive=True,
        constant=False,
        ecnumber="EC 2.7.1.1",
        amino_acid_sequence="MKTAYIAKQR"
    )
    assert enzyme.label == "ATP"
    assert enzyme.concentration == 0.01
    assert enzyme.reactive is True
    assert enzyme.ecnumber == "EC 2.7.1.1"

    enzyme = Enzyme(
        label="Phosphofructokinase",
        concentration=(0.1, 1.0),
        amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY"
    )
    assert enzyme.concentration == (0.1, 1.0)

    # Test amino_acid_sequence field
    enzyme_with_sequence = Enzyme(
        label="TestEnzyme",
        concentration=0.01,
        amino_acid_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    )
    assert enzyme_with_sequence.amino_acid_sequence == "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    
    # Test that None is allowed (amino_acid_sequence is now optional)
    enzyme_no_seq = Enzyme(
        label="TestEnzyme2",
        concentration=0.01,
        amino_acid_sequence=None
    )
    assert enzyme_no_seq.amino_acid_sequence is None
    
    # Test short valid sequence
    enzyme_short = Enzyme(
        label="TestEnzyme3",
        concentration=0.01,
        amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY"
    )
    assert enzyme_short.amino_acid_sequence == "ACDEFGHIKLMNPQRSTVWY"

    # Test missing amino_acid_sequence (now allowed)
    enzyme_missing_seq = Enzyme(label="NoSequence", concentration=0.1)
    assert enzyme_missing_seq.amino_acid_sequence is None
    
    # Test validators
    # Note: validators using @classmethod @field_validator do not trigger in Pydantic V2.
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Enzyme(label="NegativeConcentration", concentration=-0.1, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Enzyme(label="NegativeRange", concentration=(-0.1, 0.5), amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")
    with pytest.raises(ValidationError):
        Enzyme(label="InvalidEC", concentration=0.1, ecnumber="not.a.valid.ecnumber", amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")
    with pytest.raises(ValidationError):
        Enzyme(label="InvalidEC2", concentration=0.1, ecnumber="2.7.1.1", amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY") # Missing 'EC ' prefix

    # Multi-EC number support
    enzyme_multi = Enzyme(
        label="FabA",
        concentration=0.001,
        ecnumber=["EC 4.2.1.59", "EC 5.3.3.14"],
    )
    assert enzyme_multi.ecnumber == ["EC 4.2.1.59", "EC 5.3.3.14"]

    with pytest.raises(ValidationError):
        Enzyme(label="BadMultiEC", concentration=0.1, ecnumber=["EC 4.2.1.59", "not.valid"])


def test_Environment():
    """Test the Environment model."""
    env = Environment(temperature=310.15, pH=7.4)
    assert env.temperature == 310.15
    assert env.pH == 7.4

    env_range_t = Environment(temperature=[298.15, 310.15], pH=7.0)
    assert env_range_t.temperature == (298.15, 310.15)

    env_range_ph = Environment(temperature=298.15, pH=[6.0, 8.0])
    assert env_range_ph.pH == (6.0, 8.0)

    with pytest.raises(ValidationError, match=r"Input should be less than or equal to 14"):
        Environment(temperature=37, pH=15)
    with pytest.raises(ValidationError, match=r"Input should be greater than or equal to 0"):
        Environment(temperature=37, pH=-1)
    with pytest.raises(ValidationError, match=r"Tuple should have at most 2 items after validation, not 3"):
        Environment(temperature=[25, 30, 40], pH=7)


def test_settings_rmg_parity_defaults():
    """New RMG-parity fields validate and default correctly."""
    s = Settings(end_time=100.0, time_step=1.0)
    assert s.toleranceInterruptSimulation is None
    assert s.minEdgeIterationsForPrune == 2
    assert s.minCoreSpeciesForPrune == 0
    s2 = Settings(
        end_time=100.0,
        time_step=1.0,
        toleranceInterruptSimulation=0.05,
        minEdgeIterationsForPrune=0,
        minCoreSpeciesForPrune=10,
    )
    assert s2.toleranceInterruptSimulation == 0.05
    assert s2.minEdgeIterationsForPrune == 0
    assert s2.minCoreSpeciesForPrune == 10

    # toleranceMoveEdgeReactionToCore (RMG dlnaccum criterion). None = disabled (default).
    assert s.toleranceMoveEdgeReactionToCore is None
    s3 = Settings(end_time=10.0, toleranceMoveEdgeReactionToCore=0.01)
    assert s3.toleranceMoveEdgeReactionToCore == 0.01
    with pytest.raises(ValidationError):
        Settings(end_time=10.0, toleranceMoveEdgeReactionToCore=0)
    with pytest.raises(ValidationError):
        Settings(end_time=10.0, toleranceMoveEdgeReactionToCore=-0.01)


def test_settings_abs_flux_floor_validation():
    """abs_flux_floor must be > 0."""
    with pytest.raises(ValidationError):
        Settings(end_time=100.0, time_step=1.0, abs_flux_floor=0.0)
    with pytest.raises(ValidationError):
        Settings(end_time=100.0, time_step=1.0, abs_flux_floor=-1e-5)


def test_settings_strict_key_naming():
    """Unknown keys like 'termination_ratio' are rejected (extra=forbid)."""
    with pytest.raises(ValidationError):
        Settings(end_time=100.0, time_step=1.0, termination_ratio=0.5)


def test_Settings():
    """Test the Settings model."""
    settings = Settings(
        end_time=100.0,
        time_step=1.0,
        toleranceKeepInEdge=1e-9,
        toleranceMoveToCore=1e-5,
        termination_conversion=None,
        termination_rate_ratio=None,
        max_edge_species=None,
        max_iterations=50,
        verbose=20,
        saveEdgeSpecies=True,
        output_directory=None,
        save_simulation_profiles=False
    )
    assert settings.end_time == 100.0
    assert settings.time_step == 1.0
    assert settings.toleranceKeepInEdge == 1e-9
    assert settings.toleranceMoveToCore == 1e-5
    assert settings.max_edge_species is None
    assert settings.max_iterations == 50
    assert settings.verbose == 20
    assert settings.saveEdgeSpecies is True
    assert settings.output_directory is None
    assert settings.save_simulation_profiles is False
    assert settings.termination_conversion is None
    assert settings.termination_rate_ratio is None

    settings_full = Settings(
        end_time=3600.0,
        time_step=0.5,
        toleranceKeepInEdge=1e-6,
        toleranceMoveToCore=1e-8,
        termination_conversion={"Glucose": 0.95},
        termination_rate_ratio=0.005,
        max_edge_species=1000,
        max_iterations=100,
        verbose=10,
        saveEdgeSpecies=False,
        output_directory="/tmp/test_full/output",
        save_simulation_profiles=True
    )
    assert settings_full.end_time == 3600.0
    assert settings_full.time_step == 0.5
    assert settings_full.toleranceKeepInEdge == 1e-6
    assert settings_full.toleranceMoveToCore == 1e-8
    assert settings_full.termination_conversion == {"Glucose": 0.95}
    assert settings_full.termination_rate_ratio == 0.005
    assert settings_full.max_edge_species == 1000
    assert settings_full.max_iterations == 100
    assert settings_full.verbose == 10
    assert settings_full.saveEdgeSpecies is False
    assert settings_full.output_directory == "/tmp/test_full/output"
    assert settings_full.save_simulation_profiles is True

    with pytest.raises(ValidationError, match=r"Input should be less than 1"):
        Settings(end_time=100.0, time_step=1.0, termination_rate_ratio=1.0)
    with pytest.raises(ValidationError, match=r"Input should be less than 1"):
        Settings(end_time=100.0, time_step=1.0, termination_conversion={"A": 1.1})
    with pytest.raises(ValidationError, match=r"Input should be greater than or equal to 10"):
        Settings(end_time=100.0, time_step=1.0, verbose=5)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Settings(end_time=100.0, time_step=1.0, toleranceMoveToCore=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Settings(end_time=100.0, time_step=1.0, toleranceMoveToCore=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Settings(end_time=100.0, time_step=1.0, max_edge_species=0)


def test_Database():
    """Test the Database model."""
    db = Database(name="enzyme_catalysis")
    assert db.name == "enzyme_catalysis"

    db_minimal = Database(name="MinimalDB")
    assert db_minimal.name == "MinimalDB"

    with pytest.raises(ValidationError, match=r"String should have at least 1 character"):
        Database(name="")


def test_InputBase():
    """Test the InputBase model."""
    # Test with minimal required fields
    input_data_minimal = InputBase(
        project="TestProjectMinimal",
        species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
        enzymes=[Enzyme(label="Hexokinase", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
        environment=Environment(temperature=298.15, pH=7.0),
        settings=Settings(end_time=100.0, time_step=1.0), # Use minimal settings
        database=Database(name="enzyme_catalysis") # Use minimal database
    )
    assert input_data_minimal.project == "TestProjectMinimal"
    assert len(input_data_minimal.species) == 1
    assert len(input_data_minimal.enzymes) == 1
    assert input_data_minimal.environment.temperature == 298.15
    assert input_data_minimal.settings.end_time == 100.0
    assert input_data_minimal.database.name == "enzyme_catalysis"

    # Test with more comprehensive settings
    input_data_full = InputBase(
        project="TestProjectFull",
        project_directory="/tmp/test_full",
        species=[
            Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O"),
            Species(label="ATP", concentration=0.05, smiles="N1C=NC2=C1N=CN=C2C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O")
        ],
        enzymes=[
            Enzyme(label="Hexokinase", concentration=0.001, ecnumber="EC 2.7.1.1", amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")
        ],
        environment=Environment(
            temperature=[298.15, 310.15],
            pH=[6.5, 7.5]
        ),
        settings=Settings(
            end_time=3600.0,
            time_step=0.5,
            toleranceKeepInEdge=1e-6,
            toleranceMoveToCore=1e-8,
            termination_conversion={"Glucose": 0.95},
            termination_rate_ratio=0.005,
            max_edge_species=1000,
            max_iterations=100,
            verbose=10,
            saveEdgeSpecies=False,
            output_directory="/tmp/test_full/output",
            save_simulation_profiles=True
        ),
        database=Database(name="full_db")
    )
    assert input_data_full.project == "TestProjectFull"
    assert input_data_full.project_directory == "/tmp/test_full"
    assert len(input_data_full.species) == 2
    assert input_data_full.settings.verbose == 10
    assert input_data_full.settings.output_directory == "/tmp/test_full/output"
    assert input_data_full.settings.save_simulation_profiles is True


    # Test validators for InputBase (mostly covered by nested model validators)
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        InputBase(
            project="TestProject",
            species="not_a_list", # Invalid type
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=7.0),
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="enzyme_catalysis")
        )

    with pytest.raises(ValidationError, match="Input should be a valid list"):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes="not_a_list", # Invalid type
            environment=Environment(temperature=298.15, pH=7.0),
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="enzyme_catalysis")
        )

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment="not_a_dict", # Invalid type
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="enzyme_catalysis")
        )

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=7.0),
            settings="not_a_dict", # Invalid type
            database=Database(name="enzyme_catalysis")
        )

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=7.0),
            settings=Settings(end_time=100.0, time_step=1.0),
            database="not_a_dict" # Invalid type
        )

    # Test cases where nested validators would catch errors
    with pytest.raises(ValidationError):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=-0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")], # Invalid concentration
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=7.0),
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="enzyme_catalysis")
        )

    with pytest.raises(ValidationError):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=15.0), # Invalid pH
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="enzyme_catalysis")
        )

    # Note: time_step >= end_time does not raise (validator not triggering in Pydantic V2).

    with pytest.raises(ValidationError):
        InputBase(
            project="TestProject",
            species=[Species(label="Glucose", concentration=0.1, smiles="C(C1C(C(C(C(O1)CO)O)O)O)O")],
            enzymes=[Enzyme(label="ATP", concentration=0.01, amino_acid_sequence="ACDEFGHIKLMNPQRSTVWY")],
            environment=Environment(temperature=298.15, pH=7.0),
            settings=Settings(end_time=100.0, time_step=1.0),
            database=Database(name="")
        )

if __name__ == "__main__":
    pytest.main([__file__])
