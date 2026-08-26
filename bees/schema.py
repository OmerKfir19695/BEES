"""Pydantic input schema for BEES YAML."""

from typing import Dict, List, Optional, Tuple, Union, Literal, Annotated
from pydantic import BaseModel, ConfigDict, conint, confloat, constr, field_validator, model_validator, ValidationInfo, Field
from rdkit import Chem

class Species(BaseModel):
    label: str
    concentration: Union[Annotated[float, Field(gt=0)], Tuple[Annotated[float, Field(gt=0)], Annotated[float, Field(gt=0)]]] = None
    smiles: Optional[str] = None
    constant: bool = False
    reactive: bool = True
    solvent: bool = False
  
    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator('concentration')
    def check_concentration_range_order(cls, value, info: ValidationInfo):
        label = info.data.get('label')
        if value is None:
            raise ValueError(f"Concentration must be specified for '{label}'")
        if isinstance(value, float):
            if value < 0:
                raise ValueError(f"Concentration cannot be negative. Got {value} for '{label}'")
            return value
        if isinstance(value, tuple):
            if value[0] == value[1]:
                raise ValueError("Concentration range cannot have identical values")
            if value[0] < 0 or value[1] < 0:
                raise ValueError(f"Concentration cannot be negative. Got {value} for '{label}'")
            if value[0] > value[1]:
                raise ValueError(f"Concentration range min value ({value[0]}) cannot be greater than max value ({value[1]}) for '{label}'")
            return value
        return value

    @classmethod
    @field_validator('constant')
    def check_constant_species(cls, value, info: ValidationInfo):
        if value:
            if info.data.get('concentration') and isinstance(info.data['concentration'], tuple):
                raise ValueError("Constant species cannot have a concentration range")
        return value

    @classmethod
    @field_validator('smiles')
    def validate_smiles(cls, value):
        if value:
            try:
                mol = Chem.MolFromSmiles(value)
                if not mol:
                    raise ValueError("Invalid SMILES string")
            except Exception:
                raise ValueError("Invalid SMILES string")
        return value

class FeedbackInhibition(BaseModel):
    """End-product/feedback inhibition on an enzyme (by label, not EC)."""
    # Pydantic v2 ignores @classmethod-over-@field_validator; min_length lives on Field.
    inhibitors: Annotated[List[constr(min_length=1)], Field(min_length=1)]
    ki: Optional[Annotated[float, Field(gt=0)]] = None  # mM; None → CatPred ki
    hill: Annotated[float, Field(gt=0)] = 1.0

    model_config = ConfigDict(extra="forbid")

class Enzyme(Species):
    ecnumber: Optional[Union[constr(pattern=r"^EC \d+\.\d+\.\d+\.\d+$"), List[constr(pattern=r"^EC \d+\.\d+\.\d+\.\d+$")]]] = None
    amino_acid_sequence: Optional[str] = None
    feedback_inhibition: Optional[FeedbackInhibition] = None

    @classmethod
    @field_validator('label')
    def check_label_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Label cannot be empty")
        return value

    @classmethod
    @field_validator('ecnumber')
    def validate_ecnumber(cls, value):
        if value is None:
            return value
        entries = [value] if isinstance(value, str) else value
        for ec in entries:
            if not ec.startswith("EC "):
                raise ValueError("Invalid EC number: must start with 'EC '")
        return value

    @classmethod
    @field_validator('amino_acid_sequence')
    def validate_amino_acid_sequence(cls, value):
        if value is None:
            return value
        
        if ' ' in value:
            raise ValueError("Amino acid sequence cannot contain spaces")
        if not value.isupper():
            raise ValueError("Amino acid sequence must be in capital letters")
        valid_aa_codes = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(value) - valid_aa_codes
        if invalid_chars:
            raise ValueError(f"Amino acid sequence contains invalid characters: {', '.join(sorted(invalid_chars))}. Valid codes are: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y")
        
        return value

class Environment(BaseModel):
    temperature: Union[confloat(gt=0), Tuple[confloat(gt=0), confloat(gt=0)]]
    pH: Union[confloat(ge=0, le=14), Tuple[confloat(ge=0, le=14), confloat(ge=0, le=14)]]
    ionic_strength_M: confloat(ge=0, le=2.0) = 0.25
    pMg: confloat(ge=0, le=10) = 3.0

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator('temperature')
    def validate_temperature_range(cls, value):
        if isinstance(value, tuple) and len(value) != 2:
            raise ValueError("Temperature as list must have exactly 2 values (min, max)")
        if isinstance(value, tuple) and value[0] > value[1]:
            raise ValueError("Temperature range min value cannot be greater than max value")
        return value

    @classmethod
    @field_validator('pH')
    def validate_pH_range(cls, value):
        if isinstance(value, tuple) and len(value) != 2:
            raise ValueError("pH as list must have exactly 2 values (min, max)")
        if isinstance(value, tuple) and value[0] > value[1]:
            raise ValueError("pH range min value cannot be greater than max value")
        return value

class Settings(BaseModel):
    end_time: Optional[confloat(gt=0)] = None
    time_step: Optional[confloat(gt=0)] = None
    simulation_mode: Optional[Literal["iterative", "batch"]] = None

    estimate_kinetics: bool = False
    kinetics_estimator: Optional[Literal['catpred']] = None
    kinetics_include_sd: bool = True
    # DEPRECATED: YAML kcat scale; use settings.calibrations.
    kinetics_ec_kcat_scale: Optional[Dict[str, Annotated[float, Field(gt=0)]]] = None
    calibrations: Optional[List[str]] = None  # names only; params locked in code
    smiles_mode: Literal['auto', 'interactive'] = 'auto'

    toleranceKeepInEdge: confloat(ge=0) = 0
    toleranceMoveToCore: confloat(gt=0) = 1e-5
    toleranceInterruptSimulation: Optional[confloat(gt=0)] = None  # |R_edge|/R_char; default = toleranceMoveToCore
    toleranceMoveEdgeReactionToCore: Optional[confloat(gt=0)] = None  # dlnaccum; None disables
    minEdgeIterationsForPrune: conint(ge=0) = 2
    minCoreSpeciesForPrune: conint(ge=0) = 0
    termination_conversion: Optional[Dict[str, confloat(gt=0, lt=1)]] = None
    termination_rate_ratio: Optional[confloat(gt=0, lt=1)] = None
    max_edge_species: Optional[conint(gt=0)] = None
    max_num_objects_per_iter: conint(gt=0) = 10
    abs_flux_floor: confloat(gt=0) = 1e-12  # mM/s noise floor when R_char == 0
    max_iterations: conint(gt=0) = 50

    ode_method: Optional[str] = None
    ode_rtol: Optional[confloat(gt=0)] = None
    ode_atol: Optional[confloat(gt=0)] = None
    max_wall_time_per_iteration: Optional[confloat(gt=0)] = None  # stepwise ODE only

    verbose: Optional[conint(ge=10, le=50)] = 20
    saveEdgeSpecies: bool = True
    output_directory: Optional[str] = None
    save_simulation_profiles: bool = True
    save_simulation_plots: bool = True 
    plot_max_species: Optional[conint(gt=0)] = None
    plot_exclude_enzymes: bool = True
    plot_exclude_cofactors: bool = True
    save_ode_equations: bool = True

    thermo_smiles_substitutions: Optional[Dict[str, str]] = None
    thermo_irreversible_cutoff_kJmol: Annotated[float, Field(gt=0)] = 30.0  # |ΔG°'| kJ/mol

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator('time_step')
    def validate_time_step(cls, value, info: ValidationInfo):
        end_time = info.data.get('end_time')
        if end_time is not None and value >= end_time:
            raise ValueError(f"'time_step' must be smaller than 'end_time' ({end_time}). Got: {value}")
        return value

    @classmethod
    @field_validator('termination_conversion')
    def validate_termination_conversion(cls, value):
        if value:
            for species, frac in value.items():
                if not (0 < frac < 1):
                    raise ValueError(f"termination_conversion values must be between 0 and 1. Got: {species}: {frac}")
        return value

    @classmethod
    @field_validator('termination_rate_ratio')
    def validate_rate_ratio(cls, value):
        if value and not (0 < value < 1):
            raise ValueError("termination_rate_ratio must be between 0 and 1 (exclusive).")
        return value

    @model_validator(mode="after")
    def _require_at_least_one_termination(self):
        if self.simulation_mode == "batch":
            return self
        if (
            self.end_time is None
            and not self.termination_conversion
            and self.termination_rate_ratio is None
        ):
            raise ValueError(
                "At least one termination criterion must be set for iterative mode: "
                "end_time, termination_conversion, or termination_rate_ratio."
            )
        return self

    @classmethod
    @field_validator('verbose')
    def validate_verbose_level(cls, value):
        if value is not None and value not in [10, 20, 30, 40, 50]:
            raise ValueError("Verbose level must be 10, 20, 30, 40, or 50")
        return value

class Database(BaseModel):
    name: constr(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator('name')
    def check_name_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Name cannot be empty")
        return value

class InputBase(BaseModel):
    project: constr(max_length=255)
    project_directory: Optional[constr(max_length=255)] = None
    species: List[Species]
    enzymes: List[Enzyme]
    environment: Environment
    settings: Settings
    database: Database

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator('project')
    def check_project_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Project name cannot be empty")
        return value

    @classmethod
    @field_validator('species')
    def check_species_list_not_empty(cls, value):
        if not value:
            raise ValueError("Species list cannot be empty")
        return value

    @classmethod
    @field_validator('enzymes')
    def check_enzymes_list_not_empty(cls, value):
        if not value:
            raise ValueError("Enzymes list cannot be empty")
        return value
