"""Tests for MeasuredTableEstimator (kinetics_estimator: measured_table).

Covers: EC+chain lookup (chain-resolved & single-value), units (mM/s as written),
cofactor Km mapping, CatPred fallback for absent EC, ec_kcat_scale applied exactly
once (measured AND fallback rows), build_estimator wiring, and schema acceptance.
The CatPred delegate is stubbed so no subprocess/model is invoked.
"""

from __future__ import annotations

import textwrap

import pytest

from bees.kinetics_estimator import (
    EstimatedKinetics,
    MeasuredTableEstimator,
    build_estimator,
)


@pytest.fixture
def table_csv(tmp_path):
    p = tmp_path / "measured.csv"
    p.write_text(textwrap.dedent("""\
        # citation comment line — must be ignored by the parser
        ec,chain,kcat_s,km_acyl_mM,km_malonyl_mM,km_cofactor_mM,cofactor,source
        3.1.2.14,8,11.2,0.05293,,,,TesA C8
        3.1.2.14,16,108,0.0022448,,,,TesA C16
        1.1.1.100,*,0.59,0.017,,0.01,nadph,FabG
        2.3.1.41,12,2.90,0.038,0.0082,,,FabB C12
    """))
    return str(p)


class _StubDelegate:
    """Stands in for the internal CatPred delegate; returns a fixed prediction."""
    def __init__(self):
        self.calls = 0

    def estimate(self, **kwargs):
        self.calls += 1
        return EstimatedKinetics(kcat=100.0, km_per_substrate={"glucose": 0.5},
                                 km=0.5, source="catpred(stub)")


def _est(table_csv, scale=None):
    e = MeasuredTableEstimator(table_path=table_csv, include_sd=False, ec_kcat_scale=scale)
    e._fallback = _StubDelegate()  # never hit the real CatPred in tests
    return e


# -- chain-resolved measured lookup ----------------------------------------
def test_chain_resolved_lookup_units_mM(table_csv):
    e = _est(table_csv)
    r = e.estimate(enzyme_sequence="X",
                   reactant_smiles={"hexadecanoyl-[ACP]": "", "H2O": "O"},
                   ec_number="EC 3.1.2.14")
    assert r.kcat == pytest.approx(108.0)
    # Km is in mM (the uM->mM conversion is baked into the CSV) — not 2.24
    assert r.km_per_substrate["hexadecanoyl-[ACP]"] == pytest.approx(0.0022448)
    assert e._fallback.calls == 0  # measured, not fallback


def test_chain_picks_nearest_row(table_csv):
    e = _est(table_csv)
    r = e.estimate(enzyme_sequence="X", reactant_smiles={"octanoyl-[ACP]": ""},
                   ec_number="EC 3.1.2.14")
    assert r.kcat == pytest.approx(11.2)
    assert r.km_per_substrate["octanoyl-[ACP]"] == pytest.approx(0.05293)


# -- single-value (*) row + cofactor + malonyl -----------------------------
def test_single_value_row_with_cofactor(table_csv):
    e = _est(table_csv)
    r = e.estimate(enzyme_sequence="X",
                   reactant_smiles={"3-oxooctanoyl-[ACP]": "", "NADPH": "x"},
                   ec_number="EC 1.1.1.100")
    assert r.kcat == pytest.approx(0.59)
    assert r.km_per_substrate["3-oxooctanoyl-[ACP]"] == pytest.approx(0.017)
    assert r.km_per_substrate["NADPH"] == pytest.approx(0.01)


def test_malonyl_km_mapped(table_csv):
    e = _est(table_csv)
    r = e.estimate(enzyme_sequence="X",
                   reactant_smiles={"dodecanoyl-[ACP]": "", "malonyl-[ACP]": ""},
                   ec_number="EC 2.3.1.41")
    assert r.kcat == pytest.approx(2.90)
    assert r.km_per_substrate["dodecanoyl-[ACP]"] == pytest.approx(0.038)
    assert r.km_per_substrate["malonyl-[ACP]"] == pytest.approx(0.0082)


# -- fallback to CatPred for EC not in table -------------------------------
def test_fallback_for_absent_ec(table_csv):
    e = _est(table_csv)
    r = e.estimate(enzyme_sequence="X", reactant_smiles={"glucose": "C"},
                   ec_number="EC 9.9.9.9")
    assert e._fallback.calls == 1
    assert r.kcat == pytest.approx(100.0)  # the stub's value, unmodified


# -- ec_kcat_scale applied exactly once (measured AND fallback) ------------
def test_scale_applied_once_measured(table_csv):
    e = _est(table_csv, scale={"EC 3.1.2.14": 0.5})
    r = e.estimate(enzyme_sequence="X", reactant_smiles={"hexadecanoyl-[ACP]": ""},
                   ec_number="EC 3.1.2.14")
    assert r.kcat == pytest.approx(108.0 * 0.5)  # 54, scaled once


def test_scale_applied_once_fallback_not_double(table_csv):
    # Fallback delegate is unscaled; the wrapper scales once -> 100*0.5=50 (not 25).
    e = _est(table_csv, scale={"EC 9.9.9.9": 0.5})
    r = e.estimate(enzyme_sequence="X", reactant_smiles={"glucose": "C"},
                   ec_number="EC 9.9.9.9")
    assert r.kcat == pytest.approx(50.0)


# -- build_estimator wiring -------------------------------------------------
def test_build_estimator_requires_file():
    with pytest.raises(ValueError, match="measured_kinetics_file"):
        build_estimator("measured_table", measured_kinetics_file=None)


def test_build_estimator_constructs(table_csv):
    e = build_estimator("measured_table", measured_kinetics_file=table_csv)
    assert isinstance(e, MeasuredTableEstimator)


# -- schema acceptance ------------------------------------------------------
def test_schema_accepts_measured_table():
    from bees.schema import Settings
    s = Settings(kinetics_estimator="measured_table",
                 measured_kinetics_file="measured_kinetics_fas.csv",
                 end_time=800)  # satisfy the iterative-mode termination validator
    assert s.kinetics_estimator == "measured_table"
    assert s.measured_kinetics_file == "measured_kinetics_fas.csv"


def test_schema_rejects_unknown_estimator():
    from bees.schema import Settings
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(kinetics_estimator="bogus")


# -- cross-check against the real FAS table --------------------------------
def test_real_fas_table_tesa_c16():
    import bees.common as common
    import os
    path = os.path.join(common.BEES_PATH, "projects/fattyAcidSynthesis/"
                        "fattyAcidSynthesis_ecoli/measured_kinetics_fas.csv")
    if not os.path.isfile(path):
        pytest.skip("FAS measured-kinetics CSV not present")
    e = MeasuredTableEstimator(table_path=path)
    e._fallback = _StubDelegate()
    r = e.estimate(enzyme_sequence="X", reactant_smiles={"hexadecanoyl-[ACP]": "", "H2O": "O"},
                   ec_number="EC 3.1.2.14")
    assert r.kcat == pytest.approx(108.0)            # measured TesA C16 kcat (s^-1)
    assert r.km_per_substrate["hexadecanoyl-[ACP]"] == pytest.approx(0.0022448, rel=1e-3)  # mM
