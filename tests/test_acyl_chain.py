"""Stage 3a — acyl-chain-length detector validation corpus.

Gates the HydrophobicChainLengthKm rule: the detector must return the correct
carbon count for every acyl species the rule will encounter, and None for
non-acyl species, BEFORE the rule is allowed to act on Km. Kept in its own file
so the detector can be validated independently of the rule.

The critical regression (Cursor must-fix) is `test_fallback_acyl_coa_smiles_*`:
the RDKit fallback must return the ACYL chain length, never the total molecule
carbon count (a full acyl-CoA carries ~20 extra carbons in the pantetheine tail).
"""

from __future__ import annotations

import pytest

from bees.cofactors import ACYL_CHAIN_SMILES, COA_TAIL
from bees.rules.physics_rules import detect_acyl_chain_length


# ---------------------------------------------------------------------------
# Table path — saturated acyl, all carriers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("acetyl-[ACP]", 2),
    ("butanoyl-[ACP]", 4),
    ("hexanoyl-[ACP]", 6),
    ("octanoyl-[ACP]", 8),
    ("decanoyl-[ACP]", 10),
    ("dodecanoyl-[ACP]", 12),
    ("tetradecanoyl-[ACP]", 14),
    ("hexadecanoyl-[ACP]", 16),
    ("octadecanoyl-[ACP]", 18),
    ("icosanoyl-[ACP]", 20),
    # CoA carrier
    ("acetyl-CoA", 2),
    ("butanoyl-CoA", 4),
    # free acid (no carrier suffix) — looked up directly
    ("butanoyl", 4),
    ("hexadecanoyl", 16),
])
def test_saturated_chain_lengths(label, expected):
    assert detect_acyl_chain_length(label) == expected


# ---------------------------------------------------------------------------
# Modifiers do not change the carbon count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("3-oxobutanoyl-[ACP]", 4),
    ("(3R)-hydroxybutanoyl-[ACP]", 4),
    ("(2E)-butenoyl-[ACP]", 4),
    ("3-oxohexanoyl-[ACP]", 6),
    ("(3R)-hydroxyhexanoyl-[ACP]", 6),
    ("3-oxotetradecanoyl-[ACP]", 14),
    ("(3R)-hydroxytetradecanoyl-[ACP]", 14),
])
def test_modifier_chain_lengths(label, expected):
    assert detect_acyl_chain_length(label) == expected


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("TETRADECANOYL-[ACP]", 14),
    ("Butanoyl-[Acp]", 4),
])
def test_case_insensitive(label, expected):
    assert detect_acyl_chain_length(label) == expected


# ---------------------------------------------------------------------------
# Non-acyl / cofactors / bare carriers → None (rule must skip)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "NADPH", "NADP+", "NADH", "NAD+",
    "H2O", "water", "H+",
    "CO2", "carbon dioxide",
    "holo-[ACP]", "ACP", "CoA",
    "",
])
def test_non_acyl_returns_none(label):
    assert detect_acyl_chain_length(label) is None


# ---------------------------------------------------------------------------
# RDKit fallback — acyl-moiety extraction, NOT total carbon count
# (the key Cursor must-fix)
# ---------------------------------------------------------------------------

class TestFallbackAcylMoietyExtraction:
    def test_fallback_acyl_coa_smiles_returns_acyl_length_not_total(self):
        # Full butanoyl-CoA SMILES: acyl fragment (4 C) + ~20-C pantetheine tail.
        # Force the table path to miss by using an off-table label.
        coa_smiles = ACYL_CHAIN_SMILES["butanoyl"] + COA_TAIL
        assert coa_smiles.count("C") > 15  # the trap: total carbons are many
        n = detect_acyl_chain_length("mystery-acyl-not-in-table", coa_smiles)
        assert n == 4, f"expected acyl length 4, got {n} (must not count CoA tail)"

    def test_fallback_long_acyl_coa(self):
        coa_smiles = ACYL_CHAIN_SMILES["tetradecanoyl"] + COA_TAIL
        assert detect_acyl_chain_length("mystery-acyl-x", coa_smiles) == 14

    def test_fallback_free_acid_smiles(self):
        # Butanoate free acid: carboxyl anchor, flood gives 4.
        assert detect_acyl_chain_length("mystery-acid", "CCCC(=O)O") == 4

    def test_fallback_unparseable_smiles_returns_none(self):
        assert detect_acyl_chain_length("mystery", "not_a_smiles") is None

    def test_fallback_no_anchor_returns_none(self):
        # A plain alkane has no thioester/carboxyl carbonyl → no acyl anchor.
        assert detect_acyl_chain_length("mystery-alkane", "CCCCCC") is None

    def test_table_path_takes_precedence_over_smiles(self):
        # An in-table label must use the table even if a (wrong) SMILES is passed.
        assert detect_acyl_chain_length("butanoyl-[ACP]", "CCCCCCCCCCCC(=O)O") == 4
