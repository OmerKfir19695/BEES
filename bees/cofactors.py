"""
Cofactor and carrier: classification and utilities
"""

import re
from typing import Dict, List

# General cofactors: energy carriers, redox agents, vitamin-derived, metals.
# These do NOT trigger new reaction discovery or ontology expansion.
GENERAL_COFACTORS = {
    'atp', 'adp', 'amp', 'gtp', 'gdp', 'gmp', 'utp', 'udp', 'ump', 'ctp', 'cdp', 'cmp',
    'itp', 'idp', 'imp',
    'dttp', 'dtdp', 'dtmp', 'datp', 'dadp', 'damp',
    'dgtp', 'dgdp', 'dgmp', 'dctp', 'dcdp', 'dcmp',
    'nadh', 'nad+', 'nad', 'nadh2', 'nadph', 'nadp+', 'nadp', 'nadph2',
    'fad', 'fadh2', 'coa', 'coenzyme a', 'coash',
    'h2o', 'water', 'h+', 'proton',
    'phosphate', 'pi', 'orthophosphate', 'pyrophosphate', 'ppi', 'diphosphate',
    'co2', 'carbon dioxide', 'hco3-', 'bicarbonate', 'o2', 'oxygen', 'h2', 'hydrogen',
    'tpp', 'thiamine pyrophosphate', 'thiamin pyrophosphate',
    'plp', 'pyridoxal phosphate', 'pyridoxal 5-phosphate',
    'thf', 'tetrahydrofolate', 'tetrahydrofolic acid', 'h4folate',
    'sam', 's-adenosylmethionine', 's-adenosyl-l-methionine',
    'lipoic acid', 'lipoamide', 'lipoyl',
    'cobalamin', 'vitamin b12', 'adenosylcobalamin', 'methylcobalamin',
    'mg2+', 'magnesium', 'mn2+', 'manganese', 'zn2+', 'zinc',
    'fe2+', 'fe3+', 'iron', 'cu2+', 'copper', 'ca2+', 'calcium',
}

# Subset treated as "always available" — do not need to appear in the input
# species list. 
COFACTORS_ALWAYS_AVAILABLE = {
    'h2o', 'water',
    'h+', 'proton',
    'phosphate', 'pi', 'orthophosphate', 'pyrophosphate', 'ppi', 'diphosphate',
    'co2', 'carbon dioxide', 'hco3-', 'bicarbonate',
    'o2', 'oxygen',
    'h2', 'hydrogen',
    'mg2+', 'magnesium',
    'mn2+', 'manganese',
    'zn2+', 'zinc',
    'fe2+', 'fe3+', 'iron',
    'cu2+', 'copper',
    'ca2+', 'calcium',
}

# High-energy phosphate carriers — strict matching, used separately from the
# general cofactor filter.
ENERGY_CARRIERS = {
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "itp", "idp", "imp",
    "utp", "udp", "ump", "ctp", "cdp", "cmp",
    "dttp", "dtdp", "dtmp", "datp", "dadp", "damp",
    "dgtp", "dgdp", "dgmp", "dctp", "dcdp", "dcmp",
}

# Subset invariants — caught at import time, not silently at runtime.
assert COFACTORS_ALWAYS_AVAILABLE <= GENERAL_COFACTORS, (
    "COFACTORS_ALWAYS_AVAILABLE contains entries absent from GENERAL_COFACTORS"
)
assert ENERGY_CARRIERS <= GENERAL_COFACTORS, (
    "ENERGY_CARRIERS contains entries absent from GENERAL_COFACTORS"
)


def is_rate_law_exempt_cofactor(label: str) -> bool:
    """True for buffered / always-available cofactors (H2O, H+, CO2, metals, …).

    These species participate in stoichiometry and Q/Keq but must not gate the
    Liebermeister / reversible-MM product-Km completeness check — requiring a
    CatPred Km for H2O or CO2 silently collapses nearly every FAS reaction onto
    the legacy forward-only rate law. Regulatory cofactors (CoA, NAD, NADP,
    ATP, …) are NOT exempt: they need explicit product Kms for back-pressure.
    """
    if not label:
        return False
    return label.lower().strip() in COFACTORS_ALWAYS_AVAILABLE

# ---------------------------------------------------------------------------
# ACP/CoA thioester chemistry
# ---------------------------------------------------------------------------

# Acyl prefixes recognised as acyl-ACP thioester names (not free-carrier).
_ACP_ACYL_PREFIXES = frozenset({
    "acetyl", "malonyl", "propionyl", "butyryl", "butanoyl", "acetoacetyl",
    "hexanoyl", "octanoyl", "decanoyl", "dodecanoyl", "tetradecanoyl",
    "hexadecanoyl", "octadecanoyl", "icosanoyl", "docosanoyl", "tetracosanoyl",
    "hexacosanoyl", "lauroyl", "myristoyl", "palmitoyl", "stearoyl",
    "cerotoyl", "lignoceroyl", "hydroxybutyryl", "crotonyl",
})

# 3'-phospho-ADP-pantetheine moiety shared by all acyl-CoA thioesters.
# Stereochemistry matches ChEBI:57288 (natural acetyl-CoA): (3R) pantothenate
# carbon and D-ribose. 
COA_TAIL = (
    "SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)"
    "OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(=O)(O)O)n1cnc2c(N)ncnc12"
)

# Truncated 4'-phosphopantetheine handle for ACP-thioester *database* SMILES
# (no adenine-ribose-ADP). Not the same as COA_TAIL, which is for eQuilibrator.
# ACP proxy in the CSV = ACYL_CHAIN_SMILES[acyl] + PPANT_HANDLE.
# No-op for non-ACP networks (labels that do not end with ACP_SUFFIX).
PPANT_HANDLE = "SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)O"
ACP_SUFFIX = "-[acp]"

# Acyl chain SMILES fragments (ending in C(=O)) keyed by lowercase acyl group name.
# Concatenate with COA_TAIL to get the full CoA-thioester SMILES for eQuilibrator.
ACYL_CHAIN_SMILES: Dict[str, str] = {
    "acetyl":                        "CC(=O)",
    "malonyl":                       "OC(=O)CC(=O)",
    "3-oxobutanoyl":                 "CC(=O)CC(=O)",
    "acetoacetyl":                   "CC(=O)CC(=O)",
    "(3r)-hydroxybutanoyl":          "C[C@@H](O)CC(=O)",
    "(2e)-butenoyl":                 "C/C=C/C(=O)",
    "butanoyl":                      "CCCC(=O)",
    "3-oxohexanoyl":                 "CCCC(=O)CC(=O)",
    "(3r)-hydroxyhexanoyl":          "CCC[C@@H](O)CC(=O)",
    "(2e)-hexenoyl":                 "CCC/C=C/C(=O)",
    "hexanoyl":                      "CCCCCC(=O)",
    "3-oxooctanoyl":                 "CCCCCC(=O)CC(=O)",
    "(3r)-hydroxyoctanoyl":          "CCCCC[C@@H](O)CC(=O)",
    "(2e)-octenoyl":                 "CCCCC/C=C/C(=O)",
    "octanoyl":                      "CCCCCCCC(=O)",
    "3-oxodecanoyl":                 "CCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxydecanoyl":          "CCCCCCC[C@@H](O)CC(=O)",
    "(2e)-decenoyl":                 "CCCCCCC/C=C/C(=O)",
    "decanoyl":                      "CCCCCCCCCC(=O)",
    "3-oxododecanoyl":               "CCCCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxydodecanoyl":        "CCCCCCCCC[C@@H](O)CC(=O)",
    "(2e)-dodecenoyl":               "CCCCCCCCC/C=C/C(=O)",
    "dodecanoyl":                    "CCCCCCCCCCCC(=O)",
    "3-oxotetradecanoyl":            "CCCCCCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxytetradecanoyl":     "CCCCCCCCCCC[C@@H](O)CC(=O)",
    "(2e)-tetradecenoyl":            "CCCCCCCCCCC/C=C/C(=O)",
    "tetradecanoyl":                 "CCCCCCCCCCCCCC(=O)",
    "3-oxohexadecanoyl":             "CCCCCCCCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxyhexadecanoyl":      "CCCCCCCCCCCCC[C@@H](O)CC(=O)",
    "(2e)-hexadecenoyl":             "CCCCCCCCCCCCC/C=C/C(=O)",
    "hexadecanoyl":                  "CCCCCCCCCCCCCCCC(=O)",
    "3-oxooctadecanoyl":             "CCCCCCCCCCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxyoctadecanoyl":      "CCCCCCCCCCCCCCC[C@@H](O)CC(=O)",
    "(2e)-octadecenoyl":             "CCCCCCCCCCCCCCC/C=C/C(=O)",
    "octadecanoyl":                  "CCCCCCCCCCCCCCCCCC(=O)",
    # C20:0 (icosanoyl / arachidoyl) saturated cycle intermediates.
    # SMILES sourced from PubChem CID 10467 (icosanoic acid) and chain-extended
    # using the standard FAS cycle 3-oxo / (3R)-hydroxy / (2E)-enoyl patterns.
    "3-oxoicosanoyl":                "CCCCCCCCCCCCCCCCCC(=O)CC(=O)",
    "(3r)-hydroxyicosanoyl":         "CCCCCCCCCCCCCCCCC[C@@H](O)CC(=O)",
    "(2e)-icosenoyl":                "CCCCCCCCCCCCCCCCC/C=C/C(=O)",
    "icosanoyl":                     "CCCCCCCCCCCCCCCCCCCC(=O)",
    "(9z)-hexadecenoyl":             "CCCCCC/C=C\\CCCCCCCC(=O)",
    "(9z)-octadecenoyl":             "CCCCCCCC/C=C\\CCCCCCCC(=O)",
    "(11z)-octadecenoyl":            "CCCCCC/C=C\\CCCCCCCCCC(=O)",
    # Odd-chain
    "3-oxopentanoyl":                "CCC(=O)CC(=O)",
    # Unsaturated (cis) branch intermediates — E. coli FAS II
    "(3z)-decenoyl":                 "CCCCCC/C=C\\CC(=O)",
    "3-oxo-(5z)-dodecenoyl":         "CCCCCC/C=C\\CC(=O)CC(=O)",
    "(3r,5z)-3-hydroxydodecenoyl":   "CCCCCC/C=C\\C[C@@H](O)CC(=O)",
    "(2e,5z)-dodeca-2,5-dienoyl":    "CCCCCC/C=C\\C/C=C/C(=O)",
    "(5z)-dodecenoyl":               "CCCCCC/C=C\\CCCC(=O)",
    "(7z)-3-oxotetradecenoyl":       "CCCCCC/C=C\\CCCC(=O)CC(=O)",
    "(3r,7z)-3-hydroxytetradecenoyl":"CCCCCC/C=C\\CCC[C@@H](O)CC(=O)",
    "(2e,7z)-tetradeca-2,7-dienoyl": "CCCCCC/C=C\\CCC/C=C/C(=O)",
    "(7z)-tetradecenoyl":            "CCCCCC/C=C\\CCCCCC(=O)",
    "3-oxo-(11z)-octadecenoyl":      "CCCCCC/C=C\\CCCCCCCC(=O)CC(=O)",
    "(3r,11z)-3-hydroxyoctadecenoyl":"CCCCCC/C=C\\CCCCCCC[C@@H](O)CC(=O)",
    "(2e,11z)-octadeca-2,11-dienoyl":"CCCCCC/C=C\\CCCCCCC/C=C/C(=O)",
    # C20:1 (13Z) paullinoyl branch — elongation of (11Z)-octadecenoyl-ACP
    # (cis-vaccenoyl-ACP) by FabF/FabB shifts the cis double bond +2 to Δ13Z.
    # SMILES sourced from PubChem CID 5312547 (cis-13-eicosenoic acid).
    "(13z)-3-oxoicosenoyl":          "CCCCCC/C=C\\CCCCCCCCCC(=O)CC(=O)",
    "(3r,13z)-3-hydroxyicosenoyl":   "CCCCCC/C=C\\CCCCCCCCC[C@@H](O)CC(=O)",
    "(2e,13z)-icosa-2,13-dienoyl":   "CCCCCC/C=C\\CCCCCCCCC/C=C/C(=O)",
    "(13z)-icosenoyl":               "CCCCCC/C=C\\CCCCCCCCCCCC(=O)",
    # C16:1 (9Z) branch — intermediates between (7Z)-C14:1 elongation and (11Z)-C18:1.
    # The DB has two naming variants for each of these compounds (split-locant and
    # combined-locant); both must be present so `_correct_acp_smiles` rebuilds
    # them identically — otherwise the two name variants end up with DIFFERENT
    # SMILES (the un-rebuilt one keeps a DB bug putting OH at α-position instead
    # of β) and `_register_species_label` registers them as separate species,
    # splitting the C16:1 branch and breaking pathway topology.
    "3-oxo-(9z)-hexadecenoyl":       "CCCCCC/C=C\\CCCCCC(=O)CC(=O)",
    "(9z)-3-oxohexadecenoyl":        "CCCCCC/C=C\\CCCCCC(=O)CC(=O)",
    "(3r)-hydroxy-(9z)-hexadecenoyl":"CCCCCC/C=C\\CCCCC[C@@H](O)CC(=O)",
    "(3r,9z)-3-hydroxyhexadecenoyl": "CCCCCC/C=C\\CCCCC[C@@H](O)CC(=O)",
    "(2e,9z)-hexadecadienoyl":       "CCCCCC/C=C\\CCCCC/C=C/C(=O)",
    "(2e,9z)-hexadeca-2,9-dienoyl":  "CCCCCC/C=C\\CCCCC/C=C/C(=O)",
}

# Free-acid SMILES for each acyl group above (acyl-OH instead of acyl-S-CoA).
# eQuilibrator's ComponentContribution database covers free fatty acids /
# 3-oxo / 3-hydroxy / 2-enoyl species far more reliably than synthesised
# CoA-thioester SMILES, so we prefer this form when computing ΔG°' for
# acyl-[ACP] substrates. The thioester ΔG°' equals the free-acid ΔG°' up to
# a constant per thioester bond (~−31 kJ/mol) which cancels for every
# FAS-cycle reaction (transfer / reduction / dehydration / reduction) because
# the same number of thioesters appear on both sides.
ACYL_FREE_ACID_SMILES: Dict[str, str] = {
    "acetyl":                        "CC(O)=O",
    "malonyl":                       "OC(=O)CC(O)=O",
    "3-oxobutanoyl":                 "CC(=O)CC(O)=O",
    "acetoacetyl":                   "CC(=O)CC(O)=O",
    "(3r)-hydroxybutanoyl":          "C[C@@H](O)CC(O)=O",
    "(2e)-butenoyl":                 "C/C=C/C(O)=O",
    "butanoyl":                      "CCCC(O)=O",
    "3-oxohexanoyl":                 "CCCC(=O)CC(O)=O",
    "(3r)-hydroxyhexanoyl":          "CCC[C@@H](O)CC(O)=O",
    "(2e)-hexenoyl":                 "CCC/C=C/C(O)=O",
    "hexanoyl":                      "CCCCCC(O)=O",
    "3-oxooctanoyl":                 "CCCCCC(=O)CC(O)=O",
    "(3r)-hydroxyoctanoyl":          "CCCCC[C@@H](O)CC(O)=O",
    "(2e)-octenoyl":                 "CCCCC/C=C/C(O)=O",
    "octanoyl":                      "CCCCCCCC(O)=O",
    "3-oxodecanoyl":                 "CCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxydecanoyl":          "CCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-decenoyl":                 "CCCCCCC/C=C/C(O)=O",
    "decanoyl":                      "CCCCCCCCCC(O)=O",
    "3-oxododecanoyl":               "CCCCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxydodecanoyl":        "CCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-dodecenoyl":               "CCCCCCCCC/C=C/C(O)=O",
    "dodecanoyl":                    "CCCCCCCCCCCC(O)=O",
    "3-oxotetradecanoyl":            "CCCCCCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxytetradecanoyl":     "CCCCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-tetradecenoyl":            "CCCCCCCCCCC/C=C/C(O)=O",
    "tetradecanoyl":                 "CCCCCCCCCCCCCC(O)=O",
    "3-oxohexadecanoyl":             "CCCCCCCCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxyhexadecanoyl":      "CCCCCCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-hexadecenoyl":             "CCCCCCCCCCCCC/C=C/C(O)=O",
    "hexadecanoyl":                  "CCCCCCCCCCCCCCCC(O)=O",
    "3-oxooctadecanoyl":             "CCCCCCCCCCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxyoctadecanoyl":      "CCCCCCCCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-octadecenoyl":             "CCCCCCCCCCCCCCC/C=C/C(O)=O",
    "octadecanoyl":                  "CCCCCCCCCCCCCCCCCC(O)=O",
    # C20:0 (icosanoyl / arachidoyl) saturated cycle — free-acid forms.
    "3-oxoicosanoyl":                "CCCCCCCCCCCCCCCCCC(=O)CC(O)=O",
    "(3r)-hydroxyicosanoyl":         "CCCCCCCCCCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e)-icosenoyl":                "CCCCCCCCCCCCCCCCC/C=C/C(O)=O",
    "icosanoyl":                     "CCCCCCCCCCCCCCCCCCCC(O)=O",
    "(9z)-hexadecenoyl":             "CCCCCC/C=C\\CCCCCCCC(O)=O",
    "(9z)-octadecenoyl":             "CCCCCCCC/C=C\\CCCCCCCC(O)=O",
    "(11z)-octadecenoyl":            "CCCCCC/C=C\\CCCCCCCCCC(O)=O",
    # Odd-chain
    "3-oxopentanoyl":                "CCC(=O)CC(O)=O",
    # Unsaturated (cis) branch intermediates — E. coli FAS II
    "(3z)-decenoyl":                 "CCCCCC/C=C\\CC(O)=O",
    "3-oxo-(5z)-dodecenoyl":         "CCCCCC/C=C\\CC(=O)CC(O)=O",
    "(3r,5z)-3-hydroxydodecenoyl":   "CCCCCC/C=C\\C[C@@H](O)CC(O)=O",
    "(2e,5z)-dodeca-2,5-dienoyl":    "CCCCCC/C=C\\C/C=C/C(O)=O",
    "(5z)-dodecenoyl":               "CCCCCC/C=C\\CCCC(O)=O",
    "(7z)-3-oxotetradecenoyl":       "CCCCCC/C=C\\CCCC(=O)CC(O)=O",
    "(3r,7z)-3-hydroxytetradecenoyl":"CCCCCC/C=C\\CCC[C@@H](O)CC(O)=O",
    "(2e,7z)-tetradeca-2,7-dienoyl": "CCCCCC/C=C\\CCC/C=C/C(O)=O",
    "(7z)-tetradecenoyl":            "CCCCCC/C=C\\CCCCCC(O)=O",
    "3-oxo-(11z)-octadecenoyl":      "CCCCCC/C=C\\CCCCCCCC(=O)CC(O)=O",
    "(3r,11z)-3-hydroxyoctadecenoyl":"CCCCCC/C=C\\CCCCCCC[C@@H](O)CC(O)=O",
    "(2e,11z)-octadeca-2,11-dienoyl":"CCCCCC/C=C\\CCCCCCC/C=C/C(O)=O",
    # C20:1 (13Z) paullinoyl branch — free-acid forms (PubChem CID 5312547).
    "(13z)-3-oxoicosenoyl":          "CCCCCC/C=C\\CCCCCCCCCC(=O)CC(O)=O",
    "(3r,13z)-3-hydroxyicosenoyl":   "CCCCCC/C=C\\CCCCCCCCC[C@@H](O)CC(O)=O",
    "(2e,13z)-icosa-2,13-dienoyl":   "CCCCCC/C=C\\CCCCCCCCC/C=C/C(O)=O",
    "(13z)-icosenoyl":               "CCCCCC/C=C\\CCCCCCCCCCCC(O)=O",
    # C16:1 (9Z) branch — free-acid variants matching the CoA-form keys above.
    # Both naming variants present so _correct_acp_smiles unifies the DB rows.
    "3-oxo-(9z)-hexadecenoyl":       "CCCCCC/C=C\\CCCCCC(=O)CC(O)=O",
    "(9z)-3-oxohexadecenoyl":        "CCCCCC/C=C\\CCCCCC(=O)CC(O)=O",
    "(3r)-hydroxy-(9z)-hexadecenoyl":"CCCCCC/C=C\\CCCCC[C@@H](O)CC(O)=O",
    "(3r,9z)-3-hydroxyhexadecenoyl": "CCCCCC/C=C\\CCCCC[C@@H](O)CC(O)=O",
    "(2e,9z)-hexadecadienoyl":       "CCCCCC/C=C\\CCCCC/C=C/C(O)=O",
    "(2e,9z)-hexadeca-2,9-dienoyl":  "CCCCCC/C=C\\CCCCC/C=C/C(O)=O",
}

assert set(ACYL_FREE_ACID_SMILES) == set(ACYL_CHAIN_SMILES), (
    "ACYL_FREE_ACID_SMILES and ACYL_CHAIN_SMILES must cover the same acyl groups"
)

_ACP_SUFFIX_RE = re.compile(r"-\[(?:acp)\]$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Carrier-protein classification helpers
# ---------------------------------------------------------------------------

def _has_acyl_attachment_to_acp(label: str) -> bool:
    """True if label describes acyl-ACP (e.g. acetyl-ACP), not carrier-only."""
    normalized = str(label).lower().strip().replace("-", " ").replace("_", " ")
    for prefix in _ACP_ACYL_PREFIXES:
        if prefix in normalized:
            return True
    if "3 oxoacyl" in normalized or "3 oxo" in normalized and "acp" in normalized:
        return True
    if "3 hydroxyacyl" in normalized or "3 hydroxy" in normalized and "acp" in normalized:
        return True
    if "enoyl" in normalized or "dehydroacyl" in normalized:
        return True
    if normalized.startswith("acyl ") or " acyl " in normalized:
        return True
    return False


def get_coenzyme_like_flags(label: str) -> Dict[str, bool]:
    """Get flags indicating if compound is a coenzyme-like molecule."""
    normalized = str(label).lower().strip().replace("-", " ").replace("_", " ")
    is_free_coa = normalized in {"coa", "coenzyme a", "co enzyme a", "coenzyme a (coa)"}
    is_acp_like = ("acyl carrier protein" in normalized or "acp" in normalized.split())
    is_acyl_acp = is_acp_like and _has_acyl_attachment_to_acp(label)
    is_acp_carrier_only = is_acp_like and not is_acyl_acp
    is_biotin_like = ("biotin" in normalized or "carboxyl carrier protein" in normalized)
    is_coenzyme_like = is_free_coa or is_acp_carrier_only
    return {
        "normalized": normalized,
        "is_free_coa": is_free_coa,
        "is_acp_like": is_acp_like,
        "is_acyl_acp": is_acyl_acp,
        "is_acp_carrier_only": is_acp_carrier_only,
        "is_biotin_like": is_biotin_like,
        "is_coenzyme_like": is_coenzyme_like,
    }


def _acp_thioester_label_permutations(label_lc: str) -> List[str]:
    """
    Generate narrowly-scoped equivalence permutations for ACP-thioester labels.

    Targets common UniProt/Rhea naming re-orderings such as:
      - `3-oxo-(5z)-dodecenoyl-[acp]` <-> `(5z)-3-oxododecenoyl-[acp]`
      - `3-hydroxy-(5z)-dodecenoyl-[acp]` <-> `(5z)-3-hydroxydodecenoyl-[acp]`
    """
    if not label_lc:
        return []
    s = str(label_lc).lower().strip()
    if not _ACP_SUFFIX_RE.search(s):
        return []

    out: List[str] = []

    m = re.match(r"^(3-(?:oxo|hydroxy))-\(([^)]+)\)-(.+)$", s)
    if m:
        group = m.group(1)
        stereo = m.group(2)
        tail = m.group(3)
        out.append(f"({stereo})-{group}{tail}")

    m = re.match(r"^\(([^)]+)\)-(3-(?:oxo|hydroxy))(.+)$", s)
    if m:
        stereo = m.group(1)
        group = m.group(2)
        tail = m.group(3)
        out.append(f"{group}-({stereo})-{tail.lstrip('-')}")

    return list(dict.fromkeys(out))


def is_general_cofactor_label(label: str) -> bool:
    """Return True if label appears to be a general cofactor/carrier species.

    Uses GENERAL_COFACTORS plus ontology equivalents to catch naming variants.
    """
    from bees.common import get_ontology_equivalents  # deferred to avoid circular import

    def _variants(text: str) -> List[str]:
        normalized = " ".join(
            str(text).lower().strip().replace("_", " ").replace("-", " ").split()
        )
        compact = normalized.replace(" ", "")
        base = [normalized, compact]
        out: List[str] = []
        for v in base:
            out.append(v)
            out.append(v.replace("(+)", "+").replace("(-)", "-"))
            out.append(v.replace("(", "").replace(")", ""))
        # De-duplicate while preserving order
        return list(dict.fromkeys(out))

    normalized = " ".join(
        str(label).lower().strip().replace("_", " ").replace("-", " ").split()
    )

    for cand in _variants(normalized):
        if cand in GENERAL_COFACTORS:
            return True

    equivalents = get_ontology_equivalents(normalized)
    for eq in equivalents:
        for cand in _variants(eq):
            if cand in GENERAL_COFACTORS:
                return True
    return False
