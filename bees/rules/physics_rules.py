"""BEES physics rules — always-on physical-chemistry corrections.

Rationale: knowledge/functions/PHYSICS_RULES.md
Opt-in FAS calibrations: bees.rules.calibrations.fas (CALIBRATIONS.md).
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE
from bees.rules.base import RULES, Reference, Rule
from bees.rules.helpers import (  # noqa: F401  — re-export detect_acyl_chain_length
    detect_acyl_chain_length,
    _CO2_LABELS,
    _CO2_SMILES,
)

logger = logging.getLogger("BEES.rules.physics")

KCAT_REV_MAX = 1e4
DGR_IRREVERSIBLE_KJMOL = 30.0
_DG_PER_CH2_KJMOL = -1.0
_HYDROPHOBIC_REFERENCE_CHAIN = 2
_HYDROPHOBIC_TEMPERATURE_K = 298.15
_R_KJ = 8.314462618e-3


def compute_haldane_reverse_kcat(
    kcat_fwd: float,
    keq: float,
    km_substrates: Iterable[float],
    km_products: Iterable[float],
) -> Optional[float]:
    """kcat_rev = kcat_fwd · ∏Km_p / (Keq · ∏Km_s). No ceiling. See PHYSICS_RULES.md."""
    if not (
        math.isfinite(keq)
        and keq > 0.0
        and math.isfinite(kcat_fwd)
        and kcat_fwd > 0.0
    ):
        return None
    prod_p = 1.0
    for km in km_products:
        if km is None or km <= 0 or not math.isfinite(km):
            return None
        prod_p *= km
    prod_s = 1.0
    for km in km_substrates:
        if km is None or km <= 0 or not math.isfinite(km):
            return None
        prod_s *= km
    if prod_s == 0.0:
        return None
    val = kcat_fwd * prod_p / (keq * prod_s)
    if not (math.isfinite(val) and val > 0.0):
        return None
    return val


def _filter_km_for_haldane(reaction, side: str) -> list[float]:
    """Haldane Km filter by stoich side; skips always-available cofactors."""
    kin = getattr(reaction, "kinetics", None)
    if kin is None:
        return []
    km_per = getattr(kin, "km_per_substrate", None) or {}
    stoich = getattr(reaction, "stoichiometry", None) or {}
    kept: list[float] = []
    for lab, km in km_per.items():
        if km is None or km <= 0 or not math.isfinite(km):
            continue
        if lab.lower().strip() in COFACTORS_ALWAYS_AVAILABLE:
            continue
        coeff = stoich.get(lab, 0)
        if side == "substrate" and coeff >= 0:
            continue
        if side == "product" and coeff <= 0:
            continue
        kept.append(km)
    return kept


def _has_co2_product(reaction) -> bool:
    """True if CO2 is a product (label or SMILES). See PHYSICS_RULES.md."""
    stoich = getattr(reaction, "stoichiometry", None) or {}
    product_labels = getattr(reaction, "product_labels", None) or [
        lab for lab, c in stoich.items() if c > 0
    ]
    kin = getattr(reaction, "kinetics", None)
    smiles_map = (getattr(kin, "compound_smiles", None) or {}) if kin is not None else {}

    co2_canon = None
    for lab in product_labels:
        if stoich.get(lab, 0) <= 0 and lab in stoich:
            continue
        if lab.lower().strip() in _CO2_LABELS:
            return True
        smi = smiles_map.get(lab) or smiles_map.get(lab.lower().strip())
        if smi:
            from bees.common import canonical_smiles
            if co2_canon is None:
                co2_canon = canonical_smiles(_CO2_SMILES)
            if canonical_smiles(smi) == co2_canon:
                return True
    return False


class DgrIrreversibility(Rule):
    """Flag irreversible when |ΔG°'| > cutoff or Keq invalid. See PHYSICS_RULES.md."""

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None:
            return False
        return isinstance(getattr(thermo, "dgr_prime_kJmol", None), (int, float)) and \
            isinstance(getattr(thermo, "keq", None), (int, float))

    def apply(self, reaction) -> None:
        thermo = reaction.thermo
        dgr = thermo.dgr_prime_kJmol
        keq = thermo.keq
        cutoff = float(self.params["dgr_kjmol_cutoff"])
        if (
            (math.isfinite(dgr) and abs(dgr) > cutoff)
            or not math.isfinite(keq)
            or keq <= 0.0
        ):
            thermo.irreversible = True
            thermo.kcat_rev = None


class DecarboxylationIrreversible(Rule):
    """Flag irreversible when CO2 is a product. See PHYSICS_RULES.md."""

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None or getattr(thermo, "irreversible", True):
            return False
        return _has_co2_product(reaction)

    def apply(self, reaction) -> None:
        reaction.thermo.irreversible = True
        reaction.thermo.kcat_rev = None


class HydrophobicChainLengthKm(Rule):
    """Scale acyl-substrate Km by chain length. See PHYSICS_RULES.md."""

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None:
            return False
        km_per = getattr(kin, "km_per_substrate", None)
        if not km_per:
            return False
        n_ref = int(self.params["reference_chain_length"])
        stoich = getattr(reaction, "stoichiometry", None) or {}
        smiles_map = getattr(kin, "compound_smiles", None) or {}
        for lab in km_per:
            if stoich.get(lab, 0) >= 0:
                continue
            if lab.lower().strip() in COFACTORS_ALWAYS_AVAILABLE:
                continue
            n = detect_acyl_chain_length(lab, smiles_map.get(lab))
            if n is not None and n >= n_ref:
                return True
        return False

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        km_per = kin.km_per_substrate
        stoich = getattr(reaction, "stoichiometry", None) or {}
        smiles_map = getattr(kin, "compound_smiles", None) or {}
        dG = float(self.params["dG_per_CH2_kJmol"])
        n_ref = int(self.params["reference_chain_length"])
        rt = _R_KJ * float(self.params["temperature_K"])
        for lab, km in list(km_per.items()):
            if km is None or km <= 0:
                continue
            if stoich.get(lab, 0) >= 0:
                continue
            if lab.lower().strip() in COFACTORS_ALWAYS_AVAILABLE:
                continue
            n = detect_acyl_chain_length(lab, smiles_map.get(lab))
            if n is None or n < n_ref:
                continue
            km_per[lab] = km * math.exp(dG * (n - n_ref) / rt)


class HaldaneReverseKcat(Rule):
    """Recompute kcat_rev via Haldane (reverse-queried product Kms). See PHYSICS_RULES.md."""

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None or getattr(thermo, "irreversible", True):
            return False
        kin = getattr(reaction, "kinetics", None)
        if kin is None:
            return False
        kcat = getattr(kin, "kcat", None)
        if not isinstance(kcat, (int, float)):
            return False
        keq = getattr(thermo, "keq", None)
        if not isinstance(keq, (int, float)):
            return False
        return kcat > 0

    def apply(self, reaction) -> None:
        thermo = reaction.thermo
        kin = reaction.kinetics
        km_substrates = list(
            (getattr(kin, "_substrate_kms_for_haldane", None) or {}).values()
        )
        km_products = [
            km for lab, km in (getattr(kin, "_product_kms_for_haldane", None) or {}).items()
            if lab.lower().strip() not in COFACTORS_ALWAYS_AVAILABLE
        ]
        kcat_rev = compute_haldane_reverse_kcat(
            kcat_fwd=kin.kcat,
            keq=thermo.keq,
            km_substrates=km_substrates,
            km_products=km_products,
        )
        if kcat_rev is None:
            thermo.kcat_rev = None
            thermo.irreversible = True
        else:
            thermo.kcat_rev = kcat_rev


class ReverseKcatCeiling(Rule):
    """Clear absurd kcat_rev (>cap) → irreversible. See PHYSICS_RULES.md."""

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None:
            return False
        krev = getattr(thermo, "kcat_rev", None)
        if not isinstance(krev, (int, float)):
            return False
        cap = float(self.params["kcat_rev_max"])
        if not (math.isfinite(krev) and krev > cap):
            return False
        # Skip when forward kcat itself already exceeds the cap (CatPred
        # over-prediction) — supra-cap reverse is then Haldane-consistent.
        kin = getattr(reaction, "kinetics", None)
        kcat_fwd = getattr(kin, "kcat", None) if kin is not None else None
        if isinstance(kcat_fwd, (int, float)) and math.isfinite(kcat_fwd) and kcat_fwd > cap:
            return False
        return True

    def apply(self, reaction) -> None:
        thermo = reaction.thermo
        logger.warning(
            "kcat_rev=%0.3g exceeds cap %.3g; treating reaction as irreversible "
            "(likely Haldane input inconsistent with rate-law form)",
            thermo.kcat_rev,
            float(self.params["kcat_rev_max"]),
        )
        thermo.kcat_rev = None
        thermo.irreversible = True


_alberty_2003 = Reference(
    authors=("Alberty, R. A.",),
    title="Thermodynamics of Biochemical Reactions",
    year="2003",
    journal="Wiley-Interscience",
)
_haldane_1930 = Reference(
    authors=("Haldane, J. B. S.",),
    title="Enzymes",
    year="1930",
    journal="Longmans, Green and Co.",
)
_ruppe_fox_2018 = Reference(
    authors=("Ruppe, A.", "Fox, J. M."),
    title=(
        "Analysis of interdependent kinetic controls of fatty acid synthases"
    ),
    year="2018",
    journal="ACS Catalysis",
    volume="8",
    pages="11722-11734",
    doi="10.1021/acscatal.8b03171",
)
_tanford_1980 = Reference(
    authors=("Tanford, C.",),
    title=(
        "The Hydrophobic Effect: Formation of Micelles and Biological Membranes"
    ),
    year="1980",
    journal="Wiley",
)
_bar_even_2011 = Reference(
    authors=(
        "Bar-Even, A.",
        "Noor, E.",
        "Savir, Y.",
        "Liebermeister, W.",
        "Davidi, D.",
        "Tawfik, D. S.",
        "Milo, R.",
    ),
    title=(
        "The moderately efficient enzyme: evolutionary and physicochemical "
        "trends shaping enzyme parameters"
    ),
    year="2011",
    journal="Biochemistry",
    volume="50",
    pages="4402-4410",
    doi="10.1021/bi2002289",
)

RULES.register(
    DgrIrreversibility(
        name="dgr_irreversibility",
        description="Flag irreversible when |ΔG°'| > 30 kJ/mol or Keq invalid.",
        reference=_alberty_2003,
        reference_type="textbook",
        params={"dgr_kjmol_cutoff": DGR_IRREVERSIBLE_KJMOL},
    )
)

RULES.register(
    DecarboxylationIrreversible(
        name="decarboxylation_irreversible",
        description="Flag irreversible when CO₂ is a product (gas escape).",
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={},
    )
)

RULES.register(
    HydrophobicChainLengthKm(
        name="hydrophobic_chain_length_km",
        description="Scale acyl-substrate Km by exp(ΔG_CH2·(n−n_ref)/RT); ΔG_CH2≈−1.0 kJ/mol.",
        reference=_tanford_1980,
        reference_type="textbook",
        params={
            "dG_per_CH2_kJmol": _DG_PER_CH2_KJMOL,
            "reference_chain_length": _HYDROPHOBIC_REFERENCE_CHAIN,
            "temperature_K": _HYDROPHOBIC_TEMPERATURE_K,
        },
    )
)

RULES.register(
    HaldaneReverseKcat(
        name="haldane_reverse_kcat",
        description="Compute kcat_rev via Haldane; flag irreversible if unevaluable.",
        reference=_haldane_1930,
        reference_type="textbook",
        params={},
    )
)

RULES.register(
    ReverseKcatCeiling(
        name="reverse_kcat_ceiling",
        description="Clear kcat_rev and flag irreversible when kcat_rev > 10^4 s⁻¹.",
        reference=_bar_even_2011,
        reference_type="review",
        params={"kcat_rev_max": KCAT_REV_MAX},
    )
)
