"""BEES physics rules — predictive corrections from physical chemistry.

Three sections, separated by visible markers so the file stays scannable:
  * Helpers — pure functions used by Rule subclasses
  * Rules — Rule subclasses, in registration order
  * Registration — bottom-of-file manifest of what runs and in what order

Adding a fourth helper, or letting any single Rule exceed ~150 lines, is
the signal to split the file (one rule per file). For ~3 rules it is more
readable kept together than fragmented.

0
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE
from bees.rules.base import RULES, Reference, Rule
# Shared acyl/EC/CO2 helpers live in bees.rules.helpers (used by both laws and
# calibrations). detect_acyl_chain_length is re-exported here for back-compat
# (callers/tests import it from physics_rules).
from bees.rules.helpers import (  # noqa: F401
    detect_acyl_chain_length,
    _CO2_LABELS,
    _CO2_SMILES,
)

logger = logging.getLogger("BEES.rules.physics")


# ============================================================
# Helpers — pure functions, no Rule subclasses
# ============================================================

# Above ~10^4 s⁻¹ a reverse-direction kcat is faster than any known enzyme.
# Moved here from bees.thermodynamics.KCAT_REV_MAX so the constant lives next
# to the rule that enforces it. The original module-level constant is left
# in place for now to avoid breaking the (test-only) legacy haldane_kcat_rev
# code path; it will be removed once the legacy callers are migrated.
KCAT_REV_MAX = 1e4

# |ΔG°'| above this is treated as effectively irreversible at any plausible
# cellular concentration ratio Q. Conservative bar that matches the
# historical cutoff in bees.thermodynamics.ThermoEngine.compute_keq.
DGR_IRREVERSIBLE_KJMOL = 30.0

# (CO2 constants _CO2_SMILES/_CO2_LABELS moved to bees.rules.helpers, imported above.)

# --- Hydrophobic chain-length Km correction (HydrophobicChainLengthKm) -------
#
# Effective methylene (CH2) binding free-energy increment for an acyl chain in a
# finite enzyme pocket. This is a FIXED PHYSICAL CONSTANT of the rule, not a
# user/runtime knob: a physics rule encodes a law. The dev-only sensitivity path
# is editing this one named constant and re-running.
#
# Magnitude derivation (physics-informed prior, pre-registered −0.5…−1.5 range):
#   ΔG ≈ γ·ΔA_nonpolar, γ ≈ 25 cal/mol/Å² ≈ 0.105 kJ/mol/Å² (Chothia 1974;
#   Eisenberg–McLachlan 1986). A FULLY buried CH2 (~25–30 Å²) → ≈ −2.6…−3.4 kJ/mol
#   (recovers the bulk Tanford value). In a groove the chain contacts the protein
#   on ~one face → ~10 Å² buried per CH2 → ≈ −1.0 kJ/mol. Consistent with the
#   general protein–ligand "methylene increment" (−0.5…−1.5 kJ/mol). The bulk
#   Tanford −3.4 would predict ~10^7–10^9× selectivity over C4→C20 (all flux to the
#   longest chain) — the wrong regime for an enzyme.
# Corroboration (not the source): Ruppe 2020 TesA Kd ladder tightens 671× over
#   C4→C20 (koff constant, kon-driven) → also ≈ −1.0 kJ/mol per CH2.
_DG_PER_CH2_KJMOL = -1.0

# Anchor chain length: every acyl substrate's Km is corrected RELATIVE to this
# length, so a chain of exactly this length is left unchanged. C2 = acetyl, the
# natural shortest acyl / primer.
_HYDROPHOBIC_REFERENCE_CHAIN = 2

# Temperature (K) for the RT term. Matches the FAS project environment (298 K).
_HYDROPHOBIC_TEMPERATURE_K = 298.15

# Gas constant in kJ/mol/K.
_R_KJ = 8.314462618e-3



def compute_haldane_reverse_kcat(
    kcat_fwd: float,
    keq: float,
    km_substrates: Iterable[float],
    km_products: Iterable[float],
) -> Optional[float]:
    """Multi-substrate Haldane: kcat_rev = kcat_fwd · ∏Km_p / (Keq · ∏Km_s).

    Returns None if Keq is non-finite/zero, kcat_fwd is invalid, or any Km
    is missing/non-positive. Does NOT apply the kcat_rev > KCAT_REV_MAX
    ceiling — that is ReverseKcatCeiling's job. The split is what lets the
    ceiling be toggled independently via `bees validate-rule`.

    Equivalent to bees.thermodynamics.haldane_kcat_rev but without the
    ceiling guard. The two implementations are intentionally separate so
    the legacy test path can keep its existing behaviour while the rule
    layer uses the un-clamped math.
    """
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
    """Replicate the eager-path Km filtering used by Haldane.

    side='substrate': only species with negative stoichiometry (reactants),
    positive finite Km, and label not in COFACTORS_ALWAYS_AVAILABLE.
    side='product':   only species with positive stoichiometry (products),
    positive finite Km, and label not in COFACTORS_ALWAYS_AVAILABLE.

    Mirrors the filters at bees.reaction_generator lines 744-749 and 802-805.
    """
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
    """True if the reaction releases CO2 as a product.

    Detects CO2 two ways, both restricted to the product side (positive
    stoichiometric coefficient):
      * label match against _CO2_LABELS (works without SMILES), and
      * canonical-SMILES match against _CO2_SMILES (robust to label variants).

    Mirrors the detection the dead engine path used at
    bees.thermodynamics.attach_to_reactions (the `_has_co2_product` block).
    """
    stoich = getattr(reaction, "stoichiometry", None) or {}
    product_labels = getattr(reaction, "product_labels", None) or [
        lab for lab, c in stoich.items() if c > 0
    ]
    kin = getattr(reaction, "kinetics", None)
    smiles_map = (getattr(kin, "compound_smiles", None) or {}) if kin is not None else {}

    co2_canon = None
    for lab in product_labels:
        # Products only: a CO2 that appears as a reactant (carboxylation) must
        # not trip this rule.
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


# ============================================================
# Rules — Rule subclasses, in registration order
# ============================================================
#
# Order summary (full rationale in the Registration block at the bottom):
#   1. DgrIrreversibility           — flag from |ΔG°'| and Keq sanity
#   2. DecarboxylationIrreversible  — flag from CO2 product (gas escape)
#   3. HydrophobicChainLengthKm     — scale acyl-substrate Km by chain length
#   4. HaldaneReverseKcat           — recompute kcat_rev from current inputs
#   5. ReverseKcatCeiling           — clamp absurd kcat_rev to "irreversible"
#
# This module holds only physics LAWS (always on). The opt-in FAS calibrations
# (elongation_chain_cliff, tesa_long_chain_preference) live in
# bees.rules.calibrations and register themselves *before* haldane_reverse_kcat
# (via register(before=...)), so a corrected forward kcat still feeds kcat_rev.


class DgrIrreversibility(Rule):
    """Flag a reaction irreversible when |ΔG°'| exceeds the cutoff or Keq
    is non-finite / non-positive.

    Replicates the cutoff that previously lived inline in
    bees.thermodynamics.ThermoEngine.compute_keq (lines 511-515). Biological
    reactions with |ΔG°'| > ~30 kJ/mol are unidirectional under any plausible
    cellular concentration ratio Q.
    """

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None:
            return False
        # Defensive: tests and partial data may give non-numeric values.
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
    """Flag a reaction irreversible when it releases CO2 as a product.

    Trigger is any reaction with CO2 as a product (`_has_co2_product`), not
    hardcoded to specific enzymes — the rule fires identically in any pathway
    with decarboxylative steps. FabH, FabB, FabF (fatty-acid synthesis) are
    the reactions that happen to trigger it in this model, not the condition
    itself. The CO2 escapes the system, so the reverse (carboxylative)
    direction does not run at cellular CO2 partial pressure regardless of the
    computed ΔG°' — the reaction is physiologically one-way. Applying Haldane
    to the overall stoichiometry (with CO2 as a product whose Km is unknown /
    unconstrained) otherwise yields a spurious kcat_rev of 10^5–10^6 s⁻¹.

    Runs BEFORE HaldaneReverseKcat so the Haldane rule sees irreversible=True
    and skips. Replaces the dead engine-path CO2 override that previously lived
    in bees.thermodynamics.attach_to_reactions.

    Reference: Ruppe & Fox 2018 (ACS Catal.), decarboxylative condensation has
    a single forward arrow.
    """

    def applies_to(self, reaction) -> bool:
        thermo = getattr(reaction, "thermo", None)
        if thermo is None or getattr(thermo, "irreversible", True):
            return False
        return _has_co2_product(reaction)

    def apply(self, reaction) -> None:
        reaction.thermo.irreversible = True
        reaction.thermo.kcat_rev = None


class HydrophobicChainLengthKm(Rule):
    """Scale each acyl substrate's Km by chain length (hydrophobic binding).

    Longer acyl chains bury more nonpolar surface in the enzyme pocket and bind
    tighter, so Km should fall with chain length. catpred does not reliably
    capture this, which inverts TesA's chain-length selectivity. This rule injects
    it from a single physical constant:

        Km(n) = Km_baseline · exp(ΔG_CH2 · (n − n_ref) / RT)

    with ΔG_CH2 = _DG_PER_CH2_KJMOL (an effective partial-burial methylene
    increment, ≈ −1.0 kJ/mol; see the constant's derivation above). ΔG_CH2 < 0 and
    n > n_ref give a factor < 1 → tighter Km → higher forward specificity for long
    chains.

    UNIVERSAL SCOPE: applies to any reaction with a detectable acyl substrate, not
    a single enzyme. Hydrophobic chain-length binding selectivity is a general
    property of acyl-binding enzymes (Ruppe's FabA/FabZ/FabF/FabB binding data show
    it too), so this is a genuine rule, not a TesA special-case.

    FORWARD-ONLY: mutates kinetics.km_per_substrate, which the forward rate law
    (compute_mm_rate) and the SBML exporter read. It deliberately does NOT touch
    HaldaneReverseKcat's frozen `_substrate_kms_for_haldane` snapshot — reverse
    flux is negligible for the affected reactions , 
    so propagating the correction into Haldane is out of scope.

    IDEMPOTENT: reads the baseline Km (the registry restores km_per_substrate from
    baseline before each apply_all pass) and writes the corrected value, so
    repeated application across enlarger iterations does not compound.

    References: Tanford 1980 (methylene transfer); magnitude from buried-surface
    area, Chothia 1974 / Eisenberg–McLachlan 1986.
    """

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
                continue  # products / cofactors with non-negative coeff
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
            # Baseline-relative (km is the restored baseline); idempotent.
            km_per[lab] = km * math.exp(dG * (n - n_ref) / rt)


# (ElongationChainCliff and TesALongChainPreference — the fitted FAS calibrations —
# moved to bees.rules.calibrations, along with the _ec_norm / _reaction_acyl_substrate_n
# helpers they use. This module now holds only physics laws.)


class HaldaneReverseKcat(Rule):
    """Recompute reverse kcat from current Km / kcat / Keq via Haldane.

    Runs after every input-mutating rule (hydrophobic Km correction)
    so the Km values fed into Haldane are the rule-corrected ones. If
    Haldane returns None (missing Km, non-finite inputs), the reaction is
    flagged irreversible — matching the live behaviour at
    bees.reaction_generator line 812.

    IMPORTANT: Uses ONLY reverse-queried product Kms (stored in
    kinetics._product_kms_for_haldane), not original product Kms from
    forward catalytic data. This replicates the old eager-path logic at
    reaction_generator.py line 802-805 where km_products_haldane was
    filtered from product_kms (reverse-query results), not merged km_per_substrate.
    """

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
        # Use substrate Kms snapshotted in _attach_raw_thermo_eagerly BEFORE the
        # product Km merge — exactly when the old eager path captured km_substrates.
        km_substrates = list(
            (getattr(kin, "_substrate_kms_for_haldane", None) or {}).values()
        )
        # Use ONLY reverse-queried product Kms (cofactor-filtered), not all entries
        # in km_per_substrate. Mirrors old eager-path km_products_haldane filter.
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
            # thermo.irreversible stays as DgrIrreversibility left it (False).


class ReverseKcatCeiling(Rule):
    """Clamp absurd reverse kcat values to "effectively irreversible".

    A reverse-direction kcat > ~10⁴ s⁻¹ is faster than any known enzyme;
    when Haldane returns one, it almost always indicates inconsistent
    Haldane inputs (e.g. cofactor Km leaking into ∏Km_p, sign-flipped ΔG°').
    Clamping silently would still violate the Haldane equality, so we
    treat the reaction as irreversible instead.
    """

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
        # If the FORWARD kcat itself already exceeds the cap (a CatPred
        # over-prediction that BEES nonetheless accepted), a supra-cap reverse
        # kcat is the Haldane-consistent consequence of that inflated forward
        # plus Keq — not independent evidence of bad Haldane inputs (cofactor Km
        # leak / sign-flipped ΔG°'). The genuine inconsistency cases this rule
        # targets have a normal-magnitude forward kcat and an anomalously large
        # reverse. Flagging here would wrongly cut a (near-equilibrium) reaction
        # out of the network — e.g. transketolase, whose CatPred kcat_fwd alone
        # is ~9e4 s⁻¹ — so skip and keep the reaction reversible.
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


# ============================================================
# Registration — the manifest of what runs and in what order
# ============================================================
#
# Dependency chain:
#
#   1. DgrIrreversibility           — flag set from |ΔG°'| / Keq, independent
#                                     of Km and kcat.
#   2. DecarboxylationIrreversible  — flag set from CO2 product. Must run
#                                     BEFORE Haldane so it skips (CO2's escape,
#                                     not its Km, is what makes it one-way).
#   3. HydrophobicChainLengthKm     — mutates km_per_substrate from baseline by
#                                     acyl chain length. Forward-only (affects the
#                                     forward rate law, not Haldane's frozen
#                                     snapshot). Registered before Haldane to keep
#                                     the flag→Km→Haldane→ceiling order coherent.
#   4. HaldaneReverseKcat       — recomputes kcat_rev from current Km, kcat
#                                 and Keq. Must run AFTER every rule that
#                                 mutates those inputs and after every flag
#                                 rule (it skips already-irreversible ones).
#   5. ReverseKcatCeiling       — clamps Haldane's output. Must run AFTER
#                                 HaldaneReverseKcat.

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
        description=(
            "Flag a reaction irreversible when |ΔG°'| exceeds 30 kJ/mol, or "
            "when Keq is non-finite or non-positive. Replicates the cutoff "
            "previously inlined in compute_keq."
        ),
        reference=_alberty_2003,
        reference_type="textbook",
        params={"dgr_kjmol_cutoff": DGR_IRREVERSIBLE_KJMOL},
    )
)

RULES.register(
    DecarboxylationIrreversible(
        name="decarboxylation_irreversible",
        description=(
            "Flag a reaction irreversible when it releases CO2 as a product. "
            "Decarboxylative condensations (FabH/FabB/FabF) are physiologically "
            "one-way because the CO2 escapes; Haldane would otherwise assign a "
            "spurious 10^5–10^6 s⁻¹ reverse kcat. Replaces the dead engine-path "
            "CO2 override in thermodynamics.attach_to_reactions."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={},
    )
)

RULES.register(
    HydrophobicChainLengthKm(
        name="hydrophobic_chain_length_km",
        description=(
            "Scale each acyl substrate's Km by exp(ΔG_CH2·(n − n_ref)/RT) so longer "
            "acyl chains bind tighter (hydrophobic burial). ΔG_CH2 ≈ −1.0 kJ/mol is "
            "an effective partial-burial methylene increment (Chothia/Eisenberg "
            "surface-area γ; ~30% of bulk Tanford), corroborated by Ruppe's TesA "
            "binding ladder. Universal scope (any acyl substrate), forward-only "
            "(mutates km_per_substrate, not Haldane's frozen snapshot), idempotent "
            "via the registry's baseline restore of km_per_substrate."
        ),
        reference=_tanford_1980,
        reference_type="textbook",
        params={
            "dG_per_CH2_kJmol": _DG_PER_CH2_KJMOL,
            "reference_chain_length": _HYDROPHOBIC_REFERENCE_CHAIN,
            "temperature_K": _HYDROPHOBIC_TEMPERATURE_K,
        },
        designed_from=("FAS_ecoli",),
    )
)

# (elongation_chain_cliff / tesa_long_chain_preference are registered in
# bees.rules.calibrations with kind="calibration", enabled=False.)

RULES.register(
    HaldaneReverseKcat(
        name="haldane_reverse_kcat",
        description=(
            "Compute kcat_rev = kcat_fwd · ∏Km_p / (Keq · ∏Km_s) from current "
            "Km / kcat / Keq inputs. If Haldane cannot be evaluated (missing "
            "Km, non-finite inputs), flag the reaction irreversible. "
            "Replicates the eager-path Haldane logic at "
            "reaction_generator.py:796-820."
        ),
        reference=_haldane_1930,
        reference_type="textbook",
        params={},
    )
)

RULES.register(
    ReverseKcatCeiling(
        name="reverse_kcat_ceiling",
        description=(
            "Clear kcat_rev to None and flag irreversible when kcat_rev "
            "exceeds 10^4 s⁻¹. Above this rate, Haldane inputs are almost "
            "certainly inconsistent (cofactor Km leakage, sign-flipped ΔG°')."
        ),
        reference=_bar_even_2011,
        reference_type="review",
        params={"kcat_rev_max": KCAT_REV_MAX},
    )
)
