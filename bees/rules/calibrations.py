"""BEES calibration rules — fitted, system-specific kinetic corrections.

These are NOT physics laws. They are empirical corrections (calibrated to
measured data / mechanistic arguments) for a specific pathway+predictor context,
so they ship `enabled=False` and run ONLY when a model input declares them via
`settings.calibrations`. The `kind="calibration"` tag (and this separate module)
keep a fit from being mistaken for a universal law.

Parameters are LOCKED here with their provenance — they are deliberately not
settable from the input YAML, so a production run cannot quietly re-tune them to
match a target (that would be fitting-to-the-answer). Param sweeps are a dev-only
path, not a config surface.
"""

from __future__ import annotations

import math
import re

from bees.rules.base import RULES, Reference, Rule
from bees.rules.helpers import _ec_norm, _reaction_acyl_substrate_n

# --- ElongationChainCliff (synthases lose activity on long acyl-ACP) ---------
# catpred predicts a ~flat synthase kcat across chain length (acyl-CoA proxy);
# the real β-ketoacyl-ACP synthases (FabB, FabF) fall off a cliff at C14–C16.
# Calibrated to the measured FabB/FabF kcat ratio vs C12 (Ruppe 2020 S3B):
#   C14/C12 = 0.12–0.32, C16/C12 = 0.021–0.025 → long-chain floor ≈ 0.02.
#   n_cliff=13.5, k=2.7 reproduce f(12)=0.98, f(14)=0.22, f(16)=0.02.
_CLIFF_F_MIN = 0.02
_CLIFF_N_HALF = 13.5
_CLIFF_SHARPNESS = 2.7
_ELONGATION_SYNTHASE_ECS = frozenset({"2.3.1.41", "2.3.1.179"})

# --- TesALongChainPreference (channeling: TesA harvests long chains) ---------
# catpred predicts TesA's chain-length trend BACKWARDS (kcat falls with length);
# measured kcat/Km rises ~18,000× C4→C20 (Ruppe 2020 S4B), half-rise ≈ C14.
# Independent anchor: UniProt P0ADA1 (EC 3.1.2.14). On physiological acyl-ACP the
# measured kcat is monotone-increasing to C20 (so this monotone sigmoid FORM is
# correct); on the acyl-CoA proxy substrate activity peaks at C16 — different
# substrate class, do not conflate. The final distribution's C16 peak is set by
# the FabB/FabF elongation cliff above (_CLIFF_*), NOT by TesA preference.
# g_min = short-chain in-pathway effective floor. The measured assay ratio (0.057)
# is insufficient (~90% C4); the LOCKED value 0.001 is the channeling-strength
# correction validated for THIS FAS / Ruppe-2020 context — a fitted effective
# correction, NOT a universal TesA constant. Finite-E/tQSSA does NOT let us drop
# this fit: TesA is dosed 10× the Fab enzymes (10 µM vs 1 µM), so a carrier-pool
# conservation law only shifts the C4 leak ~77%→71% (probe 2026-07-22). The fit
# stands in for substrate channeling, which the lumped-MM cannot express.
_TESA_PREF_N_HALF = 14.0
_TESA_PREF_SHARPNESS = 1.0
_TESA_PREF_G_MIN = 0.001
_TESA_THIOESTERASE_ECS = frozenset({"3.1.2.14"})

# --- FabAIsomerizationBranchRatio (commitment to the unsaturated branch) ------
# At C10, (2E)-decenoyl-[ACP] partitions between two fates on the SAME substrate:
#   * FabA isomerase (EC 5.3.3.14): (2E)-decenoyl → (3Z)-decenoyl  → UNSATURATED branch
#   * FabI reductase (EC 1.3.1.9):  (2E)-decenoyl + NADH → decanoyl → SATURATED branch
# catpred predicts the isomerase kcat ≈ 10.45 s⁻¹, ~12× faster than its FabI
# competitor (0.83 s⁻¹), so BEES commits ~90% of C10 flux to the unsaturated branch —
# the opposite of E. coli, where unsaturated FA are the minority product. The error is
# a catpred OVER-prediction of the isomerase: the measured FabA anchor (Ruppe/Fox 2018
# kinetics_parameters.csv, the bifunctional 3-hydroxydecanoyl-ACP dehydratase/isomerase)
# is 0.272 s⁻¹ — ~38× lower. This calibration caps the isomerase kcat at that MEASURED
# value; it is a measured-anchored ceiling, not a fit to the product distribution.
# SCOPE: EC 5.3.3.14 only (the isomerization is unique to C10) — the FabA dehydratase
# (EC 4.2.1.59, needed on every saturated chain) is deliberately untouched.
_FABA_ISOMERASE_ECS = frozenset({"5.3.3.14"})
_FABA_ISOMERASE_KCAT_MEASURED = 0.272

# --- FabFUnsaturatedCliffExemption (FabF alone preserves activity on C16:1) --
# `elongation_chain_cliff` applies ONE cliff (f_min≈0.02 at C16) to both synthases
# and both substrate states. Measured data (kcat_by_chain.csv, Ruppe 2020 S3B) shows
# this is right for FabB and for saturated substrates, but wrong for FabF on
# unsaturated acyl-ACP: kcat(FabF, C16:1 cis-9) = 1.06 s⁻¹ vs kcat(FabF, C12) = 2.86,
# a ratio of 0.371 — barely suppressed, not cliffed to 0.02. FabB shows no such
# preservation (cis-9/C16-sat ratio ≈ 1.10, i.e. no rescue) — the source table notes
# explicitly "FabF preserves activity on C16:1 (key for E. coli UFA); FabB does NOT".
# No chain-resolved unsaturated measurement exists beyond C16:1, so the 0.371 floor
# is applied flat for all n ≥ n_cliff on FabF-unsaturated reactions (an extrapolation
# from the single measured anchor, not a fit to the product distribution).
_FABF_EC = "2.3.1.179"
_FABF_UNSAT_FLOOR = 0.371
_UNSAT_LOCANT_RE = re.compile(r"\(\d+z\)", re.IGNORECASE)

# --- FabIEnoylReductaseMeasuredKcat (diagnostic: correct the CatPred lag) -----
# FabI (EC 1.3.1.9, enoyl-ACP reductase) is the last step of every elongation
# cycle. catpred predicts kcat ≈ 0.78 s⁻¹ — ~5× BELOW the measured 4.0 s⁻¹
# (kinetics_parameters.csv kcat6). Because catpred simultaneously OVER-predicts the
# dehydratase (EC 4.2.1.59) at ~4.2–5.2 s⁻¹ (measured 0.272), FabI's underprediction
# makes it BEES's actual per-cycle bottleneck — a candidate cause of the time-course
# lag. FabI is chain-length-INDEPENDENT (one measured value for all chains), so a
# kcat anchor here cannot distort the chain-length distribution (unlike FabB/FabF/TesA).
# Snap to the measured value: kcat ← measured (raises the underpredicted CatPred kcat).
_FABI_ENOYL_REDUCTASE_ECS = frozenset({"1.3.1.9"})
_FABI_ENOYL_REDUCTASE_KCAT_MEASURED = 4.0


def _reaction_has_unsaturated_substrate(reaction) -> bool:
    stoich = getattr(reaction, "stoichiometry", None) or {}
    for lab, coeff in stoich.items():
        if coeff >= 0:
            continue
        if _UNSAT_LOCANT_RE.search(str(lab)):
            return True
    return False


_ruppe_fox_2018 = Reference(
    authors=("Ruppe, A.", "Fox, J. M."),
    title="Analysis of interdependent kinetic controls of fatty acid synthases",
    year="2018",
    journal="ACS Catalysis",
    volume="8",
    pages="11722-11734",
    doi="10.1021/acscatal.8b03171",
)


class ElongationChainCliff(Rule):
    """Drop β-ketoacyl-ACP synthase kcat for long acyl-ACP (the measured cliff).

    FabB (EC 2.3.1.41) and FabF (EC 2.3.1.179) lose activity on long acyl-ACP
    substrates: measured kcat falls ~50× from C12 to C16 (Ruppe 2020 S3B). catpred,
    scoring the acyl-CoA proxy, predicts a ~flat kcat across chain length, so nothing
    caps elongation and chains run away to C20. This rule reinstates the cliff as a
    descending logistic on the forward kcat:

        kcat(n) = kcat_baseline · [ f_min + (1 − f_min)/(1 + exp(k·(n − n_cliff))) ]

    so kcat is ≈ unchanged below n_cliff and drops to the floor f_min above it,
    calibrated to the measured FabB/FabF kcat ratio vs C12 (f_min = C16/C12 ≈ 0.02).

    SCOPE: the synthase EC class (params['ec_numbers']), not a single enzyme.
    FORWARD-ONLY (mutates kinetics.kcat). IDEMPOTENT via the registry's baseline
    restore of kcat. Multiplicative factor in (0, 1] → only ever reduces kcat.
    This is a measured-calibrated correction of a documented catpred ACP-blindness
    (a fit, not a law); catpred's own kcat SD for these enzymes contains it.
    """

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None or not isinstance(getattr(kin, "kcat", None), (int, float)):
            return False
        if _ec_norm(getattr(reaction, "ec_number", None)) not in self.params["ec_numbers"]:
            return False
        return _reaction_acyl_substrate_n(reaction) is not None

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        if kin.kcat is None or kin.kcat <= 0:
            return
        n = _reaction_acyl_substrate_n(reaction)
        if n is None:
            return
        f_min = float(self.params["f_min"])
        n_cliff = float(self.params["n_cliff"])
        k = float(self.params["sharpness_k"])
        factor = f_min + (1.0 - f_min) / (1.0 + math.exp(k * (n - n_cliff)))
        kin.kcat = kin.kcat * factor


class TesALongChainPreference(Rule):
    """Suppress thioesterase kcat for short acyl-ACP (TesA prefers long chains).

    TesA (EC 3.1.2.14) preferentially hydrolyses long acyl-ACP: measured kcat/Km
    rises ~18,000× C4→C20 (Ruppe 2020 S4B). catpred predicts the trend BACKWARDS
    (kcat falls with chain length), so BEES over-terminates short chains → ~90% C4.
    This rule encodes the long-chain preference as an ascending logistic on the
    forward kcat:

        kcat(n) = kcat_baseline · [ g_min + (1 − g_min)/(1 + exp(−k·(n − n_half))) ]

    → g_min for short chains, → 1 for long chains. The shape (n_half, k) matches the
    measured specificity (half-rise ≈ C14); the strength g_min is the short-chain
    effective floor. Measured assay strength (g_min ≈ 0.057 = kcat C4/C16) alone
    still leaves ~90% C4, because short acyl-ACPs are abundant; the in-pathway floor
    is lower because of ACP substrate CHANNELING (short chains stay protected on
    ACP), so the LOCKED g_min = 0.001.

    This is an explicit, channeling-justified, FAS-in-sample calibration — a fitted
    effective correction validated for this FAS / Ruppe-2020 context, NOT a universal
    TesA constant or a law (hence enabled=False by default). SCOPE: the TesA EC class.
    FORWARD-ONLY (mutates kinetics.kcat), IDEMPOTENT via the registry's baseline
    restore. The Km side of the preference is supplied separately by the
    HydrophobicChainLengthKm physics law.
    """

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None or not isinstance(getattr(kin, "kcat", None), (int, float)):
            return False
        if _ec_norm(getattr(reaction, "ec_number", None)) not in self.params["ec_numbers"]:
            return False
        return _reaction_acyl_substrate_n(reaction) is not None

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        if kin.kcat is None or kin.kcat <= 0:
            return
        n = _reaction_acyl_substrate_n(reaction)
        if n is None:
            return
        g_min = float(self.params["g_min"])
        n_half = float(self.params["n_half"])
        k = float(self.params["sharpness_k"])
        factor = g_min + (1.0 - g_min) / (1.0 + math.exp(-k * (n - n_half)))
        kin.kcat = kin.kcat * factor


class FabAIsomerizationBranchRatio(Rule):
    """Cap the FabA isomerase (EC 5.3.3.14) kcat at its measured value.

    The (2E)→(3Z)-decenoyl-[ACP] isomerization is the committed step into E. coli's
    unsaturated fatty-acid branch (unique to C10). It competes for the same
    (2E)-decenoyl-[ACP] pool with FabI (enoyl reductase, the saturated route).
    catpred over-predicts the isomerase kcat (~10.45 s⁻¹) ~12× above its FabI
    competitor and ~38× above the measured FabA anchor (0.272 s⁻¹, Ruppe/Fox 2018),
    so BEES sends ~90% of flux to the unsaturated branch — inverting the real
    saturated-majority distribution.

    This rule caps kcat at the MEASURED isomerase value (a measured-anchored ceiling,
    not a fit to the product distribution): kcat ← min(kcat, kcat_measured). Only ever
    reduces kcat, so it is IDEMPOTENT via the registry's baseline restore.

    SCOPE: EC 5.3.3.14 only. The FabA dehydratase activity (EC 4.2.1.59), required on
    every saturated chain, is a different EC and is untouched. OPT-IN (enabled=False).
    FORWARD-ONLY (mutates kinetics.kcat); registered before haldane_reverse_kcat so the
    capped forward kcat feeds the reverse recompute.
    """

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None or not isinstance(getattr(kin, "kcat", None), (int, float)):
            return False
        return _ec_norm(getattr(reaction, "ec_number", None)) in self.params["ec_numbers"]

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        if kin.kcat is None or kin.kcat <= 0:
            return
        target = float(self.params["kcat_measured"])
        if kin.kcat > target:
            kin.kcat = target


class MeasuredKcatAnchor(Rule):
    """Snap kcat to a measured value for chain-length-INDEPENDENT FAS enzymes.

    catpred's FAS kcat errors are bidirectional and largest on the non-chain-
    resolved steps (FabI 5× low, FabD 32× low, FabG 20× high, dehydratase 16× high).
    For enzymes whose measured kcat is a single chain-independent constant, snapping
    to the measured value corrects the error WITHOUT any risk of distorting the
    chain-length distribution (that is carried only by the chain-resolved FabB/FabF/
    TesA, which this rule never touches). params['ec_kcat'] maps EC → measured kcat.

    FORWARD-ONLY (mutates kinetics.kcat), IDEMPOTENT via the registry baseline restore.
    Registered before haldane_reverse_kcat so the anchored forward kcat feeds Haldane.
    """

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None or not isinstance(getattr(kin, "kcat", None), (int, float)):
            return False
        return _ec_norm(getattr(reaction, "ec_number", None)) in self.params["ec_kcat"]

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        if kin.kcat is None or kin.kcat <= 0:
            return
        target = self.params["ec_kcat"].get(_ec_norm(reaction.ec_number))
        if target is not None:
            kin.kcat = float(target)


class FabFUnsaturatedCliffExemption(Rule):
    """Restore FabF kcat toward its measured C16:1-preserved value.

    `elongation_chain_cliff` applies one descending logistic to both FabB and FabF,
    with no substrate-saturation awareness. Measured data shows FabF uniquely
    preserves activity on unsaturated acyl-ACP at the cliff point (kcat C16:1/C12 =
    1.06/2.86 ≈ 0.371) while FabB does not (cis-9/sat ratio ≈ 1.10 — no rescue,
    behaves like the saturated cliff). This rule recomputes the pre-cliff kcat
    (deterministically, from the same n/f_min/n_cliff/k the cliff rule used) and
    reapplies a shallower floor for FabF + unsaturated substrate only, leaving FabB
    and all saturated-substrate reactions untouched.

    SCOPE: EC 2.3.1.179 (FabF) with an unsaturated acyl-ACP substrate (detected via
    the (nZ)- locant naming used throughout this DB for FabA-isomerized species).
    Measured anchor exists only at C16:1 (Ruppe 2020 S3B); the 0.371 floor is
    extrapolated flat above n_cliff, not fit to the product distribution. OPT-IN
    (enabled=False). Must run AFTER elongation_chain_cliff in the same pass.
    """

    def applies_to(self, reaction) -> bool:
        kin = getattr(reaction, "kinetics", None)
        if kin is None or not isinstance(getattr(kin, "kcat", None), (int, float)):
            return False
        if _ec_norm(getattr(reaction, "ec_number", None)) != self.params["fabf_ec"]:
            return False
        if not _reaction_has_unsaturated_substrate(reaction):
            return False
        return _reaction_acyl_substrate_n(reaction) is not None

    def apply(self, reaction) -> None:
        kin = reaction.kinetics
        if kin.kcat is None or kin.kcat <= 0:
            return
        n = _reaction_acyl_substrate_n(reaction)
        if n is None:
            return
        f_min = float(self.params["cliff_f_min"])
        n_cliff = float(self.params["cliff_n_cliff"])
        k = float(self.params["cliff_sharpness_k"])
        applied_factor = f_min + (1.0 - f_min) / (1.0 + math.exp(k * (n - n_cliff)))
        if applied_factor <= 0:
            return
        raw_kcat = kin.kcat / applied_factor
        unsat_floor = float(self.params["unsat_floor"])
        unsat_factor = max(applied_factor, unsat_floor)
        kin.kcat = raw_kcat * unsat_factor


RULES.register(
    ElongationChainCliff(
        name="elongation_chain_cliff",
        kind="calibration",
        description=(
            "Drop β-ketoacyl-ACP synthase (FabB EC 2.3.1.41, FabF EC 2.3.1.179) "
            "kcat for long acyl-ACP via a descending logistic "
            "f_min+(1−f_min)/(1+exp(k(n−n_cliff))), reinstating the measured "
            "C14–C16 activity cliff that catpred (acyl-CoA proxy) flattens. "
            "Calibrated to the measured FabB/FabF kcat ratio vs C12 (f_min=C16/C12≈0.02). "
            "OPT-IN calibration (enabled=False): a measured-calibrated FIT correcting "
            "catpred ACP-blindness for the FAS chain-length study."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": _ELONGATION_SYNTHASE_ECS,
            "f_min": _CLIFF_F_MIN,
            "n_cliff": _CLIFF_N_HALF,
            "sharpness_k": _CLIFF_SHARPNESS,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",  # run after the Km law, before Haldane
)

RULES.register(
    FabFUnsaturatedCliffExemption(
        name="fabF_unsaturated_cliff_exemption",
        kind="calibration",
        description=(
            "Restore FabF (EC 2.3.1.179) kcat toward its measured C16:1-preserved "
            "value (kcat C16:1/C12 = 1.06/2.86 ≈ 0.371, Ruppe 2020 S3B) when "
            "elongation_chain_cliff's uniform floor (≈0.02) has over-suppressed it. "
            "FabB shows no equivalent preservation and is untouched. Recomputes the "
            "pre-cliff kcat and reapplies the shallower measured floor. Extrapolated "
            "flat above n_cliff from the single measured C16:1 anchor (no chain-"
            "resolved unsaturated series exists) — not a fit to the product "
            "distribution. OPT-IN calibration (enabled=False). Requires "
            "elongation_chain_cliff to run first in the same pass."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "fabf_ec": _FABF_EC,
            "unsat_floor": _FABF_UNSAT_FLOOR,
            "cliff_f_min": _CLIFF_F_MIN,
            "cliff_n_cliff": _CLIFF_N_HALF,
            "cliff_sharpness_k": _CLIFF_SHARPNESS,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",  # after elongation_chain_cliff (registered just above)
)

RULES.register(
    MeasuredKcatAnchor(
        name="fabI_enoyl_reductase_measured_kcat",
        kind="calibration",
        description=(
            "Snap FabI (EC 1.3.1.9, enoyl-ACP reductase) kcat to its measured "
            "4.0 s⁻¹ (Ruppe/Fox 2018 kcat6), correcting catpred's ~5× under-"
            "prediction (~0.78 s⁻¹). FabI is the last step of each elongation cycle "
            "and — because catpred also over-predicts the dehydratase — BEES's actual "
            "per-cycle bottleneck, a candidate cause of the S2A time-course lag. "
            "Chain-length-INDEPENDENT, so the anchor cannot distort the product "
            "distribution. OPT-IN calibration (enabled=False)."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_kcat": {
                next(iter(_FABI_ENOYL_REDUCTASE_ECS)): _FABI_ENOYL_REDUCTASE_KCAT_MEASURED,
            },
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",
)

RULES.register(
    FabAIsomerizationBranchRatio(
        name="fabA_isomerization_branch_ratio",
        kind="calibration",
        description=(
            "Cap the FabA isomerase (EC 5.3.3.14) kcat at its measured anchor "
            "(0.272 s⁻¹, Ruppe/Fox 2018), correcting catpred's ~38× over-prediction "
            "(~10.45 s⁻¹) that over-commits C10 (2E)-decenoyl-[ACP] flux to the "
            "unsaturated branch vs its FabI competitor. Measured-anchored ceiling "
            "(kcat←min(kcat, measured)), NOT a fit to the product distribution. "
            "SCOPE: EC 5.3.3.14 only (isomerization is unique to C10); the FabA "
            "dehydratase EC 4.2.1.59 is untouched. OPT-IN calibration (enabled=False)."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": _FABA_ISOMERASE_ECS,
            "kcat_measured": _FABA_ISOMERASE_KCAT_MEASURED,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",  # run after the Km law, before Haldane
)

RULES.register(
    TesALongChainPreference(
        name="tesa_long_chain_preference",
        kind="calibration",
        description=(
            "Suppress TesA (EC 3.1.2.14) kcat for short acyl-ACP via an ascending "
            "logistic g_min+(1−g_min)/(1+exp(−k(n−n_half))), encoding TesA's "
            "measured long-chain preference (kcat/Km rises ~18,000× C4→C20) that "
            "catpred predicts backwards. g_min (locked 0.001) is the channeling-strength "
            "short-chain effective floor — a fitted FAS/Ruppe-2020 correction, NOT a "
            "universal TesA constant. OPT-IN calibration (enabled=False), channeling-"
            "justified FIT. Km side handled by the hydrophobic_chain_length_km law."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": _TESA_THIOESTERASE_ECS,
            "g_min": _TESA_PREF_G_MIN,
            "n_half": _TESA_PREF_N_HALF,
            "sharpness_k": _TESA_PREF_SHARPNESS,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",  # run after the Km law, before Haldane
)
