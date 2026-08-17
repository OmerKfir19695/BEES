"""FAS-specific calibrations (E. coli FAS / Ruppe-2020).

Opt-in via `settings.calibrations`; params locked here.
Rationale: knowledge/functions/CALIBRATIONS.md
"""

from __future__ import annotations

import math

from bees.rules.base import RULES, Reference, Rule
from bees.rules.helpers import _ec_norm, _reaction_acyl_substrate_n

# Shape (n_half, k): OLS logistic to Ruppe 2020 SI S4B kcat (norm. to C20),
# all chain lengths including C10. Pure logistic (no short-chain floor): S4B
# three-param OLS wants g≈0.03, but that collapses FAS under lumped MM; dropping
# the pathway g_min is nearly identical to the old g_min=0.001 lock on S4B shape.
# See knowledge/functions/CALIBRATIONS.md.
_TESA_PREF_N_HALF = 15.3
_TESA_PREF_SHARPNESS = 0.64
_TESA_THIOESTERASE_ECS = frozenset({"3.1.2.14"})

_FABA_ISOMERASE_ECS = frozenset({"5.3.3.14"})
_FABA_ISOMERASE_KCAT_MEASURED = 0.272

_FABI_ENOYL_REDUCTASE_ECS = frozenset({"1.3.1.9"})
_FABI_ENOYL_REDUCTASE_KCAT_MEASURED = 4.0

_ruppe_fox_2018 = Reference(
    authors=("Ruppe, A.", "Fox, J. M."),
    title="Analysis of interdependent kinetic controls of fatty acid synthases",
    year="2018",
    journal="ACS Catalysis",
    volume="8",
    pages="11722-11734",
    doi="10.1021/acscatal.8b03171",
)


class TesALongChainPreference(Rule):
    """Logistic long-chain preference on TesA forward kcat. See CALIBRATIONS.md."""

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
        n_half = float(self.params["n_half"])
        k = float(self.params["sharpness_k"])
        factor = 1.0 / (1.0 + math.exp(-k * (n - n_half)))
        kin.kcat = kin.kcat * factor


class FabAIsomerizationBranchRatio(Rule):
    """Cap FabA isomerase kcat at measured value. See CALIBRATIONS.md."""

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
    """Snap kcat to measured values for chain-length-independent ECs. See CALIBRATIONS.md."""

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


RULES.register(
    MeasuredKcatAnchor(
        name="fabI_enoyl_reductase_measured_kcat",
        kind="calibration",
        description="Snap FabI (EC 1.3.1.9) kcat to measured 4.0 s⁻¹ (Ruppe/Fox 2018).",
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
        description="Cap FabA isomerase (EC 5.3.3.14) kcat at measured 0.272 s⁻¹.",
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": _FABA_ISOMERASE_ECS,
            "kcat_measured": _FABA_ISOMERASE_KCAT_MEASURED,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",
)

RULES.register(
    TesALongChainPreference(
        name="tesa_long_chain_preference",
        kind="calibration",
        description="TesA long-chain preference logistic (S4B shape n_half=15.3, k=0.64; no g_min).",
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": _TESA_THIOESTERASE_ECS,
            "n_half": _TESA_PREF_N_HALF,
            "sharpness_k": _TESA_PREF_SHARPNESS,
        },
        designed_from=("FAS_ecoli",),
        enabled=False,
    ),
    before="haldane_reverse_kcat",
)
