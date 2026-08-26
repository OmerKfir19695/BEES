"""FAS-specific calibrations (E. coli FAS)

Opt-in via `settings.calibrations`; params locked here and fully literature cited.
"""

from __future__ import annotations

import math

from bees.rules.base import RULES, Reference, Rule
from bees.rules.helpers import _ec_norm, _reaction_acyl_substrate_n

# TesA shape: OLS logistic to Ruppe 2020 PNAS SI Fig. S4B (kcat vs chain length).
# Rule Reference below is Ruppe & Fox 2018 (ACS Catal.) — same context citation.
_TESA_EC = "3.1.2.14"
_TESA_PREF_N_HALF = 15.3
_TESA_PREF_SHARPNESS = 0.64

# Rafi 2006: E. coli FabI on trans-2-dodecenoyl-ACP, 900 min^-1 → 15 s^-1.
_FABI_EC = "1.3.1.9"
_FABI_KCAT_MEASURED = 15.0

_ruppe_fox_2018 = Reference(
    authors=("Ruppe, A.", "Fox, J. M."),
    title="Analysis of interdependent kinetic controls of fatty acid synthases",
    year="2018",
    journal="ACS Catalysis",
    volume="8",
    pages="11722-11734",
    doi="10.1021/acscatal.8b03171",
)

_rafi_2006 = Reference(
    authors=(
        "Rafi, S.",
        "Novichenok, P.",
        "Kolappan, S.",
        "Zhang, X.",
        "Stratton, C. F.",
        "Rawat, R.",
        "Kisker, C.",
        "Simmerling, C.",
        "Tonge, P. J.",
    ),
    title=(
        "Structure of acyl carrier protein bound to FabI, "
        "the FASII enoyl reductase from Escherichia coli"
    ),
    year="2006",
    journal="J. Biol. Chem.",
    volume="281",
    pages="39285-39293",
    doi="10.1074/jbc.M608758200",
)


class TesALongChainPreference(Rule):
    """Logistic long-chain preference on TesA forward kcat."""

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


class MeasuredKcatAnchor(Rule):
    """Snap kcat to measured values for chain-length-independent ECs."""

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
        description="Snap FabI (EC 1.3.1.9) kcat to 15.0 s⁻¹ (Rafi et al. 2006, E. coli ACP).",
        reference=_rafi_2006,
        reference_type="experimental",
        params={"ec_kcat": {_FABI_EC: _FABI_KCAT_MEASURED}},
        enabled=False,
    ),
    before="haldane_reverse_kcat",
)

RULES.register(
    TesALongChainPreference(
        name="tesa_long_chain_preference",
        kind="calibration",
        description=(
            "TesA long-chain preference logistic "
            "(n_half=15.3, k=0.64 from Ruppe 2020 SI S4B context)."
        ),
        reference=_ruppe_fox_2018,
        reference_type="experimental",
        params={
            "ec_numbers": frozenset({_TESA_EC}),
            "n_half": _TESA_PREF_N_HALF,
            "sharpness_k": _TESA_PREF_SHARPNESS,
        },
        enabled=False,
    ),
    before="haldane_reverse_kcat",
)
