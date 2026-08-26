"""ΔG°′ / Keq via equilibrator-api; reverse kcat via Haldane."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from bees.common import canonical_smiles, smiles_to_inchi, smiles_to_inchikey,R

class CompoundSubstitutor:
    """Replace unresolved SMILES before equilibrator lookup. None = pass-through."""

    def substitute(self, label: str, smiles: Optional[str]) -> Optional[str]:
        """Return a replacement SMILES, or None to leave it unchanged."""
        return None

_DEFAULT_CACHE_DIR = Path(os.environ.get(
    "BEES_THERMO_CACHE",
    str(Path.home() / ".cache" / "bees" / "equilibrator"),
))

_DGR_IRREVERSIBLE_KJMOL = 30.0  # |ΔG°'| above this ⇒ treat as irreversible
_SIGMA_WARN_KJMOL = 10.0        # σ above this ⇒ Keq uncertain by >50×
# Above KCAT_REV_MAX, return None (irreversible) rather than clamp — a clamp still violates Haldane equality.
KCAT_REV_MAX = 1e4

logger = logging.getLogger("BEES")

def _is_disabled() -> bool:
    return os.environ.get("BEES_DISABLE_THERMO", "").strip() not in ("", "0", "false", "False")

def haldane_kcat_rev(
    kcat_fwd: float,
    keq: float,
    km_substrates: Iterable[float],
    km_products: Iterable[float],
) -> Optional[float]:
    """kcat_rev = kcat_fwd · ∏Km_p / (Keq · ∏Km_s). None = irreversible."""
    if not (math.isfinite(keq) and keq > 0.0 and math.isfinite(kcat_fwd) and kcat_fwd > 0.0):
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
    if val > KCAT_REV_MAX:
        logger.warning(
            "haldane_kcat_rev=%0.3g exceeds cap %.3g; treating reaction as "
            "irreversible (likely Haldane input inconsistent with rate-law form)",
            val, KCAT_REV_MAX,
        )
        return None
    return val

@dataclass
class ThermoData:
    """Per-reaction ΔG°′ / Keq / kcat_rev. Attached as reaction.thermo (runtime-only)."""
    dgr_prime_kJmol: float
    sigma_kJmol: float
    keq: float
    kcat_rev: Optional[float]
    irreversible: bool
    source: str

def _disabled_thermo() -> ThermoData:
    return ThermoData(
        dgr_prime_kJmol=float("nan"),
        sigma_kJmol=float("nan"),
        keq=float("nan"),
        kcat_rev=None,
        irreversible=True,
        source="disabled",
    )

def _fallback_thermo(reason: str) -> ThermoData:
    logger.info("Thermo fallback: %s", reason)
    return ThermoData(
        dgr_prime_kJmol=float("nan"),
        sigma_kJmol=float("nan"),
        keq=float("nan"),
        kcat_rev=None,
        irreversible=True,
        source="fallback",
    )

class _DiskCache:
    """Pickle-on-disk cache for equilibrator lookups. Caches compound hits and misses."""

    def __init__(self, root: Path = _DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{h}.pkl"

    def get(self, namespace: str, key: str, default=...):
        p = self._path(namespace, key)
        if not p.exists():
            return default
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return default

    def set(self, namespace: str, key: str, value) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open("wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning("Thermo cache write failed (%s): %s", p, e)

class ThermoEngine:
    """Compute ΔG°′ / Keq for reactions via equilibrator-api. One instance per run."""

    def __init__(
        self,
        pH: float = 7.0,
        ionic_strength_M: float = 0.25,
        pMg: float = 3.0,
        T_K: float = 310.15,
        irreversible_cutoff_kJmol: float = _DGR_IRREVERSIBLE_KJMOL,
        cache: Optional[_DiskCache] = None,
        substitutor: Optional["CompoundSubstitutor"] = None,
    ):
        self.pH = float(pH)
        self.ionic_strength_M = float(ionic_strength_M)
        self.pMg = float(pMg)
        self.T_K = float(T_K)
        self.irreversible_cutoff_kJmol = float(irreversible_cutoff_kJmol)
        self.cache = cache or _DiskCache()
        self._cc = None
        self._cc_load_failed = False
        # None = pass-through; enlarger injects a CompoundSubstitutor at init.
        self.substitutor: Optional[CompoundSubstitutor] = substitutor

    def _load_cc(self):
        """Load ComponentContribution and set environmental conditions."""
        if self._cc is not None or self._cc_load_failed:
            return self._cc
        try:
            from equilibrator_api import ComponentContribution, Q_  # type: ignore
        except ImportError as e:
            logger.warning(
                "equilibrator-api not installed (%s). "
                "Run: conda install equilibrator-api  "
                "or set BEES_DISABLE_THERMO=1 to suppress this warning.",
                e,
            )
            self._cc_load_failed = True
            return None
        try:
            logger.info(
                "Loading ComponentContribution (first run downloads ~1.3 GB from Zenodo; "
                "subsequent loads are fast)..."
            )
            cc = ComponentContribution()
            cc.p_h = Q_(self.pH)
            cc.p_mg = Q_(self.pMg)
            cc.ionic_strength = Q_(f"{self.ionic_strength_M} M")
            cc.temperature = Q_(f"{self.T_K} K")
            self._cc = cc
            logger.info("ComponentContribution ready (pH=%.1f, I=%.3f M, T=%.1f K).",
                        self.pH, self.ionic_strength_M, self.T_K)
        except Exception as e:
            logger.warning("ComponentContribution init failed (%s); thermo disabled.", e)
            self._cc_load_failed = True
            return None
        return self._cc

    def _resolve_compound(self, smiles: Optional[str]):
        """SMILES → equilibrator Compound. Returns None on miss."""
        cc = self._load_cc()
        if cc is None:
            return None

        canon = canonical_smiles(smiles)
        if not canon:
            return None

        inchi = smiles_to_inchi(canon)
        ikey = smiles_to_inchikey(canon)
        cache_key = inchi or ikey or canon
        if not cache_key:
            return None

        # Do not pickle Compound objects (SQLAlchemy DetachedInstanceError); cache misses only.
        cached = self.cache.get("compound", cache_key, default=...)
        if cached is None:
            return None  # cached miss

        cpd = None
        try:
            if inchi:
                cpd = cc.get_compound_by_inchi(inchi)

            # InChIKey 14-char prefix fallback (connectivity; ignores stereo/protonation).
            if cpd is None and ikey:
                ikey_prefix = ikey.split("-")[0]  # first 14-char block
                matches = cc.search_compound_by_inchi_key(ikey_prefix)
                if matches:
                    if len(matches) == 1:
                        cpd = matches[0]
                    else:
                        for m in matches:
                            if getattr(m, "inchi_key", None) == ikey:
                                cpd = m
                                break
                        if cpd is None:
                            cpd = matches[0]
                            logger.debug(
                                "Ambiguous InChIKey prefix %s → %d matches; "
                                "using first (%s).",
                                ikey_prefix, len(matches),
                                getattr(cpd, "inchi_key", "?"),
                            )

            if cpd is None:
                try:
                    cpd = cc.get_compound(canon)
                except Exception:
                    cpd = None
        except Exception as e:
            logger.debug("Compound lookup failed for smiles=%r: %s", smiles, e)
            cpd = None

        if cpd is None:
            self.cache.set("compound", cache_key, None)  # persist miss only
            logger.info(
                "Compound not found in eQuilibrator database: smiles=%r (inchi=%r)",
                smiles, inchi,
            )
        return cpd

    def _compute_dgr_prime(
        self,
        stoichiometry: Dict[str, int],
        smiles_map: Dict[str, str],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Build equilibrator Reaction and return (ΔG°′ kJ/mol, σ kJ/mol), or (None, None).

        H+ and H2O must stay in the stoichiometry (needed for is_balanced(); equilibrator accounts for both).
        """
        cc = self._cc
        if cc is None:
            return None, None

        try:
            from equilibrator_api import Reaction  # type: ignore
        except ImportError:
            return None, None

        effective_smi_map: Dict[str, Optional[str]] = {}
        for label in stoichiometry:
            smi = smiles_map.get(label)
            sub_smi = self.substitutor.substitute(label, smi) if self.substitutor else None
            effective_smi = sub_smi if sub_smi is not None else smi
            if sub_smi is not None and sub_smi != smi:
                logger.debug(
                    "CompoundSubstitutor: replaced SMILES for %r (was %r)",
                    label, smi,
                )
            effective_smi_map[label] = effective_smi

        # Carrier swaps collapse to a null reaction under substitution; short-circuit before equilibrator.
        smiles_coeffs: Dict[str, float] = {}
        all_have_smiles = True
        for label, coeff in stoichiometry.items():
            can = canonical_smiles(effective_smi_map.get(label))
            if not can:
                all_have_smiles = False
                break
            smiles_coeffs[can] = smiles_coeffs.get(can, 0) + coeff
        if all_have_smiles and not any(c != 0 for c in smiles_coeffs.values()):
            return 0.0, 0.0

        compound_map: Dict[str, object] = {}
        for label in stoichiometry:
            effective_smi = effective_smi_map[label]
            cpd = self._resolve_compound(effective_smi)
            if cpd is None:
                logger.debug(
                    "Cannot compute ΔG°': compound not resolved for label=%r (smiles=%r)",
                    label, effective_smi,
                )
                return None, None
            compound_map[label] = cpd

        try:
            # Accumulate coeffs per CC compound (don't overwrite when labels resolve to the same id).
            accumulated: Dict[object, float] = {}
            for lab, coeff in stoichiometry.items():
                cpd = compound_map[lab]
                accumulated[cpd] = accumulated.get(cpd, 0) + coeff
            # Drop net-zero spectators from CoA substitution.
            accumulated = {cpd: c for cpd, c in accumulated.items() if c != 0}
            if not accumulated:
                return 0.0, 0.0
            rxn = Reaction(accumulated)

            # Check balance before computing — unbalanced reactions return garbage ΔG°' with no warning.
            if not rxn.is_balanced():
                logger.warning(
                    "Reaction is not balanced (atoms/charge); skipping ΔG°' computation. "
                    "Stoichiometry: %s", dict(stoichiometry)
                )
                return None, None

            res = cc.standard_dg_prime(rxn)
            dgr = float(res.value.m_as("kJ/mol"))
            sigma = float(res.error.m_as("kJ/mol"))
            return dgr, sigma

        except Exception as e:
            logger.warning(
                "standard_dg_prime failed for stoichiometry %s: %s",
                dict(stoichiometry), e,
            )
            return None, None

  
    def compute_keq(
        self,
        stoichiometry: Dict[str, int],
        smiles_map: Dict[str, str],
    ) -> ThermoData:
        """Compute ΔG°′ and Keq. kcat_rev is always None; irreversibility is decided by the rule layer."""
        if _is_disabled():
            return _disabled_thermo()

        cc = self._load_cc()
        if cc is None:
            return _fallback_thermo("equilibrator unavailable")

        canonical_pairs = sorted(
            ((canonical_smiles(smiles_map.get(lab)) or lab, c)
             for lab, c in stoichiometry.items()),
            key=lambda p: p[0],
        )
        rxn_key = json.dumps(
            [canonical_pairs, self.pH, self.ionic_strength_M, self.pMg, self.T_K],
            default=str,
        )

        cached = self.cache.get("reaction", rxn_key, default=...)
        if cached is not ... and cached is not None and cached != (None, None):
            result = cached
        else:
            result = self._compute_dgr_prime(stoichiometry, smiles_map)
            # Cache reaction hits only (substitutor is not in the key; caching misses would trap later runs).
            if result is not None and result != (None, None):
                self.cache.set("reaction", rxn_key, result)

        if result is None or result == (None, None):
            stoich_str = " + ".join(
                f"{c} {lab}" for lab, c in stoichiometry.items() if c != 0
            )
            return _fallback_thermo(
                f"equilibrator returned no ΔG°' for: {stoich_str}"
            )

        dgr, sigma = result

        if sigma is not None and sigma > _SIGMA_WARN_KJMOL:
            logger.warning(
                "High ΔG°' uncertainty (σ=%.1f kJ/mol) for reaction %s — "
                "Keq may be unreliable.",
                sigma, canonical_pairs,
            )

        try:
            keq = math.exp(-dgr * 1000 / (R * self.T_K))
        except OverflowError:
            keq = float("inf")

        # Irreversibility is rule-layer; compute_keq returns irreversible=False.
        td = ThermoData(
            dgr_prime_kJmol=float(dgr),
            sigma_kJmol=float(sigma) if sigma is not None else float("nan"),
            keq=float(keq),
            kcat_rev=None,
            irreversible=False,
            source="equilibrator",
        )
        logger.debug(
            "thermo: ΔG°'=%.2f kJ/mol (σ=%.2f), Keq=%.3g, irreversible=%s",
            td.dgr_prime_kJmol, td.sigma_kJmol, td.keq, td.irreversible,
        )
        return td

    def attach_to_reactions(
        self,
        reactions,
        global_smiles_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Legacy/test path: populate `.thermo` on each reaction."""
        global_smiles_map = global_smiles_map or {}
        global_smiles_lc = {
            (k.lower().strip()): v for k, v in global_smiles_map.items() if v
        }
        for rxn in reactions:
            if not getattr(rxn.template, "reversible", False):
                rxn.thermo = None
                continue

            kin = rxn.kinetics
            if kin is None or rxn.rate_law is None:
                rxn.thermo = _fallback_thermo("no kinetics")
                continue

            smiles_map = dict(getattr(kin, "compound_smiles", None) or {})
            for label in rxn.stoichiometry:
                if smiles_map.get(label):
                    continue
                fallback = global_smiles_lc.get(label.lower().strip())
                if fallback:
                    smiles_map[label] = fallback
            stoich = dict(rxn.stoichiometry)
            kcat_fwd = kin.kcat
            km_per = getattr(kin, "km_per_substrate", None) or {}

            def _km_for(label: str) -> Optional[float]:
                if km_per:
                    val = km_per.get(label)
                    if val is not None:
                        return val
                    label_lc = label.lower().strip()
                    for k, v in km_per.items():
                        if k.lower().strip() == label_lc:
                            return v
                    return None
                return kin.km

            km_substrates = {lab: _km_for(lab) for lab in rxn.reactant_labels}
            km_products = {lab: _km_for(lab) for lab in rxn.product_labels}

            _missing_km_labels = [
                lab for lab, v in (*km_substrates.items(), *km_products.items())
                if v is None
            ]
            _km_complete = not _missing_km_labels

            # attach_to_reactions is the legacy/test path; production uses _attach_raw_thermo_eagerly + RULES.apply_all.
            td = self.compute_keq(stoichiometry=stoich, smiles_map=smiles_map)

            if not td.irreversible and kcat_fwd is not None and kcat_fwd > 0:
                if not _km_complete:
                    # Haldane needs Km on both sides (missing Kms leave un-cancelled c° factors).
                    _dg = td.dgr_prime_kJmol
                    _near_eq = _dg is not None and abs(_dg) <= 5.0
                    logger.debug(
                        "Haldane gated off for reaction %s: missing Km for %s "
                        "(|dG'|=%.2f kJ/mol, near_equilibrium=%s).",
                        getattr(rxn, "label", "?"),
                        _missing_km_labels,
                        _dg if _dg is not None else float("nan"),
                        _near_eq,
                    )
                    td = ThermoData(
                        dgr_prime_kJmol=td.dgr_prime_kJmol,
                        sigma_kJmol=td.sigma_kJmol,
                        keq=td.keq,
                        kcat_rev=None,
                        irreversible=not _near_eq,
                        source=td.source,
                    )
                else:
                    kcat_rev = haldane_kcat_rev(
                        kcat_fwd=kcat_fwd,
                        keq=td.keq,
                        km_substrates=km_substrates.values(),
                        km_products=km_products.values(),
                    )
                    if kcat_rev is None:
                        td = ThermoData(
                            dgr_prime_kJmol=td.dgr_prime_kJmol,
                            sigma_kJmol=td.sigma_kJmol,
                            keq=td.keq,
                            kcat_rev=None,
                            irreversible=True,
                            source=td.source,
                        )
                    else:
                        td = ThermoData(
                            dgr_prime_kJmol=td.dgr_prime_kJmol,
                            sigma_kJmol=td.sigma_kJmol,
                            keq=td.keq,
                            kcat_rev=kcat_rev,
                            irreversible=False,
                            source=td.source,
                        )

            rxn.thermo = td
