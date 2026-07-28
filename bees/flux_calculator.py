#!/usr/bin/env python3

"""Flux and characteristic-rate calculations for the core/edge reaction network."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from bees.cofactors import is_rate_law_exempt_cofactor
from bees.reaction_generator import GeneratedReaction


@dataclass
class SpeciesFlux:
    label: str
    rate: float  # mM/s  (dC/dt)
    normalized_rate: float = 0.0  # |rate| / R_char
    time: float = 0.0


def explicit_product_km(reaction: GeneratedReaction, label: str) -> Optional[float]:
    """
    Strict product-Km lookup: km_per_substrate ONLY, no fallback to kin.km.

    kin.km is the substrate Km; using it for a product would produce fake
    product inhibition. Returns None when no explicit entry is found.
    """
    kin = reaction.kinetics
    if kin is None:
        return None
    km_per = getattr(kin, "km_per_substrate", None) or {}
    if not km_per:
        return None
    km = km_per.get(label)
    if km is None:
        label_lc = label.lower().strip()
        km = next(
            (v for k, v in km_per.items() if k.lower().strip() == label_lc),
            None,
        )
    return km if (km is not None and km > 0) else None


def _product_labels_requiring_km(reaction: GeneratedReaction) -> List[str]:
    """Product labels that must have explicit Kms for Liebermeister / reversible MM.

    Buffered cofactors (H2O, H+, CO2, …) are omitted — they stay in stoichiometry
    and Q/Keq but must not gate the rate-law form.
    """
    products = getattr(reaction, "product_labels", None) or []
    return [p for p in products if not is_rate_law_exempt_cofactor(p)]


def has_complete_explicit_product_kms(reaction: GeneratedReaction) -> bool:
    """True iff every non-exempt product has a positive explicit Km in km_per_substrate.

    Exempt = buffered always-available cofactors (see is_rate_law_exempt_cofactor).
    Returns False when the reaction has no products at all.
    """
    products = getattr(reaction, "product_labels", None) or []
    if not products:
        return False
    required = _product_labels_requiring_km(reaction)
    # Only buffered products (e.g. pure CO2 release with nothing else) — treat
    # as complete so we do not silently fall back to legacy for lack of a
    # meaningless H2O/CO2 Km.
    if not required:
        return True
    return all(explicit_product_km(reaction, p) is not None for p in required)


def _compute_mm_rate_legacy(
    reaction: GeneratedReaction,
    concentrations: Dict[str, float],
    v_max_eff: float,
) -> float:
    """
    Legacy irreversible MM: v = Vmax * prod_i( S_i/(Km_i+S_i) )^νi.
    Products are ignored entirely. Used as the fallback when product Kms
    are incomplete.
    """
    kin = reaction.kinetics
    km_per = getattr(kin, "km_per_substrate", None) or {}
    km_single = kin.km
    stoich = reaction.stoichiometry

    saturation = 1.0
    for reactant in reaction.reactant_labels:
        reactant_lc = reactant.lower().strip()
        s_conc = max(concentrations.get(reactant_lc, 0.0), 0.0)

        km_val = None
        if km_per:
            km_val = km_per.get(reactant)
            if km_val is None:
                km_val = next(
                    (v for k, v in km_per.items() if k.lower().strip() == reactant_lc),
                    None,
                )
            if km_val is None:
                continue  # cofactor-skip: assume saturated
        else:
            km_val = km_single
        if km_val is None or km_val <= 0:
            continue

        nu = abs(stoich.get(reactant, 1))
        ratio = s_conc / (km_val + s_conc)
        saturation *= ratio ** nu
        if saturation == 0.0:
            return 0.0

    return v_max_eff * saturation


def compute_mm_rate(
    reaction: GeneratedReaction,
    concentrations: Dict[str, float],
    enzyme_concentrations: Dict[str, float],
) -> float:
    """
    Compute the irreversible reaction rate.

    When all product Kms are explicitly available in km_per_substrate, uses
    the Liebermeister symmetric denominator with forward-only flux:

        numerator   = Vmax * prod_i (S_i/Km_s,i)^νi
        denominator = prod_i (1+S_i/Km_s,i)^νi + prod_j (1+P_j/Km_p,j)^νj - 1
        v = numerator / denominator

    Falls back to legacy v = Vmax * prod_i (S_i/(Km_i+S_i))^νi when any
    product Km is missing or the reaction has no products.

    Returns 0.0 when required kinetic parameters are missing.
    """
    kin = reaction.kinetics
    if kin is None or reaction.rate_law is None:
        return 0.0

    enzyme_key = reaction.enzyme_label.lower().strip()
    e_conc = enzyme_concentrations.get(enzyme_key, 0.0)

    kcat = kin.kcat
    vmax = kin.vmax
    if kcat is not None and e_conc > 0:
        v_max_eff = kcat * e_conc
    elif vmax is not None:
        v_max_eff = vmax
    else:
        return 0.0

    if not has_complete_explicit_product_kms(reaction):
        return _compute_mm_rate_legacy(reaction, concentrations, v_max_eff)

    km_per = getattr(kin, "km_per_substrate", None) or {}
    km_single = kin.km
    stoich = reaction.stoichiometry

    # ---- Substrate terms ------------------------------------------------
    # Numerator:   prod_i (S_i/Km_s,i)^νi
    # Denominator: prod_i (1 + S_i/Km_s,i)^νi
    # Substrate Km semantics preserved: cofactor-skip when not in km_per.
    sub_sat_num = 1.0
    sub_sat_den = 1.0
    for reactant in reaction.reactant_labels:
        reactant_lc = reactant.lower().strip()
        s_conc = max(concentrations.get(reactant_lc, 0.0), 0.0)

        km_val = None
        if km_per:
            km_val = km_per.get(reactant)
            if km_val is None:
                km_val = next(
                    (v for k, v in km_per.items() if k.lower().strip() == reactant_lc),
                    None,
                )
            if km_val is None:
                continue  # cofactor-skip: assume saturated (factor 1 in num & den)
        else:
            km_val = km_single
        if km_val is None or km_val <= 0:
            continue

        nu = abs(stoich.get(reactant, 1))
        ratio = s_conc / km_val
        sub_sat_num *= ratio ** nu
        sub_sat_den *= (1.0 + ratio) ** nu
        if sub_sat_num == 0.0:
            return 0.0

    # ---- Product denominator terms --------------------------------------
    # prod_j (1 + P_j/Km_p,j)^νj  — skip buffered cofactors (H2O/H+/CO2/…)
    prod_sat_den = 1.0
    for product in _product_labels_requiring_km(reaction):
        km_p = explicit_product_km(reaction, product)
        p_conc = max(concentrations.get(product.lower().strip(), 0.0), 0.0)
        nu = abs(stoich.get(product, 1))
        prod_sat_den *= (1.0 + p_conc / km_p) ** nu

    denom = sub_sat_den + prod_sat_den - 1.0
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0

    return v_max_eff * sub_sat_num / denom


# Below this magnitude, |1 − Q/Keq| is treated as exact zero — protects the
# rate-law evaluation from catastrophic cancellation near equilibrium.
_EQUILIBRIUM_CLAMP = 1e-12


def _km_for(reaction: GeneratedReaction, label: str) -> Optional[float]:
    """Find Km for a single label using the same lookup rules as compute_mm_rate."""
    kin = reaction.kinetics
    if kin is None:
        return None
    km_per = getattr(kin, "km_per_substrate", None) or {}
    if km_per:
        km = km_per.get(label)
        if km is None:
            label_lc = label.lower().strip()
            km = next(
                (v for k, v in km_per.items() if k.lower().strip() == label_lc),
                None,
            )
        return km
    return kin.km


def compute_reversible_mm_rate(
    reaction: GeneratedReaction,
    concentrations: Dict[str, float],
    enzyme_concentrations: Dict[str, float],
) -> float:
    """
    Reversible Michaelis-Menten rate — multi-reactant convenience kinetics
    (Liebermeister & Klipp 2006), corrected for stoichiometric coefficients.

    Requires reaction.thermo with irreversible=False and a finite Keq.
    Falls back to compute_mm_rate() when thermo is absent/irreversible, or
    when any *non-exempt* product Km is missing. Buffered cofactors (H2O,
    H+, CO2, …) are omitted from saturation terms but still enter Q via
    stoichiometry.

    Full general form with stoichiometric exponents νi:

        numerator   = ∏_i ([S_i]/Km_s,i)^νi  ·  (1 − Q/Keq)
        denominator = ∏_i (1 + [S_i]/Km_s,i)^νi
                    + ∏_j (1 + [P_j]/Km_p,j)^νj  − 1
        v = Vmax_f · numerator / denominator

    Q is computed in log-space from stoichiometry to avoid cancellation near
    equilibrium. Negative v means net reverse flux (Q > Keq).
    """
    kin = reaction.kinetics
    thermo = reaction.thermo
    if (
        kin is None
        or reaction.rate_law is None
        or thermo is None
        or thermo.irreversible
        or not math.isfinite(thermo.keq)
        or thermo.keq <= 0.0
    ):
        return compute_mm_rate(reaction, concentrations, enzyme_concentrations)

    enzyme_key = reaction.enzyme_label.lower().strip()
    e_conc = enzyme_concentrations.get(enzyme_key, 0.0)
    kcat_fwd = kin.kcat
    if kcat_fwd is None or e_conc <= 0:
        return compute_mm_rate(reaction, concentrations, enzyme_concentrations)

    vmax_f = kcat_fwd * e_conc

    # Stoichiometric coefficients per label (abs values; sign tracked by
    # reactant_labels / product_labels membership).
    stoich = reaction.stoichiometry  # label → signed int

    # ---- Substrate saturation (numerator and denominator terms) ------------
    # Buffered cofactors without Km are skipped (assume saturated), matching
    # legacy MM. Missing Km on a non-exempt substrate → fall back.
    sub_sat_num = 1.0   # ∏ ([S_i]/Km_s,i)^νi
    sub_sat_den = 1.0   # ∏ (1 + [S_i]/Km_s,i)^νi
    for reactant in reaction.reactant_labels:
        km = _km_for(reaction, reactant)
        if km is None or km <= 0:
            if is_rate_law_exempt_cofactor(reactant):
                continue
            return compute_mm_rate(reaction, concentrations, enzyme_concentrations)
        c = max(concentrations.get(reactant.lower().strip(), 0.0), 0.0)
        nu = abs(stoich.get(reactant, 1))        # stoichiometric exponent
        ratio = c / km
        sub_sat_num *= ratio ** nu
        sub_sat_den *= (1.0 + ratio) ** nu

    # ---- Product saturation (denominator term only) ------------------------
    # Skip buffered cofactors; require Km for every regulatory / organic product.
    prod_sat_den = 1.0   # ∏ (1 + [P_j]/Km_p,j)^νj
    for product in _product_labels_requiring_km(reaction):
        km_p = _km_for(reaction, product)
        if km_p is None or km_p <= 0:
            # No product Km → Haldane was not applied; fall back to forward-only.
            return compute_mm_rate(reaction, concentrations, enzyme_concentrations)
        c = max(concentrations.get(product.lower().strip(), 0.0), 0.0)
        nu = abs(stoich.get(product, 1))
        prod_sat_den *= (1.0 + c / km_p) ** nu

    # ---- Disequilibrium ratio Q/Keq (log-space to avoid cancellation) ------
    # Q = ∏ [P_j]^νp,j / ∏ [S_i]^νs,i  (stoich coefficients are signed).
    # Buffered always-available species (H2O, H+, CO2, …) are omitted: their
    # activities are already baked into the biochemical K'eq from equilibrator.
    # Including [H2O]=55 (mM pool) would push every dehydration far reverse.
    log_q = 0.0
    finite = True
    for label, coeff in stoich.items():
        if is_rate_law_exempt_cofactor(label):
            continue
        c = max(concentrations.get(label.lower().strip(), 0.0), 0.0)
        if c <= 0.0:
            if coeff > 0:
                # A product is absent → Q = 0 → full forward driving force
                log_q = float("-inf")
            else:
                # A substrate is absent → v = 0 (numerator already 0)
                return 0.0
            finite = False
            break
        log_q += coeff * math.log(c)

    if finite:
        log_q_over_keq = log_q - math.log(thermo.keq)
        q_over_keq = math.exp(log_q_over_keq) if log_q_over_keq < 700 else float("inf")
    else:
        q_over_keq = 0.0  # Q = 0 → disequilibrium = 1

    disequilibrium = 1.0 - q_over_keq
    if abs(disequilibrium) < _EQUILIBRIUM_CLAMP:
        return 0.0

    denom = sub_sat_den + prod_sat_den - 1.0
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0

    return vmax_f * sub_sat_num * disequilibrium / denom


def calculate_species_rates(
    reactions: List[GeneratedReaction],
    concentrations: Dict[str, float],
    enzyme_concentrations: Dict[str, float],
) -> Dict[str, float]:
    """Calculate net dC/dt (mM/s) for every species that appears in reactions."""
    rates: Dict[str, float] = {}

    for rxn in reactions:
        # Reversible MM only if (a) the template flag opts in, AND
        # (b) we have ThermoData that isn't flagged irreversible.
        # Otherwise fall back to the legacy irreversible rate law.
        use_reversible = (
            getattr(rxn.template, "reversible", False)
            and rxn.thermo is not None
            and not rxn.thermo.irreversible
        )
        if use_reversible:
            v = compute_reversible_mm_rate(rxn, concentrations, enzyme_concentrations)
        else:
            v = compute_mm_rate(rxn, concentrations, enzyme_concentrations)
        if v == 0.0:
            continue

        stoich = rxn.stoichiometry
        for species_label, coeff in stoich.items():
            lc = species_label.lower().strip()
            rates[lc] = rates.get(lc, 0.0) + coeff * v

    return rates


def calculate_characteristic_rate(core_species_rates: Dict[str, float]) -> float:
    """R_char = sqrt(Σ R_j²) over core species (mM/s)."""
    if not core_species_rates:
        return 0.0
    return math.sqrt(sum(r * r for r in core_species_rates.values()))


def identify_significant_species_at_interrupt(
    edge_rates: Dict[str, float],
    char_rate: float,
    tol_move_to_core: float,
    max_objects: int = 10,
    abs_flux_floor: float = 1e-12,
) -> List[SpeciesFlux]:
    """
     At the exact moment the solver is interrupted (``t_interrupt``), compute
    ``rr_i = |R_i| / R_char`` for each edge species *i*.  Species whose
    ``rr_i >= toleranceMoveToCore`` are candidates.  The list is sorted by
    ``rr_i`` descending and truncated to ``max_objects``.

    Flat-core safeguard (``char_rate <= 0`` but edge flux exists): ratio-based
    promotion is meaningless because any tiny ``|R_i|`` would produce an
    infinite ratio.  Instead, promote edge species whose ``|R_i|`` exceeds the
    absolute flux floor ``abs_flux_floor``, sorted by ``|R_i|`` descending and
    capped to ``max_objects``.  This avoids spurious promotions from numerical
    noise while still catching species with real flux.

    Args:
        edge_rates: label_lc -> instantaneous dC/dt (mM/s) at interrupt time.
        char_rate: Instantaneous R_char at interrupt time.
        tol_move_to_core: Tolerance epsilon (toleranceMoveToCore).
        max_objects: Maximum number of species to return per interrupt.
        abs_flux_floor: Absolute |rate| threshold used when R_char is zero.

    Returns:
        Sorted list (descending by rr_i or |rate|) of SpeciesFlux candidates,
        truncated to *max_objects*.
    """
    if not edge_rates:
        return []

    if char_rate <= 0.0:
        candidates: List[SpeciesFlux] = []
        for label, rate in edge_rates.items():
            if abs(rate) > abs_flux_floor:
                candidates.append(
                    SpeciesFlux(
                        label=label,
                        rate=rate,
                        normalized_rate=float("inf"),
                    )
                )
        candidates.sort(key=lambda sf: abs(sf.rate), reverse=True)
        return candidates[:max_objects]

    candidates = []
    for label, rate in edge_rates.items():
        rr = abs(rate) / char_rate
        if rr >= tol_move_to_core:
            candidates.append(
                SpeciesFlux(label=label, rate=rate, normalized_rate=rr)
            )

    candidates.sort(key=lambda sf: sf.normalized_rate, reverse=True)
    return candidates[:max_objects]


def identify_insignificant_species_from_peak_ratios(
    max_edge_rate_ratio: Dict[str, float],
    max_char_rate: float,
    tol_keep_in_edge: float,
    ineligible_for_prune: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Prune edge species whose *peak* |R_edge|/R_char falls below tol_keep_in_edge.

    Uses aggregated peak rate ratios (not a single end-time snapshot); skips species in ineligible_for_prune.
    """
    if max_char_rate <= 0.0 or tol_keep_in_edge <= 0.0:
        return set()

    ineligible = ineligible_for_prune or set()
    to_remove: Set[str] = set()

    for label_lc, rr in max_edge_rate_ratio.items():
        if label_lc in ineligible:
            continue
        if rr < tol_keep_in_edge:
            to_remove.add(label_lc)

    return to_remove
