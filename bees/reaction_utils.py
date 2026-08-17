#!/usr/bin/env python3

from typing import List, Set, Optional
from bees.common import get_ontology_equivalents, load_ontology_categories
from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE
from bees.cofactors import get_coenzyme_like_flags


_EC_ALIASES_CACHE: Optional[dict] = None
_ENZYME_DOMAIN_COFACTORS_CACHE: Optional[dict] = None


def _load_ec_aliases() -> dict:
    """Load EC aliases from db/ontology.yaml (cached after first call).

    Reads the raw YAML directly to preserve EC number casing (e.g. 'EC 2.3.1.85').
    load_ontology_categories() lowercases all keys/values, which would break EC lookups.
    """
    global _EC_ALIASES_CACHE
    if _EC_ALIASES_CACHE is not None:
        return _EC_ALIASES_CACHE
    import os
    from bees.common import BEES_PATH, read_yaml_file
    path = os.path.join(BEES_PATH, "db", "ontology.yaml")
    if not os.path.exists(path):
        _EC_ALIASES_CACHE = {}
        return _EC_ALIASES_CACHE
    try:
        data = read_yaml_file(path)
        _EC_ALIASES_CACHE = data.get("ec_aliases", {}) if isinstance(data, dict) else {}
    except Exception:
        _EC_ALIASES_CACHE = {}
    return _EC_ALIASES_CACHE


def get_ec_aliases(ec_number: Optional[str]) -> List[str]:
    """
    Get all EC number aliases for a given EC number.

    Args:
        ec_number (str): Primary EC number (e.g., "EC 2.3.1.85")

    Returns:
        List[str]: List of EC numbers to try, including the primary EC number first
    """
    if not ec_number:
        return []

    ec_numbers_to_try = [ec_number]
    aliases = _load_ec_aliases()
    if ec_number in aliases:
        ec_numbers_to_try.extend(aliases[ec_number])
    return ec_numbers_to_try


def _load_enzyme_domain_cofactors() -> dict:
    """Load enzyme domain cofactors from db/ontology.yaml (cached after first call)."""
    global _ENZYME_DOMAIN_COFACTORS_CACHE
    if _ENZYME_DOMAIN_COFACTORS_CACHE is not None:
        return _ENZYME_DOMAIN_COFACTORS_CACHE
    import os
    from bees.common import BEES_PATH, read_yaml_file
    path = os.path.join(BEES_PATH, "db", "ontology.yaml")
    if not os.path.exists(path):
        _ENZYME_DOMAIN_COFACTORS_CACHE = {}
        return _ENZYME_DOMAIN_COFACTORS_CACHE
    try:
        data = read_yaml_file(path)
        _ENZYME_DOMAIN_COFACTORS_CACHE = (
            data.get("enzyme_domain_cofactors", {}) if isinstance(data, dict) else {}
        )
    except Exception:
        _ENZYME_DOMAIN_COFACTORS_CACHE = {}
    return _ENZYME_DOMAIN_COFACTORS_CACHE


def get_enzyme_domain_cofactors(enzyme_label: str) -> List[str]:
    """
    Get domain cofactors for an enzyme (cofactors that are part of the enzyme structure).

    Args:
        enzyme_label (str): Enzyme name/label

    Returns:
        List[str]: List of cofactor patterns that are part of this enzyme's structure
    """
    enzyme_lc = enzyme_label.lower()
    domain_cofactors = []
    for enzyme_pattern, cofactor_patterns in _load_enzyme_domain_cofactors().items():
        if enzyme_pattern in enzyme_lc:
            domain_cofactors.extend(cofactor_patterns)
    return domain_cofactors


def check_reactant_availability(
    reactant: str,
    available_species_labels_lc: Set[str],
    enzyme_label: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check whether a reactant is "available" for reaction generation.

    Checks are applied in this order:
    1) Always-available cofactors (e.g., H2O, H+, Pi; see COFACTORS_ALWAYS_AVAILABLE)
    2) Direct match in `available_species_labels_lc`
    3) Enzyme domain cofactors (if `enzyme_label` given; skipped for acyl-ACP reactants)
    4) Ontology equivalents — with concrete vs category asymmetry:
       - Category reactant (e.g. "an acyl-CoA"): available if any member / alias is present.
       - Concrete reactant (e.g. "propanoyl-CoA"): only synonyms / ACP permutations count;
         shared parent categories (e.g. having Acetyl-CoA expand to "an acyl-CoA") must
         NOT make sibling molecules available.

    Args:
        reactant (str): Reactant name to check
        available_species_labels_lc (Set[str]): Set of available species (lowercase)
        enzyme_label (str, optional): Enzyme label for domain cofactor checking
        
    Returns:
        tuple[bool, Optional[str]]: (is_available, reason)
            - is_available: True if reactant is available
            - reason: One of:
              "always_available_cofactor", "direct_match", "domain_cofactor",
              "ontology", or None if unavailable.
    """
    r_lc = str(reactant).lower().strip()
    coenzyme_flags = get_coenzyme_like_flags(reactant)

    # Check if it's an always-available cofactor (implicitly available).
    # NOTE: This uses COFACTORS_ALWAYS_AVAILABLE, which is a restricted subset
    # of GENERAL_COFACTORS (H2O, H+, Pi, inorganic ions, etc.), so that high‑
    # energy carriers like ATP / NAD(H)/NADP(H) still need to be provided in
    # the input species list or produced in the network.
    if r_lc in COFACTORS_ALWAYS_AVAILABLE:
        return True, "always_available_cofactor"
    # Check if it's directly available
    if r_lc in available_species_labels_lc:
        return True, "direct_match"

    # Check if it's a domain cofactor (part of enzyme structure)
    # Skip for acyl-ACP: acetyl-ACP, hexanoyl-ACP etc. must come from prior reactions, not the domain
    if enzyme_label and not coenzyme_flags.get("is_acyl_acp", False):
        domain_cofactors = get_enzyme_domain_cofactors(enzyme_label)
        # Ensure patterns are lowercase for consistent matching
        domain_cofactors_lc = [pattern.lower() for pattern in domain_cofactors]
        if any(pattern in r_lc for pattern in domain_cofactors_lc):
            return True, "domain_cofactor"

    # Ontology equivalents. Category parents in the *available* set must not make
    # sibling concrete molecules look present (Acetyl-CoA ≠ propanoyl-CoA).
    # Category keys from load_ontology_categories are lowercase; aliases from
    # get_ontology_equivalents may preserve YAML casing — compare lowercased.
    equivalents = get_ontology_equivalents(r_lc)
    categories = load_ontology_categories()
    if r_lc in categories:
        match_labels = equivalents
    else:
        match_labels = [
            eq for eq in equivalents
            if str(eq).lower().strip() not in categories
        ]
    if any(eq in available_species_labels_lc for eq in match_labels):
        return True, "ontology"

    return False, None


def validate_reaction_reactants(
    reactants: List[str],
    available_species_labels_lc: Set[str],
    enzyme_label: Optional[str] = None,
) -> tuple[bool, List[str]]:
    """
    Validate that all reactants for a reaction are available.
    
    Args:
        reactants (List[str]): List of reactant names
        available_species_labels_lc (Set[str]): Set of available species (lowercase)
        enzyme_label (str, optional): Enzyme label for domain cofactor checking
        
    Returns:
        tuple[bool, List[str]]: (all_available, missing_reactants)
            - all_available: True if all reactants are available
            - missing_reactants: List of reactants that are not available
    """
    missing_reactants = []
    
    for reactant in reactants:
        is_available, reason = check_reactant_availability(
            reactant,
            available_species_labels_lc,
            enzyme_label=enzyme_label,
        )
        
        if not is_available:
            missing_reactants.append(reactant)
    
    return len(missing_reactants) == 0, missing_reactants
