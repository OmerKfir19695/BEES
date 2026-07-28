"""Shared helpers for the BEES rule layer.

Pure functions and chemistry constants used by BOTH the physics laws
(`physics_rules.py`, e.g. HydrophobicChainLengthKm, DecarboxylationIrreversible)
and the fitted calibrations (`calibrations.py`). Kept in their own module so the
calibration layer does not have to import private helpers from the law layer
(and to avoid an import cycle, since the law layer also uses these).
"""

from __future__ import annotations

from typing import Optional

from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE

# Canonical SMILES for CO2. A reaction that releases CO2 (decarboxylation) is
# physiologically one-way: the gas escapes, so the reverse condensation never
# runs at cellular CO2 partial pressure regardless of the computed ΔG°'.
_CO2_SMILES = "O=C=O"

# Labels treated as CO2 even when no SMILES is available. Lower-cased / stripped
# before comparison. Kept alongside the SMILES check so detection still works
# when a CO2 species was added without a resolved SMILES.
_CO2_LABELS = frozenset({"carbon dioxide", "co2", "carbon-dioxide", "co₂"})

# Carrier suffixes stripped to recover the bare acyl name (mirrors the
# AcylACPSubstitutor pattern in bees.substitutor_registry). Order matters:
# try the bracketed forms before the bare "-coa".
_CARRIER_SUFFIXES = ("-[acp]", "-[coa]", "-coa")

# Residual names that are bare carriers / not an acyl chain → no chain length.
_BARE_CARRIERS = frozenset({"", "holo", "acp", "coa", "holo-"})


def detect_acyl_chain_length(label: str, smiles: Optional[str] = None) -> Optional[int]:
    """Return the acyl carbon count (incl. the carbonyl carbon) for a fatty-acyl
    species, or None if the species is not an acyl chain.

    Primary route (authoritative for all FAS species): strip the carrier suffix
    (-[acp]/-coa) and look the residual acyl name up in bees.cofactors.
    ACYL_CHAIN_SMILES, returning the carbon count of the fragment. Functional-group
    modifiers (3-oxo / 3-hydroxy / 2-enoyl / cis) do not change the carbon count.

    Fallback (only when not in the table and a SMILES is given): extract the acyl
    moiety from the SMILES — find the thioester carbonyl (C(=O) bonded to S) or the
    free-acid carboxyl (C(=O) bonded to a single-bonded O), then count carbons in
    the carbon-connected component containing that carbonyl carbon. The S / carboxyl
    oxygens break the carbon path, so this yields ONLY the acyl carbons and never
    the ~20+ carbons of a CoA/pantetheine tail. Returns None if no anchor is found
    or the SMILES fails to parse — it NEVER falls back to total-molecule carbons.

    Returns None for cofactors and bare carriers (NADPH, H2O, CO2, holo-[ACP],
    ACP, CoA) so the rule naturally skips them.
    """
    if not label:
        return None
    lab = label.lower().strip()
    if lab in COFACTORS_ALWAYS_AVAILABLE or lab in _CO2_LABELS:
        return None

    acyl = lab
    for suf in _CARRIER_SUFFIXES:
        if acyl.endswith(suf):
            acyl = acyl[: -len(suf)].strip()
            break
    if acyl in _BARE_CARRIERS:
        return None

    from bees.cofactors import ACYL_CHAIN_SMILES

    frag = ACYL_CHAIN_SMILES.get(acyl)
    if frag is not None:
        n = frag.count("C")
        return n if n > 0 else None

    if smiles:
        return _acyl_chain_from_smiles(smiles)
    return None


def _acyl_chain_from_smiles(smiles: str) -> Optional[int]:
    """Count acyl carbons by extracting the acyl moiety from a full SMILES.

    Anchors on the thioester carbonyl (C(=O)–S) or free-acid carboxyl (C(=O)–O),
    then floods the carbon-only connected component from that carbonyl carbon.
    Heteroatoms (S, the carboxyl/carbonyl O, N, P) break the carbon path, so the
    component is exactly the acyl chain — not the CoA/ACP scaffold.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    anchor_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        has_carbonyl = False
        bonded_S = False
        bonded_O_single = False
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if nb.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
                has_carbonyl = True
            elif nb.GetSymbol() == "S":
                bonded_S = True
            elif nb.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 1.0:
                bonded_O_single = True
        if has_carbonyl and (bonded_S or bonded_O_single):
            anchor_idx = atom.GetIdx()
            break

    if anchor_idx is None:
        return None

    # Flood-fill carbons reachable from the anchor via C–C bonds only.
    seen: set[int] = set()
    stack = [anchor_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == "C" and nb.GetIdx() not in seen:
                stack.append(nb.GetIdx())
    return len(seen) if seen else None


def _ec_norm(ec) -> Optional[str]:
    """Normalize an EC string to bare dotted form ('EC 3.1.2.14' -> '3.1.2.14')."""
    if not ec:
        return None
    s = str(ec).strip().upper()
    if s.startswith("EC "):
        s = s[3:].strip()
    return s or None


def _reaction_acyl_substrate_n(reaction) -> Optional[int]:
    """Longest acyl-chain length among the reaction's substrates (negative stoich,
    non-cofactor), or None if no acyl substrate is detectable.

    Using the longest substrate picks the elongating acyl chain (e.g. dodecanoyl-
    [ACP], n=12) over the small extender (malonyl-[ACP], n=3) in a condensation,
    and the acyl-[ACP] over H2O in a hydrolysis.
    """
    kin = getattr(reaction, "kinetics", None)
    stoich = getattr(reaction, "stoichiometry", None) or {}
    smiles_map = (getattr(kin, "compound_smiles", None) or {}) if kin is not None else {}
    best: Optional[int] = None
    for lab, coeff in stoich.items():
        if coeff >= 0:
            continue  # products / cofactors with non-negative coeff
        if str(lab).lower().strip() in COFACTORS_ALWAYS_AVAILABLE:
            continue
        n = detect_acyl_chain_length(lab, smiles_map.get(lab))
        if n is not None and (best is None or n > best):
            best = n
    return best
