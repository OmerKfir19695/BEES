"""Shared helpers for the BEES rule layer (laws + calibrations)."""

from __future__ import annotations

from typing import Optional

from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE

_CO2_SMILES = "O=C=O"
_CO2_LABELS = frozenset({"carbon dioxide", "co2", "carbon-dioxide", "co₂"})
_CARRIER_SUFFIXES = ("-[acp]", "-[coa]", "-coa")
_BARE_CARRIERS = frozenset({"", "holo", "acp", "coa", "holo-"})

def detect_acyl_chain_length(label: str, smiles: Optional[str] = None) -> Optional[int]:
    """Acyl carbon count (incl. carbonyl), or None."""
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
    """Count acyl carbons from SMILES (thioester/carboxyl anchor)."""
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
    """'EC 3.1.2.14' -> '3.1.2.14'."""
    if not ec:
        return None
    s = str(ec).strip().upper()
    if s.startswith("EC "):
        s = s[3:].strip()
    return s or None

def _reaction_acyl_substrate_n(reaction) -> Optional[int]:
    """Longest acyl-chain length among substrates, or None."""
    kin = getattr(reaction, "kinetics", None)
    stoich = getattr(reaction, "stoichiometry", None) or {}
    smiles_map = (getattr(kin, "compound_smiles", None) or {}) if kin is not None else {}
    best: Optional[int] = None
    for lab, coeff in stoich.items():
        if coeff >= 0:
            continue
        if str(lab).lower().strip() in COFACTORS_ALWAYS_AVAILABLE:
            continue
        n = detect_acyl_chain_length(lab, smiles_map.get(lab))
        if n is not None and (best is None or n > best):
            best = n
    return best
