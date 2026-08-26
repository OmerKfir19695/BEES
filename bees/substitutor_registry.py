"""
Concrete CompoundSubstitutor implementations for BEES.

This module contains biology-specific SMILES substitution strategies.
The abstract base class (CompoundSubstitutor). It's can be relvent in 
many occasions and each class should be for each occasion.

To add a new substitutor for a different pathway or carrier system:
  1. Maske a new class CompoundSubstitutor here
  2. Import and use it in bees/enlarger.py _build_substitutor()
"""

from typing import Dict, Optional

from bees.thermodynamics import CompoundSubstitutor
from bees.cofactors import ACP_SUFFIX, COA_TAIL, ACYL_CHAIN_SMILES, ACYL_FREE_ACID_SMILES

import logging
logger = logging.getLogger("BEES")


class AcylACPSubstitutor(CompoundSubstitutor):
    """Maps acyl-[ACP] labels to a SMILES eQuilibrator can resolve.

    Justify:
    The ΔG°' of an acyl-ACP thioester reaction equals the ΔG°' of the
    corresponding acyl-CoA reaction — the phosphopantetheine thioester bond
    energy is the same whether the carrier is ACP or CoA; the protein scaffold
    does not contribute to the reaction free energy. This holds for every
    reaction class (transfer, condensation, reduction, dehydration), so the
    acyl-CoA substitution is thermodynamically exact rather than approximate.

    Resolution strategy:
      - For ACP labels (`-[acp]` suffix), the substitutor ALWAYS overrides
        whatever SMILES the caller passed in. Database rows frequently attach
        a truncated phosphopantetheine-only fragment that eQuilibrator cannot
        resolve; replacing it with the full acyl-CoA SMILES (built from
        `ACYL_CHAIN_SMILES[acyl] + COA_TAIL`) makes the look-up succeed.
        Order of preference:
          1. User-supplied explicit override (extra_mappings).
          2. Full acyl-CoA thioester (ACYL_CHAIN_SMILES + COA_TAIL).
          3. Free-acid SMILES (ACYL_FREE_ACID_SMILES) as last-resort fallback
             for acyl groups missing from ACYL_CHAIN_SMILES.
          4. None if the acyl fragment is unknown.
      - For non-ACP labels, do nothing (pass through whatever the caller had).

    extra_mappings: label → SMILES dict injected from input YAML field
    `thermo_smiles_substitutions`. Takes precedence over the built-in tables.
    """

    def __init__(self, extra_mappings: Optional[Dict[str, str]] = None):
        self._extra: Dict[str, str] = {
            k.lower().strip(): v
            for k, v in (extra_mappings or {}).items()
        }

    def substitute(self, label: str, smiles: Optional[str]) -> Optional[str]:
        label_lc = label.lower().strip()

        if label_lc in self._extra:
            return self._extra[label_lc]

        if not label_lc.endswith(ACP_SUFFIX):
            return None

        acyl = label_lc[: -len(ACP_SUFFIX)]

        # holo-[ACP] (the bare carrier) → free CoA, so the ACP-side scaffold
        # matches the full acyl-CoA we substitute on the other side. Without
        # this, holo-[ACP] keeps its user-supplied phosphopantetheine fragment
        # (no adenine-ribose-ADP) and the reaction is rejected as unbalanced
        # against the full acyl-CoAs.
        if acyl == "holo":
            return COA_TAIL

        chain = ACYL_CHAIN_SMILES.get(acyl)
        if chain is not None:
            if smiles:
                logger.debug(
                    "AcylACPSubstitutor: overriding DB-supplied SMILES for %r "
                    "with acyl-CoA construction (DB form was %r)",
                    label, smiles,
                )
            return chain + COA_TAIL

        free_acid = ACYL_FREE_ACID_SMILES.get(acyl)
        if free_acid is not None:
            return free_acid

        logger.debug(
            "AcylACPSubstitutor: no SMILES mapping for acyl group %r (label=%r)",
            acyl, label,
        )
        return None
