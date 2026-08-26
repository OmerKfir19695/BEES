#!/usr/bin/env python3

"""Local CSV reaction database (load + query by EC / substrate)."""

import os
import csv
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass
from bees.logger import Logger
from bees.common import get_ontology_equivalents, canonical_smiles, get_chemical_aliases
from bees.cofactors import (
    ACYL_CHAIN_SMILES,
    ACP_SUFFIX,
    GENERAL_COFACTORS,
    PPANT_HANDLE,
)

@dataclass
class KineticData:
    """One reaction row from the CSV. Units: mM, kJ/mol, K."""
    
    ec_number: str
    enzyme_name: str
    reaction_string: str
    stoichiometry: Dict[str, int]
    meta: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    ph: Optional[float] = None
    km: Optional[float] = None
    km_per_substrate: Optional[Dict[str, float]] = None
    vmax: Optional[float] = None
    kcat: Optional[float] = None
    delta_g: Optional[float] = None
    kinetic_parameters_source: Optional[str] = None
    source: str = "database"
    km_sd: Optional[float] = None
    km_sd_per_substrate: Optional[Dict[str, float]] = None
    kcat_sd: Optional[float] = None
    ki_sd: Optional[float] = None
    compound_smiles: Optional[Dict[str, str]] = None
    
    def __repr__(self):
        return (f"KineticData(ec={self.ec_number}, reaction={self.reaction_string}, "
                f"Km={self.km} mM, ΔG={self.delta_g} kJ/mol)")

class ReactionDatabase:
    """Load/query enzyme reactions from CSV."""
    
    def __init__(self, logger: Logger, ontology: Optional[Dict[str, List[str]]] = None):
        if not isinstance(logger, Logger):
            raise TypeError(
                f"logger must be a BEES Logger instance, got {type(logger).__name__}. "
                f"Import from bees.logger and pass a Logger object."
            )
        
        self.data: List[Dict[str, Any]] = []
        self.reactions: List[KineticData] = []
        self._ec_index: Dict[str, List[KineticData]] = {}
        self.source_file: Optional[str] = None
        self.logger = logger
        self.ontology = ontology or {}
    
    def load_from_csv(self, csv_path: str) -> int:
        """Load reactions from CSV; return count."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Reaction database file not found: {csv_path}")
        
        self.source_file = csv_path
        self.data = []
        self.reactions = []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append(row)
                    
                    # Parse row into KineticData object
                    try:
                        kinetic_data = self._parse_row(row, len(self.reactions) + 2)
                        self.reactions.append(kinetic_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse row in {csv_path}: {row}. Error: {e}")
                        continue
            
            self._ec_index = {}
            for rxn in self.reactions:
                self._ec_index.setdefault(rxn.ec_number, []).append(rxn)

            self.logger.info(f"Loaded {len(self.reactions)} reactions from {csv_path}")
            return len(self.reactions)
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {e}")

    def _parse_row(self, row: Dict[str, Any], row_index: int) -> KineticData:
        """Parse a single CSV row into a KineticData object."""
        # Pull meta JSON OR fall back to legacy `kinetic_parameters_source`
        meta_raw = (row.get("meta") or "").strip()
        kinetic_parameters_source = (row.get("kinetic_parameters_source") or "").strip()
        meta = self._parse_json_blob(meta_raw) or self._parse_json_blob(kinetic_parameters_source)

        # Reaction string: explicit column OR meta["reaction"]
        reaction_string = (row.get("reaction") or row.get("reaction_string") or "").strip()
        if not reaction_string and isinstance(meta, dict):
            reaction_string = str(meta.get("reaction") or "").strip()

        # Stoichiometry: explicit column OR meta["stoichiometry"] OR legacy parse fallback
        stoich_raw = (row.get("stoichiometry") or "").strip()
        stoichiometry = self._parse_stoichiometry(
            stoich_raw=stoich_raw,
            meta=meta,
            fallback_blob=kinetic_parameters_source,
            row_index=row_index,
        )

        ec_number = (row.get("ec_number") or "").strip()
        enzyme_name = (row.get("enzyme") or row.get("enzyme_name") or "").strip()

        compound_smiles = self._correct_acp_smiles(self._parse_smiles_map(
            row.get("compound_smiles"),
            row.get("reactants_smiles"),
            row.get("products_smiles"),
        ))

        return KineticData(
            ec_number=ec_number,
            enzyme_name=enzyme_name,
            reaction_string=reaction_string,
            stoichiometry=stoichiometry or {},
            meta=meta if isinstance(meta, dict) else None,
            kinetic_parameters_source=kinetic_parameters_source or None,
            source="database",
            compound_smiles=compound_smiles,
        )

    def _parse_json_blob(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        raw = raw.strip()
        if not raw.startswith("{"):
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _parse_smiles_map(
        self,
        compound_smiles_raw: Any,
        reactants_smiles_raw: Any,
        products_smiles_raw: Any,
    ) -> Optional[Dict[str, str]]:
        compound_smiles: Dict[str, str] = {}

        def merge_smiles(raw: Any) -> None:
            if not raw:
                return
            parsed = self._parse_json_blob(str(raw).strip())
            if not isinstance(parsed, dict):
                return
            for key, value in parsed.items():
                if key and value:
                    compound_smiles[str(key)] = str(value)

        merge_smiles(compound_smiles_raw)
        if not compound_smiles:
            merge_smiles(reactants_smiles_raw)
            merge_smiles(products_smiles_raw)

        return compound_smiles or None

    def _correct_acp_smiles(
        self,
        compound_smiles: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Rebuild ACP-thioester SMILES from ACYL_CHAIN_SMILES + PPANT_HANDLE.

        ACP-pathway identity fix (FAS-II style labels). No-op unless the
        compound name ends with ACP_SUFFIX and the acyl prefix is in the table.
        """
        if not compound_smiles:
            return compound_smiles
        corrected: Dict[str, str] = {}
        for name, smi in compound_smiles.items():
            name_lc = name.lower().strip()
            if name_lc.endswith(ACP_SUFFIX):
                acyl = name_lc[: -len(ACP_SUFFIX)]
                chain = ACYL_CHAIN_SMILES.get(acyl)
                if chain is not None:
                    corrected[name] = chain + PPANT_HANDLE
                    continue
            corrected[name] = smi
        return corrected

    def _parse_stoichiometry(
        self,
        stoich_raw: str,
        meta: Optional[Dict[str, Any]],
        fallback_blob: Optional[str],
        row_index: int,
    ) -> Optional[Dict[str, int]]:
        """Parse stoichiometry from column / meta / legacy blob."""
        candidates: List[Any] = []
        if stoich_raw:
            candidates.append(self._parse_json_blob(stoich_raw) or stoich_raw)
        if isinstance(meta, dict) and "stoichiometry" in meta:
            candidates.append(meta.get("stoichiometry"))
        if fallback_blob:
            fb = self._parse_json_blob(fallback_blob)
            if fb is not None:
                candidates.append(fb.get("stoichiometry", fb))

        for cand in candidates:
            if isinstance(cand, dict):
                stoich: Dict[str, int] = {}
                for species, coeff in cand.items():
                    if not species:
                        continue
                    try:
                        coeff_int = int(coeff)
                    except (ValueError, TypeError):
                        continue
                    if coeff_int == 0:
                        continue
                    stoich[str(species)] = coeff_int
                if stoich:
                    return stoich

        self.logger.debug(f"Row {row_index}: No valid stoichiometry found.")
        return None
    
    def query_by_enzyme_substrate(
        self, 
        ec_number: str, 
        substrate_label: str,
        cofactor: Optional[str] = None,
        strict: bool = False,
        temperature_range: Optional[Tuple[float, float]] = None,
        ph_range: Optional[Tuple[float, float]] = None,
        available_species_labels_lc: Optional[Set[str]] = None,
        return_all: bool = False,
        substrate_smiles: Optional[str] = None,
    ) -> Union[Optional[KineticData], List[KineticData]]:
        """Query by EC + substrate (SMILES then name/ontology)."""
        self.logger.debug(f"Querying database: EC={ec_number}, Substrate={substrate_label}, "
                         f"Cofactor={cofactor}, TempRange={temperature_range}, pHRange={ph_range}")

        
        # Normalize inputs
        if not ec_number:
            return [] if return_all else None
        ec_number = ec_number.strip()
        substrate_label = substrate_label.strip()
        substrate_canonical_smi = canonical_smiles(substrate_smiles) if substrate_smiles else None

        # Get all equivalents for this substrate (name + ontology fallback)
        substrate_equivalents = get_ontology_equivalents(substrate_label)
        self.logger.debug(f"Substrate '{substrate_label}' equivalents: {substrate_equivalents}")
        
        # Collect all matching reactions and prioritize them
        # Match by SMILES first (structural identity), then by name + ontology
        matches = []
        for reaction in self._ec_index.get(ec_number, []):

            is_match = False
            needs_reversal = False

            # 1) SMILES-based match (primary)
            if substrate_canonical_smi and reaction.stoichiometry and getattr(reaction, "compound_smiles", None):
                stoich_lc = {str(k).lower().strip(): v for k, v in reaction.stoichiometry.items()}
                for name, smi in reaction.compound_smiles.items():
                    if not smi:
                        continue
                    name_lc = str(name).lower().strip()
                    if name_lc in GENERAL_COFACTORS:
                        continue
                    coeff = stoich_lc.get(name_lc)
                    if coeff is None:
                        continue
                    can_smi = canonical_smiles(smi)
                    if can_smi and can_smi == substrate_canonical_smi:
                        if coeff < 0:
                            is_match = True
                            break
                        if coeff > 0:
                            is_match = True
                            needs_reversal = True
                            break
                if is_match:
                    pass  # use this match
                else:
                    # 2) Name + ontology match (fallback)
                    if reaction.stoichiometry:
                        stoich_lc = {str(k).lower().strip(): v for k, v in reaction.stoichiometry.items()}
                        for equiv in substrate_equivalents:
                            e_lc = str(equiv).lower().strip()
                            if e_lc in stoich_lc:
                                if stoich_lc[e_lc] < 0:
                                    if e_lc in GENERAL_COFACTORS:
                                        continue
                                    is_match = True
                                    break
                                elif stoich_lc[e_lc] > 0:
                                    if e_lc in GENERAL_COFACTORS:
                                        continue
                                    is_match = True
                                    needs_reversal = True
                                    break
            else:
                # No substrate SMILES or no compound_smiles: use name + ontology only
                if reaction.stoichiometry:
                    stoich_lc = {str(k).lower().strip(): v for k, v in reaction.stoichiometry.items()}
                    for equiv in substrate_equivalents:
                        e_lc = str(equiv).lower().strip()
                        if e_lc in stoich_lc:
                            if stoich_lc[e_lc] < 0:
                                if e_lc in GENERAL_COFACTORS:
                                    continue
                                is_match = True
                                break
                            elif stoich_lc[e_lc] > 0:
                                if e_lc in GENERAL_COFACTORS:
                                    continue
                                is_match = True
                                needs_reversal = True
                                break
            
            if is_match:
                # Check temperature range if specified
                if temperature_range and reaction.temperature is not None:
                    temp_min, temp_max = temperature_range
                    if not (temp_min <= reaction.temperature <= temp_max):
                        self.logger.debug(f"Database entry temperature {reaction.temperature} K outside "
                                        f"range [{temp_min}, {temp_max}] K")
                        continue
                
                # Check pH range if specified
                if ph_range and reaction.ph is not None:
                    ph_min, ph_max = ph_range
                    if not (ph_min <= reaction.ph <= ph_max):
                        self.logger.debug(f"Database entry pH {reaction.ph} outside "
                                        f"range [{ph_min}, {ph_max}]")
                        continue
                
                # Calculate priority score for ranking matches
                # Higher score = better match
                priority_score = 0
                
                # High priority: prefer reactions where all/most reactants are in provided species
                if available_species_labels_lc and reaction.stoichiometry:
                    # reactants are species with negative coefficients
                    reaction_reactants_lc = {
                        str(k).lower().strip() 
                        for k, v in reaction.stoichiometry.items() 
                        if (v < 0 if not needs_reversal else v > 0)
                    }
                    
                    # Count how many of the reaction's reactants are in our available set
                    provided_match_count = sum(
                        1 for r in reaction_reactants_lc if r in available_species_labels_lc
                    )
                    
                    # significantly boost score if more reactants match our inputs
                    priority_score += provided_match_count * 100
                
                # Prefer reactions with more complete stoichiometry
                if reaction.stoichiometry:
                    num_species = len(reaction.stoichiometry)
                    priority_score += num_species * 0.1  # More species = slightly higher priority
                
                # Prefer normal direction over reversed
                if not needs_reversal:
                    priority_score += 10
                
                # Prefer reactions with kinetic parameters
                has_km = reaction.km is not None or (getattr(reaction, "km_per_substrate", None) and reaction.km_per_substrate)
                if has_km or reaction.kcat is not None:
                    priority_score += 5
                
                matches.append((reaction, needs_reversal, priority_score))

        # If we have matches, sort by priority (highest first) then deduplicate by (EC, substrate identity)
        if matches:
            matches.sort(key=lambda x: -x[2])  # Sort by priority_score descending

            def _dedup_key(rxn: KineticData, needs_rev: bool) -> Tuple:
                stoich = rxn.stoichiometry or {}
                cs = getattr(rxn, "compound_smiles", None) or {}
                items = []
                for name, coeff in sorted(stoich.items(), key=lambda x: x[0].lower()):
                    c = -coeff if needs_rev else coeff
                    smi = cs.get(name)
                    can = canonical_smiles(smi) if smi else None
                    items.append((can or name.lower().strip(), c))
                return (rxn.ec_number, tuple(items))

            seen_keys: Set[Tuple[str, Tuple[str, ...]]] = set()
            result_list = []
            for match_rxn, needs_reversal, score in matches:
                # Skip reversed matches: the forward direction will be discovered
                # naturally when the DB row's true reactants are processed as
                # substrates in a later iteration. Emitting a reversed clone
                # creates duplicate forward/reverse pairs that require post-hoc
                # deduplication and inflates the reaction count.
                if needs_reversal:
                    continue

                dkey = _dedup_key(match_rxn, needs_reversal)
                if dkey in seen_keys:
                    continue
                seen_keys.add(dkey)

                final_rxn = match_rxn

                # Filter by cofactor if strict
                if strict and cofactor:
                    stoich_lc = {str(k).lower().strip(): v for k, v in final_rxn.stoichiometry.items()}
                    c_lc = str(cofactor).lower().strip()
                    if not (c_lc in stoich_lc and stoich_lc[c_lc] < 0):
                        continue
                
                result_list.append(final_rxn)
                if not return_all:
                    break
            
            if not result_list:
                return [] if return_all else None
                
            return result_list if return_all else result_list[0]
        
        return [] if return_all else None
    
    def query_by_enzyme(self, ec_number: str) -> List[KineticData]:
        """All reactions for an EC number."""
        ec_number = ec_number.strip()
        matches = [rxn for rxn in self.reactions if rxn.ec_number == ec_number]
        self.logger.debug(f"Found {len(matches)} reactions for enzyme {ec_number}")
        return matches
    
    def query_by_substrate(self, substrate_label: str) -> List[KineticData]:
        """All reactions with substrate as reactant (ontology aliases)."""
        substrate_label = substrate_label.strip()
        
        # Get all aliases for this substrate (including chemical class categories)
        substrate_aliases = get_chemical_aliases(substrate_label)
        
        matches = []
        for rxn in self.reactions:
            if rxn.stoichiometry:
                stoich_lc = {str(k).lower().strip(): v for k, v in rxn.stoichiometry.items()}
                for alias in substrate_aliases:
                    a_lc = str(alias).lower().strip()
                    if a_lc in stoich_lc and stoich_lc[a_lc] < 0:
                        if a_lc not in GENERAL_COFACTORS:
                            matches.append(rxn)
                            break  # Avoid duplicate matches

        self.logger.debug(f"Found {len(matches)} reactions for substrate {substrate_label} (checked {len(substrate_aliases)} aliases)")
        return matches
    
    def get_all_reactions(self) -> List[KineticData]:
        """All loaded reactions."""
        return self.reactions
    
    def summary(self) -> Dict[str, Any]:
        """Counts: reactions, unique enzymes, unique non-cofactor substrates."""
        total = len(self.reactions)
        unique_enzymes = len(set(r.ec_number for r in self.reactions))
        unique_substrates_set = set()
        for r in self.reactions:
            if not r.stoichiometry:
                continue
            for name, coeff in r.stoichiometry.items():
                if coeff < 0 and name.lower().strip() not in GENERAL_COFACTORS:
                    unique_substrates_set.add(name)
        unique_substrates = len(unique_substrates_set)
        
        return {
            'total_reactions': total,
            'unique_enzymes': unique_enzymes,
            'unique_substrates': unique_substrates,
            'source_file': self.source_file
        }
