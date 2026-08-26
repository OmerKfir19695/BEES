#!/usr/bin/env python3

"""Builds reactions from species + enzymes (batch discovery or iterative enlargement)."""

from __future__ import annotations

from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from types import SimpleNamespace

from bees.reaction_template import (
    create_reaction_from_database,
    ReactionTemplate,
)
from db.reaction_database import ReactionDatabase, KineticData
from bees.common import (
    get_ontology_equivalents, 
    canonical_smiles, 
    heavy_atom_count
)
from bees.cofactors import GENERAL_COFACTORS, COFACTORS_ALWAYS_AVAILABLE
from bees.cofactors import get_coenzyme_like_flags
from bees.reaction_utils import (
    get_ec_aliases,
    validate_reaction_reactants
)
from bees.kinetics_estimator import build_estimator
from bees.thermodynamics import ThermoData

@dataclass
class GeneratedReaction:
    enzyme_label: str
    substrate_label: str
    ec_number: Optional[str]
    template: ReactionTemplate
    kinetics: Optional[KineticData]
    reactant_labels: List[str]
    product_labels: List[str]
    stoichiometry: Dict[str, int]
    rate_law: Optional[str] = None  # Only set if kinetics available from database
    thermo: Optional[ThermoData] = None
    # Opt-in; enlarger attaches the map (match by enzyme label).
    feedback_inhibitors: Optional[Dict[str, tuple]] = None

    def __repr__(self):
        return f"Reaction: {' + '.join(self.reactant_labels)} → {' + '.join(self.product_labels)}"

class ReactionGenerator:
    """Builds biochemical reactions from BEES input species and enzymes."""

    def __init__(self, bees_object, logger, output_directory):
        self.bees_object = bees_object
        self.logger = logger
        self.output_directory = output_directory
        self.kinetic_db = None
        self.reactions: List[GeneratedReaction] = []
        self.kinetics_estimator = None
        self._species_alias_to_canonical_label: Dict[str, str] = {}
        self._species_smiles_to_canonical_label: Dict[str, str] = {}

        # Seed registry from user-provided species so their names become preferred.
        for sp in getattr(self.bees_object, "species", []) or []:
            label = getattr(sp, "label", None)
            if not label:
                continue
            smiles = getattr(sp, "smiles", None)
            self._register_species_label(label=label, smiles=smiles)
    
    def load_kinetic_database(self, db_path: str, ontology: Optional[Dict[str, List[str]]] = None) -> int:
        """Load kinetic database from CSV file."""
        self.logger.info(f"Loading kinetic database from {db_path}")
        self.kinetic_db = ReactionDatabase(logger=self.logger, ontology=ontology)
        num_reactions = self.kinetic_db.load_from_csv(db_path)
        summary = self.kinetic_db.summary()
        self.logger.info(f"Database loaded: {summary['total_reactions']} reactions, "
                        f"{summary['unique_enzymes']} unique enzymes, "
                        f"{summary['unique_substrates']} unique substrates")
        
        return num_reactions

    def ensure_estimator_initialized(self) -> None:
        if self.kinetics_estimator is not None:
            return
        try:
            if hasattr(self.bees_object, "settings") and getattr(self.bees_object.settings, "estimate_kinetics", False):
                include_sd = getattr(self.bees_object.settings, "kinetics_include_sd", False)
                ec_kcat_scale = getattr(self.bees_object.settings, "kinetics_ec_kcat_scale", None)
                if ec_kcat_scale:
                    # DEPRECATED: YAML per-EC kcat multipliers; use settings.calibrations.
                    import warnings
                    msg = (
                        "settings.kinetics_ec_kcat_scale is deprecated and experiment-only: "
                        "it applies un-reviewed per-EC kcat multipliers straight from the input "
                        "(a fitting surface). Use settings.calibrations (declared, parameter-locked "
                        "calibration rules) for production fitted corrections."
                    )
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)
                    self.logger.warning(f"DEPRECATION: {msg}")
                self.kinetics_estimator = build_estimator(
                    getattr(self.bees_object.settings, "kinetics_estimator", None),
                    include_sd=include_sd,
                    ec_kcat_scale=ec_kcat_scale,
                )
                if self.kinetics_estimator:
                    self.logger.info(f"Kinetics estimation enabled: {self.kinetics_estimator.name}")
                else:
                    self.logger.info("Kinetics estimation enabled, but no estimator selected.")
            else:
                self.kinetics_estimator = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize kinetics estimator: {e}")
            self.kinetics_estimator = None

    def generate_reactions(self) -> List[GeneratedReaction]:
        """Iteratively discover reactions; products from each round become substrates for the next."""
        self.logger.info("=" * 60)
        self.logger.info("REACTION GENERATION")
        self.logger.info("=" * 60)
        
        def reaction_signature(reaction: GeneratedReaction) -> tuple:
            # Signature includes enzyme so isozymes (FabA/FabZ, FabB/FabF) stay distinct.
            enzyme = str(reaction.enzyme_label).lower().strip()
            reactants_tuple = tuple(sorted(str(r).lower().strip() for r in reaction.reactant_labels))
            products_tuple = tuple(sorted(str(p).lower().strip() for p in reaction.product_labels))
            return (enzyme, reactants_tuple, products_tuple)

        initial_substrates = [
            s
            for s in self.bees_object.species
            if s.reactive and not s.solvent and s.label.lower().strip() not in GENERAL_COFACTORS
        ]
        excluded_general_cofactors = [
            s.label
            for s in self.bees_object.species
            if s.reactive and not s.solvent and s.label.lower().strip() in GENERAL_COFACTORS
        ]

        enzymes = [e for e in self.bees_object.enzymes if e.reactive]
        
        substrate_names = [s.label for s in initial_substrates]
        enzyme_names = [e.label for e in enzymes]
        self.logger.info(
            f"Reactive substrates ({len(initial_substrates)}): {', '.join(substrate_names)}"
        )
        self.logger.info(
            f"Reactive enzymes ({len(enzymes)}): {', '.join(enzyme_names)}"
        )
        if excluded_general_cofactors:
            self.logger.info(
                f"General cofactors (available but not triggers): {', '.join(excluded_general_cofactors)}"
            )
        self.logger.info("")

        self.ensure_estimator_initialized()

        available_species_labels_lc = set()
        provided_species_labels_lc = set()
        for s in getattr(self.bees_object, "species", []):
            if getattr(s, "label", None):
                label_lc = s.label.lower().strip()
                provided_species_labels_lc.add(label_lc)
                available_species_labels_lc.update(get_ontology_equivalents(s.label))
        
        existing_signatures = {reaction_signature(r) for r in self.reactions}
        iteration = 0
        max_iterations = int(
            getattr(getattr(self.bees_object, "settings", None), "max_iterations", 50) or 50
        )
        reaction_count = len(self.reactions)
        while iteration < max_iterations:
            iteration += 1
            new_reactions_this_iteration = 0
            substrates_this_iteration = initial_substrates.copy()
            new_products_added = 0
            substrates_this_iteration_lc = {
                s.label.lower().strip()
                for s in substrates_this_iteration
                if getattr(s, "label", None)
            }
            
            for reaction in self.reactions:
                for product_label in reaction.product_labels:
                    product_lc = product_label.lower().strip()
                    if product_lc not in substrates_this_iteration_lc:
                        coenzyme_flags = get_coenzyme_like_flags(product_label)
                        is_coenzyme_like = coenzyme_flags["is_coenzyme_like"]
                        if product_lc not in GENERAL_COFACTORS and not is_coenzyme_like:
                            virtual_substrate = SimpleNamespace(
                                label=product_label,
                                reactive=True,
                                solvent=False,
                                smiles=None
                            )
                            substrates_this_iteration.append(virtual_substrate)
                            substrates_this_iteration_lc.add(product_lc)
                            new_products_added += 1
                        
                        provided_species_labels_lc.add(product_lc)
                        available_species_labels_lc.update(get_ontology_equivalents(product_label))
              
            
            self.logger.info(f"Iteration {iteration}: Checking {len(substrates_this_iteration)} potential substrate(s) x {len(enzymes)} enzyme(s)...")
            
            for enzyme in enzymes:
                for substrate in substrates_this_iteration:
                    new_reactions = self._generate_reactions(
                        enzyme, 
                        substrate, 
                        available_species_labels_lc,
                        provided_species_labels_lc
                    )
                    for reaction in new_reactions:
                        signature = reaction_signature(reaction)
                        if signature in existing_signatures:
                            continue
                        self.reactions.append(reaction)
                        existing_signatures.add(signature)
                        reaction_count += 1
                        new_reactions_this_iteration += 1
                        self.logger.info(f"  [{reaction_count}] {reaction}")
                        if reaction.kinetics:
                            kin = reaction.kinetics
                            substrate_label_for_display = reaction.substrate_label
                            enzyme_label_for_display = reaction.enzyme_label
                            kin_parts = []
                            km_per = getattr(kin, "km_per_substrate", None)
                            if km_per:
                                for rname, val in km_per.items():
                                    kin_parts.append(f"Km({rname})={val} mM")
                            elif kin.km is not None:
                                kin_parts.append(f"Km({substrate_label_for_display})={kin.km} mM")
                            if kin.kcat is not None:
                                kin_parts.append(f"kcat({enzyme_label_for_display})={kin.kcat} 1/s")
                            if kin.vmax is not None:
                                kin_parts.append(f"Vmax={kin.vmax} mM/s")
                            if kin.delta_g is not None:
                                kin_parts.append(f"dG={kin.delta_g} kJ/mol")
                            sd_parts = []
                            km_sd_per = getattr(kin, "km_sd_per_substrate", None)
                            if km_sd_per:
                                for rname, val in km_sd_per.items():
                                    sd_parts.append(f"Km_sd({rname})={val:.4g} mM")
                            elif getattr(kin, "km_sd", None) is not None:
                                sd_parts.append(f"Km_sd={kin.km_sd:.4g} mM")
                            if getattr(kin, "kcat_sd", None) is not None:
                                sd_parts.append(f"kcat_sd={kin.kcat_sd:.4g} 1/s")
                            sd_part = f" [SD: {', '.join(sd_parts)}]" if sd_parts else ""
                            self.logger.info(
                                f"      ├─ Kinetics: {', '.join(kin_parts)} "
                                f"(source={kin.source}){sd_part}"
                            )
                        else:
                            self.logger.info(f"      ├─ No database match (will need parameter estimation)")
                        self.logger.info(f"      └─ Type: {reaction.template.template_type} "
                                       f"({reaction.template.ec_class.name})")
            
            if new_reactions_this_iteration == 0:
                self.logger.info(f"No new reactions found in iteration {iteration}. Stopping discovery.")
                break
        
        self.logger.info("")
        self.logger.info(f"Generated {len(self.reactions)} total reaction(s) in {iteration} iteration(s)")
        return self.reactions
    
    def _resolve_smiles_for_compound(
        self,
        compound_label: str,
        kinetic_data,
        substrate_label: str,
    ) -> Optional[str]:
        """Resolve SMILES from species, compound_smiles, or ontology."""
        compound_lc = compound_label.lower().strip()
        for species in getattr(self.bees_object, "species", []):
            if hasattr(species, "label") and species.label.lower().strip() == compound_lc:
                if hasattr(species, "smiles") and species.smiles:
                    return species.smiles
                break
        compound_smiles = getattr(kinetic_data, "compound_smiles", None)
        if isinstance(compound_smiles, dict):
            s = compound_smiles.get(compound_label)
            if s:
                return s
            equivalents_lc = {str(e).lower().strip() for e in get_ontology_equivalents(compound_label)}
            equivalents_lc.add(compound_lc)
            for name, smiles in compound_smiles.items():
                if not smiles:
                    continue
                if str(name).lower().strip() in equivalents_lc:
                    return smiles
        return None

    def _register_species_label(
        self,
        label: str,
        smiles: Optional[str] = None,
    ) -> str:
        """Return the canonical display label, unifying by SMILES then ontology."""
        label_clean = str(label).strip()
        label_lc = label_clean.lower()
        if not label_clean:
            return label_clean

        existing = self._species_alias_to_canonical_label.get(label_lc)
        if existing is not None:
            return existing

        can_smi = canonical_smiles(smiles) if smiles else None
        if can_smi:
            existing_from_smiles = self._species_smiles_to_canonical_label.get(can_smi)
            if existing_from_smiles is not None:
                self._species_alias_to_canonical_label[label_lc] = existing_from_smiles
                for eq in get_ontology_equivalents(label_clean):
                    self._species_alias_to_canonical_label[str(eq).lower().strip()] = existing_from_smiles
                return existing_from_smiles

        # Skip ontology unify when existing canonical has SMILES (categories mix structures).
        for eq in get_ontology_equivalents(label_clean):
            eq_lc = str(eq).lower().strip()
            existing_from_alias = self._species_alias_to_canonical_label.get(eq_lc)
            if existing_from_alias is not None:
                existing_smiles = self._species_smiles_to_canonical_label
                existing_can_smi = next(
                    (s for s, lbl in existing_smiles.items() if lbl == existing_from_alias),
                    None,
                )
                if existing_can_smi is not None:
                    if can_smi is None or can_smi != existing_can_smi:
                        continue
                self._species_alias_to_canonical_label[label_lc] = existing_from_alias
                return existing_from_alias

        canonical_label = label_clean
        self._species_alias_to_canonical_label[label_lc] = canonical_label
        for eq in get_ontology_equivalents(label_clean):
            self._species_alias_to_canonical_label[str(eq).lower().strip()] = canonical_label
        if can_smi:
            self._species_smiles_to_canonical_label[can_smi] = canonical_label
        return canonical_label

    def _canonicalize_species_label(
        self,
        label: str,
        kinetic_data: Optional[KineticData],
        substrate_label: str,
    ) -> str:
        """Canonicalize via alias registry, SMILES identity, then ontology."""
        smiles = self._resolve_smiles_for_compound(
            compound_label=label,
            kinetic_data=kinetic_data,
            substrate_label=substrate_label,
        )
        return self._register_species_label(label=label, smiles=smiles)
    
    def _check_heavy_atom_balance(
        self,
        stoichiometry: Dict[str, int],
        kinetic_data,
        substrate_label: str,
    ) -> Optional[bool]:
        """True if conserved, False if unbalanced (skip), None if check not possible."""
        
        for species in stoichiometry.keys():
            if isinstance(species, str) and ("[ACP]" in species or "[acp]" in species):
                return None

        left = 0.0
        right = 0.0
        for species, coeff in stoichiometry.items():
            try:
                c = float(coeff)
            except (TypeError, ValueError):
                return None
            if c == 0:
                continue
            smiles = self._resolve_smiles_for_compound(
                compound_label=species,
                kinetic_data=kinetic_data,
                substrate_label=substrate_label,
            )
            count = heavy_atom_count(smiles)
            if count is None:
                return None
            if c < 0:
                left += (-c) * count
            else:
                right += c * count
        return abs(left - right) < 1e-9

    def _generate_reactions(
        self,
        enzyme,
        substrate,
        available_species_labels_lc: Optional[set] = None,
        provided_species_labels_lc: Optional[set] = None,
        thermo_engine=None,
        global_smiles_map: Optional[Dict[str, str]] = None,
        substitutor=None,
    ) -> List[GeneratedReaction]:
        """Query the DB and build GeneratedReaction objects for all matching enzyme–substrate pairs."""
        enzyme_label = enzyme.label
        substrate_label = substrate.label
        ec_number = enzyme.ecnumber

        if isinstance(ec_number, list):
            ec_numbers_declared = ec_number
        elif ec_number is not None:
            ec_numbers_declared = [ec_number]
        else:
            ec_numbers_declared = [None]

        self.logger.debug(f"Generating reactions for: {substrate_label} + {enzyme_label} ({ec_number})")

        all_ec_to_try = []
        for ec in ec_numbers_declared:
            for alias in get_ec_aliases(ec):
                if alias not in all_ec_to_try:
                    all_ec_to_try.append(alias)

        all_kinetic_data = []
        if self.kinetic_db:
            temp_range = None
            if isinstance(self.bees_object.environment.temperature, tuple):
                temp_range = self.bees_object.environment.temperature
            elif self.bees_object.environment.temperature is not None:
                temp_val = self.bees_object.environment.temperature
                temp_range = (temp_val * 0.95, temp_val * 1.05)
            
            ph_range = None
            if isinstance(self.bees_object.environment.pH, tuple):
                ph_range = self.bees_object.environment.pH
            elif self.bees_object.environment.pH is not None:
                ph_val = self.bees_object.environment.pH
                ph_range = (max(0, ph_val - 0.5), min(14, ph_val + 0.5))
            
            substrate_smiles = getattr(substrate, "smiles", None) or self._resolve_smiles_for_compound(
                substrate_label, None, substrate_label
            )
            for ec_to_try in all_ec_to_try:
                matches = self.kinetic_db.query_by_enzyme_substrate(
                    ec_number=ec_to_try,
                    substrate_label=substrate_label,
                    strict=False,
                    temperature_range=temp_range,
                    ph_range=ph_range,
                    available_species_labels_lc=provided_species_labels_lc,
                    return_all=True,
                    substrate_smiles=substrate_smiles,
                )
                if matches:
                    all_kinetic_data.extend(matches)
        
        if not all_kinetic_data:
            return []

        generated_reactions = []
        for kinetic_data in all_kinetic_data:
            if self.kinetics_estimator is not None:
                try:
                    enzyme_seq = getattr(enzyme, "amino_acid_sequence", None)
                    stoich = kinetic_data.stoichiometry or {}
                    reactants = [k for k, v in stoich.items() if v < 0]
                    reactant_smiles: Dict[str, str] = {}
                    smiles_mode = getattr(
                        getattr(self.bees_object, "settings", None), "smiles_mode", "auto"
                    )
                    for rname in reactants:
                        smi = self._resolve_smiles_for_compound(
                            compound_label=rname,
                            kinetic_data=kinetic_data,
                            substrate_label=substrate_label,
                        )
                        if not smi and smiles_mode == "interactive":
                            print(f"\n[SMILES needed] Compound: \"{rname}\" "
                                  f"(enzyme: {enzyme_label}, EC {ec_number})")
                            user_smiles = input("  Enter SMILES (or press Enter to skip): ").strip()
                            if user_smiles:
                                smi = user_smiles
                                self.logger.info(f"  User provided SMILES for '{rname}': {user_smiles}")
                        if smi:
                            reactant_smiles[rname] = smi

                    _NON_SUBSTRATE_LABELS = {"h+", "h(+)", "proton", "h2o", "water"}
                    reactant_smiles_normalized: Dict[str, str] = {}
                    for _rname, _smi in reactant_smiles.items():
                        if _rname.lower().strip() in _NON_SUBSTRATE_LABELS:
                            continue
                        if substitutor is not None:
                            _sub = substitutor.substitute(_rname, _smi)
                            if _sub is not None:
                                self.logger.debug(f"SMILES: {_rname!r} normalized by substitutor")
                                _smi = _sub
                        reactant_smiles_normalized[_rname] = _smi

                    if enzyme_seq and reactant_smiles_normalized:
                        est = self.kinetics_estimator.estimate(
                            enzyme_sequence=enzyme_seq,
                            reactant_smiles=reactant_smiles_normalized,
                            inhibitor_smiles=None,
                            ec_number=getattr(kinetic_data, "ec_number", None),
                        )
                        if est.kcat is not None or est.km is not None or getattr(est, "km_per_substrate", None):
                            kinetic_data.kcat = est.kcat if est.kcat is not None else kinetic_data.kcat
                            kinetic_data.source = est.source
                            if getattr(est, "kcat_sd", None) is not None:
                                kinetic_data.kcat_sd = est.kcat_sd
                            if getattr(est, "ki_sd", None) is not None:
                                kinetic_data.ki_sd = est.ki_sd
                            if getattr(est, "km_per_substrate", None):
                                kinetic_data.km_per_substrate = est.km_per_substrate
                                if getattr(est, "km_sd_per_substrate", None):
                                    kinetic_data.km_sd_per_substrate = est.km_sd_per_substrate
                            if est.km is not None:
                                kinetic_data.km = est.km
                                if getattr(est, "km_sd", None) is not None:
                                    kinetic_data.km_sd = est.km_sd
                        else:
                            self.logger.debug(f"  Kinetics estimation returned no values (source: {est.source})")
                    else:
                        missing = []
                        if not enzyme_seq:
                            missing.append("sequence")
                        if not reactant_smiles_normalized:
                            missing.append("SMILES for reactants")
                        self.logger.debug(f"  Skipping kinetics estimation: missing {', '.join(missing)}")
                except Exception as e:
                    self.logger.warning(f"  Kinetics estimation failed: {e}")

            # Estimation on: require kcat/Km. Off: topology-only is OK.
            has_kcat = kinetic_data.kcat is not None
            has_km = kinetic_data.km is not None or bool(getattr(kinetic_data, "km_per_substrate", None))
            estimation_enabled = getattr(
                getattr(self.bees_object, "settings", None), "estimate_kinetics", False
            )
            if not has_kcat and not has_km:
                if estimation_enabled:
                    self.logger.debug(
                        f"No kinetics available for {substrate_label} / {enzyme_label} "
                        f"(EC {getattr(kinetic_data, 'ec_number', '?')}) — reaction skipped."
                    )
                    continue
                else:
                    self.logger.debug(
                        f"No kinetics for {substrate_label} / {enzyme_label} "
                        f"— keeping reaction (topology-only mode)."
                    )

            try:
                matched_ec = getattr(kinetic_data, "ec_number", None) or ec_numbers_declared[0]
                template = create_reaction_from_database(
                    substrate=substrate_label,
                    enzyme_label=enzyme_label,
                    ec_number=matched_ec,
                    cofactor=None,
                    database_products=None,
                )
                
                if kinetic_data.stoichiometry:
                    canonical_stoich: Dict[str, int] = {}
                    for raw_name, coeff in kinetic_data.stoichiometry.items():
                        canonical_name = self._canonicalize_species_label(
                            label=raw_name,
                            kinetic_data=kinetic_data,
                            substrate_label=substrate_label,
                        )
                        canonical_stoich[canonical_name] = (
                            canonical_stoich.get(canonical_name, 0) + coeff
                        )
                    canonical_stoich = {
                        name: coeff for name, coeff in canonical_stoich.items() if coeff != 0
                    }

                    template.stoichiometry = canonical_stoich
                    template.reactants = [s for s, coeff in canonical_stoich.items() if coeff < 0]
                    template.products = [s for s, coeff in canonical_stoich.items() if coeff > 0]
                    template.cofactors = [
                        s for s in template.reactants if s.lower().strip() in GENERAL_COFACTORS
                    ]

                    # Remap km_per_substrate keys to canonical labels (else unbounded accumulation).
                    if getattr(kinetic_data, "km_per_substrate", None) and kinetic_data.stoichiometry:
                        raw_to_canonical: Dict[str, str] = {}
                        for raw_name in kinetic_data.stoichiometry:
                            canon = self._canonicalize_species_label(
                                label=raw_name,
                                kinetic_data=kinetic_data,
                                substrate_label=substrate_label,
                            )
                            if canon != raw_name:
                                raw_to_canonical[raw_name] = canon
                        if raw_to_canonical:
                            kinetic_data.km_per_substrate = {
                                raw_to_canonical.get(k, k): v
                                for k, v in kinetic_data.km_per_substrate.items()
                            }
                            if getattr(kinetic_data, "km_sd_per_substrate", None):
                                kinetic_data.km_sd_per_substrate = {
                                    raw_to_canonical.get(k, k): v
                                    for k, v in kinetic_data.km_sd_per_substrate.items()
                                }

                    balanced = self._check_heavy_atom_balance(
                        stoichiometry=canonical_stoich,
                        kinetic_data=kinetic_data,
                        substrate_label=substrate_label,
                    )
                    if balanced is False:
                        rxn_desc = " + ".join(template.reactants) + " -> " + " + ".join(template.products)
                        self.logger.debug(
                            f"  Skipping heavy-atom-unbalanced reaction: {rxn_desc}"
                        )
                        continue
                    if balanced is None:
                        self.logger.debug(
                            f"  Atom-balance check skipped (missing SMILES) for "
                            f"{substrate_label} / {enzyme_label}"
                        )

                all_available, missing_reactants = validate_reaction_reactants(
                    reactants=template.reactants,
                    available_species_labels_lc=available_species_labels_lc,
                    enzyme_label=enzyme_label,
                )
                
                if not all_available:
                    self.logger.debug(f"  Skipping {substrate_label}/{enzyme_label}: missing reactants {missing_reactants}")
                    continue
                
                rate_law = self._determine_rate_law(template, kinetic_data)
                
                reaction = GeneratedReaction(
                    enzyme_label=enzyme_label,
                    substrate_label=substrate_label,
                    ec_number=matched_ec,
                    template=template,
                    kinetics=kinetic_data,
                    reactant_labels=template.reactants,
                    product_labels=template.products,
                    stoichiometry=template.stoichiometry,
                    rate_law=rate_law
                )

                if thermo_engine is not None:
                    self._attach_raw_thermo_eagerly(
                        reaction=reaction,
                        thermo_engine=thermo_engine,
                        global_smiles_map=global_smiles_map or {},
                        ec_numbers_to_try=all_ec_to_try,
                        temp_range=temp_range,
                        ph_range=ph_range,
                        provided_species_labels_lc=provided_species_labels_lc,
                        substitutor=substitutor,
                    )

                generated_reactions.append(reaction)

            except Exception as e:
                self.logger.warning(f"Failed to create reaction for {substrate_label}: {e}")
                continue

        return generated_reactions
    
    def _attach_raw_thermo_eagerly(
        self,
        reaction: "GeneratedReaction",
        thermo_engine,
        global_smiles_map: Dict[str, str],
        ec_numbers_to_try: list,
        temp_range,
        ph_range,
        provided_species_labels_lc,
        substitutor=None,
    ) -> None:
        """Attach raw ΔG°′ / Keq only; no cutoff, Haldane, or kcat_rev ceiling."""
        kin = reaction.kinetics
        stoich = reaction.stoichiometry

        smiles_map: Dict[str, str] = dict(global_smiles_map)
        kin_smiles = getattr(kin, "compound_smiles", None) or {}
        for lab, smi in kin_smiles.items():
            if smi:
                smiles_map[lab] = smi

        # Snapshot substrate Kms BEFORE product Km merge (Haldane needs the original set).
        from bees.cofactors import COFACTORS_ALWAYS_AVAILABLE as _COFACTORS
        _km_sub = getattr(kin, "km_per_substrate", None) or {}
        kin._substrate_kms_for_haldane = {
            lab: km for lab, km in _km_sub.items()
            if stoich.get(lab, 0) < 0
            and km and km > 0
            and lab.lower().strip() not in _COFACTORS
        }

        td = thermo_engine.compute_keq(stoichiometry=stoich, smiles_map=smiles_map)
        reaction.thermo = td

        # Always reverse-query even if irreversible (product inhibition).
        enzyme_seq: Optional[str] = None
        for enz in getattr(self.bees_object, "enzymes", []):
            if getattr(enz, "label", None) == reaction.enzyme_label:
                enzyme_seq = getattr(enz, "amino_acid_sequence", None)
                break
        product_kms, product_smiles = self._lookup_product_kms_via_reverse_query(
            reaction=reaction,
            stoich=stoich,
            ec_numbers_to_try=ec_numbers_to_try,
            temp_range=temp_range,
            ph_range=ph_range,
            provided_species_labels_lc=provided_species_labels_lc,
            enzyme_sequence=enzyme_seq,
            smiles_map=smiles_map,
            substitutor=substitutor,
        )

        if product_smiles:
            existing = getattr(kin, "compound_smiles", None) or {}
            kin.compound_smiles = {**product_smiles, **existing}
            smiles_map.update({k: v for k, v in product_smiles.items() if k not in smiles_map})

        # Haldane uses ONLY reverse-queried product Kms, not forward catalytic product Kms.
        kin._product_kms_for_haldane = product_kms or {}
        if product_kms:
            existing_km = getattr(kin, "km_per_substrate", None) or {}
            for lab, km in product_kms.items():
                if lab not in existing_km:
                    existing_km[lab] = km
            kin.km_per_substrate = existing_km

        td_full = thermo_engine.compute_keq(stoichiometry=stoich, smiles_map=smiles_map)
        reaction.thermo = td_full

    def _lookup_product_kms_via_reverse_query(
        self,
        reaction,
        stoich: Dict[str, int],
        ec_numbers_to_try: list,
        temp_range,
        ph_range,
        provided_species_labels_lc: set,
        enzyme_sequence: Optional[str] = None,
        smiles_map: Optional[Dict[str, str]] = None,
        substitutor=None,
    ) -> tuple:
        """Collect product Kms via reverse CatPred, then DB fallback."""
        product_kms: Dict[str, float] = {}
        product_smiles: Dict[str, str] = {}

        if enzyme_sequence and self.kinetics_estimator:
            # Skip only buffered cofactors; query CoA/NAD/NADP for product terms.
            _NON_SUBSTRATE_LABELS = {"h+", "h(+)", "proton", "h2o", "water"}
            rev_reactant_smiles: Dict[str, str] = {}
            for prod_label in reaction.product_labels:
                prod_lc = prod_label.lower().strip()
                if prod_lc in COFACTORS_ALWAYS_AVAILABLE or prod_lc in _NON_SUBSTRATE_LABELS:
                    continue
                smi = (smiles_map or {}).get(prod_label) or (smiles_map or {}).get(prod_lc)
                if smi and substitutor is not None:
                    _sub = substitutor.substitute(prod_label, smi)
                    if _sub is not None:
                        self.logger.debug(f"SMILES: {prod_label!r} normalized by substitutor")
                        smi = _sub
                if smi:
                    rev_reactant_smiles[prod_label] = smi

            if rev_reactant_smiles:
                try:
                    est = self.kinetics_estimator.estimate(
                        enzyme_sequence=enzyme_sequence,
                        reactant_smiles=rev_reactant_smiles,
                        inhibitor_smiles=None,
                        # Pass same EC as forward so measured reverse Kms resolve.
                        ec_number=getattr(reaction, "ec_number", None)
                        or (ec_numbers_to_try[0] if ec_numbers_to_try else None),
                    )
                    for lab, km in (getattr(est, "km_per_substrate", None) or {}).items():
                        if km and km > 0 and lab not in product_kms:
                            product_kms[lab] = km
                    self.logger.debug(
                        f"CatPred reverse Kms for {reaction.enzyme_label}"
                        f" ({reaction.substrate_label}): "
                        f"{ {k: f'{v:.4g} mM' for k, v in product_kms.items()} }"
                    )
                except Exception as exc:
                    self.logger.debug(
                        f"CatPred reverse estimation failed for {reaction.enzyme_label}: {exc}"
                    )

        # DB fallback never yields for ecoli.csv (forward-only rows).
        if self.kinetic_db:
            for prod_label in reaction.product_labels:
                if prod_label in product_kms:
                    continue
                for ec in ec_numbers_to_try:
                    rev_matches = self.kinetic_db.query_by_enzyme_substrate(
                        ec_number=ec,
                        substrate_label=prod_label,
                        strict=False,
                        temperature_range=temp_range,
                        ph_range=ph_range,
                        available_species_labels_lc=provided_species_labels_lc,
                        return_all=True,
                    )
                    for rev_kd in (rev_matches or []):
                        rev_stoich = getattr(rev_kd, "stoichiometry", None) or {}
                        rev_stoich_lc = {
                            str(k).lower().strip(): v for k, v in rev_stoich.items()
                        }
                        stoich_lc = {
                            str(k).lower().strip(): v for k, v in stoich.items()
                        }
                        if set(rev_stoich_lc.keys()) != set(stoich_lc.keys()):
                            continue
                        rev_km = getattr(rev_kd, "km_per_substrate", None) or {}
                        for lab, km in rev_km.items():
                            lab_lc = str(lab).lower().strip()
                            if rev_stoich_lc.get(lab_lc, 0) < 0 and km and km > 0:
                                if lab not in product_kms:
                                    product_kms[lab] = km
                        rev_smi = getattr(rev_kd, "compound_smiles", None) or {}
                        for lab, smi in rev_smi.items():
                            if smi and lab not in product_smiles:
                                product_smiles[lab] = smi
                    if prod_label in product_kms:
                        break

        return product_kms, product_smiles

    def _determine_rate_law(
        self,
        template: ReactionTemplate,
        kinetics: Optional[KineticData]
    ) -> Optional[str]:
  
        if not kinetics:
            return None

        has_km = (
            kinetics.km is not None
            or bool(getattr(kinetics, "km_per_substrate", None))
        )
        if has_km or kinetics.kcat is not None:
            return "Michaelis-Menten"

       
        return None
    
    def export_reactions_summary(
        self,
        filename="reactions_summary.txt",
        n_core: Optional[int] = None,
        core_rxn_ids: Optional[set] = None,
    ) -> str:
        """Export human-readable summary of generated reactions."""
        output_path = os.path.join(self.output_directory, filename)
        n = len(self.reactions)

        if core_rxn_ids is not None:
            core_rxns = [r for r in self.reactions if id(r) in core_rxn_ids]
            edge_rxns = [r for r in self.reactions if id(r) not in core_rxn_ids]
            total_line = f"Total Reactions: {n} ({len(core_rxns)} core, {len(edge_rxns)} edge)\n"
        else:
            core_rxns = self.reactions
            edge_rxns = []
            if n_core is not None and n_core >= 0:
                n_edge = max(0, n - n_core)
                total_line = f"Total Reactions: {n} ({n_core} core, {n_edge} edge)\n"
            else:
                total_line = f"Total Reactions: {n}\n"

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BEES GENERATED REACTIONS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(total_line)
            f.write(f"Project: {self.bees_object.project}\n")
            f.write(f"Database: {self.bees_object.database.name}\n")
            f.write(f"Temperature: {self.bees_object.environment.temperature} K\n")
            f.write(f"pH: {self.bees_object.environment.pH}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("COENZYME PARTICIPATION NOTE\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Energy carriers and redox coenzymes (e.g., ATP, NADH) are treated as\n"
                "secondary reactants. They can participate in reaction stoichiometry, but\n"
                "they do not trigger new reaction discovery on their own.\n"
            )
            f.write(
                "Common coenzymes/cofactors considered in this run: "
                f"{', '.join(sorted(GENERAL_COFACTORS))}\n"
            )
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("REACTION NETWORK SCHEMA\n")
            f.write("-" * 80 + "\n")
            f.write("This section shows the connectivity of species in the reaction network.\n")
            f.write("Species are connected through reactions where products of one reaction\n")
            f.write("may be reactants in another.\n\n")
            
            all_species = set()
            species_as_reactant: dict = {}
            species_as_product: dict = {}
            
            for i, rxn in enumerate(self.reactions, 1):
                for reactant in rxn.reactant_labels:
                    all_species.add(reactant)
                    if reactant not in species_as_reactant:
                        species_as_reactant[reactant] = []
                    species_as_reactant[reactant].append(i)
                
                for product in rxn.product_labels:
                    all_species.add(product)
                    if product not in species_as_product:
                        species_as_product[product] = []
                    species_as_product[product].append(i)
            
            sorted_species = sorted(all_species)
            f.write(f"Total Unique Species in Network: {len(sorted_species)}\n\n")
            f.write("Species Connectivity:\n")
            for species in sorted_species:
                f.write(f"  {species}:\n")
                if species in species_as_reactant:
                    rxns_consuming = species_as_reactant[species]
                    f.write(f"    Consumed in reaction(s): {', '.join(f'R{i}' for i in rxns_consuming)}\n")
                if species in species_as_product:
                    rxns_producing = species_as_product[species]
                    f.write(f"    Produced in reaction(s): {', '.join(f'R{i}' for i in rxns_producing)}\n")
                if species not in species_as_reactant and species not in species_as_product:
                    f.write(f"    (Only in stoichiometry, not in main reaction equation)\n")
            
            f.write("\n" + "-" * 80 + "\n\n")

            def _write_reactions(rxn_list, start_idx):
                for i, rxn in enumerate(rxn_list, start_idx):
                    f.write(f"Reaction {i}:\n")
                    f.write(f"  Enzyme: {rxn.enzyme_label}\n")
                    f.write(f"  EC Number: {rxn.ec_number or 'N/A'}\n")
                    f.write(f"  EC Class: {rxn.template.ec_class.name}\n")
                    f.write(f"  Type: {rxn.template.template_type}\n")
                    td = getattr(rxn, "thermo", None)
                    _is_rev = td is not None and not td.irreversible
                    arrow = "⇌" if _is_rev else "→"
                    f.write(f"  Equation: {' + '.join(rxn.reactant_labels)} {arrow} {' + '.join(rxn.product_labels)}\n")
                    f.write(f"  Stoichiometry: {rxn.stoichiometry}\n")

                    if rxn.kinetics:
                        direction = "reversible" if _is_rev else "irreversible"
                        f.write(f"  Rate Law: {rxn.rate_law} ({direction})\n")

                    if rxn.kinetics:
                        substrate = rxn.substrate_label
                        enzyme = rxn.enzyme_label
                        kin = rxn.kinetics
                        f.write(f"  Kinetic Parameters:\n")
                        km_per = getattr(kin, "km_per_substrate", None)
                        if km_per:
                            for rname, val in km_per.items():
                                f.write(f"    Km({rname}) = {val} mM\n")
                        elif kin.km is not None:
                            f.write(f"    Km({substrate}) = {kin.km} mM\n")
                        if kin.kcat is not None:
                            f.write(f"    kcat({enzyme}) = {kin.kcat} 1/s\n")
                        if kin.vmax is not None:
                            f.write(f"    Vmax = {kin.vmax} mM/s\n")
                        if kin.delta_g is not None:
                            f.write(f"    dG = {kin.delta_g} kJ/mol\n")
                        if kin.temperature is not None:
                            f.write(f"    Temperature = {kin.temperature} K\n")
                        if kin.ph is not None:
                            f.write(f"    pH = {kin.ph}\n")
                        f.write(f"    Source: {kin.source}\n")
                        km_sd_per = getattr(kin, "km_sd_per_substrate", None)
                        if km_sd_per:
                            for rname, val in km_sd_per.items():
                                f.write(f"    Km({rname}) SD = {val:.4g} mM\n")
                        elif getattr(kin, "km_sd", None) is not None:
                            f.write(f"    Km({substrate}) SD = {kin.km_sd:.4g} mM\n")
                        if getattr(kin, "kcat_sd", None) is not None:
                            f.write(f"    kcat({enzyme}) SD = {kin.kcat_sd:.4g} 1/s\n")
                        if getattr(kin, "ki_sd", None) is not None:
                            f.write(f"    Ki SD = {kin.ki_sd:.4g} mM\n")
                    else:
                        f.write(f"  Kinetic Parameters: NOT FOUND IN DATABASE\n")
                        f.write(f"    (Parameters will need to be estimated)\n")

                    if td is not None:
                        f.write(f"  Thermodynamics:\n")
                        f.write(f"    Reversible: {'Yes' if _is_rev else 'No'}\n")
                        if td.dgr_prime_kJmol is not None:
                            f.write(f"    ΔG°' = {td.dgr_prime_kJmol:.2f} kJ/mol\n")
                        if td.keq is not None:
                            f.write(f"    Keq = {td.keq:.4g}\n")
                        if td.kcat_rev is not None:
                            f.write(f"    kcat_rev = {td.kcat_rev:.4g} 1/s\n")
                        f.write(f"    Source: {td.source}\n")

                    if rxn.template.cofactors:
                        f.write(f"  Cofactors: {', '.join(rxn.template.cofactors)}\n")

                    f.write("\n")

            if core_rxn_ids is not None:
                f.write("=" * 80 + "\n")
                f.write(f"CORE REACTIONS ({len(core_rxns)})\n")
                f.write("=" * 80 + "\n\n")
                _write_reactions(core_rxns, 1)
                if edge_rxns:
                    f.write("=" * 80 + "\n")
                    f.write(f"EDGE REACTIONS ({len(edge_rxns)})\n")
                    f.write("=" * 80 + "\n\n")
                    _write_reactions(edge_rxns, len(core_rxns) + 1)
            else:
                _write_reactions(core_rxns, 1)

        self.logger.info(f"Exported reactions summary to {output_path}")
        return output_path

