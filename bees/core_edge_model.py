#!/usr/bin/env python3

"""
Core/Edge Model Module
----------------------
Implements the core/edge species and reaction model for rate-based algorithm.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from bees.reaction_generator import GeneratedReaction


@dataclass
class SpeciesData:
    label: str
    concentration: float = 0.0
    initial_concentration: float = 0.0
    is_enzyme: bool = False
    constant: bool = False


class CoreEdgeModel:
    """Maintains core/edge sets of species and reactions for the model enlargement."""

    def __init__(self):
        self.core_species: List[SpeciesData] = []
        self.edge_species: List[SpeciesData] = []
        self.core_reactions: List[GeneratedReaction] = []
        self.edge_reactions: List[GeneratedReaction] = []

        # Fast lookup maps (lowercase label -> index in respective list)
        self._core_species_index: Dict[str, int] = {}
        self._edge_species_index: Dict[str, int] = {}
        self._core_species_labels_lc: Set[str] = set()
        self._edge_species_labels_lc: Set[str] = set()
        self._known_reaction_signatures: Set[tuple] = set()

    # ------------------------------------------------------------------
    # Species management
    # ------------------------------------------------------------------

    def add_core_species(self, species: SpeciesData) -> None:
        """Add a species directly to the core."""
        lc = species.label.lower().strip()
        if lc in self._core_species_labels_lc:
            return  # already present
        idx = len(self.core_species)
        self.core_species.append(species)
        self._core_species_index[lc] = idx
        self._core_species_labels_lc.add(lc)
        if lc in self._edge_species_labels_lc:
            self._remove_edge_species(lc)

    def add_edge_species(self, species: SpeciesData) -> None:
        """Add a species to the edge (candidate pool)."""
        lc = species.label.lower().strip()
        if lc in self._core_species_labels_lc or lc in self._edge_species_labels_lc:
            return  # already tracked
        idx = len(self.edge_species)
        self.edge_species.append(species)
        self._edge_species_index[lc] = idx
        self._edge_species_labels_lc.add(lc)

    def promote_species_to_core(self, label: str) -> Optional[SpeciesData]:
        """Move a species from the edge to the core; returns SpeciesData or None if not in edge."""
        lc = label.lower().strip()
        if lc not in self._edge_species_labels_lc:
            return None
        idx = self._edge_species_index[lc]
        species = self.edge_species[idx]
        self._remove_edge_species(lc)
        self.add_core_species(species)
        return species

    def _remove_edge_species(self, label_lc: str) -> None:
        if label_lc not in self._edge_species_index:
            return
        idx = self._edge_species_index.pop(label_lc)
        self._edge_species_labels_lc.discard(label_lc)
        # Replace with last element (swap-remove) to keep indices dense
        last_idx = len(self.edge_species) - 1
        if idx != last_idx:
            moved = self.edge_species[last_idx]
            self.edge_species[idx] = moved
            moved_lc = moved.label.lower().strip()
            self._edge_species_index[moved_lc] = idx
        self.edge_species.pop()

    def is_core_species(self, label: str) -> bool:
        return label.lower().strip() in self._core_species_labels_lc

    def is_edge_species(self, label: str) -> bool:
        return label.lower().strip() in self._edge_species_labels_lc

    def get_core_species_by_label(self, label: str) -> Optional[SpeciesData]:
        lc = label.lower().strip()
        idx = self._core_species_index.get(lc)
        if idx is not None:
            return self.core_species[idx]
        return None

    def get_species_labels_lc(self) -> Set[str]:
        """Return all known species labels (core + edge), lowercase."""
        return self._core_species_labels_lc | self._edge_species_labels_lc

    # ------------------------------------------------------------------
    # Reaction management
    # ------------------------------------------------------------------

    def _reaction_signature(self, reaction: GeneratedReaction) -> tuple:
        """Canonical signature for deduplication.
        Uses lowercased, stripped species and enzyme labels so that reactions
        differing only by capitalization or trivial formatting are treated as
        identical.
        """
        enzyme = str(reaction.enzyme_label).lower().strip()
        reactants = tuple(sorted(str(r).lower().strip() for r in reaction.reactant_labels))
        products = tuple(sorted(str(p).lower().strip() for p in reaction.product_labels))
        return (enzyme, reactants, products)

    def add_reaction(self, reaction: GeneratedReaction) -> None:
        """
        Add a reaction, placing it in core_reactions if all participants
        are core species, otherwise in edge_reactions.
        Skips if an identical reaction (same enzyme, reactants/products) already exists.
        """
        sig = self._reaction_signature(reaction)
        if sig in self._known_reaction_signatures:
            return  # duplicate
        
        self._known_reaction_signatures.add(sig)
        
        if self._reaction_is_core(reaction):
            self.core_reactions.append(reaction)
        else:
            self.edge_reactions.append(reaction)

    def reclassify_reactions(self) -> int:
        """
        Re-examine edge reactions; promote those whose participants are
        now all in the core.  Returns number of reactions promoted.
        """
        still_edge = []
        promoted = 0
        for rxn in self.edge_reactions:
            if self._reaction_is_core(rxn):
                self.core_reactions.append(rxn)
                promoted += 1
            else:
                still_edge.append(rxn)
        self.edge_reactions = still_edge
        return promoted

    def _reaction_is_core(self, reaction: GeneratedReaction) -> bool:
        """True if every reactant and product is a core species."""
        for label in reaction.reactant_labels + reaction.product_labels:
            if not self.is_core_species(label):
                return False
        return True
    
    def prune_edge(self, labels_to_remove: Set[str]) -> int:
        """Remove species from the edge by lowercase label; returns count removed."""
        removed = 0
        for lc in list(labels_to_remove):
            if lc in self._edge_species_labels_lc:
                self._remove_edge_species(lc)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Concentration vector helpers
    # ------------------------------------------------------------------

    def get_core_concentration_vector(self) -> List[float]:
        """Return ordered concentration vector for core species."""
        return [s.concentration for s in self.core_species]

    def set_core_concentrations(self, values: List[float]) -> None:
        """Update core species concentrations from an ordered vector."""
        for i, val in enumerate(values):
            if not self.core_species[i].constant:
                self.core_species[i].concentration = val

    def get_core_species_labels(self) -> List[str]:
        """Return ordered list of core species labels."""
        return [s.label for s in self.core_species]

    def get_core_species_index_map(self) -> Dict[str, int]:
        """Return label (lowercase) -> index map for core species."""
        return dict(self._core_species_index)

    def get_all_species_labels(self) -> List[str]:
        """Return ordered list of all species labels (core first, then edge)."""
        return (
            [s.label for s in self.core_species]
            + [s.label for s in self.edge_species]
        )

    def get_all_concentration_vector(self) -> List[float]:
        """Return concentration vector for all species (core + edge)."""
        return (
            [s.concentration for s in self.core_species]
            + [s.concentration for s in self.edge_species]
        )

    def set_all_concentrations(self, values: List[float]) -> None:
        """Update core and edge species concentrations from an ordered vector."""
        n_core = len(self.core_species)
        for i, val in enumerate(values[:n_core]):
            if i < len(self.core_species) and not self.core_species[i].constant:
                self.core_species[i].concentration = val
        for i, val in enumerate(values[n_core:]):
            if i < len(self.edge_species):
                self.edge_species[i].concentration = val

    def reset_concentrations_to_initial(self) -> None:
        """
        Reset all species concentrations to their stored initial values.

        Used when restarting an ODE from t=0 after a simulation interrupt
        (model structure may have changed; concentrations must match a fresh run).
        """
        for s in self.core_species:
            if not s.constant:
                s.concentration = s.initial_concentration
        for s in self.edge_species:
            s.concentration = s.initial_concentration

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------


    def summary(self) -> Dict[str, int]:
        return {
            "core_species": len(self.core_species),
            "edge_species": len(self.edge_species),
            "core_reactions": len(self.core_reactions),
            "edge_reactions": len(self.edge_reactions),
        }
