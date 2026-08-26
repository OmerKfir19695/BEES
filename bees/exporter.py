#!/usr/bin/env python3

"""
Enlarger Exporter Module
-----------------------
Holds all export/plot functionality for the iterative enlarger.


"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bees.cofactors import is_general_cofactor_label
from bees.core_edge_model import CoreEdgeModel, SpeciesData
from bees.reaction_generator import GeneratedReaction
from bees.simulator import SimulationResult

try:
    import libsbml as _libsbml  # python-libsbml
    _LIBSBML_AVAILABLE = True
except ImportError:
    _libsbml = None
    _LIBSBML_AVAILABLE = False


def reaction_signature(
    reaction: GeneratedReaction,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Canonical reaction ID. Enzyme included so isozymes stay distinct."""
    enzyme = str(reaction.enzyme_label).lower().strip()
    reactants = tuple(sorted(str(r).lower().strip() for r in reaction.reactant_labels))
    products = tuple(sorted(str(p).lower().strip() for p in reaction.product_labels))
    return (enzyme, reactants, products)


@dataclass
class EnlargerExporter:
    """Writes IterativeEnlarger artifacts (CSVs, plots, SBML)."""

    model: CoreEdgeModel
    profiles: List[SimulationResult]
    output_directory: str
    logger: Any

    reaction_id_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int]
    reaction_first_seen_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int]
    reaction_core_enter_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Optional[int]]
    reaction_obj_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], GeneratedReaction]
    iteration_summaries: List[Dict[str, int]]

    save_simulation_plots: bool = True
    plot_max_species: Optional[int] = None
    plot_exclude_enzymes: bool = True
    plot_exclude_cofactors: bool = True

    bees_object: Optional[Any] = None

    def export_flux_analysis(self, filename: str = "flux_analysis.csv") -> Optional[str]:
        """Write per-iteration core/edge counts and reaction history CSVs."""
        output_path = os.path.join(self.output_directory, filename)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteration",
                    "core_species",
                    "edge_species",
                    "core_reactions",
                    "edge_reactions",
                ]
            )

            if self.iteration_summaries:
                for row in self.iteration_summaries:
                    writer.writerow(
                        [
                            row["iteration"],
                            row["core_species"],
                            row["edge_species"],
                            row["core_reactions"],
                            row["edge_reactions"],
                        ]
                    )
            else:
                s = self.model.summary()
                writer.writerow(
                    [
                        len(self.profiles),
                        s["core_species"],
                        s["edge_species"],
                        s["core_reactions"],
                        s["edge_reactions"],
                    ]
                )

        self.logger.info(f"Exported flux analysis to {output_path}")

        details_path = os.path.join(self.output_directory, "flux_analysis_reactions.csv")
        core_sigs = {reaction_signature(rxn) for rxn in self.model.core_reactions}
        edge_sigs = {reaction_signature(rxn) for rxn in self.model.edge_reactions}
        all_sigs = sorted(
            set(core_sigs) | set(edge_sigs),
            key=lambda sig: self.reaction_id_by_sig.get(sig, 10**9),
        )
        with open(details_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "reaction_id",
                    "classification",
                    "first_seen_iteration",
                    "core_enter_iteration",
                ]
            )
            for sig in all_sigs:
                rxn = self.reaction_obj_by_sig.get(sig)
                if rxn is None:
                    continue
                reaction_id = f"R{self.reaction_id_by_sig.get(sig, 0)}"
                classification = "core" if sig in core_sigs else "edge"
                first_seen = self.reaction_first_seen_iter.get(sig, "")
                core_enter = self.reaction_core_enter_iter.get(sig, "")
                writer.writerow(
                    [
                        reaction_id,
                        classification,
                        first_seen,
                        core_enter if core_enter is not None else "",
                    ]
                )
        self.logger.info(f"Exported reaction flux summary to {details_path}")
        return output_path

    # ------------------------------------------------------------------
    # Export helpers - core/edge CSVs
    # ------------------------------------------------------------------

    def export_core_edge_reaction_species_csvs(self) -> Dict[str, str]:
        """
        Export core/edge reaction tables, each followed by a species section.

        Returns:
            Dict with keys "core" and "edge" and absolute output file paths.
        """
        outputs: Dict[str, str] = {}

        def _write_one(
            path: str,
            reactions: List[GeneratedReaction],
            species: List[SpeciesData],
            section_name: str,
        ) -> None:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "index",
                        "reaction_id",
                        "template",
                        "ec_number",
                        "family",
                        "enzyme",
                        "substrate",
                        "reactants",
                        "products",
                    ]
                )
                for idx, rxn in enumerate(reactions, start=1):
                    sig = reaction_signature(rxn)
                    rid_num = self.reaction_id_by_sig.get(sig)
                    rid = f"R{rid_num}" if rid_num is not None else ""
                    template_type = ""
                    family = ""
                    if getattr(rxn, "template", None) is not None:
                        template_type = str(getattr(rxn.template, "template_type", "") or "")
                        ec_class_obj = getattr(rxn.template, "ec_class", None)
                        family = str(getattr(ec_class_obj, "name", "") or "")
                    writer.writerow(
                        [
                            idx,
                            rid,
                            template_type,
                            str(getattr(rxn, "ec_number", "") or ""),
                            family,
                            str(getattr(rxn, "enzyme_label", "") or ""),
                            str(getattr(rxn, "substrate_label", "") or ""),
                            " + ".join(str(r) for r in getattr(rxn, "reactant_labels", []) or []),
                            " + ".join(str(p) for p in getattr(rxn, "product_labels", []) or []),
                        ]
                    )

                writer.writerow([])
                writer.writerow([f"{section_name} species"])
                writer.writerow(["index", "label", "is_enzyme", "constant"])
                for idx, sp in enumerate(species, start=1):
                    writer.writerow(
                        [
                            idx,
                            str(getattr(sp, "label", "") or ""),
                            bool(getattr(sp, "is_enzyme", False)),
                            bool(getattr(sp, "constant", False)),
                        ]
                    )

        core_path = os.path.join(self.output_directory, "core_reactions_species.csv")
        edge_path = os.path.join(self.output_directory, "edge_reactions_species.csv")

        _write_one(
            core_path,
            list(self.model.core_reactions),
            list(self.model.core_species),
            section_name="core",
        )
        _write_one(
            edge_path,
            list(self.model.edge_reactions),
            list(self.model.edge_species),
            section_name="edge",
        )

        self.logger.info(f"Exported core reactions/species CSV to {core_path}")
        self.logger.info(f"Exported edge reactions/species CSV to {edge_path}")
        outputs["core"] = core_path
        outputs["edge"] = edge_path
        return outputs

    # ------------------------------------------------------------------
    # Export helpers - simulation profiles
    # ------------------------------------------------------------------

    def export_simulation_profiles(self, filename: str = "simulation_profiles.csv") -> Optional[str]:
        """Write concentration time-series from every iteration into one CSV."""
        if not self.profiles:
            return None

        output_path = os.path.join(self.output_directory, filename)
        all_labels: List[str] = []
        seen: Set[str] = set()
        for prof in self.profiles:
            for lab in prof.species_labels:
                if lab not in seen:
                    all_labels.append(lab)
                    seen.add(lab)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "time"] + all_labels)
            for it_idx, prof in enumerate(self.profiles, 1):
                label_to_row = {lab: i for i, lab in enumerate(prof.species_labels)}
                for t_idx in range(prof.t.shape[0]):
                    row = [it_idx, prof.t[t_idx]]
                    for lab in all_labels:
                        ridx = label_to_row.get(lab)
                        if ridx is not None:
                            row.append(prof.y[ridx, t_idx])
                        else:
                            row.append("")
                    writer.writerow(row)

        self.logger.info(f"Exported simulation profiles to {output_path}")
        return output_path

    def export_simulation_plots(
        self,
        filename_pattern: str = "simulation_plot_iter{}.png",
    ) -> Optional[List[str]]:
        """Plot concentration vs time for each iteration and save to PNG."""
        if not self.profiles:
            return None

        enzyme_labels: Set[str] = set()
        if self.plot_exclude_enzymes and self.model:
            for sd in self.model.core_species + self.model.edge_species:
                if sd.is_enzyme:
                    enzyme_labels.add(sd.label)

        cofactor_labels: Set[str] = set()
        if self.plot_exclude_cofactors and self.bees_object is not None:
            for sp in getattr(self.bees_object, "species", []) or []:
                if getattr(sp, "reactive", True) is False:
                    cofactor_labels.add(getattr(sp, "label", ""))

        max_species = self.plot_max_species if self.plot_max_species is not None else 12
        exclude = enzyme_labels | cofactor_labels
        paths: List[str] = []

        for it_idx, prof in enumerate(self.profiles, 1):
            candidates = []
            for i, lab in enumerate(prof.species_labels):
                if lab in exclude:
                    continue
                if self.plot_exclude_cofactors and is_general_cofactor_label(lab):
                    continue

                y = prof.y[i, :]
                span = float(y.max() - y.min())
                if span <= 1e-9:
                    continue
                candidates.append((i, lab, span))

            if not candidates:
                continue

            candidates.sort(key=lambda item: item[2], reverse=True)
            if len(candidates) > max_species:
                candidates = candidates[:max_species]

            fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
            ax.set_facecolor("white")

            colors = [
                "#0173B2",
                "#DE8F05",
                "#029E73",
                "#CC78BC",
                "#CA9161",
                "#FBAFE4",
                "#949494",
                "#ECE133",
                "#56B4E9",
                "#D55E00",
            ]
            ax.set_prop_cycle(color=colors)

            mM_to_uM = 1000.0
            t = prof.t
            for i, lab, _ in candidates:
                ax.plot(t, prof.y[i, :] * mM_to_uM, label=lab)

            ax.set_xlabel("Time (s)", fontsize=14, fontweight="medium")
            ax.set_ylabel(r"Concentration ($\mu$M)", fontsize=14, fontweight="medium")
            ax.set_title(f"Iteration {it_idx}", fontsize=16, fontweight="medium")
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=11,
            )

            fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
            out_path = os.path.join(
                self.output_directory,
                filename_pattern.format(it_idx),
            )
            fig.savefig(
                out_path,
                dpi=150,
                bbox_inches="tight",
                pad_inches=0.25,
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)
            paths.append(out_path)

        if paths:
            self.logger.info(
                f"Exported {len(paths)} simulation plot(s) to {self.output_directory}"
            )
        return paths if paths else None

    # ------------------------------------------------------------------
    # Export helpers - SBML (forany SBML-compatible tool)
    # ------------------------------------------------------------------

    def export_sbml(
        self,
        filename: str = "model.xml",
        core_only: bool = True,
        # If True, the production-only invariant raises ValueError instead of
        # warning — blocks SBML export when any core species is produced but
        # never consumed (intended for catching incomplete enlarger snapshots
        # during development; off by default so legitimate branch exits and
        # terminal products do not block export).
        strict_invariant: bool = False,
    ) -> Optional[str]:
        """Export the network as SBML Level 3 Version 2.

        One compartment ``compartment1`` (1 L so mmol = mM). Core-only by default.
        Kinetic laws: reversible Liebermeister-Klipp when thermo is usable,
        else forward-only MM. ``strict_invariant=True`` raises on production-only
        species (warns by default). See knowledge/functions/EXPORTER.md.
        """
        if not _LIBSBML_AVAILABLE:
            self.logger.warning(
                "python-libsbml is not installed – SBML export skipped. "
                "Install it with:  pip install python-libsbml"
            )
            return None

        reactions: List[GeneratedReaction] = list(self.model.core_reactions)
        species_list: List[SpeciesData] = list(self.model.core_species)
        if not core_only:
            reactions += list(self.model.edge_reactions)
            species_list += list(self.model.edge_species)

        if not reactions:
            self.logger.warning("SBML export: no reactions in model – skipping.")
            return None

        # ----------------------------------------------------------------
        # 1.  Create SBML document
        # ----------------------------------------------------------------
        doc = _libsbml.SBMLDocument(3, 2)
        model = doc.createModel()
        model.setId("BEES_model")
        model.setName("BEES auto-generated kinetic model")

        # ----------------------------------------------------------------
        # 1.  Unit Definitions
        # ----------------------------------------------------------------
        # Time units = seconds
        tu = model.createUnitDefinition()
        tu.setId("second")
        u = tu.createUnit()
        u.setKind(_libsbml.UNIT_KIND_SECOND)
        u.setExponent(1)
        u.setScale(0)
        u.setMultiplier(1.0)
        model.setTimeUnits("second")

        # Substance units = mmol
        su = model.createUnitDefinition()
        su.setId("mmol")
        u2 = su.createUnit()
        u2.setKind(_libsbml.UNIT_KIND_MOLE)
        u2.setExponent(1)
        u2.setScale(-3)
        u2.setMultiplier(1.0)
        model.setSubstanceUnits("mmol")
        model.setExtentUnits("mmol")

        # Volume units = litre
        vu = model.createUnitDefinition()
        vu.setId("litre")
        u3 = vu.createUnit()
        u3.setKind(_libsbml.UNIT_KIND_LITRE)
        u3.setExponent(1)
        u3.setScale(0)
        u3.setMultiplier(1.0)
        model.setVolumeUnits("litre")

        # ----------------------------------------------------------------
        # 2.  One compartment:  compartment1, volume = 1 L.
        #     Each kineticLaw below is multiplied by this compartment so it is a
        #     proper SBML *substance* (amount/time) rate; with size = 1 the amount
        #     in mmol equals the concentration in mM, and dC/dt comes out equal to
        #     the bare rate expression (see kineticLaw construction comment).
        # ----------------------------------------------------------------
        comp = model.createCompartment()
        comp.setId("compartment1")
        comp.setName("compartment1")
        comp.setSize(1.0)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)

        # SBML ids must start with a letter or underscore.
        def _sbml_id(label: str) -> str:
            s = re.sub(r"[^A-Za-z0-9_]", "_", str(label).strip())
            if s and s[0].isdigit():
                s = "_" + s
            return s or "_species"

        label_to_id: Dict[str, str] = {}
        used_ids: Set[str] = set()
        for sd in species_list:
            base = _sbml_id(sd.label)
            sid = base
            n = 2
            while sid in used_ids:
                sid = f"{base}_{n}"
                n += 1
            label_to_id[sd.label.lower().strip()] = sid
            used_ids.add(sid)

        input_concs = {}
        if self.bees_object is not None:
            for sp in getattr(self.bees_object, "species", []) or []:
                input_concs[str(sp.label).lower().strip()] = getattr(sp, "concentration", 0.0)
            for enz in getattr(self.bees_object, "enzymes", []) or []:
                input_concs[str(enz.label).lower().strip()] = getattr(enz, "concentration", 0.0)

        for sd in species_list:
            lc = sd.label.lower().strip()
            sid = label_to_id[lc]
            sp = model.createSpecies()
            sp.setId(sid)
            sp.setName(sd.label)
            sp.setCompartment("compartment1")
            conc = max(input_concs.get(lc, 0.0) or 0.0, 0.0)
            sp.setInitialConcentration(conc)
            sp.setHasOnlySubstanceUnits(False)
            is_enz = bool(getattr(sd, "is_enzyme", False))
            is_const = bool(getattr(sd, "constant", False)) or is_enz
            sp.setConstant(is_const)
            sp.setBoundaryCondition(is_const)

        # Set of boundary/constant species labels (H2O, H+, enzymes, buffers).
        # These are excluded from saturation terms: their concentrations are
        # fixed, so they don't limit the rate and their activities are already
        # incorporated into the biochemical Keq from eQuilibrator.
        boundary_labels: Set[str] = {
            sd.label.lower().strip()
            for sd in species_list
            if bool(getattr(sd, "constant", False)) or bool(getattr(sd, "is_enzyme", False))
        }

        # Production-only invariant (FAS 3-oxo-octadec-ACP bug); strict_invariant raise vs warn.
        produced_lcs: Set[str] = set()
        consumed_lcs: Set[str] = set()
        for rxn in reactions:
            # Reversible both-directions so isomerizations are not flagged.
            is_rev = (
                getattr(rxn.template, "reversible", False)
                and getattr(rxn, "thermo", None) is not None
                and not rxn.thermo.irreversible
            )
            for lab in getattr(rxn, "reactant_labels", ()) or ():
                lc = lab.lower().strip()
                consumed_lcs.add(lc)
                if is_rev:
                    produced_lcs.add(lc)
            for lab in getattr(rxn, "product_labels", ()) or ():
                lc = lab.lower().strip()
                produced_lcs.add(lc)
                if is_rev:
                    consumed_lcs.add(lc)

        # Edge-only consumers = branch exits, not leaks.
        edge_consumed_lcs: Set[str] = set()
        for rxn in self.model.edge_reactions:
            for lab in getattr(rxn, "reactant_labels", ()) or ():
                edge_consumed_lcs.add(lab.lower().strip())

        leaks: List[str] = []
        branch_exits: List[str] = []
        for sp_lc in sorted(produced_lcs - consumed_lcs):
            if sp_lc in boundary_labels:
                continue
            # Cofactors + FFA (ate / oic acid) exempt.
            if is_general_cofactor_label(sp_lc):
                continue
            if sp_lc.endswith("ate") or sp_lc.endswith("oic acid"):
                continue
            if sp_lc in edge_consumed_lcs:
                branch_exits.append(sp_lc)
                continue
            leaks.append(sp_lc)

        if branch_exits:
            preview = ", ".join(branch_exits[:10]) + ("…" if len(branch_exits) > 10 else "")
            self.logger.warning(
                f"SBML export: {len(branch_exits)} core species are produced by "
                f"core reactions but consumed only by edge reactions (branch exits "
                f"— enlarger terminated before promoting downstream reactions). "
                f"These species will accumulate in the core-only SBML model. "
                f"Species: {preview}"
            )

        if leaks:
            preview = ", ".join(leaks[:10]) + ("…" if len(leaks) > 10 else "")
            msg = (
                f"SBML export invariant violated: {len(leaks)} non-boundary "
                f"species are produced but never consumed (will accumulate "
                f"without bound under integration). Likely an incomplete "
                f"enlarger snapshot. Species: {preview}"
            )
            if strict_invariant:
                raise ValueError(msg)
            self.logger.warning(msg)

        enzyme_param_ids: Dict[str, str] = {}
        for sd in species_list:
            if not getattr(sd, "is_enzyme", False):
                continue
            lc = sd.label.lower().strip()
            pid = f"E_{_sbml_id(sd.label)}"
            n = 2
            orig_pid = pid
            while model.getParameter(pid) is not None:
                pid = f"{orig_pid}_{n}"
                n += 1
            p = model.createParameter()
            p.setId(pid)
            p.setName(f"[{sd.label}]")
            p.setValue(max(input_concs.get(lc, 0.0) or 0.0, 0.0))
            p.setConstant(True)
            enzyme_param_ids[lc] = pid

        used_rxn_ids: Set[str] = set()

        for rxn_idx, rxn in enumerate(reactions, start=1):
            sig = reaction_signature(rxn)
            rxn_num = self.reaction_id_by_sig.get(sig, rxn_idx)
            rid = f"R{rxn_num}"
            if rid in used_rxn_ids:
                rid = f"R{rxn_num}_{rxn_idx}"
            used_rxn_ids.add(rid)

            # Fallback thermo = irreversible forward-only MM.
            sbml_rxn = model.createReaction()
            sbml_rxn.setId(rid)
            # Reversibility flag AFTER kinetic-law branch.
            td = getattr(rxn, "thermo", None)

            for r_label in rxn.reactant_labels:
                lc = r_label.lower().strip()
                sid = label_to_id.get(lc)
                if sid is None:
                    continue
                sr = sbml_rxn.createReactant()
                sr.setSpecies(sid)
                coeff = abs(rxn.stoichiometry.get(r_label, -1))
                sr.setStoichiometry(float(coeff))
                sr.setConstant(True)

            for p_label in rxn.product_labels:
                lc = p_label.lower().strip()
                sid = label_to_id.get(lc)
                if sid is None:
                    continue
                sp2 = sbml_rxn.createProduct()
                sp2.setSpecies(sid)
                coeff = abs(rxn.stoichiometry.get(p_label, 1))
                sp2.setStoichiometry(float(coeff))
                sp2.setConstant(True)

            kin = getattr(rxn, "kinetics", None)
            if kin is None or getattr(rxn, "rate_law", None) is None:
                self.logger.debug(
                    "SBML: %s skipped kinetic law (no kinetics/rate_law) — "
                    "reaction structure exported without rate equation",
                    rid,
                )
                sbml_rxn.setName(
                    f"{rxn.enzyme_label}: {' + '.join(rxn.reactant_labels)} -> {' + '.join(rxn.product_labels)}"
                )
                sbml_rxn.setReversible(False)
                if td is not None:
                    notes_parts = [f"source={td.source}", "WARNING: no kinetics — no rate law emitted"]
                    if td.dgr_prime_kJmol is not None and math.isfinite(td.dgr_prime_kJmol):
                        notes_parts.append(f"dGr_prime={td.dgr_prime_kJmol:.2f} kJ/mol")
                    if td.keq is not None and math.isfinite(td.keq):
                        notes_parts.append(f"Keq={td.keq:.4g}")
                    notes_parts.append("irreversible=True")
                    sbml_rxn.setNotes(
                        "<body xmlns='http://www.w3.org/1999/xhtml'><p>"
                        + "; ".join(notes_parts)
                        + "</p></body>"
                    )
                continue

            kl = sbml_rxn.createKineticLaw()

            kcat_val = getattr(kin, "kcat", None)
            km_per = getattr(kin, "km_per_substrate", None) or {}
            km_single = getattr(kin, "km", None)

            def _km_for_label(label: str) -> Optional[float]:
                lc = label.lower().strip()
                if km_per:
                    v = km_per.get(label)
                    if v is None:
                        v = next(
                            (w for k, w in km_per.items() if k.lower().strip() == lc),
                            None,
                        )
                    return v
                return km_single

            # Product Km default 1.0 matches empty Haldane list.
            use_rev = (
                td is not None
                and not td.irreversible
                and td.keq is not None
                and math.isfinite(td.keq)
                and td.keq > 0.0
                and td.kcat_rev is not None
                and math.isfinite(td.kcat_rev)
                and kcat_val is not None
            )

            substrate_km_pairs: List[Tuple[str, str, int]] = []
            for r_label in rxn.reactant_labels:
                r_lc = r_label.lower().strip()
                if r_lc in boundary_labels:
                    continue
                sid = label_to_id.get(r_lc)
                if sid is None:
                    continue
                km_val: Optional[float] = _km_for_label(r_label)
                if km_val is None or km_val <= 0:
                    use_rev = False
                    continue
                km_pid = f"Km_{rid}_{_sbml_id(r_label)}"
                suffix_n = 2
                orig_km_pid = km_pid
                while model.getParameter(km_pid) is not None:
                    km_pid = f"{orig_km_pid}_{suffix_n}"
                    suffix_n += 1
                p_km = model.createParameter()
                p_km.setId(km_pid)
                p_km.setName(f"Km for {r_label} ({rxn.enzyme_label})")
                p_km.setValue(float(km_val))
                p_km.setConstant(True)
                
                nu = abs(rxn.stoichiometry.get(r_label, 1))
                substrate_km_pairs.append((sid, km_pid, nu))

            product_km_pairs: List[Tuple[str, str, int]] = []
            if use_rev:
                for p_label in rxn.product_labels:
                    p_lc = p_label.lower().strip()
                    if p_lc in boundary_labels:
                        continue
                    sid_p = label_to_id.get(p_lc)
                    if sid_p is None:
                        use_rev = False
                        break
                    km_val_p: float = _km_for_label(p_label) or 1.0
                    if km_val_p <= 0:
                        km_val_p = 1.0
                    km_pid_p = f"Km_{rid}_{_sbml_id(p_label)}_P"
                    suffix_n = 2
                    orig_p = km_pid_p
                    while model.getParameter(km_pid_p) is not None:
                        km_pid_p = f"{orig_p}_{suffix_n}"
                        suffix_n += 1
                    p_km_p = model.createParameter()
                    p_km_p.setId(km_pid_p)
                    p_km_p.setName(f"Km for {p_label} ({rxn.enzyme_label})")
                    p_km_p.setValue(float(km_val_p))
                    p_km_p.setConstant(True)
                    
                    nu = abs(rxn.stoichiometry.get(p_label, 1))
                    product_km_pairs.append((sid_p, km_pid_p, nu))
                if not product_km_pairs:
                    use_rev = False

            e_lc = rxn.enzyme_label.lower().strip()
            e_param = enzyme_param_ids.get(e_lc)
            e_token = e_param if e_param is not None else "0.001"

            pid_kcat: Optional[str] = None
            if kcat_val is not None:
                pid_kcat = f"kcat_{rid}"
                p_kcat = model.createParameter()
                p_kcat.setId(pid_kcat)
                p_kcat.setName(f"kcat ({rxn.enzyme_label})")
                p_kcat.setValue(float(kcat_val))
                p_kcat.setConstant(True)

            if use_rev and substrate_km_pairs and product_km_pairs and pid_kcat:
                pid_kcat_rev = f"kcat_rev_{rid}"
                p_kcat_rev = model.createParameter()
                p_kcat_rev.setId(pid_kcat_rev)
                p_kcat_rev.setName(f"kcat_rev ({rxn.enzyme_label})")
                p_kcat_rev.setValue(float(td.kcat_rev))
                p_kcat_rev.setConstant(True)

                pid_keq = f"Keq_{rid}"
                p_keq = model.createParameter()
                p_keq.setId(pid_keq)
                p_keq.setName(f"Keq ({rxn.enzyme_label})")
                p_keq.setValue(float(td.keq))
                p_keq.setConstant(True)

                def _pow_wrap(base_expr: str, exponent: int) -> str:
                    if exponent == 1:
                        return base_expr
                    return f"pow({base_expr}, {exponent})"

                fwd_num = " * ".join(_pow_wrap(f"({s} / {k})", nu) for s, k, nu in substrate_km_pairs)
                rev_num = " * ".join(_pow_wrap(f"({s} / {k})", nu) for s, k, nu in product_km_pairs)
                sub_den = " * ".join(_pow_wrap(f"(1 + {s} / {k})", nu) for s, k, nu in substrate_km_pairs)
                prod_den = " * ".join(_pow_wrap(f"(1 + {s} / {k})", nu) for s, k, nu in product_km_pairs)

                formula = (
                    f"({pid_kcat} * {e_token} * {fwd_num}"
                    f" - {pid_kcat_rev} * {e_token} * {rev_num})"
                    f" / ({sub_den} + {prod_den} - 1)"
                )
            elif pid_kcat and substrate_km_pairs:
                formula_parts = [pid_kcat, e_token]
                def _pow_wrap(base_expr: str, exponent: int) -> str:
                    if exponent == 1:
                        return base_expr
                    return f"pow({base_expr}, {exponent})"
                for sid, km_pid, nu in substrate_km_pairs:
                    base_expr = f"({sid} / ({km_pid} + {sid}))"
                    formula_parts.append(_pow_wrap(base_expr, nu))
                formula = " * ".join(formula_parts)
            elif pid_kcat:
                formula = f"{pid_kcat} * {e_token}"
            else:
                formula = "0"

            # Feedback = modifierSpeciesReference (not consumed).
            fb = getattr(rxn, "feedback_inhibitors", None)
            if formula != "0" and isinstance(fb, dict) and fb:
                existing_mods = {
                    sbml_rxn.getModifier(j).getSpecies()
                    for j in range(sbml_rxn.getNumModifiers())
                }
                fb_terms = []
                for inh_label, (ki_val, hill_val) in fb.items():
                    inh_sid = label_to_id.get(str(inh_label).lower().strip())
                    if inh_sid is None:
                        self.logger.debug(
                            f"SBML: {rid} feedback inhibitor {inh_label!r} "
                            "not a model species — skipped"
                        )
                        continue
                    if inh_sid not in existing_mods:
                        sbml_rxn.createModifier().setSpecies(inh_sid)
                        existing_mods.add(inh_sid)
                    pid_ki = f"Ki_fb_{rid}_{inh_sid}"
                    p_ki = model.createParameter()
                    p_ki.setId(pid_ki)
                    p_ki.setName(f"feedback Ki ({rxn.enzyme_label}<-{inh_label})")
                    p_ki.setValue(float(ki_val))
                    p_ki.setConstant(True)
                    pid_h = f"hill_fb_{rid}_{inh_sid}"
                    p_h = model.createParameter()
                    p_h.setId(pid_h)
                    p_h.setName(f"feedback Hill ({rxn.enzyme_label}<-{inh_label})")
                    p_h.setValue(float(hill_val))
                    p_h.setConstant(True)
                    fb_terms.append(f"1 / (1 + ({inh_sid} / {pid_ki})^{pid_h})")
                if fb_terms:
                    formula = f"({formula}) * ({' * '.join(fb_terms)})"

            # kineticLaw * V (substance rate; else 1/V too fast in COPASI).
            if formula != "0":
                formula = f"compartment1 * ({formula})"

            # L3 parser, not L1 setFormula.
            _ast = _libsbml.parseL3Formula(formula)
            if _ast is None:
                self.logger.warning(
                    "SBML: %s parseL3Formula failed (%s) — falling back to setFormula",
                    rid, _libsbml.getLastParseL3Error(),
                )
                kl.setFormula(formula)
            else:
                kl.setMath(_ast)

            emitted_reversible = bool(
                use_rev
                and substrate_km_pairs
                and product_km_pairs
                and pid_kcat is not None
            )
            arrow = "<=>" if emitted_reversible else "->"
            sbml_rxn.setName(
                f"{rxn.enzyme_label}: {' + '.join(rxn.reactant_labels)} {arrow} {' + '.join(rxn.product_labels)}"
            )
            sbml_rxn.setReversible(emitted_reversible)

            if td is not None:
                notes_parts = [f"source={td.source}"]
                if td.source == "fallback":
                    notes_parts.append("WARNING: no dG available — exported as irreversible forward-only")
                if td.dgr_prime_kJmol is not None and math.isfinite(td.dgr_prime_kJmol):
                    notes_parts.append(f"dGr_prime={td.dgr_prime_kJmol:.2f} kJ/mol")
                if td.keq is not None and math.isfinite(td.keq):
                    notes_parts.append(f"Keq={td.keq:.4g}")
                if td.kcat_rev is not None:
                    notes_parts.append(f"kcat_rev={td.kcat_rev:.4g} 1/s")
                notes_parts.append(f"irreversible={not emitted_reversible}")
                sbml_rxn.setNotes(
                    "<body xmlns='http://www.w3.org/1999/xhtml'><p>"
                    + "; ".join(notes_parts)
                    + "</p></body>"
                )

        # ----------------------------------------------------------------
        # FAS / Yu-2011 convenience (not generic SBML).  use for regenrate the results without
        # extreanl tools. Optional assignment rules for free-acid totals and palmitic
        # equivalents when free FA species are present. Safe no-op for
        # non-FAS models (no *-oate labels → nothing added). 
        # time-course metrics are usually computed outside this exporter;
        # keep for now.
        # ----------------------------------------------------------------
        _FA_STEMS = [  # most specific first so e.g. 'hexadec' wins over 'hex'
            ("icosen", 20), ("icosan", 20), ("octadecen", 18), ("octadecan", 18),
            ("hexadecen", 16), ("hexadecan", 16), ("tetradecen", 14), ("tetradecan", 14),
            ("dodecen", 12), ("dodecan", 12), ("decen", 10), ("decan", 10),
            ("octen", 8), ("octan", 8), ("hexen", 6), ("hexan", 6),
            ("penten", 5), ("pentan", 5), ("buten", 4), ("butan", 4),
        ]

        def _fa_carbons(label: str) -> Optional[int]:
            n = label.lower()
            if "[acp]" in n or not n.endswith("oate"):
                return None  # ACP thioester or not a free acid
            for stem, c in _FA_STEMS:
                if stem in n:
                    return c
            return None

        fa_total_terms: List[str] = []
        fa_palm_terms: List[str] = []
        fa_band_terms: List[str] = []
        for sd in species_list:
            c = _fa_carbons(sd.label)
            if c is None:
                continue
            sid = label_to_id[sd.label.lower().strip()]
            fa_total_terms.append(sid)
            fa_palm_terms.append(f"({c}/16) * {sid}")
            # Palmitic equivalents C14-C18 (Yu 2011 TLC band).
            if c in (14, 16, 18):
                fa_band_terms.append(f"({c}/16) * {sid}")

        if fa_total_terms:
            def _add_global_quantity(pid: str, name: str, infix: str) -> None:
                p = model.createParameter()
                p.setId(pid)
                p.setName(name)
                p.setConstant(False)  # set by an assignment rule each step
                p.setValue(0.0)
                rule = model.createAssignmentRule()
                rule.setVariable(pid)
                ast = _libsbml.parseL3Formula(infix)
                if ast is None:
                    self.logger.warning(
                        f"SBML: failed to parse assignment rule for {pid} "
                        f"({_libsbml.getLastParseL3Error()})"
                    )
                else:
                    rule.setMath(ast)

            _add_global_quantity(
                "total_free_FA_uM", "Total free fatty acid (uM)",
                "1000 * (" + " + ".join(fa_total_terms) + ")",
            )
            _add_global_quantity(
                "palmitic_equivalents_uM",
                "Palmitic equivalents, Yu-2011 metric (uM)",
                "1000 * (" + " + ".join(fa_palm_terms) + ")",
            )
            if fa_band_terms:
                _add_global_quantity(
                    "palmitic_equivalents_C14_C18_uM",
                    "Palmitic equivalents, Yu-2011 S2A C14-C18 band (uM)",
                    "1000 * (" + " + ".join(fa_band_terms) + ")",
                )

        doc.setConsistencyChecks(
            _libsbml.LIBSBML_CAT_GENERAL_CONSISTENCY, True
        )
        doc.setConsistencyChecks(
            _libsbml.LIBSBML_CAT_IDENTIFIER_CONSISTENCY, True
        )

        output_path = os.path.join(self.output_directory, filename)
        writer = _libsbml.SBMLWriter()
        writer.setProgramName("BEES")
        writer.setProgramVersion("0.1")
        ok = writer.writeSBMLToFile(doc, output_path)

        if ok:
            n_sp = model.getNumSpecies()
            n_rx = model.getNumReactions()
            self.logger.info(
                f"Exported SBML model to {output_path} "
                f"({n_sp} species, {n_rx} reactions) — load into COPASI or any SBML tool."
            )
            return output_path
        else:
            self.logger.warning(f"SBML export failed (libsbml writer returned error).")
            return None


