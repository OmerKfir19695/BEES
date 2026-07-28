#!/usr/bin/env python3

"""
Enlarger Exporter Module
-----------------------
Holds all export/plot functionality for the iterative enlarger.


"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
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


# Species that legitimately appear as products without a downstream consumer.
# Free fatty acids (anything ending in "ate" / "oic acid") are matched
# separately by suffix in the SBML export invariant. Anything else added here
# is asserting "yes, this accumulating product is biologically real."
TERMINAL_PRODUCT_ALLOWLIST: Set[str] = {
    "coenzyme a",
    "nadp",
    "nad",
    "carbon dioxide",
}


def reaction_signature(
    reaction: GeneratedReaction,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """
    Canonical reaction signature for stable reaction ID tracking.
    """
    enzyme = str(reaction.enzyme_label).lower().strip()
    reactants = tuple(sorted(str(r).lower().strip() for r in reaction.reactant_labels))
    products = tuple(sorted(str(p).lower().strip() for p in reaction.product_labels))
    return (enzyme, reactants, products)


@dataclass
class EnlargerExporter:
    """
    Exporter for IterativeEnlarger outputs (CSVs, plots, reaction tree).

    Reads state that was produced by the enlarger module and writes artifacts to disk.
    """

    model: CoreEdgeModel
    profiles: List[SimulationResult]
    output_directory: str
    logger: Any

    reaction_id_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int]
    reaction_first_seen_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int]
    reaction_core_enter_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Optional[int]]
    reaction_obj_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], GeneratedReaction]
    iteration_summaries: List[Dict[str, int]]

    # Plot/export settings
    save_reaction_tree_plots: bool = False
    save_simulation_plots: bool = True
    plot_max_species: Optional[int] = None
    plot_exclude_enzymes: bool = True
    plot_exclude_cofactors: bool = True
    reaction_tree_layout: str = "graphviz"
    reaction_tree_rankdir: str = "TB"
    reaction_tree_fontsize: int = 8

    # Stateful across iterations (used by reaction-tree coloring)
    core_seen_labels: Optional[Set[str]] = None
    # Optional access to original input (used for cofactor exclusion in plots)
    bees_object: Optional[Any] = None

    # ------------------------------------------------------------------
    # Export helpers - flux analysis
    # ------------------------------------------------------------------

    def export_flux_analysis(self, filename: str = "flux_analysis.csv") -> Optional[str]:
        """
        Write per-iteration flux data (core/edge growth summary + reaction history).
        """
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
        """
        Write concentration time-series to CSV.

        Concatenates profiles from every iteration into a single file.
        Returns the output path, or None if there are no profiles.
        """
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

    # ------------------------------------------------------------------
    # Export helpers - reaction tree (visualisation)
    # ------------------------------------------------------------------

    def export_reaction_tree(
        self,
        *,
        iteration: int,
        promoted_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Export a reaction tree plot for this iteration.

        The tree:
        - Includes only core species.
        - Excludes enzymes and general cofactors.
        - Colours newly promoted core species in this iteration differently
          from species that were already in the core.
        """
        if not self.save_reaction_tree_plots or not self.model:
            return

        if self.core_seen_labels is None:
            self.core_seen_labels = set()

        core_nodes: List[SpeciesData] = []
        for sd in self.model.core_species:
            if sd.is_enzyme:
                continue
            if is_general_cofactor_label(sd.label):
                continue
            core_nodes.append(sd)

        if not core_nodes:
            return

        labels = [sd.label for sd in core_nodes]
        label_set = set(labels)

        promoted_set = set(promoted_labels or [])
        new_nodes: Set[str] = set()
        for lab in labels:
            if lab in promoted_set and lab not in self.core_seen_labels:
                new_nodes.add(lab)

        self.core_seen_labels.update(labels)

        edges: List[tuple] = []
        edge_reaction_labels: Dict[tuple, Set[str]] = {}
        for rxn in self.model.core_reactions:
            sig = reaction_signature(rxn)
            reaction_id = self.reaction_id_by_sig.get(sig)
            reaction_tag = f"R{reaction_id}" if reaction_id is not None else ""
            for reactant in rxn.reactant_labels:
                if reactant not in label_set:
                    continue
                for product in rxn.product_labels:
                    if product not in label_set:
                        continue
                    if reactant == product:
                        continue
                    edge = (reactant, product)
                    edges.append(edge)
                    if reaction_tag:
                        edge_reaction_labels.setdefault(edge, set()).add(reaction_tag)

        edges = sorted(set(edges))
        if not edges:
            return

        xs: Dict[str, float] = {}
        ys: Dict[str, float] = {}

        def _simple_layout() -> None:
            old_labels = [lab for lab in labels if lab not in new_nodes]
            new_labels_ordered = [lab for lab in labels if lab in new_nodes]

            def _assign_row(row_labels: List[str], y_val: float) -> None:
                n = len(row_labels)
                if n == 0:
                    return
                if n == 1:
                    xs[row_labels[0]] = 0.5
                    ys[row_labels[0]] = y_val
                    return
                for i, lab in enumerate(row_labels):
                    xs[lab] = i / (n - 1)
                    ys[lab] = y_val

            _assign_row(old_labels, y_val=0.0)
            _assign_row(new_labels_ordered, y_val=-1.0)

        def _graphviz_layout() -> bool:
            dot_exe = shutil.which("dot")
            if not dot_exe:
                return False

            rankdir = str(self.reaction_tree_rankdir or "TB").upper()
            if rankdir not in {"TB", "BT", "LR", "RL"}:
                rankdir = "TB"

            node_id: Dict[str, str] = {}
            for i, lab in enumerate(labels):
                node_id[lab] = f"n{i}"

            dot_lines: List[str] = [
                "digraph ReactionTree {",
                f'  rankdir="{rankdir}";',
                "  splines=true;",
                "  overlap=false;",
                "  nodesep=0.35;",
                "  ranksep=0.6;",
                "  node [shape=circle];",
            ]
            for lab in labels:
                dot_lines.append(f'  {node_id[lab]} [label="{node_id[lab]}"];')
            for src, dst in edges:
                dot_lines.append(f"  {node_id[src]} -> {node_id[dst]};")
            dot_lines.append("}")
            dot = "\n".join(dot_lines)

            try:
                proc = subprocess.run(
                    [dot_exe, "-Tplain"],
                    input=dot.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except Exception:
                return False

            positions_raw: Dict[str, tuple[float, float]] = {}
            for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("node "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                name = parts[1]
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                except ValueError:
                    continue
                positions_raw[name] = (x, y)

            if not positions_raw:
                return False

            xs_vals = [p[0] for p in positions_raw.values()]
            ys_vals = [p[1] for p in positions_raw.values()]
            min_x, max_x = min(xs_vals), max(xs_vals)
            min_y, max_y = min(ys_vals), max(ys_vals)
            span_x = max(1e-9, max_x - min_x)
            span_y = max(1e-9, max_y - min_y)

            inv_node_id = {v: k for k, v in node_id.items()}
            for nid, (x, y) in positions_raw.items():
                lab = inv_node_id.get(nid)
                if not lab:
                    continue
                xs[lab] = (x - min_x) / span_x
                ys[lab] = (y - min_y) / span_y
            return len(xs) > 0 and len(ys) > 0

        used_graphviz = (
            str(self.reaction_tree_layout or "graphviz").lower() == "graphviz"
            and _graphviz_layout()
        )
        if not used_graphviz:
            _simple_layout()

        fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
        ax.set_facecolor("white")

        for src, dst in edges:
            ax.annotate(
                "",
                xy=(xs[dst], ys[dst]),
                xytext=(xs[src], ys[src]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#555555",
                    linewidth=1.0,
                    alpha=0.8,
                ),
            )
            rid_set = edge_reaction_labels.get((src, dst), set())
            if rid_set:
                rid_text = ",".join(
                    sorted(
                        rid_set,
                        key=lambda s: int(s[1:])
                        if s.startswith("R") and s[1:].isdigit()
                        else 10**9,
                    )
                )
                mid_x = (xs[src] + xs[dst]) / 2.0
                mid_y = (ys[src] + ys[dst]) / 2.0
                ax.annotate(
                    rid_text,
                    xy=(mid_x, mid_y),
                    xycoords="data",
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#222222",
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )

        old_color = "#A6CEE3"
        new_color = "#FB9A99"

        def _shorten_label(label: str) -> str:
            s = str(label).strip()
            s = re.sub(r"\s+", " ", s)
            if len(s) <= 10 and " " not in s:
                return s
            tokens = re.split(r"[\s\-_]+", s)
            keep = []
            for t in tokens:
                if not t:
                    continue
                if t.isdigit() or re.fullmatch(r"\d+[A-Za-z]*", t or ""):
                    keep.append(t)
                else:
                    keep.append(t[0].upper())
            base = "".join(keep) or s[:6].upper()
            if len(base) > 12:
                base = base[:12]
            return base

        cumulative_mapping_path = os.path.join(
            self.output_directory,
            "reaction_tree_labels.csv",
        )
        full_to_short: Dict[str, str] = {}
        first_seen_by_full: Dict[str, int] = {}
        used_short_labels: Set[str] = set()

        if os.path.exists(cumulative_mapping_path):
            try:
                with open(cumulative_mapping_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        full = str(row.get("full_label", "")).strip()
                        short = str(row.get("short_label", "")).strip()
                        first_seen_raw = str(row.get("first_seen_iteration", "")).strip()
                        if not full or not short:
                            continue
                        try:
                            first_seen = int(first_seen_raw)
                        except ValueError:
                            first_seen = iteration
                        if full not in full_to_short:
                            full_to_short[full] = short
                            first_seen_by_full[full] = first_seen
                            used_short_labels.add(short)
            except Exception:
                full_to_short = {}
                first_seen_by_full = {}
                used_short_labels = set()

        for lab in labels:
            if lab in full_to_short:
                continue
            base = _shorten_label(lab)
            candidate = base
            if candidate in used_short_labels:
                digest = hashlib.blake2s(
                    str(lab).encode("utf-8"), digest_size=2
                ).hexdigest()
                suffix = digest.upper()
                candidate = f"{base}-{suffix}"
            if candidate in used_short_labels:
                i = 2
                while f"{candidate}{i}" in used_short_labels:
                    i += 1
                candidate = f"{candidate}{i}"
            full_to_short[lab] = candidate
            first_seen_by_full[lab] = iteration
            used_short_labels.add(candidate)

        try:
            with open(cumulative_mapping_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["short_label", "full_label", "first_seen_iteration"])
                for full in sorted(
                    full_to_short.keys(),
                    key=lambda x: (first_seen_by_full.get(x, iteration), x.lower()),
                ):
                    w.writerow(
                        [
                            full_to_short[full],
                            full,
                            first_seen_by_full.get(full, iteration),
                        ]
                    )
        except Exception as exc:
            self.logger.warning("Failed to write cumulative label mapping to %s: %s", cumulative_mapping_path, exc)

        for lab in labels:
            display_label = str(full_to_short.get(lab, lab))
            color = new_color if lab in new_nodes else old_color
            lines = display_label.splitlines() if display_label else [""]
            n_lines = max(1, len(lines))
            max_line_len = max((len(line) for line in lines), default=0)
            s = 400 + 45 * (max_line_len**1.15) + 220 * n_lines
            s = max(700, min(s, 8000))
            ax.scatter(
                xs[lab],
                ys[lab],
                s=s,
                c=color,
                edgecolors="#333333",
                linewidths=1.0,
                zorder=3,
            )
            ax.text(
                xs[lab],
                ys[lab],
                display_label,
                ha="center",
                va="center",
                fontsize=max(8, int(self.reaction_tree_fontsize)),
                color="black",
                zorder=4,
                wrap=True,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.1, 1.1)
        min_y = min(ys.values())
        max_y = max(ys.values())
        ax.set_ylim(min_y - 0.5, max_y + 0.5)

        ax.set_title(f"Reaction tree – iteration {iteration}", fontsize=14)

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=old_color, edgecolor="#333333", label="Existing core species"),
            Patch(facecolor=new_color, edgecolor="#333333", label="New in this iteration"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
        )

        fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

        out_path = os.path.join(
            self.output_directory,
            f"reaction_tree_iter{iteration}.png",
        )
        fig.savefig(
            out_path,
            dpi=250,
            bbox_inches="tight",
            pad_inches=0.25,
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        self.logger.info(
            f"Exported reaction tree plot for iteration {iteration} to {out_path}"
        )

    # ------------------------------------------------------------------
    # Export helpers - simulation plots
    # ------------------------------------------------------------------

    def export_simulation_plots(
        self,
        filename_pattern: str = "simulation_plot_iter{}.png",
    ) -> Optional[List[str]]:
        """
        Plot concentration vs time for each iteration and save to PNG.
        """
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
    # Export helpers - SBML (for COPASI / any SBML-compatible tool)
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
        """
        Export the reaction network as an SBML Level 3 Version 2 file.

        Structure of the generated SBML:
        - One compartment: ``cytosol`` volume = 1 L, so concentrations in mM
          map directly to amounts in mmol).
        - One ``species`` per model species; ``initialConcentration`` is the
          value from the last iteration (mM).  Enzyme species are set
          ``constant=true, boundaryCondition=true`` so COPASI treats them as
          fixed parameters.
        - One ``parameter`` per unique enzyme holding its concentration (mM).
        - One ``reaction`` per core reaction (or core + edge if
          ``core_only=False``) with:
          - Explicit ``listOfReactants`` / ``listOfProducts`` stoichiometries.
          - A Michaelis-Menten ``kineticLaw`` written as a MathML formula:
            ``kcat * E * prod_i(S_i / (Km_i + S_i))``
          - All kinetic constants stored as local ``parameter`` elements
            inside the kineticLaw.

        Args:
            filename: Output file name inside the project output directory.
            core_only: If True (default) export only core reactions/species.
                       Set False to include edge species/reactions as well.
            strict_invariant: If True raise ``ValueError`` when any non-boundary,
                       non-terminal species is produced by some reaction but
                       never consumed by any reaction. Default False — the
                       check logs a warning instead, since with constant
                       cofactor pools an irreversible production-only species
                       would accumulate without bound (the original FAS bug
                       pattern), but most such findings in a converged model
                       are legitimate branch exits or terminal products. Set
                       True for stricter dev-time checks.

        Returns:
            Absolute path of the written SBML file, or None (if libsbml is
            not installed or the model is empty)
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
        # 2.  One compartment: cytosol, volume = 1 L.
        #     Each kineticLaw below is multiplied by this compartment so it is a
        #     proper SBML *substance* (amount/time) rate; with size = 1 the amount
        #     in mmol equals the concentration in mM, and dC/dt comes out equal to
        #     the bare rate expression (see kineticLaw construction comment).
        # ----------------------------------------------------------------
        comp = model.createCompartment()
        comp.setId("cytosol")
        comp.setName("Cytosol")
        comp.setSize(1.0)  # 1 L (see note above; size cancels once rate law carries it)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)

        # ----------------------------------------------------------------
        # 3.  Species
        # ----------------------------------------------------------------
        # Build a safe SBML id from a label (SBML ids must start with letter/underscore)
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

        # Build initial concentration mapping from original input
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
            sp.setCompartment("cytosol")
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

        # Production-only invariant: catch incomplete enlarger snapshots
        # before they reach a downstream integrator. Any non-boundary,
        # non-terminal species that gets produced but never consumed will
        # accumulate without bound under integration (the FAS bug:
        # 3-oxo-(11Z)-octadec-ACP was production-only as an edge before its
        # FabG reduction was discovered, leading to a 4× carbon overshoot).
        produced_lcs: Set[str] = set()
        consumed_lcs: Set[str] = set()
        for rxn in reactions:
            # For reversible reactions the reverse direction consumes products
            # and produces reactants — account for both directions so that a
            # species that is only the product of a reversible core reaction
            # (e.g. an isomerization with Keq≈1) is not incorrectly flagged.
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

        # Also collect consumers from edge reactions (not included in core-only
        # export). Species consumed only by edge reactions are branch exits —
        # the enlarger found them but hadn't yet promoted their downstream
        # reactions before termination. These are not structural leaks.
        edge_consumed_lcs: Set[str] = set()
        for rxn in self.model.edge_reactions:
            for lab in getattr(rxn, "reactant_labels", ()) or ():
                edge_consumed_lcs.add(lab.lower().strip())

        leaks: List[str] = []
        branch_exits: List[str] = []
        for sp_lc in sorted(produced_lcs - consumed_lcs):
            if sp_lc in boundary_labels:
                continue
            # General cofactor pools (ATP/ADP, NAD/NADH, H+, H2O, etc.) may
            # legitimately appear as net products in pathway-limited snapshots.
            # Treat them as exempt from this structural leak invariant.
            if is_general_cofactor_label(sp_lc):
                continue
            if sp_lc in TERMINAL_PRODUCT_ALLOWLIST:
                continue
            # Free fatty acid family: bare carboxylates that exit the cycle.
            if sp_lc.endswith("ate") or sp_lc.endswith("oic acid"):
                continue
            # Species consumed by an edge reaction is a branch exit: the
            # enlarger found it but termination fired before promoting the
            # downstream edge reactions. Not a structural leak.
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

        # ----------------------------------------------------------------
        # 4.  Global parameters: enzyme concentrations
        # ----------------------------------------------------------------
        enzyme_param_ids: Dict[str, str] = {}  # enzyme_label_lc -> param_id
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

        # ----------------------------------------------------------------
        # 5.  Reactions
        # ----------------------------------------------------------------
        used_rxn_ids: Set[str] = set()

        for rxn_idx, rxn in enumerate(reactions, start=1):
            sig = reaction_signature(rxn)
            rxn_num = self.reaction_id_by_sig.get(sig, rxn_idx)
            rid = f"R{rxn_num}"
            if rid in used_rxn_ids:
                rid = f"R{rxn_num}_{rxn_idx}"
            used_rxn_ids.add(rid)

            # Fallback reactions (nan ΔG°', source="fallback") are exported as
            # irreversible forward-only MM — same treatment as source="disabled".
            # The use_rev gate below rejects nan Keq, so no reversible formula
            # is written; the reaction is still topologically correct in SBML.

            sbml_rxn = model.createReaction()
            sbml_rxn.setId(rid)
            # Reversibility flag, name, arrow, and notes are emitted AFTER the
            # kinetic-law branch is chosen (further down) so they always agree
            # with the actual rate law (forward-only vs. reversible Liebermeister).
            td = getattr(rxn, "thermo", None)

            # Reactants
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

            # Products
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

            # KineticLaw: reversible Liebermeister-Klipp MM when thermo data
            # is available, otherwise forward-only MM: kcat * E * ∏(Si/(Km_Si+Si))
            kin = getattr(rxn, "kinetics", None)
            if kin is None or getattr(rxn, "rate_law", None) is None:
                self.logger.debug(
                    "SBML: %s skipped kinetic law (no kinetics/rate_law) — "
                    "reaction structure exported without rate equation",
                    rid,
                )
                # No kinetic law → reaction is structurally present but has
                # no rate. Mark irreversible and emit metadata before skipping
                # to next reaction so the file is self-consistent.
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

            # Determine whether to emit the reversible formula.
            # Requires: finite Keq, Haldane-derived kcat_rev, and kcat available.
            # Product Kms default to 1.0 mM when absent (consistent with how
            # kcat_rev is computed by haldane_kcat_rev when product list is empty).
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

            # Build substrate Km pairs. Boundary species (H+, H2O, enzymes)
            # are skipped — their fixed concentrations don't limit the rate and
            # their activities are already incorporated into the Keq.
            substrate_km_pairs: List[Tuple[str, str, int]] = []
            for r_label in rxn.reactant_labels:
                r_lc = r_label.lower().strip()
                if r_lc in boundary_labels:
                    continue  # fixed concentration, no Km term
                sid = label_to_id.get(r_lc)
                if sid is None:
                    continue  # species not exported — skip term
                km_val: Optional[float] = _km_for_label(r_label)
                if km_val is None or km_val <= 0:
                    use_rev = False  # missing Km for variable substrate → cannot write sat term
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

            # Build product Km pairs for variable (non-boundary) products.
            # When no DB-derived Km is available, default to 1.0 mM — this is
            # self-consistent because kcat_rev was computed via Haldane with
            # prod_p = 1.0 (empty product Km list → ∏Km_P = 1).
            product_km_pairs: List[Tuple[str, str, int]] = []
            if use_rev:
                for p_label in rxn.product_labels:
                    p_lc = p_label.lower().strip()
                    if p_lc in boundary_labels:
                        continue  # fixed concentration, omit from reverse numerator
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
                    use_rev = False  # all products are boundary — no thermodynamic coupling

            # Enzyme token
            e_lc = rxn.enzyme_label.lower().strip()
            e_param = enzyme_param_ids.get(e_lc)
            e_token = e_param if e_param is not None else "0.001"

            # kcat forward parameter
            pid_kcat: Optional[str] = None
            if kcat_val is not None:
                pid_kcat = f"kcat_{rid}"
                p_kcat = model.createParameter()
                p_kcat.setId(pid_kcat)
                p_kcat.setName(f"kcat ({rxn.enzyme_label})")
                p_kcat.setValue(float(kcat_val))
                p_kcat.setConstant(True)

            # Build formula
            if use_rev and substrate_km_pairs and product_km_pairs and pid_kcat:
                # Reversible Liebermeister-Klipp:
                #   v = (kcat_f * E * ∏(Si/Km_Si) − kcat_r * E * ∏(Pj/Km_Pj))
                #       / (∏(1+Si/Km_Si) + ∏(1+Pj/Km_Pj) − 1)
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

            # Opt-in end-product feedback inhibition (Enzyme.feedback_inhibition),
            # the simulator's Overlay 3. Multiply the rate by ∏_k 1/(1+([I_k]/Ki)^h)
            # over the named inhibitor species. Inhibitors are not consumed, so each
            # is declared as a modifierSpeciesReference (required for kineticLaw refs).
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

            # SBML defines a kineticLaw as a *substance* rate (amount/time); a solver
            # then computes dC/dt = kineticLaw / V.  The formulas above are
            # *concentration* rates (mM/s), so they must be multiplied by the
            # compartment volume to survive that division — exactly RMG's liquid
            # solver `res = core_species_rates * V`.  Without this the exported model
            # integrates 1/V times too fast in COPASI / roadrunner.
            if formula != "0":
                formula = f"cytosol * ({formula})"

            # Use the L3 parser (parseL3Formula + setMath) rather than the
            # legacy L1 setFormula() — L3 handles modern operators correctly
            # and keeps the AST clean for downstream tools like COPASI.
            _ast = _libsbml.parseL3Formula(formula)
            if _ast is None:
                # Fall back to legacy formula; log so we can spot it.
                self.logger.warning(
                    "SBML: %s parseL3Formula failed (%s) — falling back to setFormula",
                    rid, _libsbml.getLastParseL3Error(),
                )
                kl.setFormula(formula)
            else:
                kl.setMath(_ast)

            # Reversibility flag, name, arrow, and notes — emitted now so they
            # always match the actual kinetic law emitted above.
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
                # irreversible= reflects the emitted SBML attribute, not raw thermo
                notes_parts.append(f"irreversible={not emitted_reversible}")
                sbml_rxn.setNotes(
                    "<body xmlns='http://www.w3.org/1999/xhtml'><p>"
                    + "; ".join(notes_parts)
                    + "</p></body>"
                )

        # ----------------------------------------------------------------
        # 5b. Global quantities for the released free fatty acid, so any SBML
        #     tool reads the product directly (no post-processing):
        #       total_free_FA_uM        = 1000 * Σ [acid]              (molar total)
        #       palmitic_equivalents_uM = 1000 * Σ (carbons/16)·[acid]
        #         — the Yu-2011 / Ruppe "palmitic equivalents" metric (C16 = 1 unit).
        #     Species concentrations are mM; the ×1000 reports µM (figure units).
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
        fa_band_terms: List[str] = []  # Yu-2011 S2A TLC band = C14–C18 only
        for sd in species_list:
            c = _fa_carbons(sd.label)
            if c is None:
                continue
            sid = label_to_id[sd.label.lower().strip()]
            fa_total_terms.append(sid)
            fa_palm_terms.append(f"({c}/16) * {sid}")
            # Yu 2011 kinetic assay quantifies a single TLC band in which C14–C18
            # free acids comigrate (identical Rf); shorter acids sit elsewhere and
            # water-soluble ones are lost in hexanes extraction. So the S2A time
            # course is C14–C18 only — this is the correct like-for-like observable.
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

        # ----------------------------------------------------------------
        # 6.  Validate and write
        # ----------------------------------------------------------------
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


