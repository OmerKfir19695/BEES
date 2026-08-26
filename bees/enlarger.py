#!/usr/bin/env python3

"""
Iterative Enlarger Module
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from bees.core_edge_model import CoreEdgeModel, SpeciesData
from bees.exporter import reaction_signature
from bees.flux_calculator import (
    identify_insignificant_species_from_peak_ratios,
    identify_significant_species_at_interrupt,
)
from bees.reaction_generator import GeneratedReaction, ReactionGenerator
from bees.simulator import ODESimulator, SimulationResult

@dataclass
class EnlargerResult:
    iterations: int = 0
    converged: bool = False
    convergence_reason: str = ""
    final_core_species: int = 0
    final_edge_species: int = 0
    final_core_reactions: int = 0
    simulation_profiles: list = None  
    model: Optional[CoreEdgeModel] = None

    def __post_init__(self):
        if self.simulation_profiles is None:
            self.simulation_profiles = []


class IterativeEnlarger:
    """
    Enlargement engine. Reactions are generated for the initial input species and then for
    each batch of newly promoted species; simulation and discovery are interleaved.
    """

    def __init__(
        self,
        bees_object,
        reaction_generator: ReactionGenerator,
        logger,
        output_directory: str,
    ):
        self.bees_object = bees_object
        self.reaction_generator = reaction_generator
        self.logger = logger
        self.output_directory = output_directory

        settings = bees_object.settings
        # Safety-valve so the ODE solver always has a finite upper bound when end_time is unset.
        _DEFAULT_END_TIME_FALLBACK = 1e6
        self._end_time_user_set: bool = settings.end_time is not None
        self.end_time: float = (
            float(settings.end_time)
            if self._end_time_user_set
            else _DEFAULT_END_TIME_FALLBACK
        )
        self.time_step: Optional[float] = settings.time_step
        self.tol_move_to_core: float = settings.toleranceMoveToCore
        self.tol_keep_in_edge: float = settings.toleranceKeepInEdge
        self.tol_interrupt_simulation: float = getattr(
            settings, "toleranceInterruptSimulation", None
        )
        if self.tol_interrupt_simulation is None:
            self.tol_interrupt_simulation = self.tol_move_to_core
        _tol_rxn = getattr(settings, "toleranceMoveEdgeReactionToCore", None)
        self.tol_move_edge_reaction_to_core: Optional[float] = (
            float(_tol_rxn) if _tol_rxn is not None else None
        )
        self.min_edge_iterations_for_prune: int = int(
            getattr(settings, "minEdgeIterationsForPrune", 2) or 0
        )
        self.min_core_species_for_prune: int = int(
            getattr(settings, "minCoreSpeciesForPrune", 0) or 0
        )
        self.max_iterations: int = settings.max_iterations
        self.max_edge_species: Optional[int] = settings.max_edge_species
        self.max_num_objects_per_iter: int = int(
            getattr(settings, "max_num_objects_per_iter", 10)
        )
        self.abs_flux_floor: float = float(
            getattr(settings, "abs_flux_floor", 1e-12)
        )
        self.ode_method: str = getattr(settings, "ode_method", None) or "BDF"
        _ode_rtol = getattr(settings, "ode_rtol", None)
        _ode_atol = getattr(settings, "ode_atol", None)
        self.ode_rtol: float = float(_ode_rtol) if _ode_rtol is not None else 1e-8
        self.ode_atol: float = float(_ode_atol) if _ode_atol is not None else 1e-10
        _mwt = getattr(settings, "max_wall_time_per_iteration", None)
        self.max_wall_time_per_iteration: Optional[float] = (
            float(_mwt) if _mwt is not None else None
        )
        self.termination_conversion: Optional[Dict[str, float]] = settings.termination_conversion
        self.termination_rate_ratio: Optional[float] = settings.termination_rate_ratio
        self.save_ode_equations: bool = getattr(
            settings, "save_ode_equations", False
        )
        self.save_simulation_plots: bool = getattr(
            settings, "save_simulation_plots", True
        )
        self.plot_max_species: Optional[int] = getattr(
            settings, "plot_max_species", None
        )
        self.plot_exclude_enzymes: bool = getattr(
            settings, "plot_exclude_enzymes", True
        )
        self.plot_exclude_cofactors: bool = getattr(
            settings, "plot_exclude_cofactors", True
        )

        self.model = CoreEdgeModel()
        self._profiles: List[SimulationResult] = []
        self._edge_species_created_iter: Dict[str, int] = {}
        self._ingest_iteration: int = 0
        self._reaction_id_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int] = {}
        self._reaction_first_seen_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], int] = {}
        self._reaction_core_enter_iter: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Optional[int]] = {}
        self._reaction_obj_by_sig: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], GeneratedReaction] = {}
        self._next_reaction_id: int = 1
        self._iteration_summaries: List[Dict[str, int]] = []

        # Caches / backoff to avoid repeated slow no-hit searches.
        self._rxn_nohit_cache: Set = set()
        self._enzyme_nohit_counts: Dict = {}
        self._skip_enzyme_ec_numbers: Set = set()

        self._thermo_engine = self._build_thermo_engine()
        self._substitutor = self._build_substitutor()
        self._global_smiles_map: Dict[str, str] = self._build_global_smiles_map()

    # ------------------------------------------------------------------
    #  Entry point 
    # ------------------------------------------------------------------

    def run(self) -> EnlargerResult:
        self.logger.info("=" * 60)
        self.logger.info("BEES RATE-BASED MODEL START")
        self.logger.info("=" * 60)

        self._initialise_model()

        self.reaction_generator.ensure_estimator_initialized()

        initial_species = [
            sp.label for sp in self.bees_object.species
            if sp.reactive and not getattr(sp, "solvent", False)
        ]
        initial_reactions = self._generate_reactions_for_species(
            initial_species
        )
        self._ingest_iteration = 0
        self._ingest_reactions(initial_reactions)
        self._sync_reaction_tracking(iteration=0)
        self.logger.info(
            f"Initial discovery: {len(initial_reactions)} reaction(s) "
            f"for {len(initial_species)} input species."
        )

        self.logger.info(
            f"Initial model: {self.model.summary()}"
        )
        s0 = self.model.summary()
        self._iteration_summaries = [{
            "iteration": 0,
            "core_species": s0["core_species"],
            "edge_species": s0["edge_species"],
            "core_reactions": s0["core_reactions"],
            "edge_reactions": s0["edge_reactions"],
        }]

        result = EnlargerResult(model=self.model)

        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"--- Enlargement iteration {iteration} ---")
            result.iterations = iteration

            self.model.reset_concentrations_to_initial()
            self._attach_thermo_to_reactions()

            simulator = ODESimulator(self.model, logger=self.logger)
            sim_result = simulator.simulate(
                end_time=self.end_time,
                time_step=self.time_step,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
                interrupt_simulation_tol=self.tol_interrupt_simulation,
                tol_move_edge_reaction_to_core=self.tol_move_edge_reaction_to_core,
                max_wall_time_s=self.max_wall_time_per_iteration,
            )
            self._profiles.append(sim_result)

            promoted_labels_combined: List[str] = []

            if self.save_ode_equations and sim_result.success:
                ode_path = os.path.join(
                    self.output_directory,
                    f"ode_equations_iter{iteration}.txt",
                )
                simulator.export_ode_equations(
                    output_path=ode_path,
                    iteration=iteration,
                    end_time=self.end_time,
                )

            if not sim_result.success:
                self.logger.warning(f"ODE simulation failed: {sim_result.message}")
                result.convergence_reason = f"ODE failure: {sim_result.message}"
                break

            rxn_promote_species: List[str] = []
            if (
                self.tol_move_edge_reaction_to_core is not None
                and self.tol_move_edge_reaction_to_core > 0
                and sim_result.max_edge_reaction_dlnaccum
            ):
                violators = [
                    (sig, dln)
                    for sig, dln in sim_result.max_edge_reaction_dlnaccum.items()
                    if dln > self.tol_move_edge_reaction_to_core
                ]
                violators.sort(key=lambda kv: kv[1], reverse=True)
                edge_label_set = {sp.label for sp in self.model.edge_species}
                for sig, dln in violators[: self.max_num_objects_per_iter]:
                    rxn_obj = self._reaction_obj_by_sig.get(sig)
                    if rxn_obj is None:
                        continue
                    for sp_label in list(rxn_obj.reactant_labels) + list(rxn_obj.product_labels):
                        if sp_label in edge_label_set and sp_label not in rxn_promote_species:
                            rxn_promote_species.append(sp_label)
                if rxn_promote_species and self.logger:
                    self.logger.info(
                        f"  Reaction-level (dlnaccum) promotion: {len(violators)} edge reaction(s) "
                        f"above tol={self.tol_move_edge_reaction_to_core}; "
                        f"promoting {len(rxn_promote_species)} associated species."
                    )

            if sim_result.simulation_interrupted:
                significant_sub = identify_significant_species_at_interrupt(
                    sim_result.interrupt_edge_rates,
                    sim_result.interrupt_char_rate,
                    self.tol_move_to_core,
                    max_objects=self.max_num_objects_per_iter,
                    abs_flux_floor=self.abs_flux_floor,
                )
                if not significant_sub and not rxn_promote_species:
                    # Interrupt but nothing promotable: resume from here, don't terminate.
                    t_resume = float(sim_result.t[-1]) if len(sim_result.t) > 0 else 0.0
                    if t_resume < self.end_time * 0.99:
                        max_edge_abs = max(
                            (abs(r) for r in sim_result.interrupt_edge_rates.values()),
                            default=0.0,
                        )
                        if self.logger:
                            self.logger.info(
                                f"  Interrupt at t={t_resume:.3e} s produced no new promotable "
                                f"species (R_char={sim_result.interrupt_char_rate:.4e}, "
                                f"max |R_edge|={max_edge_abs:.4e}); resuming simulation."
                            )
                        remaining_time = self.end_time - t_resume
                        sim_result2 = simulator.simulate(
                            end_time=remaining_time,
                            time_step=self.end_time / 100.0,
                            interrupt_simulation_tol=self.tol_interrupt_simulation,
                            tol_move_edge_reaction_to_core=self.tol_move_edge_reaction_to_core,
                        )
                        if sim_result2.success and not sim_result2.simulation_interrupted:
                            sim_result = sim_result2
                        elif sim_result2.simulation_interrupted:
                            sim_result = sim_result2
                        if not sim_result.success:
                            result.convergence_reason = f"ODE failure on resume: {sim_result.message}"
                            if self.logger:
                                self.logger.warning(f"  {result.convergence_reason}")
                            break
                        significant_sub = identify_significant_species_at_interrupt(
                            sim_result.interrupt_edge_rates,
                            sim_result.interrupt_char_rate,
                            self.tol_move_to_core,
                            max_objects=self.max_num_objects_per_iter,
                            abs_flux_floor=self.abs_flux_floor,
                        )
                        rxn_promote_species = []
                        if (
                            self.tol_move_edge_reaction_to_core is not None
                            and self.tol_move_edge_reaction_to_core > 0
                            and sim_result.max_edge_reaction_dlnaccum
                        ):
                            violators2 = [
                                (sig, dln)
                                for sig, dln in sim_result.max_edge_reaction_dlnaccum.items()
                                if dln > self.tol_move_edge_reaction_to_core
                            ]
                            edge_label_set2 = {sp.label for sp in self.model.edge_species}
                            for sig, _dln in violators2[: self.max_num_objects_per_iter]:
                                rxn_obj = self._reaction_obj_by_sig.get(sig)
                                if rxn_obj is None:
                                    continue
                                for sp_label in list(rxn_obj.reactant_labels) + list(rxn_obj.product_labels):
                                    if sp_label in edge_label_set2 and sp_label not in rxn_promote_species:
                                        rxn_promote_species.append(sp_label)
                        if not significant_sub and not rxn_promote_species:
                            max_edge_abs = max(
                                (abs(r) for r in sim_result.interrupt_edge_rates.values()),
                                default=0.0,
                            )
                            result.convergence_reason = (
                                "ODE interrupted but no edge species exceeded "
                                f"threshold (R_char={sim_result.interrupt_char_rate:.4e}, "
                                f"max |R_edge|={max_edge_abs:.4e}, "
                                f"tol={self.tol_move_to_core})."
                            )
                            self.logger.warning(f"  {result.convergence_reason}")
                            break
                    else:
                        max_edge_abs = max(
                            (abs(r) for r in sim_result.interrupt_edge_rates.values()),
                            default=0.0,
                        )
                        result.convergence_reason = (
                            "ODE interrupted but no edge species exceeded "
                            f"threshold (R_char={sim_result.interrupt_char_rate:.4e}, "
                            f"max |R_edge|={max_edge_abs:.4e}, "
                            f"tol={self.tol_move_to_core})."
                        )
                        self.logger.warning(f"  {result.convergence_reason}")
                        break

                t_int = float(sim_result.t[-1]) if len(sim_result.t) > 0 else 0.0
                labels_csv = ", ".join(sf.label for sf in significant_sub[:3])
                if len(significant_sub) > 3:
                    labels_csv += f", ... (+{len(significant_sub) - 3} more)"
                self.logger.info(
                    f"  Interrupt at t={t_int:.6e} s: "
                    f"R_char={sim_result.interrupt_char_rate:.6e}, "
                    f"promoting {len(significant_sub)}: [{labels_csv}]"
                )
                for sf in significant_sub:
                    rr_s = (
                        f"{sf.normalized_rate:.4e}"
                        if sf.normalized_rate != float("inf")
                        else "inf"
                    )
                    self.logger.debug(
                        f"    {sf.label}: rr={rr_s}, |rate|={abs(sf.rate):.4e} mM/s"
                    )

                batch_promoted: List[str] = []
                for sf in significant_sub:
                    sp = self.model.promote_species_to_core(sf.label)
                    if sp is not None:
                        batch_promoted.append(sp.label)
                        promoted_labels_combined.append(sp.label)
                for sp_label in rxn_promote_species:
                    sp = self.model.promote_species_to_core(sp_label)
                    if sp is not None and sp.label not in batch_promoted:
                        batch_promoted.append(sp.label)
                        promoted_labels_combined.append(sp.label)
                if not batch_promoted:
                    result.convergence_reason = (
                        "Interrupt: significant flux reported but no edge species "
                        "could be promoted."
                    )
                    self.logger.warning(f"  {result.convergence_reason}")
                    break

                n_reclassed = self.model.reclassify_reactions()
                self.logger.info(
                    f"  Promoted {len(batch_promoted)} species to core, "
                    f"{n_reclassed} reactions moved edge->core."
                )

                new_rxns_sub = self._generate_reactions_for_species(
                    batch_promoted
                )
                self._ingest_iteration = iteration
                self._ingest_reactions(new_rxns_sub)
                self._attach_thermo_to_reactions()
                self._sync_reaction_tracking(iteration=iteration)
                self.logger.info(
                    f"  Generated {len(new_rxns_sub)} new reaction(s); restarting ODE from t=0."
                )

            else:
                r_char_final = sim_result.final_char_rate
                r_char_max = sim_result.max_char_rate
                self.logger.info(
                    f"  R_char at t_end = {r_char_final:.6e} mM/s  |  "
                    f"R_char peak (this run) = {r_char_max:.6e} mM/s"
                )

                if self._check_conversion_termination(sim_result):
                    result.converged = True
                    result.convergence_reason = "Termination conversion target reached."
                    self.logger.info(result.convergence_reason)
                elif self._check_rate_ratio_termination(sim_result):
                    result.converged = True
                    result.convergence_reason = "Termination rate ratio reached."
                    self.logger.info(result.convergence_reason)
                elif self._end_time_user_set:
                    result.converged = True
                    result.convergence_reason = (
                        "Simulation reached end_time without exceeding "
                        "toleranceInterruptSimulation."
                    )
                    self.logger.info(result.convergence_reason)
                else:
                    self.logger.info(
                        "  max end_time reached; conversion and "
                        "rate-ratio criteria not yet met — continuing iteration."
                    )

                # Optional pruning (only meaningful on a full trajectory).
                if self.tol_keep_in_edge > 0:
                    n_core = len(self.model.core_species)
                    if n_core < self.min_core_species_for_prune:
                        self.logger.info(
                            "  Skipping edge prune: core species count "
                            f"{n_core} < minCoreSpeciesForPrune "
                            f"({self.min_core_species_for_prune})."
                        )
                    else:
                        ineligible_prune: Set[str] = set()
                        for sp in self.model.edge_species:
                            lc = sp.label.lower().strip()
                            birth = self._edge_species_created_iter.get(lc, 0)
                            if iteration - birth < self.min_edge_iterations_for_prune:
                                ineligible_prune.add(lc)
                        to_prune = identify_insignificant_species_from_peak_ratios(
                            sim_result.max_edge_rate_ratio,
                            sim_result.max_char_rate,
                            self.tol_keep_in_edge,
                            ineligible_for_prune=ineligible_prune,
                        )
                        n_pruned = self.model.prune_edge(to_prune)
                        if n_pruned:
                            self.logger.info(f"  Pruned {n_pruned} edge species.")

            promoted_labels_combined = list(dict.fromkeys(promoted_labels_combined))

            if (
                self.max_edge_species is not None
                and len(self.model.edge_species) > self.max_edge_species
            ):
                result.convergence_reason = (
                    f"Exceeded max_edge_species ({self.max_edge_species})."
                )
                self.logger.info(result.convergence_reason)
                break

            self.logger.info(f"  Model status: {self.model.summary()}")
            s_it = self.model.summary()
            self._iteration_summaries.append({
                "iteration": iteration,
                "core_species": s_it["core_species"],
                "edge_species": s_it["edge_species"],
                "core_reactions": s_it["core_reactions"],
                "edge_reactions": s_it["edge_reactions"],
            })
            if result.converged:
                break

        else:
            result.convergence_reason = (
                f"Reached max_iterations ({self.max_iterations})."
            )
            self.logger.info(result.convergence_reason)

        summary = self.model.summary()
        result.final_core_species = summary["core_species"]
        result.final_edge_species = summary["edge_species"]
        result.final_core_reactions = summary["core_reactions"]
        result.simulation_profiles = self._profiles
        result.model = self.model
        if result.iterations == 0:
            result.iterations = min(
                self.max_iterations, max(1, len(self._profiles))
            )

        self.logger.info("=" * 60)
        self.logger.info("ENLARGEMENT SUMMARY")
        self.logger.info(f"  Iterations:      {result.iterations}")
        self.logger.info(f"  Converged:       {result.converged}")
        self.logger.info(f"  Reason:          {result.convergence_reason}")
        self.logger.info(f"  Core species:    {result.final_core_species}")
        self.logger.info(f"  Edge species:    {result.final_edge_species}")
        self.logger.info(f"  Core reactions:  {result.final_core_reactions}")
        self.logger.info("=" * 60)

        self._log_thermo_fallback_prevalence()

        return result

    def _log_thermo_fallback_prevalence(self) -> None:
        """Log the fraction of core reactions that used a thermodynamic fallback.

        Reactions with source='fallback' have ΔG°' = nan and run as irreversible
        with no Keq correction. A high fallback fraction (>5%) signals chemically
        inconsistent stoichiometry or unknown compounds and deserves review.

        The configurable threshold is settings.max_thermo_fallback_fraction
        (default 0.05 = 5%). Exceeding it emits a WARNING; a hard failure is
        not raised so that legitimate niche reactions lacking equilibrator data
        can still proceed.
        """
        fallback_rxns = [
            rxn for rxn in self.model.core_reactions
            if getattr(rxn.thermo, "source", None) == "fallback"
        ]
        n_fallback = len(fallback_rxns)
        n_total = len(self.model.core_reactions)
        fraction = n_fallback / n_total if n_total > 0 else 0.0

        threshold = float(getattr(
            getattr(self.bees_object, "settings", None),
            "max_thermo_fallback_fraction",
            0.05,
        ))

        if n_fallback == 0:
            self.logger.info(
                f"Thermo fallback gate: 0 / {n_total} core reactions use fallback ΔG°' (all resolved). ✓"
            )
        else:
            rxn_ids = []
            for rxn in fallback_rxns:
                sig = reaction_signature(rxn)
                rid = self._reaction_id_by_sig.get(sig)
                rxn_ids.append(f"R{rid}" if rid is not None else "R?")
            ids_str = ", ".join(rxn_ids[:10]) + (" ..." if len(rxn_ids) > 10 else "")
            msg = (
                f"Thermo fallback gate: {n_fallback} / {n_total} core reactions "
                f"({100*fraction:.1f}%) use fallback ΔG°' (nan) — reactions: [{ids_str}]"
            )
            if fraction > threshold:
                self.logger.warning(
                    f"{msg} — exceeds {100*threshold:.0f}% threshold; "
                    "exported as irreversible forward-only MM. "
                    "To resolve: add compound SMILES or review stoichiometry."
                )
            else:
                self.logger.info(f"{msg} — exported as irreversible forward-only MM.")


    # ------------------------------------------------------------------
    # Generate reactions for species
    # ------------------------------------------------------------------

    
    def _generate_reactions_for_species(
        self,
        species_labels: List[str],
    ) -> List[GeneratedReaction]:
        from types import SimpleNamespace
        from bees.cofactors import GENERAL_COFACTORS
        from bees.common import get_ontology_equivalents

        new_reactions: List[GeneratedReaction] = []
        enzymes = [e for e in self.bees_object.enzymes if e.reactive]
        available_lc = self.model.get_species_labels_lc()

        full_available: Set[str] = set()
        for lc in available_lc:
            full_available.update(get_ontology_equivalents(lc))

        provided_lc = set(available_lc)

        for label in species_labels:
            if label.lower().strip() in GENERAL_COFACTORS:
                continue
            substrate = SimpleNamespace(
                label=label,
                reactive=True,
                solvent=False,
                smiles=None,
            )
            for enzyme in enzymes:
                _ecval = getattr(enzyme, "ecnumber", None) or ""
                if isinstance(_ecval, list):
                    ec = tuple(e.strip() for e in _ecval)
                else:
                    ec = _ecval.strip()
                if ec and ec in self._skip_enzyme_ec_numbers:
                    continue
                key = (ec, label.lower().strip())
                if ec and key in self._rxn_nohit_cache:
                    continue

                t0 = time.time()
                rxns = self.reaction_generator._generate_reactions(
                    enzyme,
                    substrate,
                    available_species_labels_lc=full_available,
                    provided_species_labels_lc=provided_lc,
                    thermo_engine=self._thermo_engine,
                    global_smiles_map=self._global_smiles_map,
                    substitutor=self._substitutor,
                )
                dt = time.time() - t0
                new_reactions.extend(rxns)

                if ec and not rxns:
                    self._rxn_nohit_cache.add(key)
                    self._enzyme_nohit_counts[ec] = self._enzyme_nohit_counts.get(ec, 0) + 1

                    # FabA backoff: skip EC 4.2.1.60 after a slow no-hit (>30 s).
                    if ec == "EC 4.2.1.60" and self._enzyme_nohit_counts[ec] >= 1 and dt > 30.0:
                        self._skip_enzyme_ec_numbers.add(ec)
                        self.logger.info(
                            f"Skipping further reaction-generation calls for {ec} (repeated no-hits; last call {dt:.1f}s)."
                        )

        return new_reactions

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------

    def _check_conversion_termination(
        self, sim_result: SimulationResult
    ) -> bool:
        """X = 1 - C(t) / C(0); returns True when any target species reaches its threshold."""
        if not self.termination_conversion:
            return False

        if sim_result.y.shape[1] == 0:
            return False

        label_to_idx = {
            lab.lower().strip(): i
            for i, lab in enumerate(sim_result.species_labels)
        }
        for species_label, target_frac in self.termination_conversion.items():
            lc = species_label.lower().strip()
            idx = label_to_idx.get(lc)
            if idx is None:
                continue
            c0 = sim_result.y[idx, 0]
            if c0 <= 0:
                continue
            c_final = sim_result.y[idx, -1]
            conversion = 1.0 - (max(c_final, 0.0) / c0)
            if conversion >= target_frac:
                self.logger.info(
                    f"  Conversion of '{species_label}': "
                    f"{conversion:.4f} >= {target_frac}"
                )
                return True
        return False

    def _check_rate_ratio_termination(
        self,
        sim_result: SimulationResult,
    ) -> bool:
        """Stop when R_char(t_end) / R_char(peak) < termination_rate_ratio."""
        if self.termination_rate_ratio is None:
            return False

        max_cr = sim_result.max_char_rate
        final_cr = sim_result.final_char_rate
        if max_cr <= 0.0:
            return False

        ratio = final_cr / max_cr
        if ratio < self.termination_rate_ratio:
            self.logger.info(
                f"  Char rate ratio (R_char at t_end / peak): {ratio:.6e} "
                f"< {self.termination_rate_ratio}"
            )
            return True
        return False


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_substitutor(self):
        """Build a CompoundSubstitutor with any user-supplied extra mappings."""
        from bees.substitutor_registry import AcylACPSubstitutor
        extra = getattr(self.bees_object.settings, "thermo_smiles_substitutions", None) or {}
        return AcylACPSubstitutor(extra_mappings=extra)

    def _build_thermo_engine(self):
        """Construct ThermoEngine from Environment settings. Returns None if unavailable."""
        from bees.thermodynamics import ThermoEngine
        substitutor = getattr(self, "_substitutor", None) or self._build_substitutor()
        env = self.bees_object.environment
        T = env.temperature
        if isinstance(T, tuple):
            T = 0.5 * (T[0] + T[1])
        pH = env.pH
        if isinstance(pH, tuple):
            pH = 0.5 * (pH[0] + pH[1])
        return ThermoEngine(
            pH=float(pH),
            ionic_strength_M=float(getattr(env, "ionic_strength_M", 0.25)),
            pMg=float(getattr(env, "pMg", 3.0)),
            T_K=float(T),
            irreversible_cutoff_kJmol=float(
                getattr(self.bees_object.settings, "thermo_irreversible_cutoff_kJmol", 30.0)
            ),
            substitutor=substitutor,
        )

    def _build_global_smiles_map(self) -> Dict[str, str]:
        """Build label→SMILES map from user-declared input species and thermo substitutions."""
        smiles: Dict[str, str] = {}
        for sp in getattr(self.bees_object, "species", []) or []:
            smi = getattr(sp, "smiles", None)
            if smi:
                smiles[sp.label] = smi
        substitutions = getattr(self.bees_object.settings, "thermo_smiles_substitutions", None) or {}
        smiles.update(substitutions)
        return smiles

    def _attach_thermo_to_reactions(self) -> None:
        # RULES.apply_all restores then applies (no compounding). Sync template.reversible after.
        from bees.rules import load_rules
        rules = load_rules()
        # Re-apply every iteration so the enabled set never leaks across runs.
        rules.configure_calibrations(
            getattr(self.bees_object.settings, "calibrations", None) or []
        )
        all_reactions = list(self.model.core_reactions) + list(self.model.edge_reactions)
        rules.apply_all(all_reactions)
        self._sync_template_reversibility(all_reactions)
        self._apply_feedback_inhibition(all_reactions)

    def _apply_feedback_inhibition(self, reactions) -> None:
        """Attach opt-in end-product feedback inhibition (Enzyme.feedback_inhibition).

        Sets ``rxn.feedback_inhibitors`` to {inhibitor_label_lc: (Ki, hill)} for
        reactions whose enzyme declares a spec (and whose Ki is known), else None.
        Matched by enzyme LABEL (not EC), so isozymes sharing an EC (FabA/FabZ =
        EC 4.2.1.59) are targeted unambiguously. Re-applied every iteration so the
        configured set is deterministic per run and never leaks across runs in one
        process (mirrors calibrations). Specs with ``ki is None`` are left off here —
        those are resolved by the CatPred ki path when wired.
        """
        spec_by_label = {
            str(enz.label).lower().strip(): enz.feedback_inhibition
            for enz in getattr(self.bees_object, "enzymes", []) or []
            if getattr(enz, "feedback_inhibition", None) is not None
        }
        for rxn in reactions:
            spec = spec_by_label.get(str(getattr(rxn, "enzyme_label", "")).lower().strip())
            if spec is None or spec.ki is None:
                rxn.feedback_inhibitors = None
                continue
            rxn.feedback_inhibitors = {
                str(inh).lower().strip(): (float(spec.ki), float(spec.hill))
                for inh in spec.inhibitors
            }

    @staticmethod
    def _sync_template_reversibility(reactions) -> None:
        """Propagate thermo.irreversible onto template.reversible after rules."""
        from dataclasses import replace as dc_replace
        import copy as _copy
        for rxn in reactions:
            thermo = getattr(rxn, "thermo", None)
            if thermo is None:
                continue
            target = not thermo.irreversible
            tpl = getattr(rxn, "template", None)
            if tpl is None or getattr(tpl, "reversible", None) == target:
                continue
            try:
                rxn.template = dc_replace(tpl, reversible=target)
            except TypeError:
                try:
                    rxn.template = _copy.copy(tpl)
                    rxn.template.reversible = target
                except AttributeError:
                    pass

    def _initialise_model(self) -> None:
        for sp in self.bees_object.species:
            conc = sp.concentration
            if isinstance(conc, tuple):
                conc = conc[0]  # use lower bound of range
            sd = SpeciesData(
                label=sp.label,
                concentration=conc,
                initial_concentration=conc,
                is_enzyme=False,
                constant=getattr(sp, "constant", False),
            )
            self.model.add_core_species(sd)

        for enz in self.bees_object.enzymes:
            conc = enz.concentration
            if isinstance(conc, tuple):
                conc = conc[0]
            sd = SpeciesData(
                label=enz.label,
                concentration=conc,
                initial_concentration=conc,
                is_enzyme=True,
                constant=True, 
            )
            self.model.add_core_species(sd)

    def _ingest_reactions(self, reactions: List[GeneratedReaction]) -> None:
        for rxn in reactions:
            for label in rxn.reactant_labels + rxn.product_labels:
                label= label.lower().strip()
                if (
                    not self.model.is_core_species(label)
                    and not self.model.is_edge_species(label)
                ):
                    self.model.add_edge_species(SpeciesData(
                        label=label,
                        concentration=0.0,
                        initial_concentration=0.0,
                    ))
                    self._edge_species_created_iter[label] = self._ingest_iteration
            self.model.add_reaction(rxn)

    def _sync_reaction_tracking(self, iteration: int) -> None:
        """
        Synchronize reaction metadata from the current model state.
        Assigns stable IDs and tracks first-seen/core-entry iterations.
        """
        all_reactions = list(self.model.core_reactions) + list(self.model.edge_reactions)
        core_sigs = {
            reaction_signature(rxn) for rxn in self.model.core_reactions
        }
        for rxn in all_reactions:
            sig = reaction_signature(rxn)
            if sig not in self._reaction_id_by_sig:
                self._reaction_id_by_sig[sig] = self._next_reaction_id
                self._next_reaction_id += 1
            if sig not in self._reaction_first_seen_iter:
                self._reaction_first_seen_iter[sig] = iteration
            if sig not in self._reaction_obj_by_sig:
                self._reaction_obj_by_sig[sig] = rxn
            if sig in core_sigs and sig not in self._reaction_core_enter_iter:
                self._reaction_core_enter_iter[sig] = iteration

    
  