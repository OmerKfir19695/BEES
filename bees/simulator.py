#!/usr/bin/env python3

"""
ODE Simulator Module
--------------------
Integrates the reaction network over time using scipy's ODE solvers.

All concentrations are in mM, time in seconds, rates in mM/s.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from bees.common import get_ontology_equivalents
from bees.core_edge_model import CoreEdgeModel
from bees.flux_calculator import (
    compute_mm_rate,
    compute_reversible_mm_rate,
    has_complete_explicit_product_kms,
)
from bees.conservator import Conservator


@dataclass
class SimulationResult:
    """
    ODE simulation output.
    """
    t: np.ndarray
    y: np.ndarray
    species_labels: List[str]
    success: bool = True
    message: str = ""
    max_char_rate: float = 0.0
    final_char_rate: float = 0.0
    max_edge_rate_ratio: Dict[str, float] = field(default_factory=dict)
    simulation_interrupted: bool = False
    interrupt_char_rate: float = 0.0
    interrupt_edge_rates: Dict[str, float] = field(default_factory=dict)
    # reaction-level dynamics metric (dlnaccum) per edge reaction.
    # Keyed by reaction signature ((reactants_sorted_tuple, products_sorted_tuple)).
    # max_edge_reaction_dlnaccum is the maximum dlnaccum_j seen during the run.
    max_edge_reaction_dlnaccum: Dict[Tuple, float] = field(default_factory=dict)


class _VectorizedRHS:
    """
    Pre-compiled, NumPy-vectorized ODE right-hand side for a fixed reaction network.

    Builds all matrices and index arrays once at construction time so that
    each call to ``__call__(t, y)`` runs without any Python-level loops or
    dict lookups.
    """

    def __init__(
        self,
        species_labels: List[str],
        reactions: list,
        alias_to_model_label: Dict[str, str],
        enzyme_conc_map: Dict[str, float],
        constant_mask: np.ndarray,
        n_core_species: Optional[int] = None,
        n_core_reactions: Optional[int] = None,
    ):
        n_sp = len(species_labels)
        n_rx = len(reactions)
        label_to_idx = {lab.lower().strip(): i for i, lab in enumerate(species_labels)}

        # ---- Per-reaction Vmax ----------------------------------------
        # Two overlay paths keep the vectorized fast-path clean:
        #   1. Reversible MM: compute_reversible_mm_rate 
        #   2. Product-inhibited irreversible MM: compute_mm_rate with Liebermeister & Klipp eq
        vmax = np.zeros(n_rx, dtype=np.float64)
        reversible_indices: List[int] = []
        product_inhibited_indices: List[int] = []
        for j, rxn in enumerate(reactions):
            is_reversible = (
                getattr(rxn.template, "reversible", False)
                and getattr(rxn, "thermo", None) is not None
                and not rxn.thermo.irreversible
            )
            if is_reversible:
                reversible_indices.append(j)
                continue
            # Product-inhibited overlay: irreversible reactions with complete
            # explicit product Kms use compute_mm_rate (Liebermeister denominator).
            if has_complete_explicit_product_kms(rxn):
                product_inhibited_indices.append(j)
                continue  # leave vmax[j] = 0
            kin = rxn.kinetics
            if kin is None or rxn.rate_law is None:
                continue
            e_key = rxn.enzyme_label.lower().strip()
            e_conc = enzyme_conc_map.get(e_key, 0.0)
            if kin.kcat is not None and e_conc > 0:
                vmax[j] = kin.kcat * e_conc
            elif kin.vmax is not None:
                vmax[j] = kin.vmax

        # ---- Saturation substrate lists (ragged -> matrix) -------------
        # For each reaction we store a list of (species_idx, km_value) pairs.
        # We pack these into matrices (n_rx, K) where K is max substrates.
        sub_idx_list: List[List[int]] = []
        km_list:      List[List[float]] = []
        nu_list:      List[List[float]] = []

        for rxn in reactions:
            kin = rxn.kinetics
            pairs_idx: List[int] = []
            pairs_km:  List[float] = []
            pairs_nu:  List[float] = []
            if kin is not None and rxn.rate_law is not None:
                km_per = getattr(kin, "km_per_substrate", None) or {}
                km_single = kin.km
                stoich = rxn.stoichiometry
                for reactant in rxn.reactant_labels:
                    r_lc = reactant.lower().strip()
                    km_val = None
                    if km_per:
                        km_val = km_per.get(reactant)
                        if km_val is None:
                            km_val = next(
                                (v for k, v in km_per.items()
                                 if k.lower().strip() == r_lc),
                                None,
                            )
                        if km_val is None:
                            continue
                    else:
                        km_val = km_single
                    if km_val is None or km_val <= 0:
                        continue
                    model_lc = alias_to_model_label.get(r_lc, r_lc)
                    idx = label_to_idx.get(model_lc)
                    if idx is None:
                        idx = label_to_idx.get(r_lc)
                    if idx is not None:
                        pairs_idx.append(idx)
                        pairs_km.append(float(km_val))
                        pairs_nu.append(float(abs(stoich.get(reactant, 1))))
            sub_idx_list.append(pairs_idx)
            km_list.append(pairs_km)
            nu_list.append(pairs_nu)

        K = max((len(idxs) for idxs in sub_idx_list), default=0)
        # Pad with dummy index pointing to a constant 1.0 concentration.
        # Padded ν = 0 so (s/(Km+s))**0 = 1 contributes neutrally.
        self._sub_idx_mat = np.full((n_rx, K), fill_value=n_sp, dtype=np.int32)
        self._km_mat = np.zeros((n_rx, K), dtype=np.float64)
        self._nu_mat = np.zeros((n_rx, K), dtype=np.float64)
        for j, (idxs, kms, nus) in enumerate(zip(sub_idx_list, km_list, nu_list)):
            nj = len(idxs)
            if nj > 0:
                self._sub_idx_mat[j, :nj] = idxs
                self._km_mat[j, :nj] = kms
                self._nu_mat[j, :nj] = nus

        # ---- Feedback / end-product inhibition (opt-in) ----------------
        # For reactions carrying rxn.feedback_inhibitors ({label_lc: (Ki, hill)}),
        # the final rate is multiplied by ∏_k 1/(1+([I_k]/Ki)^hill). Inhibitor
        # species are resolved to indices here; missing species are skipped.
        fb_idx_list: List[List[int]] = []
        fb_ki_list:  List[List[float]] = []
        fb_h_list:   List[List[float]] = []
        for rxn in reactions:
            inh = getattr(rxn, "feedback_inhibitors", None)
            if not isinstance(inh, dict):  # None, or a MagicMock in tests
                inh = {}
            pi: List[int] = []
            pk: List[float] = []
            ph: List[float] = []
            for label_lc, params in inh.items():
                try:
                    ki, hill = float(params[0]), float(params[1])
                except (TypeError, IndexError, ValueError):
                    continue
                if ki <= 0:
                    continue
                model_lc = alias_to_model_label.get(label_lc, label_lc)
                idx = label_to_idx.get(model_lc, label_to_idx.get(label_lc))
                if idx is not None:
                    pi.append(idx)
                    pk.append(ki)
                    ph.append(hill)
            fb_idx_list.append(pi)
            fb_ki_list.append(pk)
            fb_h_list.append(ph)

        M = max((len(p) for p in fb_idx_list), default=0)
        self._has_feedback = M > 0 and any(fb_idx_list)
        # Pad: sentinel idx -> appended 1.0; Ki = inf so (1.0/inf)^h = 0 -> factor 1.
        self._fb_idx_mat = np.full((n_rx, M), fill_value=n_sp, dtype=np.int32)
        self._fb_ki_mat = np.full((n_rx, M), fill_value=np.inf, dtype=np.float64)
        self._fb_hill_mat = np.ones((n_rx, M), dtype=np.float64)
        for j, (idxs, kis, hs) in enumerate(zip(fb_idx_list, fb_ki_list, fb_h_list)):
            nj = len(idxs)
            if nj > 0:
                self._fb_idx_mat[j, :nj] = idxs
                self._fb_ki_mat[j, :nj] = kis
                self._fb_hill_mat[j, :nj] = hs

        # ---- Stoichiometry matrix S  (n_species x n_reactions) ---------
        rows, cols, data = [], [], []
        for j, rxn in enumerate(reactions):
            for sp_label, coeff in rxn.stoichiometry.items():
                lc = sp_label.lower().strip()
                model_lc = alias_to_model_label.get(lc, lc)
                idx = label_to_idx.get(model_lc)
                if idx is not None:
                    rows.append(idx)
                    cols.append(j)
                    data.append(float(coeff))
        S = csc_matrix((data, (rows, cols)), shape=(n_sp, n_rx), dtype=np.float64)

        # Store as dense if small, sparse otherwise
        if n_rx <= 200:
            self._S = S.toarray()
            self._sparse = False
        else:
            self._S = S
            self._sparse = True

        S_prod = S.copy()
        S_prod.data[S_prod.data < 0] = 0.0
        S_prod.eliminate_zeros()
        S_cons = S.copy()
        S_cons.data[S_cons.data > 0] = 0.0
        S_cons.data = np.abs(S_cons.data)
        S_cons.eliminate_zeros()

        if n_rx <= 200:
            self._S_prod = S_prod.toarray()
            self._S_cons = S_cons.toarray()
        else:
            self._S_prod = S_prod
            self._S_cons = S_cons

        self._vmax = vmax
        self._n_rx = n_rx
        self._constant_mask = constant_mask
        self._n_sp = n_sp
        self.label_to_idx = label_to_idx
        self.alias_to_model_label = alias_to_model_label

        # Edge isolation (RMG-style):
        #   - edge SPECIES rows are forced to zero in the residual so their
        #     concentrations never drift from the initial value;
        #   - edge REACTION fluxes are zeroed before S @ v so they cannot
        #     contribute to core species' dC/dt either.
        # When n_core_* are omitted we treat everything as core (legacy).
        self._n_core_species = n_sp if n_core_species is None else int(n_core_species)
        self._n_core_reactions = n_rx if n_core_reactions is None else int(n_core_reactions)
        self._edge_species_mask = np.zeros(n_sp, dtype=bool)
        if self._n_core_species < n_sp:
            self._edge_species_mask[self._n_core_species:] = True
        self._integration_mask = constant_mask | self._edge_species_mask

        self._reversible_indices = reversible_indices
        self._reversible_reactions = [reactions[j] for j in reversible_indices]
        self._product_inhibited_indices = product_inhibited_indices
        self._product_inhibited_reactions = [reactions[j] for j in product_inhibited_indices]
        self._species_labels = list(species_labels)
        self._enzyme_conc_map = dict(enzyme_conc_map)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate dydt = S @ v(y)."""
        return self.compute_dydt(y)

    def compute_v(self, y: np.ndarray) -> np.ndarray:
        """
        Compute reaction rate vector v(y).
        Supports y as (n_species,) or (n_species, n_timepoints).

        Mass-conservation note: the raw state y is passed unclipped through to
        the stoichiometry matrix so that S @ v preserves every conserved moiety
        analytically. 
        """
        is_mat = (y.ndim > 1)

        # Append dummy 1.0 for out-of-bounds substrate indices (padding sentinel).
        # Use raw y — DO NOT clip here; clipping breaks the mass-balance of S @ v.
        if is_mat:
            ones = np.ones((1, y.shape[1]), dtype=y.dtype)
            y_ext = np.vstack([y, ones])
        else:
            y_ext = np.append(y, 1.0)

        # Saturation: s / (Km + s), clipped to [0, 1] to keep rates non-negative.
        # Clip only inside the saturation formula — not globally on y.
        s = y_ext[self._sub_idx_mat]
        s_pos = np.maximum(s, 0.0)

        if is_mat:
            km = self._km_mat[:, :, np.newaxis]
            nu = self._nu_mat[:, :, np.newaxis]
            sat = s_pos / (km + s_pos)
            # product over K dimension (axis 1), with stoichiometric exponents
            v = self._vmax[:, np.newaxis] * np.prod(sat ** nu, axis=1)
        else:
            sat = s_pos / (self._km_mat + s_pos)
            v = self._vmax * np.prod(sat ** self._nu_mat, axis=1)

        # Pass raw (possibly negative) concentrations; overlay functions apply max(c, 0.0) themselves.
        _need_overlay = self._reversible_indices or self._product_inhibited_indices
        if _need_overlay and not is_mat:
            conc = {
                lab.lower().strip(): float(y[i])
                for i, lab in enumerate(self._species_labels)
            }

        # Overlay 1: reversible MM (may return negative v for reverse flux).
        if self._reversible_indices:
            if is_mat:
                n_t = y.shape[1]
                for col, rxn in zip(self._reversible_indices, self._reversible_reactions):
                    for k in range(n_t):
                        conc_k = {
                            lab.lower().strip(): float(y[i, k])
                            for i, lab in enumerate(self._species_labels)
                        }
                        v[col, k] = compute_reversible_mm_rate(
                            rxn, conc_k, self._enzyme_conc_map
                        )
            else:
                for col, rxn in zip(self._reversible_indices, self._reversible_reactions):
                    v[col] = compute_reversible_mm_rate(rxn, conc, self._enzyme_conc_map)

        # Overlay 2: product-inhibited irreversible MM (forward-only).
        if self._product_inhibited_indices:
            if is_mat:
                n_t = y.shape[1]
                for col, rxn in zip(self._product_inhibited_indices, self._product_inhibited_reactions):
                    for k in range(n_t):
                        conc_k = {
                            lab.lower().strip(): float(y[i, k])
                            for i, lab in enumerate(self._species_labels)
                        }
                        v[col, k] = compute_mm_rate(rxn, conc_k, self._enzyme_conc_map)
            else:
                for col, rxn in zip(self._product_inhibited_indices, self._product_inhibited_reactions):
                    v[col] = compute_mm_rate(rxn, conc, self._enzyme_conc_map)

        # Overlay 3: feedback / end-product inhibition (opt-in, multiplicative).
        # v[j] *= ∏_k 1/(1+([I_k]/Ki)^hill). Neutral (factor 1) for reactions
        # without a spec (padding uses Ki=inf -> term 1).
        if self._has_feedback:
            if is_mat:
                c = np.maximum(y_ext[self._fb_idx_mat], 0.0)        # (n_rx, M, n_t)
                ki = self._fb_ki_mat[:, :, np.newaxis]
                hill = self._fb_hill_mat[:, :, np.newaxis]
                factor = np.prod(1.0 / (1.0 + (c / ki) ** hill), axis=1)
            else:
                c = np.maximum(y_ext[self._fb_idx_mat], 0.0)        # (n_rx, M)
                factor = np.prod(
                    1.0 / (1.0 + (c / self._fb_ki_mat) ** self._fb_hill_mat), axis=1
                )
            v = v * factor

        return v

    def compute_dydt(self, y: np.ndarray) -> np.ndarray:
        """Compute dydt = S @ v(y) for the integrator (edges isolated).

        Edge-reaction fluxes are zeroed before the stoichiometric multiply so
        they cannot drain or feed core species, and edge-species rows are
        zeroed afterwards so their concentrations stay at the initial value.
        For the un-masked rate vector used by promotion scoring, call
        :meth:`compute_dydt_unmasked`.
        """
        v = self.compute_v(y)
        if self._n_core_reactions < self._n_rx:
            # Zero edge reactions on a copy so compute_v's result stays intact
            # for any caller that wants the raw rate vector.
            v = v.copy()
            if v.ndim == 1:
                v[self._n_core_reactions:] = 0.0
            else:
                v[self._n_core_reactions:, :] = 0.0
        if self._sparse:
            dydt = self._S.dot(v)
            if hasattr(dydt, "toarray"):
                dydt = dydt.toarray()
            if dydt.ndim > 1 and v.ndim == 1:
                dydt = dydt.ravel()
        else:
            dydt = self._S @ v

        if dydt.ndim > 1:
            dydt[self._integration_mask, :] = 0.0
        else:
            dydt[self._integration_mask] = 0.0
        return dydt

    def compute_dydt_unmasked(self, y: np.ndarray) -> np.ndarray:
        """Compute dydt = S @ v(y) WITHOUT edge isolation.

        Used by promotion-scoring code paths that need the would-be dC/dt of
        edge species and the would-be contribution of edge reactions. The
        constant_mask is still applied (enzymes / buffers should always read
        as flat regardless of which path computes the residual).
        """
        v = self.compute_v(y)
        if self._sparse:
            dydt = self._S.dot(v)
            if hasattr(dydt, "toarray"):
                dydt = dydt.toarray()
            if dydt.ndim > 1 and v.ndim == 1:
                dydt = dydt.ravel()
        else:
            dydt = self._S @ v

        if dydt.ndim > 1:
            dydt[self._constant_mask, :] = 0.0
        else:
            dydt[self._constant_mask] = 0.0
        return dydt

    def compute_prod_cons(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute production and consumption rates for all species."""
        v = self.compute_v(y)
        if self._sparse:
            p = self._S_prod.dot(v)
            c = self._S_cons.dot(v)
            if hasattr(p, "toarray"):
                p = p.toarray()
            if hasattr(c, "toarray"):
                c = c.toarray()
            if p.ndim > 1 and v.ndim == 1:
                p = p.ravel()
            if c.ndim > 1 and v.ndim == 1:
                c = c.ravel()
        else:
            p = self._S_prod @ v
            c = self._S_cons @ v
        return p, c


class ODESimulator:
    """Integrate the biochemical reaction network for a CoreEdgeModel."""

    def __init__(self, model: CoreEdgeModel, logger=None):
        self.model = model
        self.logger = logger

    @staticmethod
    def _concentrations_nonnegative(y: np.ndarray) -> np.ndarray:
        """Project concentrations (mM) to non-negative values, in-place safe."""
        out = np.asarray(y, dtype=float)
        return np.maximum(out, 0.0)

    def simulate(
        self,
        end_time: float,
        time_step: Optional[float] = None,
        method: str = "BDF",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        interrupt_simulation_tol: Optional[float] = None,
        tol_move_edge_reaction_to_core: Optional[float] = None,
        max_wall_time_s: Optional[float] = None,
        stepwise_heartbeat_interval_s: Optional[float] = None,
    ) -> SimulationResult:
        """Run an ODE simulation of the full model (core + edge species and reactions)."""
        species_labels = self.model.get_all_species_labels()
        n_species = len(species_labels)

        if n_species == 0:
            return SimulationResult(
                t=np.array([0.0]),
                y=np.empty((0, 1)),
                species_labels=[],
                success=True,
                message="No species to simulate.",
            )

        edge_labels_lc = {sp.label.lower().strip() for sp in self.model.edge_species}

        use_stepwise = (
            interrupt_simulation_tol is not None
            and interrupt_simulation_tol > 0
            and len(edge_labels_lc) > 0
        )
        if use_stepwise:
            return self._simulate_stepwise(
                end_time=end_time,
                time_step=time_step,
                method=method,
                rtol=rtol,
                atol=atol,
                interrupt_simulation_tol=float(interrupt_simulation_tol),
                tol_move_edge_reaction_to_core=tol_move_edge_reaction_to_core,
                max_wall_time_s=max_wall_time_s,
                stepwise_heartbeat_interval_s=stepwise_heartbeat_interval_s,
            )
        return self._simulate_continuous(
            end_time=end_time,
            time_step=time_step,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------
    # Continuous (non-interrupt) simulation path
    # ------------------------------------------------------------------

    def _simulate_continuous(
        self,
        end_time: float,
        time_step: Optional[float],
        method: str,
        rtol: float,
        atol: float,
    ) -> SimulationResult:
        """Full-span solve_ivp without flux-interrupt events."""
        species_labels = self.model.get_all_species_labels()
        n_species = len(species_labels)
        label_to_idx = {lab.lower().strip(): i for i, lab in enumerate(species_labels)}
        alias_to_model_label = self._build_alias_to_model_label(species_labels)
        enzyme_conc_map = self._build_enzyme_concentration_map()
        constant_mask = np.array(
            [s.constant for s in self.model.core_species]
            + [False] * len(self.model.edge_species),
            dtype=bool,
        )
        y0 = self._concentrations_nonnegative(
            np.array(self.model.get_all_concentration_vector(), dtype=float)
        )
        reactions = self.model.core_reactions + self.model.edge_reactions

        t_eval = None
        if time_step is not None and time_step > 0:
            t_eval = np.arange(0.0, end_time + time_step * 0.5, time_step)
            t_eval = t_eval[t_eval <= end_time]

        rhs = _VectorizedRHS(
            species_labels=species_labels,
            reactions=reactions,
            alias_to_model_label=alias_to_model_label,
            enzyme_conc_map=enzyme_conc_map,
            constant_mask=constant_mask,
            n_core_species=len(self.model.core_species),
            n_core_reactions=len(self.model.core_reactions),
        )
        conservator = Conservator(species_labels)

        if self.logger:
            self.logger.debug(
                f"Running ODE simulation (continuous): {n_species} species, "
                f"{len(reactions)} reactions, t=[0, {end_time}] s"
            )

        sol = solve_ivp(
            rhs, (0.0, end_time), y0,
            method=method, t_eval=t_eval,
            rtol=rtol, atol=atol,
            dense_output=True,
            max_step=end_time / 10 if end_time > 0 else np.inf,
        )

        t_out = sol.t
        y_raw = sol.y
        extra_msg = ""
        ok = sol.success
        if y_raw.size > 0 and ok:
            if not np.isfinite(y_raw).all():
                ok = False
                extra_msg = "Non-finite values (NaN or inf) in concentration state."
                if self.logger:
                    self.logger.error(f"ODE state invalid: {extra_msg}")
        y_out = (
            conservator.clip_2d(y_raw) if y_raw.size else y_raw
        )

        result = SimulationResult(
            t=t_out, y=y_out,
            species_labels=species_labels,
            success=ok,
            message=(
                (sol.message if hasattr(sol, "message") else "")
                + (f" {extra_msg}" if extra_msg else "")
            ).strip(),
        )

        if ok and y_out.shape[1] > 0:
            self.model.set_all_concentrations(y_out[:, -1].tolist())

        self._attach_flux_metrics(result, rhs=rhs)

        if self.logger:
            if not sol.success:
                self.logger.warning(
                    f"ODE solver FAILED: {getattr(sol, 'message', 'unknown')}"
                )
            elif ok:
                self.logger.debug(
                    f"ODE simulation completed "
                    f"({t_out.shape[0]} time points)"
                )
        return result

    # ------------------------------------------------------------------
    # Simulation with flux-ratio interrupt
    # ------------------------------------------------------------------

    def _simulate_stepwise(
        self,
        end_time: float,
        time_step: Optional[float],
        method: str,
        rtol: float,
        atol: float,
        interrupt_simulation_tol: float,
        tol_move_edge_reaction_to_core: Optional[float] = None,
        max_wall_time_s: Optional[float] = None,
        stepwise_heartbeat_interval_s: Optional[float] = None,
    ) -> SimulationResult:
        """Advance by dt; interrupt when any edge flux ratio exceeds tolerance (earliest at t > 0)."""
        species_labels = self.model.get_all_species_labels()
        n_species = len(species_labels)
        label_to_idx = {lab.lower().strip(): i for i, lab in enumerate(species_labels)}
        alias_to_model_label = self._build_alias_to_model_label(species_labels)
        enzyme_conc_map = self._build_enzyme_concentration_map()
        constant_mask = np.array(
            [s.constant for s in self.model.core_species]
            + [False] * len(self.model.edge_species),
            dtype=bool,
        )
        y0 = self._concentrations_nonnegative(
            np.array(self.model.get_all_concentration_vector(), dtype=float)
        )
        reactions = self.model.core_reactions + self.model.edge_reactions
        edge_labels_lc = {sp.label.lower().strip() for sp in self.model.edge_species}
        edge_labels_lc_list = [sp.label.lower().strip() for sp in self.model.edge_species]
        n_core = len(self.model.core_species)
        n_core_rxns = len(self.model.core_reactions)

        rhs = _VectorizedRHS(
            species_labels=species_labels,
            reactions=reactions,
            alias_to_model_label=alias_to_model_label,
            enzyme_conc_map=enzyme_conc_map,
            constant_mask=constant_mask,
            n_core_species=n_core,
            n_core_reactions=n_core_rxns,
        )
        conservator = Conservator(species_labels)

        # ---- Edge-reaction setup for dlnaccum---
        # products separately to look up consumption/production at each step.
        # Signature must match `bees.exporter.reaction_signature`:
        # (enzyme_lc, reactants_sorted_tuple, products_sorted_tuple).
        edge_rxn_sigs: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
        edge_rxn_reactant_idx: List[List[int]] = []
        edge_rxn_product_idx: List[List[int]] = []
        edge_rxn_global_idx: List[int] = []
        for j_local, rxn in enumerate(self.model.edge_reactions):
            j_global = n_core_rxns + j_local
            r_idx: List[int] = []
            p_idx: List[int] = []
            for sp_label, coeff in rxn.stoichiometry.items():
                lc = sp_label.lower().strip()
                model_lc = alias_to_model_label.get(lc, lc)
                idx = label_to_idx.get(model_lc)
                if idx is None:
                    idx = label_to_idx.get(lc)
                if idx is None:
                    continue
                # Stoichiometric multiplicity: 
                # reactant/product *index lists* (a coefficient of 2 contributes twice).
                mult = int(round(abs(float(coeff))))
                if mult <= 0:
                    mult = 1
                if float(coeff) < 0:
                    r_idx.extend([idx] * mult)
                elif float(coeff) > 0:
                    p_idx.extend([idx] * mult)
            sig = (
                str(getattr(rxn, "enzyme_label", "")).lower().strip(),
                tuple(sorted(str(r).lower().strip() for r in rxn.reactant_labels)),
                tuple(sorted(str(p).lower().strip() for p in rxn.product_labels)),
            )
            edge_rxn_sigs.append(sig)
            edge_rxn_reactant_idx.append(r_idx)
            edge_rxn_product_idx.append(p_idx)
            edge_rxn_global_idx.append(j_global)

        max_edge_rxn_dlnaccum: Dict[Tuple, float] = {sig: 0.0 for sig in edge_rxn_sigs}
        # Small floor to keep ln(1 + v/R) finite when R ~ 0.
        _RATE_FLOOR = 1e-30

        if stepwise_heartbeat_interval_s is None:
            heartbeat_interval = 30.0
        else:
            heartbeat_interval = float(stepwise_heartbeat_interval_s)

        if self.logger:
            self.logger.debug(
                f"Running ODE simulation (stepwise): {n_species} species, "
                f"{len(reactions)} reactions, t=[0, {end_time}] s, "
                f"tol_interrupt={interrupt_simulation_tol}"
            )
            if max_wall_time_s is not None:
                self.logger.debug(
                    f"Stepwise wall-clock limit: {max_wall_time_s} s per pass"
                )
            if heartbeat_interval > 0:
                self.logger.debug(
                    f"Stepwise heartbeat interval: {heartbeat_interval} s"
                )

        t_history: List[float] = [0.0]
        y_history: List[np.ndarray] = [y0.copy()]

        wall_start = time.monotonic()
        last_heartbeat = wall_start
        wall_time_exceeded = False

        current_t = 0.0
        current_y = y0.copy()
        # Outer-step size bounds. A floor avoids spending wall time on 1e-12 s
        # segments when max_rr sits below interrupt_tol (no interrupt possible
        # yet) but old backoff logic kept shrinking dt.
        dt_max = 5.0
        if end_time > 0.0:
            dt_floor_outer = max(1e-12, min(1e-6, end_time * 1e-8))
        else:
            dt_floor_outer = 1e-12
        dt = dt_floor_outer
        dt_min = dt_floor_outer
        dt_growth = 2.0
        near_frac = 0.5
        max_char_rate = 0.0
        interrupted = False
        interrupt_char = 0.0
        interrupt_edge: Dict[str, float] = {}
        first_step = True
        outer_step_idx = 0
        state_invalid = False
        invalid_message = ""

        while current_t < end_time:
            now_wall = time.monotonic()
            if max_wall_time_s is not None and (now_wall - wall_start) > max_wall_time_s:
                wall_time_exceeded = True
                if self.logger:
                    self.logger.warning(
                        "ODE stepwise pass stopped: exceeded "
                        f"max_wall_time_per_iteration ({max_wall_time_s} s wall-clock) "
                        f"at simulation t={current_t:.6e} s."
                    )
                break

            dt = min(max(dt, dt_min), dt_max)
            target_t = min(current_t + dt, end_time)
            if target_t <= current_t:
                break

            try:
                sol = solve_ivp(
                    rhs, (current_t, target_t), current_y,
                    method=method, rtol=rtol, atol=atol,
                    dense_output=False,
                    max_step=target_t - current_t,
                )
            except Exception:
                if self.logger:
                    self.logger.warning(
                        f"ODE solver exception at t={current_t:.6e} s"
                    )
                break

            if not sol.success or sol.y.shape[1] == 0:
                if self.logger:
                    self.logger.warning(
                        f"ODE solver failed at t={current_t:.6e} s: "
                        f"{getattr(sol, 'message', 'unknown')}"
                    )
                break

            current_t = float(sol.t[-1])
            current_y = sol.y[:, -1].copy()
            outer_step_idx += 1
            if not np.isfinite(current_y).all():
                state_invalid = True
                invalid_message = (
                    "Non-finite values (NaN or inf) in concentration state."
                )
                if self.logger:
                    self.logger.error(
                        f"ODE state invalid at t={current_t:.6e} s after outer step. "
                        f"{invalid_message}"
                    )
                break
            # Do NOT clip current_y here. Clipping mid-integration breaks
            # conservation: when BDF temporarily places a species at a tiny
            # negative value (numerical roundoff), `np.maximum(y, 0)` adds
            # that negative mass back as positive without subtracting it
            # from any conjugate species. With small initial pools,
            # this compounds across many outer steps
            # and many species into massive drift.

            t_history.append(current_t)
            y_history.append(conservator.clip(current_y).copy())

            # Enlarger uses the UNMASKED residual for both R_char and edge
            # rates. The isolated residual is for the integrator only. With
            # the bootstrap (core empty, all edges) the isolated R_char is
            # zero and the rate-ratio interrupt could never fire; unmasked
            # gives the right signal for promotion. See _attach_flux_metrics.
            dydt_unmasked_step = rhs.compute_dydt_unmasked(current_y)
            char_rate = float(np.linalg.norm(dydt_unmasked_step[:n_core]))
            if edge_labels_lc_list:
                edge_rates = {
                    lab: float(dydt_unmasked_step[n_core + i])
                    for i, lab in enumerate(edge_labels_lc_list)
                }
            else:
                edge_rates = {}
            max_char_rate = max(max_char_rate, char_rate)

            max_rr = 0.0
            if char_rate > 0.0 and edge_rates:
                max_rr = max((abs(r) / char_rate for r in edge_rates.values()), default=0.0)

            # ---- reaction-level dlnaccum---
            # dlnaccum_j = Σ_i ln(1 + v_j / R_i) over species i touched by edge
            # reaction j; R_i is consumption (L_i) for reactants, production
            # (P_i) for products. Tracks per-step max over the run.
            max_rxn_dlnaccum_step = 0.0
            max_rxn_dlnaccum_sig: Optional[Tuple] = None
            if edge_rxn_sigs:
                v_all = rhs.compute_v(current_y)
                prod_all, cons_all = rhs.compute_prod_cons(current_y)
                # prod_all/cons_all are species production/consumption magnitudes.
                for k, sig in enumerate(edge_rxn_sigs):
                    j_global = edge_rxn_global_idx[k]
                    v_j = float(v_all[j_global])
                    if v_j <= 0.0:
                        continue
                    dln = 0.0
                    for i_idx in edge_rxn_reactant_idx[k]:
                        L_i = float(cons_all[i_idx])
                        dln += math.log1p(v_j / max(L_i, _RATE_FLOOR))
                    for i_idx in edge_rxn_product_idx[k]:
                        P_i = float(prod_all[i_idx])
                        dln += math.log1p(v_j / max(P_i, _RATE_FLOOR))
                    if dln > max_edge_rxn_dlnaccum[sig]:
                        max_edge_rxn_dlnaccum[sig] = dln
                    if dln > max_rxn_dlnaccum_step:
                        max_rxn_dlnaccum_step = dln
                        max_rxn_dlnaccum_sig = sig

            now_wall = time.monotonic()
            if heartbeat_interval > 0 and self.logger and (
                now_wall - last_heartbeat >= heartbeat_interval
            ):
                elapsed = now_wall - wall_start
                self.logger.info(
                    "ODE stepwise progress: "
                    f"sim_t={current_t:.6e} s, wall_elapsed={elapsed:.1f} s, "
                    f"dt={dt:.4e} s, "
                    f"max_rr={max_rr:.4e}, R_char={char_rate:.4e} mM/s"
                )
                last_heartbeat = now_wall

            if max_wall_time_s is not None and (now_wall - wall_start) > max_wall_time_s:
                wall_time_exceeded = True
                if self.logger:
                    self.logger.warning(
                        "ODE stepwise pass stopped: exceeded "
                        f"max_wall_time_per_iteration ({max_wall_time_s} s wall-clock) "
                        f"after completing outer step at simulation t={current_t:.6e} s "
                                            )
                break

            if not first_step and max_rr > interrupt_simulation_tol:
                interrupted = True
                interrupt_char = char_rate
                interrupt_edge = edge_rates
                if self.logger:
                    self.logger.info(
                        f"ODE interrupted at t={current_t:.6e} s "
                        f"(max_rr={max_rr:.4e}, "
                        f"tol={interrupt_simulation_tol}, "
                        f"{len(t_history)} time points)"
                    )
                break

            # Reaction-level interrupt: if any edge reaction's dlnaccum exceeds
            # the threshold, halt and record candidates for promotion.
            if (
                not first_step
                and tol_move_edge_reaction_to_core is not None
                and tol_move_edge_reaction_to_core > 0
                and max_rxn_dlnaccum_step > tol_move_edge_reaction_to_core
            ):
                interrupted = True
                interrupt_char = char_rate
                interrupt_edge = edge_rates
                if self.logger:
                    self.logger.info(
                        f"ODE interrupted (reaction dlnaccum) at t={current_t:.6e} s "
                        f"(max_dlnaccum={max_rxn_dlnaccum_step:.4e}, "
                        f"tol={tol_move_edge_reaction_to_core}, "
                        f"sig={max_rxn_dlnaccum_sig})"
                    )
                break

            first_step = False

            # Outer-step adaptation (no shrink while max_rr <= tol):
            # Interrupt fires only when max_rr > interrupt_simulation_tol. If
            # max_rr is below tol, shrinking dt cannot create an interrupt; it
            # only traps the integrator at dt_min. 
            dt = max(dt, dt_floor_outer)
            if max_rr >= near_frac * interrupt_simulation_tol:
                pass
            else:
                dt = min(dt * dt_growth, dt_max)

        t_arr = np.array(t_history)
        y_arr = np.column_stack(y_history)

        fail_msg = ""
        if wall_time_exceeded:
            fail_msg = (
                f"Exceeded max_wall_time_per_iteration ({max_wall_time_s} s wall-clock)."
            )
        if state_invalid:
            fail_msg = (
                f"{fail_msg + ' ' if fail_msg else ''}"
                f"Invalid concentration state: {invalid_message}"
            ).strip()

        result = SimulationResult(
            t=t_arr,
            y=y_arr,
            species_labels=species_labels,
            success=(not wall_time_exceeded and not state_invalid),
            message=fail_msg,
            simulation_interrupted=interrupted,
            max_char_rate=max_char_rate,
        )

        if result.success and y_arr.shape[1] > 0:
            self.model.set_all_concentrations(
                conservator.clip(current_y).tolist()
            )
        # Trajectory-based max/final R_char and max_edge_rate_ratio; interrupt_* stay unclipped.
        self._attach_flux_metrics(result, rhs=rhs)

        if interrupted:
            result.interrupt_char_rate = interrupt_char
            result.interrupt_edge_rates = interrupt_edge
        else:
            if self.logger and not wall_time_exceeded and not state_invalid:
                self.logger.debug(
                    f"ODE simulation completed (stepwise) "
                    f"({len(t_history)} time points)"
                )

        # Always expose the lifetime-max dlnaccum so the enlarger can promote
        # reactions discovered above tolerance even on a non-interrupted pass.
        result.max_edge_reaction_dlnaccum = max_edge_rxn_dlnaccum

        return result

    # ------------------------------------------------------------------
    # ODE export
    # ------------------------------------------------------------------

    def export_ode_equations(
        self,
        output_path: str,
        iteration: int,
        end_time: float,
    ) -> str:
        """Export ODE equations solved at this iteration to a text file."""

        species_labels = self.model.get_all_species_labels()
        reactions = self.model.core_reactions + self.model.edge_reactions
        label_to_idx = {lab.lower().strip(): i for i, lab in enumerate(species_labels)}
        edge_labels_lc = {sp.label.lower().strip() for sp in self.model.edge_species}
        enzyme_map = self._build_enzyme_concentration_map()

        species_terms: Dict[str, List[Tuple[int, int]]] = {
            lab.lower().strip(): [] for lab in species_labels
        }
        constant_mask = {}
        for i, sp in enumerate(self.model.core_species):
            constant_mask[sp.label.lower().strip()] = sp.constant
        for sp in self.model.edge_species:
            constant_mask[sp.label.lower().strip()] = False

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"ODE EQUATIONS - Enlargement Iteration {iteration}")
        lines.append("=" * 80)
        lines.append(
            f"Species: {len(species_labels)}  |  Reactions: {len(reactions)}  |  t in [0, {end_time}] s"
        )
        lines.append("")

        # --- Reaction rate laws ---
        lines.append("-" * 80)
        lines.append("Reactions (rate laws with parameters)")
        lines.append("-" * 80)

        for r_idx, rxn in enumerate(reactions, 1):
            reactants = " + ".join(
                s for s, c in rxn.stoichiometry.items() if c < 0
            )
            products = " + ".join(
                s for s, c in rxn.stoichiometry.items() if c > 0
            )
            td = getattr(rxn, "thermo", None)
            _is_rev = td is not None and not td.irreversible
            arrow = "<=>" if _is_rev else "->"
            lines.append(f"R{r_idx}: {reactants} {arrow} {products}")
            lines.append(f"    Enzyme: {rxn.enzyme_label}  ({rxn.ec_number or 'N/A'})")

            kin = rxn.kinetics
            if kin is None or rxn.rate_law is None:
                lines.append("    v = (no kinetics - rate = 0)")
            else:
                e_conc = enzyme_map.get(rxn.enzyme_label.lower().strip(), 0.0)
                kcat = kin.kcat
                vmax = kin.vmax
                km_per = getattr(kin, "km_per_substrate", None) or {}
                km_single = kin.km

                # Buffered species (H+, H2O, enzymes) are omitted: activities baked into K'eq; keeps equations consistent with SBML.
                def _is_buffered(lab: str) -> bool:
                    return constant_mask.get(lab.lower().strip(), False)

                if _is_rev:
                    # Reversible MM (Liebermeister & Klipp 2006), kinetic form:
                    # v = (kcat_fwd * [E] * prod([S]/Km_s) - kcat_rev * [E] * prod([P]/Km_p))
                    #     / (prod(1+[S]/Km_s) + prod(1+[P]/Km_p) - 1)
                    sub_terms = []
                    for r in rxn.reactant_labels:
                        if _is_buffered(r):
                            continue
                        km_val = km_per.get(r) if km_per else km_single
                        if km_val and km_val > 0:
                            sub_terms.append(f"[{r}]/Km_{r}")
                        else:
                            sub_terms.append(f"[{r}]")
                    prod_num_terms = []
                    prod_terms = []
                    for p in rxn.product_labels:
                        if _is_buffered(p):
                            continue
                        km_val = km_per.get(p) if km_per else None
                        if km_val and km_val > 0:
                            prod_num_terms.append(f"[{p}]/Km_{p}")
                            prod_terms.append(f"(1+[{p}]/Km_{p})")
                        else:
                            prod_num_terms.append(f"[{p}]")
                            prod_terms.append(f"(1+[{p}])")
                    sub_num = " * ".join(sub_terms) if sub_terms else "1"
                    prod_num = " * ".join(prod_num_terms) if prod_num_terms else "1"
                    sub_den = " * ".join(
                        f"(1+[{r}]/Km_{r})" if (km_per.get(r) or km_single) else f"(1+[{r}])"
                        for r in rxn.reactant_labels
                        if not _is_buffered(r)
                    )
                    prod_den = " * ".join(prod_terms) if prod_terms else "1"
                    rate_str = (
                        f"v{r_idx} = (kcat_fwd * [{rxn.enzyme_label}] * ({sub_num})"
                        f" - kcat_rev * [{rxn.enzyme_label}] * ({prod_num}))"
                        f" / ({sub_den} + {prod_den} - 1)"
                    )
                else:
                    if kcat is not None and e_conc > 0:
                        rate_pre = f"v{r_idx} = kcat * [{rxn.enzyme_label}] * "
                    elif vmax is not None:
                        rate_pre = f"v{r_idx} = Vmax * "
                    else:
                        rate_pre = f"v{r_idx} = (missing kcat/Vmax)"
                    sat_parts = []
                    for r in rxn.reactant_labels:
                        if _is_buffered(r):
                            continue
                        km_val = km_per.get(r) if km_per else km_single
                        if km_val is not None and km_val > 0:
                            sat_parts.append(f"[{r}]/(Km_{r}+[{r}])")
                        else:
                            sat_parts.append(f"[{r}]")
                    rate_str = rate_pre + " * ".join(sat_parts) if sat_parts else rate_pre.rstrip(" * ")

                lines.append(f"    {rate_str}")

                params = []
                if kcat is not None:
                    if _is_rev:
                        params.append(f"kcat_fwd={kcat:.6g} 1/s")
                    else:
                        params.append(f"kcat={kcat:.6g} 1/s")
                    if getattr(kin, "kcat_sd", None) is not None:
                        params.append(f"kcat_sd={kin.kcat_sd:.6g} 1/s")
                if _is_rev and td is not None and td.kcat_rev is not None:
                    params.append(f"kcat_rev={td.kcat_rev:.6g} 1/s")
                if vmax is not None:
                    params.append(f"Vmax={vmax:.6g} mM/s")
                if km_per:
                    km_sd_per = getattr(kin, "km_sd_per_substrate", None) or {}
                    for r, k in km_per.items():
                        if k is not None:
                            params.append(f"Km({r})={k:.6g} mM")
                            if km_sd_per and r in km_sd_per:
                                params.append(f"Km_sd({r})={km_sd_per[r]:.6g} mM")
                elif km_single is not None:
                    params.append(f"Km={km_single:.6g} mM")
                    if getattr(kin, "km_sd", None) is not None:
                        params.append(f"Km_sd={kin.km_sd:.6g} mM")
                if params:
                    lines.append(f"    Parameters: {', '.join(params)}")
            if td is not None:
                thermo_parts = []
                if td.dgr_prime_kJmol is not None:
                    thermo_parts.append(f"ΔG°'={td.dgr_prime_kJmol:.2f} kJ/mol")
                    if td.sigma_kJmol is not None and math.isfinite(td.sigma_kJmol):
                        thermo_parts.append(f"σ={td.sigma_kJmol:.2f} kJ/mol")
                thermo_parts.append(f"source={td.source}")
                lines.append(f"    Thermodynamics: {', '.join(thermo_parts)}")
            lines.append("")

            for species_label, coeff in rxn.stoichiometry.items():
                lc = species_label.lower().strip()
                if lc in label_to_idx and lc in species_terms:
                    species_terms[lc].append((coeff, r_idx))

        # --- ODEs for each species ---
        lines.append("-" * 80)
        lines.append("ODEs (dC/dt for each species)")
        lines.append("-" * 80)

        for lab in species_labels:
            lc = lab.lower().strip()
            terms = species_terms.get(lc, [])
            is_edge = lc in edge_labels_lc
            is_const = constant_mask.get(lc, False)

            if is_const:
                lines.append(f"d[{lab}]/dt = 0  (constant)")
            elif not terms:
                lines.append(f"d[{lab}]/dt = 0")
            else:
                term_strs = []
                for coeff, r_idx in terms:
                    if coeff > 0:
                        term_strs.append(f"+{coeff}*v{r_idx}")
                    else:
                        term_strs.append(f"{coeff}*v{r_idx}")
                rhs = " ".join(term_strs).lstrip("+") or "0"
                suffix = "  (edge)" if is_edge else ""
                lines.append(f"d[{lab}]/dt = {rhs}{suffix}")
        lines.append("")
        lines.append("=" * 80)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        if self.logger:
            self.logger.info(f"Exported ODE equations to {output_path}")

        return output_path

    # ------------------------------------------------------------------
    #  Flux trajectory metrics
    # ------------------------------------------------------------------

    def _attach_flux_metrics(self, sim_result: SimulationResult, rhs: _VectorizedRHS) -> None:
        """
        Populate max_char_rate, final_char_rate, and max_edge_rate_ratio by
        scanning the stored trajectory using vectorized RHS.
        """
        if not sim_result.success or sim_result.y.size == 0:
            sim_result.max_char_rate = 0.0
            sim_result.final_char_rate = 0.0
            sim_result.max_edge_rate_ratio = {}
            return

        n_core = len(self.model.core_species)
        n_edge = len(self.model.edge_species)
        edge_labels = [sp.label.lower().strip() for sp in self.model.edge_species]

        # Promotion scoring uses the UNMASKED residual end-to-end. Reason:
        # the enlarger bootstraps from zero core reactions — all reactions
        # start as edges and get promoted by flux. If we measured R_char from
        # the isolated residual, it would be identically zero while the core
        # is empty, the rate ratio `edge_rate / R_char` would never fire, and
        # the enlarger would converge trivially on iteration 1. The unmasked
        # residual answers the right question for promotion: "what WOULD the
        # dynamics be if these edges were core?"
        dydt_unmasked = rhs.compute_dydt_unmasked(sim_result.y)
        dydt_core_mat = dydt_unmasked[:n_core, :]
        char_rates = np.linalg.norm(dydt_core_mat, axis=0)

        sim_result.max_char_rate = float(np.max(char_rates))
        sim_result.final_char_rate = float(char_rates[-1])

        if n_edge > 0:
            dydt_edge_mat = dydt_unmasked[n_core:, :]
            abs_edge_mat = np.abs(dydt_edge_mat)
            safe_char = char_rates.copy()
            safe_char[safe_char == 0] = 1e-100
            rr_mat = abs_edge_mat / safe_char
            max_ratios = np.max(rr_mat, axis=1)
            sim_result.max_edge_rate_ratio = {
                edge_labels[i]: float(max_ratios[i]) for i in range(n_edge)
            }
        else:
            sim_result.max_edge_rate_ratio = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_alias_to_model_label(
        self,
        species_labels: List[str],
    ) -> Dict[str, str]:
        """Map ontology aliases to model species labels so stoichiometry lookups resolve correctly."""
        alias_to_model: Dict[str, str] = {}
        for lab in species_labels:
            model_lc = lab.lower().strip()
            alias_to_model[model_lc] = model_lc
            for equiv in get_ontology_equivalents(lab):
                alias_to_model[equiv.lower().strip()] = model_lc
        return alias_to_model

    def _build_enzyme_concentration_map(self) -> Dict[str, float]:
        """Return enzyme label (lc) -> concentration (mM) for all enzyme core species."""
        enzyme_map: Dict[str, float] = {}
        for sp in self.model.core_species:
            if sp.is_enzyme:
                enzyme_map[sp.label.lower().strip()] = sp.concentration
        return enzyme_map
