#!/usr/bin/env python3
r"""
local (one-factor-at-a-time) sensitivity analysis for the E. coli
FAS-II model.

Provenance: knowledge/summaries/scripts/sensitivity_analysis.md

Run (repo root, bees_env):
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --tornado-only --bare --jobs 16
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --feedback-only   # skip tornado
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --bare           # no titles (Ki legends stay)
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --end-time 3000 # 50 min horizon
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --jobs 16       # parallel tornado

Feedback Ki panels are per-enzyme: FabH and FabI are swept separately; the
other enzyme's Ki stays at the production lock. Hill-n panel is retired.
"""

import argparse
import os
import re
import sys
import math
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Load .env.bees BEFORE importing any bees.* module (class-level env vars
#     such as CATPRED_PYTHON are resolved at class-definition time). Mirrors
#     BEES.py exactly. ---------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ENV_BEES = os.path.join(_REPO_ROOT, ".env.bees")
if os.path.isfile(_ENV_BEES):
    with open(_ENV_BEES) as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            _m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", _line)
            if _m:
                _k, _v = _m.group(1), _m.group(2).strip()
                if (_v.startswith('"') and _v.endswith('"')) or (
                    _v.startswith("'") and _v.endswith("'")
                ):
                    _v = _v[1:-1]
                os.environ.setdefault(_k, _v)

sys.path.insert(0, _REPO_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bees.common as common
from bees.main import BEES
from bees.reaction_generator import ReactionGenerator
from bees.enlarger import IterativeEnlarger
from bees.simulator import ODESimulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_YML = os.path.join(_PROJECT_DIR, "input.yml")
_OUT_DIR = os.path.join(_PROJECT_DIR, "output", "sensitivity")
os.makedirs(_OUT_DIR, exist_ok=True)
OUT_STEM = os.path.join(_OUT_DIR, "sensitivity_analysis")
OUT_KI_FABH_STEM = os.path.join(_OUT_DIR, "sensitivity_ki_fabh_timecourse")
OUT_KI_FABI_STEM = os.path.join(_OUT_DIR, "sensitivity_ki_fabi_timecourse")
# Legacy aliases (FabH panel also written here for older deliverable paths).
OUT_KI_STEM = OUT_KI_FABH_STEM
OUT_HILL_STEM = os.path.join(_OUT_DIR, "sensitivity_hill_timecourse")

KCAT_LN_STEP = 0.05          # h: perturb kcat/Km/Ki by exp(+/- h) in ln-space
KM_LN_STEP = KCAT_LN_STEP
KI_LN_STEP = KCAT_LN_STEP
DG_STEP_KCAL = 0.5           # ΔG°′ perturbation, kcal/mol
KCAL_PER_KJ = 1.0 / 4.184
R_KJ = 8.314462618e-3        # gas constant, kJ / (mol K)
DYNAMIC_FRAC = 0.5           # t* = time baseline PE reaches this fraction of final
TOP_N = 10                   # bars to show per tornado panel

# Publication mode: drop figure/panel titles (caption carries them). Ki
# legends stay on (2-column boxed style). Set by --bare; numbers still go to stdout.
BARE = False

# Horizon override (--end-time). Applied to settings BEFORE the enlarger runs,
# so it changes the network that gets built, not just the plotted window.
END_TIME_OVERRIDE = None

# Tornado OFAT parallelism (--jobs). Ki panels stay serial. Default: all but one core.
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# Fork-worker globals (set in parent before ProcessPoolExecutor; children inherit).
_W_MODEL = None
_W_SIM_KW = None
_W_TEMPERATURE = None

# Corrected Yu 2011 Fig S2A digitization (t=9/t=12 plateau) — same points as
# plot_vs_yu_s2a_ruppe_s2b.py. Used for Ki re-fit RMSE at fixed hill=2.
S2A_EXP_T_MIN = np.array([1.5, 3.0, 5.0, 7.0, 9.0, 12.0])
S2A_EXP_UM = np.array([6.5, 17.0, 28.5, 30.0, 35.5, 35.5])
# Ki grid at hill=2 (µM): coarse then fine around the coarse minimum.
KI_REFIT_COARSE_UM = list(range(2, 17))          # 2, 3, ..., 16
KI_REFIT_FINE_HALFWIDTH_UM = 2.0
KI_REFIT_FINE_STEP_UM = 0.25

# Free-fatty-acid recognition (copied from exporter.py:_fa_carbons) ----------
_FA_STEMS = [
    ("icosen", 20), ("icosan", 20), ("octadecen", 18), ("octadecan", 18),
    ("hexadecen", 16), ("hexadecan", 16), ("tetradecen", 14), ("tetradecan", 14),
    ("dodecen", 12), ("dodecan", 12), ("decen", 10), ("decan", 10),
    ("octen", 8), ("octan", 8), ("hexen", 6), ("hexan", 6),
    ("penten", 5), ("pentan", 5), ("buten", 4), ("butan", 4),
]


def _fa_carbons(label):
    n = label.lower()
    if "[acp]" in n or not n.endswith("oate"):
        return None
    for stem, c in _FA_STEMS:
        if stem in n:
            return c
    return None


# ---------------------------------------------------------------------------
# Build the final network once
# ---------------------------------------------------------------------------
def build_model():
    """Run the BEES pipeline up to the final enlarged model; return (enlarger,
    model, temperature_K). CatPred results come from the shared on-disk cache."""
    input_data = common.read_yaml_file(INPUT_YML)
    input_data.setdefault("settings", {})
    if END_TIME_OVERRIDE is not None:
        print(
            f"end_time override: {input_data['settings'].get('end_time')} "
            f"-> {END_TIME_OVERRIDE:g} s",
            flush=True,
        )
        input_data["settings"]["end_time"] = float(END_TIME_OVERRIDE)

    bees = BEES(input_data=input_data)
    obj = bees.bees_object

    db_name = obj.database.name.strip()
    db_path = os.path.join(common.BEES_PATH, "db", f"{db_name}.csv")
    if not os.path.exists(db_path):
        db_path = os.path.join(common.BEES_PATH, "db", "db.csv")

    rgen = ReactionGenerator(
        bees_object=obj, logger=bees.logger, output_directory=bees.output_directory
    )
    rgen.load_kinetic_database(db_path, ontology=bees.ontology)

    enlarger = IterativeEnlarger(
        bees_object=obj,
        reaction_generator=rgen,
        logger=bees.logger,
        output_directory=bees.output_directory,
    )
    result = enlarger.run()
    temperature = float(getattr(obj.environment, "temperature", 298.0))
    return enlarger, result.model, temperature


# ---------------------------------------------------------------------------
# Simulation + target extraction
# ---------------------------------------------------------------------------
def _simulate(enlarger, model, end_time=None):
    """Fresh ODESimulator each call so baked-in RHS (incl. feedback Ki) is rebuilt.

    Tornado sweeps only need PE at fixed t*, so pass end_time slightly past t*
    to avoid integrating the full 3000 s horizon ~200–300 times.
    """
    model.reset_concentrations_to_initial()
    sim = ODESimulator(model, logger=None)
    t_end = float(enlarger.end_time if end_time is None else end_time)
    return sim.simulate(
        end_time=t_end,
        time_step=enlarger.time_step,
        method=enlarger.ode_method,
        rtol=enlarger.ode_rtol,
        atol=enlarger.ode_atol,
    )


def _try_simulate(enlarger, model, end_time=None):
    """As _simulate, but returns None when the stiff solver diverges.

    A perturbed parameter can push the BDF Jacobian to inf/NaN, which raises out
    of solve_ivp. That is a property of the perturbed point, not a script bug, so
    the sweep records the parameter as unscored instead of aborting the whole run.
    """
    try:
        return _simulate(enlarger, model, end_time=end_time)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        print(f"    ! solver diverged ({type(exc).__name__}); parameter unscored", flush=True)
        return None


def _sweep_end_time(enlarger, t_star):
    """Integrate just past fixed t* (with a floor) for OFAT tornado sweeps."""
    pad = max(60.0, 0.25 * float(t_star))
    return min(float(enlarger.end_time), float(t_star) + pad)


def _sim_kwargs(enlarger):
    """Scalar ODE settings needed by workers (no enlarger object in the pool)."""
    return {
        "time_step": enlarger.time_step,
        "method": enlarger.ode_method,
        "rtol": enlarger.ode_rtol,
        "atol": enlarger.ode_atol,
    }


def _pe_at_model(model, t_end, t_star, sim_kw):
    """PE(t*) for a (possibly perturbed) model copy; None if solver diverges."""
    try:
        model.reset_concentrations_to_initial()
        sim = ODESimulator(model, logger=None)
        result = sim.simulate(end_time=float(t_end), **sim_kw)
        return _pe_at(result, t_star)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def _fd_pe(pe_p, pe_m, denom):
    if pe_p is None or pe_m is None or pe_p <= 0 or pe_m <= 0:
        return 0.0
    return (math.log(pe_p) - math.log(pe_m)) / denom


def _pe_series(sim):
    """Palmitate-equivalent free-FA titer (uM) vs time."""
    weights = {}
    for i, lab in enumerate(sim.species_labels):
        c = _fa_carbons(lab)
        if c is not None:
            weights[i] = c / 16.0
    if not weights:
        return np.zeros_like(sim.t)
    idx = list(weights.keys())
    w = np.array([weights[i] for i in idx])
    return 1000.0 * (sim.y[idx, :].T * w).sum(axis=1)


def _pe_at(sim, t_eval):
    return float(np.interp(t_eval, sim.t, _pe_series(sim)))


def _finite_diff(sim_p, sim_m, t_eval, denom):
    """Central FD of ln(PE) at fixed clock time t_eval (never recompute t*)."""
    if sim_p is None or sim_m is None:
        return 0.0
    cp = _pe_at(sim_p, t_eval)
    cm = _pe_at(sim_m, t_eval)
    return _fd_pe(cp, cm, denom)


# ---------------------------------------------------------------------------
# Parallel tornado workers (fork: inherit _W_* from parent)
# ---------------------------------------------------------------------------
def _worker_kcat(job):
    """job = (rxn_idx, label, h, t_star, t_end) -> (rxn_idx, label, S)."""
    rxn_idx, label, h, t_star, t_end = job
    model = copy.deepcopy(_W_MODEL)
    kin = model.core_reactions[rxn_idx].kinetics
    k0 = float(kin.kcat)
    kin.kcat = k0 * math.exp(h)
    pe_p = _pe_at_model(model, t_end, t_star, _W_SIM_KW)
    kin.kcat = k0 * math.exp(-h)
    pe_m = _pe_at_model(model, t_end, t_star, _W_SIM_KW)
    return rxn_idx, label, _fd_pe(pe_p, pe_m, 2.0 * h)


def _worker_km(job):
    """job = (rxn_idx, label, h, t_star, t_end) -> (rxn_idx, label, S)."""
    rxn_idx, label, h, t_star, t_end = job
    model = copy.deepcopy(_W_MODEL)
    kin = model.core_reactions[rxn_idx].kinetics
    km0 = getattr(kin, "km", None)
    per0_raw = getattr(kin, "km_per_substrate", None) or {}
    per0 = dict(per0_raw)
    has_scalar = km0 is not None and km0 > 0
    has_per = any(v is not None and v > 0 for v in per0.values())

    def _apply(factor):
        if has_scalar:
            kin.km = float(km0) * factor
        if has_per:
            kin.km_per_substrate = {
                k: (float(v) * factor if v is not None and v > 0 else v)
                for k, v in per0.items()
            }

    _apply(math.exp(h))
    pe_p = _pe_at_model(model, t_end, t_star, _W_SIM_KW)
    _apply(math.exp(-h))
    pe_m = _pe_at_model(model, t_end, t_star, _W_SIM_KW)
    return rxn_idx, label, _fd_pe(pe_p, pe_m, 2.0 * h)


def _worker_dg(job):
    """job = (rxn_idx, label, dg_step_kj, t_star, t_end) -> (rxn_idx, label, S)."""
    rxn_idx, label, dg_step_kj, t_star, t_end = job
    model = copy.deepcopy(_W_MODEL)
    thermo = model.core_reactions[rxn_idx].thermo
    dg0 = float(thermo.dgr_prime_kJmol)
    T = float(_W_TEMPERATURE)

    thermo.dgr_prime_kJmol = dg0 + dg_step_kj
    thermo.keq = math.exp(-(dg0 + dg_step_kj) / (R_KJ * T))
    pe_p = _pe_at_model(model, t_end, t_star, _W_SIM_KW)

    thermo.dgr_prime_kJmol = dg0 - dg_step_kj
    thermo.keq = math.exp(-(dg0 - dg_step_kj) / (R_KJ * T))
    pe_m = _pe_at_model(model, t_end, t_star, _W_SIM_KW)

    return rxn_idx, label, _fd_pe(pe_p, pe_m, 2.0 * DG_STEP_KCAL)


def _run_tornado_jobs(tag, jobs, worker, model, sim_kw, temperature=None, n_jobs=None):
    """Run OFAT jobs; parallel via fork when n_jobs > 1.

    Returns list of (label, S) in job submission order.
    """
    n_jobs = N_JOBS if n_jobs is None else int(n_jobs)
    n = len(jobs)
    if n == 0:
        return []

    global _W_MODEL, _W_SIM_KW, _W_TEMPERATURE
    _W_MODEL = model
    _W_SIM_KW = sim_kw
    _W_TEMPERATURE = temperature

    results = [None] * n
    if n_jobs <= 1:
        for i, job in enumerate(jobs):
            print(f"  {tag} [{i + 1}/{n}] {job[1]}", flush=True)
            _, label, s = worker(job)
            results[i] = (label, s)
        return results

    # fork: children inherit _W_* without pickling the (large) model through the pipe.
    ctx = mp.get_context("fork")
    print(f"  {tag}: {n} parameters x 2 sims, {n_jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        fut_to_i = {pool.submit(worker, job): i for i, job in enumerate(jobs)}
        done = 0
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            _, label, s = fut.result()
            results[i] = (label, s)
            done += 1
            print(f"  {tag} [{done}/{n}] {label}  S={s:+.4f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Sensitivity loops (score at fixed baseline t*)
# ---------------------------------------------------------------------------
def _reaction_equation(rxn):
    """Stoichiometric equation: reactants -> products."""
    left = " + ".join(rxn.reactant_labels) if rxn.reactant_labels else "?"
    right = " + ".join(rxn.product_labels) if rxn.product_labels else "?"
    return f"{left} -> {right}"


def _reaction_label(rxn, seen):
    """Unique bar label: enzyme + full reaction equation."""
    base = f"{rxn.enzyme_label}: {_reaction_equation(rxn)}"
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else f"{base} ({n + 1})"


def kcat_sensitivity(enlarger, model, t_star, n_jobs=None):
    seen = {}
    t_end = _sweep_end_time(enlarger, t_star)
    sim_kw = _sim_kwargs(enlarger)
    jobs = []
    for idx, rxn in enumerate(model.core_reactions):
        kin = getattr(rxn, "kinetics", None)
        if getattr(kin, "kcat", None) is None or kin.kcat <= 0:
            continue
        jobs.append((idx, _reaction_label(rxn, seen), KCAT_LN_STEP, t_star, t_end))
    return _run_tornado_jobs("kcat", jobs, _worker_kcat, model, sim_kw, n_jobs=n_jobs)


def km_sensitivity(enlarger, model, t_star, n_jobs=None):
    """Scale kin.km and kin.km_per_substrate together (reaction-level affinity)."""
    seen = {}
    t_end = _sweep_end_time(enlarger, t_star)
    sim_kw = _sim_kwargs(enlarger)
    jobs = []
    for idx, rxn in enumerate(model.core_reactions):
        kin = getattr(rxn, "kinetics", None)
        if kin is None:
            continue
        km0 = getattr(kin, "km", None)
        per0 = getattr(kin, "km_per_substrate", None) or {}
        has_scalar = km0 is not None and km0 > 0
        has_per = any(v is not None and v > 0 for v in per0.values())
        if has_scalar or has_per:
            jobs.append((idx, _reaction_label(rxn, seen), KM_LN_STEP, t_star, t_end))
    return _run_tornado_jobs("Km", jobs, _worker_km, model, sim_kw, n_jobs=n_jobs)


def dg_sensitivity(enlarger, model, temperature, t_star, n_jobs=None):
    seen = {}
    t_end = _sweep_end_time(enlarger, t_star)
    sim_kw = _sim_kwargs(enlarger)
    dg_step_kj = DG_STEP_KCAL / KCAL_PER_KJ
    jobs = []
    for idx, rxn in enumerate(model.core_reactions):
        thermo = getattr(rxn, "thermo", None)
        if (
            thermo is None
            or getattr(thermo, "irreversible", True)
            or not math.isfinite(getattr(thermo, "keq", float("nan")))
            or thermo.keq <= 0
        ):
            continue
        jobs.append((idx, _reaction_label(rxn, seen), dg_step_kj, t_star, t_end))
    return _run_tornado_jobs(
        "ΔG°′", jobs, _worker_dg, model, sim_kw, temperature=temperature, n_jobs=n_jobs
    )


# ---------------------------------------------------------------------------
# Ki time-course figure
# ---------------------------------------------------------------------------
def _feedback_reactions(model, enzyme=None):
    """Core reactions with feedback_inhibitors. Optionally filter by enzyme label."""
    key = None if enzyme is None else str(enzyme).lower().strip()
    out = []
    for rxn in model.core_reactions:
        fb = getattr(rxn, "feedback_inhibitors", None)
        if not isinstance(fb, dict) or not fb:
            continue
        if key is not None:
            enz = str(getattr(rxn, "enzyme_label", "")).lower().strip()
            if enz != key:
                continue
        out.append(rxn)
    return out


def _snapshot_feedback(rxns):
    return {id(rxn): copy.deepcopy(rxn.feedback_inhibitors) for rxn in rxns}


def _restore_feedback(rxns, snap):
    for rxn in rxns:
        rxn.feedback_inhibitors = copy.deepcopy(snap[id(rxn)])


def _scale_feedback_ki(rxns, factor):
    """Scale every (Ki, hill) tuple's Ki; hill unchanged. Dict shape: {inh: (Ki, hill)}."""
    for rxn in rxns:
        fb = rxn.feedback_inhibitors
        if not isinstance(fb, dict):
            continue
        rxn.feedback_inhibitors = {
            lab: (float(ki) * factor, float(hill)) for lab, (ki, hill) in fb.items()
        }


def _baseline_ki_uM(rxns):
    """Report the (single) baseline Ki in uM for legend labeling."""
    for rxn in rxns:
        fb = rxn.feedback_inhibitors
        if isinstance(fb, dict) and fb:
            ki_mM = float(next(iter(fb.values()))[0])
            return ki_mM * 1000.0
    return float("nan")


def _set_feedback_ki_hill(rxns, ki_mM=None, hill=None):
    """Set absolute Ki (mM) and/or hill on every feedback entry; leave others unchanged."""
    for rxn in rxns:
        fb = rxn.feedback_inhibitors
        if not isinstance(fb, dict):
            continue
        rxn.feedback_inhibitors = {
            lab: (
                float(ki_mM) if ki_mM is not None else float(ki),
                float(hill) if hill is not None else float(h),
            )
            for lab, (ki, h) in fb.items()
        }


def _s2a_rmse(sim):
    """RMSE of PE(t) vs corrected Yu S2A 6-point curve (µM palmitate-eq)."""
    pe = _pe_series(sim)
    t_min = sim.t / 60.0
    pred = np.interp(S2A_EXP_T_MIN, t_min, pe)
    return float(np.sqrt(np.mean((pred - S2A_EXP_UM) ** 2)))


def _refit_ki_at_hill(enlarger, model, fb_rxns, snap, hill=2.0):
    """Coarse-then-fine Ki grid at fixed hill vs Yu S2A. Returns (Ki*_uM, RMSE*, table)."""
    print(f"\n=== Ki re-fit at hill={hill:g} vs Yu S2A ===", flush=True)
    results = []  # (ki_uM, rmse)

    def _eval(ki_uM):
        _restore_feedback(fb_rxns, snap)
        _set_feedback_ki_hill(fb_rxns, ki_mM=ki_uM / 1000.0, hill=hill)
        sim = _simulate(enlarger, model)
        rmse = _s2a_rmse(sim)
        results.append((ki_uM, rmse))
        print(f"  Ki={ki_uM:6.2f} µM  RMSE={rmse:6.3f} µM", flush=True)
        return rmse

    print("  coarse grid...", flush=True)
    coarse_results = []
    for ki_uM in KI_REFIT_COARSE_UM:
        rmse = _eval(float(ki_uM))
        coarse_results.append((float(ki_uM), rmse))
    ki_c, rmse_c = min(coarse_results, key=lambda kr: kr[1])
    print(
        f"  coarse minimum: Ki={ki_c:.2f} µM (RMSE={rmse_c:.3f})",
        flush=True,
    )

    print("  fine grid...", flush=True)
    lo = max(0.25, ki_c - KI_REFIT_FINE_HALFWIDTH_UM)
    hi = ki_c + KI_REFIT_FINE_HALFWIDTH_UM
    fine_vals = np.arange(lo, hi + 0.5 * KI_REFIT_FINE_STEP_UM, KI_REFIT_FINE_STEP_UM)
    seen = {round(k, 4) for k, _ in results}
    for ki_uM in fine_vals:
        if round(float(ki_uM), 4) in seen:
            continue
        _eval(float(ki_uM))

    ki_star, rmse_star = min(results, key=lambda kr: kr[1])
    print(
        f"  winner: Ki*={ki_star:.2f} µM  RMSE={rmse_star:.3f} µM (hill={hill:g})",
        flush=True,
    )
    _restore_feedback(fb_rxns, snap)
    return ki_star, rmse_star, results


def _style_legend(ax, fontsize=9):
    """Two-column boxed legend (paper style: frame, ncol=2, no fancybox)."""
    ax.legend(
        loc="best",
        ncol=2,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        handlelength=2.0,
        columnspacing=1.2,
        borderpad=0.5,
    )


def _panel_letter(fig, letter):
    """Bold panel tag at true figure top-left."""
    fig.text(
        0.01,
        0.99,
        f"({letter})",
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def ki_timecourse(enlarger, model, t_star, enzyme="FabH", out_stem=None, panel_letter="A"):
    """PE(t) for baseline / +/- ln-step Ki / enzyme-feedback-off; print S_Ki at t*.

    Only reactions for ``enzyme`` are scaled or cleared; the other enzyme's Ki
    stays at the production lock.
    """
    if out_stem is None:
        out_stem = OUT_KI_FABH_STEM if enzyme.lower() == "fabh" else OUT_KI_FABI_STEM
    fb_rxns = _feedback_reactions(model, enzyme=enzyme)
    if not fb_rxns:
        print(
            f"No {enzyme} reactions with feedback_inhibitors; "
            f"skipping {enzyme} Ki time course."
        )
        return None

    snap = _snapshot_feedback(fb_rxns)
    ki_uM = _baseline_ki_uM(fb_rxns)

    # Baseline
    sim_base = _simulate(enlarger, model)
    pe_base = _pe_series(sim_base)

    # Ki * exp(+h) — this enzyme only
    _scale_feedback_ki(fb_rxns, math.exp(KI_LN_STEP))
    sim_p = _simulate(enlarger, model)
    pe_p = _pe_series(sim_p)
    _restore_feedback(fb_rxns, snap)

    # Ki * exp(-h)
    _scale_feedback_ki(fb_rxns, math.exp(-KI_LN_STEP))
    sim_m = _simulate(enlarger, model)
    pe_m = _pe_series(sim_m)
    _restore_feedback(fb_rxns, snap)

    # This enzyme's feedback off (other enzyme unchanged)
    for rxn in fb_rxns:
        rxn.feedback_inhibitors = None
    sim_off = _simulate(enlarger, model)
    pe_off = _pe_series(sim_off)
    _restore_feedback(fb_rxns, snap)

    s_ki = _finite_diff(sim_p, sim_m, t_star, 2.0 * KI_LN_STEP)
    print(
        f"\n=== {enzyme} feedback Ki: S_Ki = dln(PE(t*))/dln(Ki) = {s_ki:.4f} ==="
    )
    print(f"  baseline Ki = {ki_uM:.3f} uM; t* = {t_star:.1f} s (fixed)")
    print(f"  other enzyme Ki held at production lock", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    t_min = sim_base.t / 60.0
    ax.plot(
        t_min, pe_base,
        color="#2c3e50", linewidth=2.0,
        label=f"Ki = {ki_uM:.1f} µM",
    )
    ax.plot(
        sim_p.t / 60.0, pe_p,
        color="#c0392b", linewidth=1.5, linestyle="--",
        label=rf"Ki × $e^{{+{KI_LN_STEP}}}$",
    )
    ax.plot(
        sim_m.t / 60.0, pe_m,
        color="#2980b9", linewidth=1.5, linestyle="--",
        label=rf"Ki × $e^{{-{KI_LN_STEP}}}$",
    )
    ax.plot(
        sim_off.t / 60.0, pe_off,
        color="#7f8c8d", linewidth=1.8, linestyle=":",
        label=f"no {enzyme} feedback",
    )
    ax.scatter(
        S2A_EXP_T_MIN, S2A_EXP_UM,
        color="black", s=28, zorder=5,
        label="Experimental",
    )
    ax.axvline(t_star / 60.0, color="black", linewidth=0.8, linestyle="-.", alpha=0.6)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("PE (µM)")
    if not BARE:
        ax.set_title(
            f"{enzyme} feedback Ki: PE(t) shape/timing\n"
            f"(S_Ki at fixed t*={t_star:.0f}s = {s_ki:.3f}; "
            f"dotted = no {enzyme} feedback, other Ki held)"
        )
    _style_legend(ax, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    _panel_letter(fig, panel_letter)
    for ext in ("png", "pdf"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return s_ki


def hill_timecourse(enlarger, model):
    """DEPRECATED: Hill-n panel retired in favor of per-enzyme FabI Ki SA.

    Kept for optional re-runs; not called from main().
    """
    fb_rxns = _feedback_reactions(model, enzyme="FabH")
    if not fb_rxns:
        print("No reactions with feedback_inhibitors; skipping Hill time course.")
        return None

    snap = _snapshot_feedback(fb_rxns)
    ki0_uM = _baseline_ki_uM(fb_rxns)
    ki0_mM = ki0_uM / 1000.0

    # --- scenarios at held Ki ---
    def _sim_at(hill, ki_mM):
        _restore_feedback(fb_rxns, snap)
        _set_feedback_ki_hill(fb_rxns, ki_mM=ki_mM, hill=hill)
        return _simulate(enlarger, model)

    print("Hill scenarios (Ki held at baseline)...", flush=True)
    sim_n1 = _sim_at(1.0, ki0_mM)
    pe_n1 = _pe_series(sim_n1)
    rmse_n1 = _s2a_rmse(sim_n1)

    sim_n2 = _sim_at(2.0, ki0_mM)
    pe_n2 = _pe_series(sim_n2)
    rmse_n2_held = _s2a_rmse(sim_n2)

    sim_n05 = _sim_at(0.5, ki0_mM)
    pe_n05 = _pe_series(sim_n05)
    rmse_n05 = _s2a_rmse(sim_n05)

    _restore_feedback(fb_rxns, snap)
    for rxn in fb_rxns:
        rxn.feedback_inhibitors = None
    sim_off = _simulate(enlarger, model)
    pe_off = _pe_series(sim_off)
    _restore_feedback(fb_rxns, snap)

    # --- Ki re-fit at n=2 ---
    ki_star, rmse_star, _table = _refit_ki_at_hill(
        enlarger, model, fb_rxns, snap, hill=2.0
    )
    _restore_feedback(fb_rxns, snap)
    _set_feedback_ki_hill(fb_rxns, ki_mM=ki_star / 1000.0, hill=2.0)
    sim_refit = _simulate(enlarger, model)
    pe_refit = _pe_series(sim_refit)
    _restore_feedback(fb_rxns, snap)

    print(
        f"\n=== Hill robustness RMSE vs Yu S2A ===\n"
        f"  n=1, Ki={ki0_uM:.1f} µM (baseline):  RMSE={rmse_n1:.3f} µM\n"
        f"  n=2, Ki={ki0_uM:.1f} µM (held):      RMSE={rmse_n2_held:.3f} µM\n"
        f"  n=0.5, Ki={ki0_uM:.1f} µM (held):    RMSE={rmse_n05:.3f} µM\n"
        f"  n=2, Ki*={ki_star:.2f} µM (re-fit):   RMSE={rmse_star:.3f} µM",
        flush=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        sim_n1.t / 60.0, pe_n1,
        color="#2c3e50", linewidth=2.0,
        label=f"n=1, Ki={ki0_uM:.1f} µM",
    )
    ax.plot(
        sim_n2.t / 60.0, pe_n2,
        color="#c0392b", linewidth=1.5, linestyle="--",
        label=f"n=2, Ki={ki0_uM:.1f} µM",
    )
    ax.plot(
        sim_n05.t / 60.0, pe_n05,
        color="#2980b9", linewidth=1.5, linestyle="--",
        label=f"n=0.5, Ki={ki0_uM:.1f} µM",
    )
    ax.plot(
        sim_refit.t / 60.0, pe_refit,
        color="#27ae60", linewidth=2.0,
        label=f"n=2, Ki={ki_star:.2f} µM",
    )
    ax.plot(
        sim_off.t / 60.0, pe_off,
        color="#7f8c8d", linewidth=1.8, linestyle=":",
        label="no FabH feedback",
    )
    ax.scatter(
        S2A_EXP_T_MIN, S2A_EXP_UM,
        color="black", s=28, zorder=5,
        label="Experimental",
    )
    ax.set_xlabel("time (min)")
    ax.set_ylabel("PE (µM)")
    if not BARE:
        ax.set_title(
            "FabH feedback Hill coefficient: PE(t) cooperativity robustness\n"
            f"RMSE: baseline={rmse_n1:.2f}, n=2 held={rmse_n2_held:.2f}, "
            f"n=2 re-fit={rmse_star:.2f} µM  "
            "(dotted = structural comparator)"
        )
    _style_legend(ax, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    _panel_letter(fig, "B")
    for ext in ("png", "pdf"):
        path = f"{OUT_HILL_STEM}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return {
        "rmse_n1": rmse_n1,
        "rmse_n2_held": rmse_n2_held,
        "rmse_n05": rmse_n05,
        "ki_star": ki_star,
        "rmse_star": rmse_star,
    }


# ---------------------------------------------------------------------------
# Plotting — Figure 1 tornado (one figure per parameter class)
# ---------------------------------------------------------------------------
def _tornado(ax, rows, xlabel, title, panel_letter=None):
    import textwrap

    data = sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:TOP_N]
    data = list(reversed(data))
    labels = [
        "\n".join(textwrap.wrap(r[0], width=70, break_long_words=False))
        for r in data
    ]
    vals = [r[1] for r in data]
    colors = ["#c0392b" if v >= 0 else "#2c7fb8" for v in vals]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", labelsize=11)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    if not BARE:
        ax.set_title(title, fontsize=12)
    return panel_letter


def _save_tornado_csv(stem, rows):
    path = f"{stem}.csv"
    ranked = sorted(rows, key=lambda r: abs(r[1]), reverse=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("rank,label,S\n")
        for i, (lab, s) in enumerate(ranked, 1):
            safe = lab.replace('"', "'")
            fh.write(f'{i},"{safe}",{s:.8g}\n')
    print(f"wrote {path}")


def _load_tornado_csv(stem):
    """Load (label, S) rows from a previously written tornado CSV."""
    import csv

    path = f"{stem}.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            rows.append((rec["label"], float(rec["S"])))
    return rows


def make_figure(kcat_rows, km_rows, dg_rows, t_star, write_csv=True):
    """Write three separate tornado figures (readable fonts + full equations).

    Panel letters (paper order): (A) ΔG°′, (B) k_cat, (C) K_m.
    """
    panels = [
        (
            dg_rows,
            r"$d\ln(\mathrm{PE})\,/\,d(\Delta G^{\circ\prime})$  [(kcal/mol)$^{-1}$]",
            f"Thermodynamic sensitivity at fixed t*={t_star:.0f}s",
            f"{OUT_STEM}_dgr",
            "A",
        ),
        (
            kcat_rows,
            r"$d\ln(\mathrm{PE})\,/\,d\ln(k_{\mathrm{cat}})$",
            f"kcat sensitivity at fixed t*={t_star:.0f}s "
            f"({int(DYNAMIC_FRAC * 100)}% of baseline final)",
            f"{OUT_STEM}_kcat",
            "B",
        ),
        (
            km_rows,
            r"$d\ln(\mathrm{PE})\,/\,d\ln(K_m)$",
            f"Km sensitivity at fixed t*={t_star:.0f}s",
            f"{OUT_STEM}_km",
            "C",
        ),
    ]
    for rows, xlabel, title, stem, letter in panels:
        if write_csv:
            _save_tornado_csv(stem, rows)
        fig, ax = plt.subplots(figsize=(12, 7))
        _tornado(ax, rows, xlabel, title)
        fig.subplots_adjust(left=0.48, right=0.98, top=0.92, bottom=0.12)
        # True figure top-left (outside the crowded y-tick region).
        fig.text(
            0.01,
            0.99,
            f"({letter})",
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )
        for ext in ("png", "pdf"):
            path = f"{stem}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


def replot_tornado_from_csv(t_star=None):
    """Redraw tornado PNGs/PDFs from saved CSVs (no ODE re-run)."""
    kcat_rows = _load_tornado_csv(f"{OUT_STEM}_kcat")
    km_rows = _load_tornado_csv(f"{OUT_STEM}_km")
    dg_rows = _load_tornado_csv(f"{OUT_STEM}_dgr")
    # t* only used in non-bare titles; bare mode ignores it.
    make_figure(kcat_rows, km_rows, dg_rows, t_star=t_star or 0.0, write_csv=False)


def _print_table(name, rows):
    print(f"\n=== {name}: top by |S| at fixed t* ===")
    print(f"{'parameter':<48}{'S':>12}")
    for lab, s in sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:TOP_N]:
        print(f"{lab[:47]:<48}{s:>12.4f}")


# ---------------------------------------------------------------------------
def main(feedback_only=False, tornado_only=False):
    print("Building final FAS-II network (CatPred cached)...", flush=True)
    enlarger, model, temperature = build_model()

    print("Baseline full-horizon PE trajectory...", flush=True)
    base = _simulate(enlarger, model)  # full end_time — need PE_final for t*
    pe = _pe_series(base)
    pe_final = float(pe[-1])
    t_final = float(base.t[-1])
    target = DYNAMIC_FRAC * pe_final
    reach = np.where(pe >= target)[0]
    # Fixed baseline clock time — held for all subsequent perturbed evals.
    t_star = float(base.t[reach[0]]) if len(reach) else t_final
    t_sweep = _sweep_end_time(enlarger, t_star)
    print(
        f"Baseline: final PE = {pe_final:.3f} uM; "
        f"t*({int(DYNAMIC_FRAC*100)}%) = {t_star:.1f} s (FIXED for all sweeps); "
        f"tornado sims end at {t_sweep:.1f} s; T = {temperature:.1f} K; "
        f"tornado jobs = {N_JOBS}",
        flush=True,
    )

    if not feedback_only:
        print("kcat sensitivity...", flush=True)
        kcat_rows = kcat_sensitivity(enlarger, model, t_star, n_jobs=N_JOBS)
        print("Km sensitivity...", flush=True)
        km_rows = km_sensitivity(enlarger, model, t_star, n_jobs=N_JOBS)
        print("ΔG°′ sensitivity...", flush=True)
        dg_rows = dg_sensitivity(
            enlarger, model, temperature, t_star, n_jobs=N_JOBS
        )

        _print_table("kcat (dln PE / dln kcat)", kcat_rows)
        _print_table("Km (dln PE / dln Km)", km_rows)
        _print_table("ΔG°′ (dln PE / d ΔG°′, per kcal/mol)", dg_rows)
        make_figure(kcat_rows, km_rows, dg_rows, t_star)
    else:
        print("Skipping tornado sweeps (--feedback-only).", flush=True)

    if tornado_only:
        print("Skipping FabH/FabI Ki figures (--tornado-only).", flush=True)
        return

    print("FabH Ki PE(t) time course (FabI held)...", flush=True)
    ki_timecourse(
        enlarger, model, t_star,
        enzyme="FabH", out_stem=OUT_KI_FABH_STEM, panel_letter="A",
    )

    print("FabI Ki PE(t) time course (FabH held)...", flush=True)
    ki_timecourse(
        enlarger, model, t_star,
        enzyme="FabI", out_stem=OUT_KI_FABI_STEM, panel_letter="B",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FAS-II sensitivity analysis "
        "(tornado + FabH/FabI Ki feedback figures)."
    )
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Skip kcat/Km/ΔG°′ tornado; only build model and run FabH + FabI Ki figures.",
    )
    parser.add_argument(
        "--tornado-only",
        action="store_true",
        help="Skip FabH/FabI Ki figures; only build model and run tornado panels.",
    )
    parser.add_argument(
        "--replot-tornado",
        action="store_true",
        help="Redraw tornado figures from saved CSVs only (no model/ODE run).",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Drop figure/panel titles (paper figures; caption carries them). "
        "Ki legends still drawn.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override settings.end_time from input.yml (affects the built network too).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers for kcat/Km/ΔG°′ tornado (default: CPU count - 1). "
        "Use 1 for serial. FabH/FabI Ki figures stay serial.",
    )
    args = parser.parse_args()
    if args.feedback_only and args.tornado_only:
        parser.error("Use only one of --feedback-only / --tornado-only")
    BARE = args.bare
    END_TIME_OVERRIDE = args.end_time
    if args.jobs is not None:
        if args.jobs < 1:
            parser.error("--jobs must be >= 1")
        N_JOBS = args.jobs
    if args.replot_tornado:
        replot_tornado_from_csv()
    else:
        main(feedback_only=args.feedback_only, tornado_only=args.tornado_only)
