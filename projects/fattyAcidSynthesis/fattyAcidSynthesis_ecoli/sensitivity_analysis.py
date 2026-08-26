#!/usr/bin/env python3
r"""
Local (one-factor-at-a-time) sensitivity analysis for the E. coli FAS-II model.

Run (repo root, bees_env):
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --tornado-only --bare --jobs 16
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --feedback-only
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --bare
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --end-time 3000
    python projects/fattyAcidSynthesis/fattyAcidSynthesis_ecoli/sensitivity_analysis.py --jobs 16

Feedback Ki panels are per-enzyme: FabH and FabI are swept separately; the
other enzyme's Ki stays at the production lock.
"""

import argparse
import os
import re
import sys
import math
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load .env.bees before any bees.* import (class-level env vars resolve at definition time).
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

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_YML = os.path.join(_PROJECT_DIR, "input.yml")
_OUT_DIR = os.path.join(_PROJECT_DIR, "output", "sensitivity")
os.makedirs(_OUT_DIR, exist_ok=True)
OUT_STEM = os.path.join(_OUT_DIR, "sensitivity_analysis")
OUT_KI_FABH_STEM = os.path.join(_OUT_DIR, "sensitivity_ki_fabh_timecourse")
OUT_KI_FABI_STEM = os.path.join(_OUT_DIR, "sensitivity_ki_fabi_timecourse")
OUT_KI_STEM = OUT_KI_FABH_STEM

KCAT_LN_STEP = 0.05          # perturb kcat/Km/Ki by exp(+/- h) in ln-space
KM_LN_STEP = KCAT_LN_STEP
KI_LN_STEP = KCAT_LN_STEP
DG_STEP_KCAL = 0.5           # ΔG°′ perturbation, kcal/mol
KCAL_PER_KJ = 1.0 / 4.184
R_KJ = 8.314462618e-3        # kJ / (mol K)
DYNAMIC_FRAC = 0.5           # t* = time baseline PE reaches this fraction of final
TOP_N = 10

BARE = False
END_TIME_OVERRIDE = None
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# Fork workers inherit these from the parent (set before ProcessPoolExecutor).
_W_MODEL = None
_W_SIM_KW = None
_W_TEMPERATURE = None


def _load_s2a_experiment():
    """Ruppe 2018 SI Experimental_Dataset.csv (S2A; min, µM palmitic eq)."""
    candidates = (
        os.path.join(
            _REPO_ROOT,
            "knowledge",
            "references",
            "ecoli_fas2_project",
            "ruppe_2018_pnas",
            "Model and Solver",
            "Experimental_Dataset.csv",
        ),
        os.path.join(
            _REPO_ROOT,
            "docs",
            "deliverables",
            "tIme_course_simulation",
            "Experimental_Dataset.csv",
        ),
    )
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            "Ruppe S2A Experimental_Dataset.csv not found in knowledge/ or docs/deliverables/"
        )
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]


S2A_EXP_T_MIN, S2A_EXP_UM = _load_s2a_experiment()

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


def build_model():
    """Run the BEES pipeline up to the final enlarged model; return (enlarger, model, T_K)."""
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


def _simulate(enlarger, model, end_time=None):
    """Fresh ODESimulator each call so the baked-in RHS (including feedback Ki) is rebuilt."""
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
    """As _simulate, but return None if the stiff solver diverges."""
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


def _worker_kcat(job):
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
    """Run OFAT jobs; parallel via fork when n_jobs > 1. Returns (label, S) in submission order."""
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


def _reaction_equation(rxn):
    left = " + ".join(rxn.reactant_labels) if rxn.reactant_labels else "?"
    right = " + ".join(rxn.product_labels) if rxn.product_labels else "?"
    return f"{left} -> {right}"


def _reaction_label(rxn, seen):
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


def _feedback_reactions(model, enzyme=None):
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
    for rxn in rxns:
        fb = rxn.feedback_inhibitors
        if not isinstance(fb, dict):
            continue
        rxn.feedback_inhibitors = {
            lab: (float(ki) * factor, float(hill)) for lab, (ki, hill) in fb.items()
        }


def _baseline_ki_uM(rxns):
    for rxn in rxns:
        fb = rxn.feedback_inhibitors
        if isinstance(fb, dict) and fb:
            ki_mM = float(next(iter(fb.values()))[0])
            return ki_mM * 1000.0
    return float("nan")


def _style_legend(ax, fontsize=9):
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
    """PE(t) for baseline Ki, ln-step Ki band (×e^{±h}), and enzyme-feedback-off."""
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

    sim_base = _simulate(enlarger, model)
    pe_base = _pe_series(sim_base)

    _scale_feedback_ki(fb_rxns, math.exp(KI_LN_STEP))
    sim_p = _simulate(enlarger, model)
    pe_p = _pe_series(sim_p)
    _restore_feedback(fb_rxns, snap)

    _scale_feedback_ki(fb_rxns, math.exp(-KI_LN_STEP))
    sim_m = _simulate(enlarger, model)
    pe_m = _pe_series(sim_m)
    _restore_feedback(fb_rxns, snap)

    for rxn in fb_rxns:
        rxn.feedback_inhibitors = None
    sim_off = _simulate(enlarger, model)
    pe_off = _pe_series(sim_off)
    _restore_feedback(fb_rxns, snap)

    s_ki = _finite_diff(sim_p, sim_m, t_star, 2.0 * KI_LN_STEP)
    print(
        f"\n=== {enzyme} feedback Ki: S_Ki = dln(PE(t*))/dln(Ki) = {s_ki:.4f} ==="
    )
    ki_hi = ki_uM * math.exp(KI_LN_STEP)
    ki_lo = ki_uM * math.exp(-KI_LN_STEP)
    rel_pct = 100.0 * (math.exp(KI_LN_STEP) - 1.0)
    print(f"  baseline Ki = {ki_uM:.3f} uM; t* = {t_star:.1f} s (fixed)")
    print(
        f"  ln-step h={KI_LN_STEP}: Ki scaled to {ki_lo:.3f} and {ki_hi:.3f} uM "
        f"(×e^{{±h}} ≈ ±{rel_pct:.1f}% relative, not ±{KI_LN_STEP} uM)"
    )
    print(f"  other enzyme Ki held at production lock", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    t_min = sim_base.t / 60.0
    pe_p_i = np.interp(sim_base.t, sim_p.t, pe_p)
    pe_m_i = np.interp(sim_base.t, sim_m.t, pe_m)
    rel_pct = 100.0 * (math.exp(KI_LN_STEP) - 1.0)
    ax.fill_between(
        t_min,
        np.minimum(pe_p_i, pe_m_i),
        np.maximum(pe_p_i, pe_m_i),
        color="#2980b9",
        alpha=0.35,
        linewidth=0,
        zorder=1,
        label=rf"Ki × $e^{{\pm {KI_LN_STEP}}}$ ($\approx \pm${rel_pct:.1f}%)",
    )
    ax.plot(
        t_min, pe_base,
        color="#2c3e50", linewidth=2.0, zorder=3,
        label=f"Ki = {ki_uM:.1f} µM",
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
    import csv

    path = f"{stem}.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            rows.append((rec["label"], float(rec["S"])))
    return rows


def make_figure(kcat_rows, km_rows, dg_rows, t_star, write_csv=True):
    """Write three standalone tornado figures (ΔG°′, k_cat, K_m); no panel letters."""
    panels = [
        (
            dg_rows,
            r"$d\ln(\mathrm{PE})\,/\,d(\Delta G^{\circ\prime})$  [(kcal/mol)$^{-1}$]",
            f"Thermodynamic sensitivity at fixed t*={t_star:.0f}s",
            f"{OUT_STEM}_dgr",
        ),
        (
            kcat_rows,
            r"$d\ln(\mathrm{PE})\,/\,d\ln(k_{\mathrm{cat}})$",
            f"kcat sensitivity at fixed t*={t_star:.0f}s "
            f"({int(DYNAMIC_FRAC * 100)}% of baseline final)",
            f"{OUT_STEM}_kcat",
        ),
        (
            km_rows,
            r"$d\ln(\mathrm{PE})\,/\,d\ln(K_m)$",
            f"Km sensitivity at fixed t*={t_star:.0f}s",
            f"{OUT_STEM}_km",
        ),
    ]
    for rows, xlabel, title, stem in panels:
        if write_csv:
            _save_tornado_csv(stem, rows)
        fig, ax = plt.subplots(figsize=(12, 7))
        _tornado(ax, rows, xlabel, title)
        fig.subplots_adjust(left=0.48, right=0.98, top=0.92, bottom=0.12)
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
    make_figure(kcat_rows, km_rows, dg_rows, t_star=t_star or 0.0, write_csv=False)


def _print_table(name, rows):
    print(f"\n=== {name}: top by |S| at fixed t* ===")
    print(f"{'parameter':<48}{'S':>12}")
    for lab, s in sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:TOP_N]:
        print(f"{lab[:47]:<48}{s:>12.4f}")


def main(feedback_only=False, tornado_only=False):
    print("Building final FAS-II network (CatPred cached)...", flush=True)
    enlarger, model, temperature = build_model()

    print("Baseline full-horizon PE trajectory...", flush=True)
    base = _simulate(enlarger, model)
    pe = _pe_series(base)
    pe_final = float(pe[-1])
    t_final = float(base.t[-1])
    target = DYNAMIC_FRAC * pe_final
    reach = np.where(pe >= target)[0]
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
