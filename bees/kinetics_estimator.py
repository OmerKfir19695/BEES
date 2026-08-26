"""

This module defines a small abstraction layer so BEES can fill missing kinetic
parameters (kcat, Km, Ki) using external tools (e.g., CatPred) without coupling
the core generator to a specific backend.

Current status:
- Provides an interface integrated with CatPred for kcat and Km estimation.
"""

from __future__ import annotations
import dataclasses
import shutil
import hashlib
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from bees.common import canonical_smiles, log10_sd_to_linear_sd

@dataclass(frozen=True)
class EstimatedKinetics:
    """
    Estimated kinetic parameters.

    Units:
    - km: mM (single value, for backward compat when only one substrate)
    - km_per_substrate: mM per substrate (CatPred per-substrate Km)
    - ki: mM
    - kcat: 1/s (one per reaction)
    - *_sd: standard deviation of the prediction (same units as the parameter)
    """

    km: Optional[float] = None
    km_per_substrate: Optional[Dict[str, float]] = None
    km_sd: Optional[float] = None
    km_sd_per_substrate: Optional[Dict[str, float]] = None
    kcat: Optional[float] = None
    ki: Optional[float] = None
    kcat_sd: Optional[float] = None
    ki_sd: Optional[float] = None
    source: str = "estimator"

class BaseKineticsEstimator:
    """Adaptor base class for external kinetics estimators."""

    name: str = "base"

    def estimate(
        self,
        *,
        enzyme_sequence: str,
        reactant_smiles: Dict[str, str],
        inhibitor_smiles: Optional[str] = None,
        ec_number: Optional[str] = None,
    ) -> EstimatedKinetics:
        """
        Estimate kinetics from enzyme and reactant SMILES.

        Args:
            enzyme_sequence: Enzyme amino acid sequence
            reactant_smiles: Map of compound name -> SMILES for all reactants.
                Used for: kcat = concatenated SMILES; Km = one per substrate.
            inhibitor_smiles: Optional inhibitor SMILES for Ki
            ec_number: EC of the reaction, used to look up per-EC kcat scaling.
        """
        raise NotImplementedError(
            "BaseKineticsEstimator is an interface. Use build_estimator('catpred') "
        )

class CatPredEstimator(BaseKineticsEstimator):
    """
    CatPred kinetics estimator. This class is used to estimate the kinetics of a reaction using CatPred.

    Args:
        include_sd: If True, include SD_total (standard deviation) from CatPred output for each parameter.
    Attributes:
        include_sd: If True, include SD_total (standard deviation) from CatPred output for each parameter.
        CATPRED_DIR: Path to CatPred installation.
        CHECKPOINT_BASE: Path to pretrained production models.
        CONDA_ENV: Name of conda environment containing CatPred.

    Methods:
    """

    name = "catpred"

    def __init__(
        self,
        include_sd: bool = False,
        ec_kcat_scale: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            include_sd: If True, include SD_total (standard deviation) from CatPred output for each parameter.
            ec_kcat_scale: Optional EC-string -> multiplier dict. Applied to kcat
                only (not Km, not SD) as a post-hoc calibration. Keys are EC strings
                as they appear in the reaction DB (e.g. "EC 2.3.1.41").
        """
        self.include_sd = include_sd
        self.ec_kcat_scale: Dict[str, float] = dict(ec_kcat_scale or {})
        if self.ec_kcat_scale:
            msg = (
                f"CatPred kcat calibration active for {len(self.ec_kcat_scale)} EC(s): "
                + ", ".join(f"{k}x{v}" for k, v in sorted(self.ec_kcat_scale.items()))
            )
            logging.getLogger(__name__).info(msg)
            print(f"[CatPredEstimator] {msg}", flush=True)
        # In-process memoization: repeated CatPred calls are expensive.
        # Keyed by (enzyme sequence, reactant set, inhibitor, include_sd).
        # ec_number is NOT in the key — the cached entry is the unscaled CatPred
    
        self._memo: Dict[tuple, EstimatedKinetics] = {}
        # BEES_CATPRED_CACHE: unset -> ~/.cache/bees/...; <path> -> that file; off/0/none/false/disable -> off.
        _cache_env = os.environ.get("BEES_CATPRED_CACHE")
        if _cache_env is None:
            _cache_dir = os.environ.get("XDG_CACHE_HOME") or os.path.join(
                os.path.expanduser("~"), ".cache")
            self._cache_path: Optional[str] = os.path.join(
                _cache_dir, "bees", "catpred_predictions.pkl")
        elif _cache_env.strip().lower() in {
            "", "0", "off", "none", "false", "disable", "disabled",
        }:
            self._cache_path = None
        else:
            self._cache_path = _cache_env
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                import pickle
                with open(self._cache_path, "rb") as fh:
                    self._memo.update(pickle.load(fh))
                logging.getLogger(__name__).info(
                    "Loaded %d CatPred cache entries from %s",
                    len(self._memo), self._cache_path,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Could not load CatPred cache %s: %s", self._cache_path, exc
                )

    def _save_persistent_cache(self) -> None:
        """Atomically write the memo to disk (best-effort)."""
        if not self._cache_path:
            return
        try:
            import pickle
            import tempfile
            d = os.path.dirname(self._cache_path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(self._memo, fh)
            os.replace(tmp, self._cache_path)
        except Exception as exc:
            logging.getLogger(__name__).debug("CatPred cache save failed: %s", exc)

    CATPRED_DIR = os.environ.get("CATPRED_DIR", "/path/to/CatPred")
    CHECKPOINT_BASE = os.environ.get(
        "CATPRED_CHECKPOINT_BASE",
        "/path/to/pretrained/production"
    )
    CONDA_ENV = os.environ.get("CATPRED_CONDA_ENV", "catpred")
    CONDA_BIN = os.environ.get("CATPRED_CONDA_BIN", "conda")
    CATPRED_PYTHON = os.environ.get("CATPRED_PYTHON", "")

    def _apply_kcat_scaling(
        self,
        raw: EstimatedKinetics,
        ec_number: Optional[str],
    ) -> EstimatedKinetics:
        """Multiply kcat by the EC-specific factor; kcat_sd is not scaled."""
        if raw.kcat is None or not ec_number:
            return raw
        factor = self.ec_kcat_scale.get(ec_number)
        if factor is None:
            return raw
        scaled = raw.kcat * factor
        msg = (
            f"Applied ec_kcat_scale[{ec_number}]={factor:g}: "
            f"kcat {raw.kcat:.4g} -> {scaled:.4g} s^-1"
        )
        logging.getLogger(__name__).info(msg)
        print(f"[CatPredEstimator] {msg}", flush=True)
        return EstimatedKinetics(
            km=raw.km,
            km_per_substrate=raw.km_per_substrate,
            km_sd=raw.km_sd,
            km_sd_per_substrate=raw.km_sd_per_substrate,
            kcat=scaled,
            ki=raw.ki,
            kcat_sd=raw.kcat_sd,
            ki_sd=raw.ki_sd,
            source=f"{raw.source} [kcat scaled x{factor} for {ec_number}]",
        )

    def estimate(
        self,
        *,
        enzyme_sequence: str,
        reactant_smiles: Dict[str, str],
        inhibitor_smiles: Optional[str] = None,
        ec_number: Optional[str] = None,
    ) -> EstimatedKinetics:
        """
        Estimate kcat and Km using CatPred.

        - kcat: concatenated SMILES of all reactants (one kcat per reaction)
        - Km: one value per substrate (individual substrate SMILES)
        - ec_number: if matches a key in self.ec_kcat_scale, the returned kcat is
          multiplied by that factor (Km and SDs unchanged).
        """
        if self.ec_kcat_scale:
            logging.getLogger(__name__).debug(
                "estimate() called with ec_number=%r (scale dict has %d keys)",
                ec_number, len(self.ec_kcat_scale),
            )
        if not reactant_smiles:
            return EstimatedKinetics(source="catpred(no reactants)")

        memo_key = None
        _omit_cof = os.environ.get(
            "BEES_CATPRED_KCAT_OMIT_COFACTORS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            memo_key = (
                enzyme_sequence,
                tuple(
                    sorted(
                        (str(name), canonical_smiles(smi) or str(smi))
                        for name, smi in reactant_smiles.items()
                        if smi
                    )
                ),
                canonical_smiles(inhibitor_smiles) if inhibitor_smiles else None,
                bool(self.include_sd),
                self.CHECKPOINT_BASE,  # model identity: a checkpoint change invalidates
                bool(_omit_cof),  # kcat SMILES filter mode
            )
            cached = self._memo.get(memo_key)
            if cached is not None:
                return self._apply_kcat_scaling(cached, ec_number)
        except Exception:
            memo_key = None

        if not os.path.isdir(self.CATPRED_DIR):
            raise FileNotFoundError(
                f"CatPred directory not found: {self.CATPRED_DIR!r}. "
                "Set CATPRED_DIR to your CatPred clone path. "
                "Run ./install.sh to install CatPred, or manually: clone CatPred, create the catpred conda env, "
                "download pretrained data. Then run ./install.sh to create .env.bees (auto-loaded by BEES), "
                "or export CATPRED_DIR=... CATPRED_CHECKPOINT_BASE=... CATPRED_CONDA_ENV=catpred."
            )

        # 1. Create a unique ID for this prediction request to avoid file collisions
        run_id = f"bees_{uuid.uuid4().hex[:8]}"
        
        # Ensure a local writable temp directory exists
        local_tmp = os.path.join(os.getcwd(), ".tmp")
        os.makedirs(local_tmp, exist_ok=True)
        
        # Create a writable working directory for CatPred execution
        catpred_work_dir = os.path.join(local_tmp, f"work_{run_id}")
        os.makedirs(catpred_work_dir, exist_ok=True)
        
        # Symlink necessary CatPred files to the writable work dir
        for item in os.listdir(self.CATPRED_DIR):
            if item.startswith('.') or item == "output" or item == "demo":
                continue
            src = os.path.normpath(os.path.join(self.CATPRED_DIR, item))
            dst = os.path.join(catpred_work_dir, item)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except Exception as exc:
                    # Some environments (including sandboxed execution) can prevent creating
                    # symlinks to paths outside the workspace. Fall back to copying.
                    try:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(src, dst)
                    except Exception as copy_exc:
                        logging.getLogger(__name__).warning(
                            "Failed to stage CatPred item %r "
                            "(symlink error: %s; copy error: %s)",
                            item, exc, copy_exc,
                        )
        
        os.makedirs(os.path.join(catpred_work_dir, "output"), exist_ok=True)
        os.makedirs(os.path.join(catpred_work_dir, "demo"), exist_ok=True)
        # CatPred demo_run.py writes ./predict.sh; a read-only symlink from the repo raises PermissionError.
        staged_predict_sh = os.path.join(catpred_work_dir, "predict.sh")
        try:
            if os.path.exists(staged_predict_sh):
                os.remove(staged_predict_sh)
        except Exception as exc:
            logging.getLogger(__name__).debug("Best-effort cleanup: could not remove %s: %s", staged_predict_sh, exc)

        results = {}

        # ESM embedding cache is keyed by pdbpath; key by sequence so enzymes don't collide.
        pdb_id = "bees_" + hashlib.md5(enzyme_sequence.encode("utf-8")).hexdigest()[:16]

        # BEES_CATPRED_KCAT_OMIT_COFACTORS=1 drops GENERAL_COFACTORS from kcat SMILES only (Km unchanged).
        if _omit_cof:
            from bees.cofactors import GENERAL_COFACTORS as _GEN_COF

            _kcat_smiles = [
                s
                for name, s in reactant_smiles.items()
                if s and str(name).lower().strip() not in _GEN_COF
            ]
            # Fall back to all reactants if filtering emptied the list.
            if not _kcat_smiles:
                _kcat_smiles = [s for s in reactant_smiles.values() if s]
        else:
            _kcat_smiles = [s for s in reactant_smiles.values() if s]
        concatenated_smiles = ".".join(_kcat_smiles)
        if concatenated_smiles:
            kcat_csv = os.path.join(catpred_work_dir, f"{run_id}_kcat.csv")
            df_kcat = pd.DataFrame([{
                "SMILES": concatenated_smiles,
                "sequence": enzyme_sequence,
                "pdbpath": pdb_id
            }])
            try:
                df_kcat.to_csv(kcat_csv, index=False)
            except Exception as e:
                return EstimatedKinetics(source=f"catpred(error writing kcat input: {str(e)})")

            param = "kcat"
            checkpoint_dir = os.path.join(self.CHECKPOINT_BASE, param)
            if self.CATPRED_PYTHON:
                cmd = [self.CATPRED_PYTHON, "demo_run.py", "--parameter", param,
                       "--input_file", kcat_csv, "--checkpoint_dir", checkpoint_dir]
            else:
                cmd = [self.CONDA_BIN, "run", "-n", self.CONDA_ENV,
                       "python", "demo_run.py", "--parameter", param,
                       "--input_file", kcat_csv, "--checkpoint_dir", checkpoint_dir]
            try:
                env = os.environ.copy()
                env["TMPDIR"] = local_tmp
                env["TEMP"] = local_tmp
                env["TMP"] = local_tmp
                # When using a direct python path, prepend its directory to PATH so that
                # predict.sh (written by demo_run.py) also picks up the correct python.
                if self.CATPRED_PYTHON:
                    catpred_bin = os.path.dirname(self.CATPRED_PYTHON)
                    env["PATH"] = catpred_bin + os.pathsep + env.get("PATH", "")
                subprocess.run(cmd, cwd=catpred_work_dir, check=True, capture_output=True, text=True, env=env)
                kcat_basename = os.path.splitext(os.path.basename(kcat_csv))[0]
                output_path = os.path.join(catpred_work_dir, "output", kcat_basename, f"{kcat_basename}_{param}_output.csv")
                if os.path.exists(output_path):
                    df_res = pd.read_csv(output_path)
                    if not df_res.empty:
                        col_name = "Prediction_(s^(-1))"
                        if col_name in df_res.columns:
                            results["kcat"] = float(df_res[col_name].iloc[0])
                        if self.include_sd and "SD_total" in df_res.columns:
                            pred_val = results.get("kcat")
                            sd_log10 = float(df_res["SD_total"].iloc[0])
                            if pred_val is not None and pred_val > 0 and sd_log10 > 0:
                                results["kcat_sd"] = log10_sd_to_linear_sd(pred_val, sd_log10)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.strip() if e.stderr else ""
                results["kcat_error"] = stderr or str(e)
                logging.getLogger(__name__).debug("CatPred kcat stderr:\n%s", stderr)
            except Exception as e:
                results["kcat_error"] = str(e)

        # --- km: one row per reactant (individual substrate SMILES) ---
        reactant_names = list(reactant_smiles.keys())
        reactant_smiles_list = [reactant_smiles.get(n, "") for n in reactant_names]
        if any(reactant_smiles_list):
            km_csv = os.path.join(catpred_work_dir, f"{run_id}_km.csv")
            df_km = pd.DataFrame([
                {"SMILES": smi, "sequence": enzyme_sequence, "pdbpath": pdb_id}
                for smi in reactant_smiles_list if smi
            ])
            # Track which reactant name corresponds to each row (only rows with SMILES)
            km_reactant_order = [n for n, smi in zip(reactant_names, reactant_smiles_list) if smi]
            try:
                df_km.to_csv(km_csv, index=False)
            except Exception as e:
                pass  # results already populated for kcat
            else:
                param = "km"
                checkpoint_dir = os.path.join(self.CHECKPOINT_BASE, param)
                if self.CATPRED_PYTHON:
                    cmd = [self.CATPRED_PYTHON, "demo_run.py", "--parameter", param,
                           "--input_file", km_csv, "--checkpoint_dir", checkpoint_dir]
                else:
                    cmd = [self.CONDA_BIN, "run", "-n", self.CONDA_ENV,
                           "python", "demo_run.py", "--parameter", param,
                           "--input_file", km_csv, "--checkpoint_dir", checkpoint_dir]
                try:
                    env = os.environ.copy()
                    env["TMPDIR"] = local_tmp
                    env["TEMP"] = local_tmp
                    env["TMP"] = local_tmp
                    if self.CATPRED_PYTHON:
                        catpred_bin = os.path.dirname(self.CATPRED_PYTHON)
                        env["PATH"] = catpred_bin + os.pathsep + env.get("PATH", "")
                    subprocess.run(cmd, cwd=catpred_work_dir, check=True, capture_output=True, text=True, env=env)
                    km_basename = os.path.splitext(os.path.basename(km_csv))[0]
                    output_path = os.path.join(catpred_work_dir, "output", km_basename, f"{km_basename}_{param}_output.csv")
                    if os.path.exists(output_path):
                        df_res = pd.read_csv(output_path)
                        col_name = "Prediction_(mM)"
                        if not df_res.empty and col_name in df_res.columns:
                            km_per_substrate = {}
                            km_sd_per_substrate = {}
                            for i, rname in enumerate(km_reactant_order):
                                if i < len(df_res):
                                    val = float(df_res[col_name].iloc[i])
                                    km_per_substrate[rname] = val
                                    if self.include_sd and "SD_total" in df_res.columns and i < len(df_res):
                                        sd_log10 = float(df_res["SD_total"].iloc[i])
                                        if val > 0 and sd_log10 > 0:
                                            km_sd_per_substrate[rname] = log10_sd_to_linear_sd(val, sd_log10)
                            results["km_per_substrate"] = km_per_substrate
                            if km_sd_per_substrate:
                                results["km_sd_per_substrate"] = km_sd_per_substrate
                            # Backward compat: single km = first substrate.
                            if km_per_substrate:
                                first = next(iter(km_per_substrate.values()))
                                results["km"] = first
                                if km_sd_per_substrate:
                                    results["km_sd"] = next(iter(km_sd_per_substrate.values()))
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr.strip() if e.stderr else ""
                    results["km_error"] = stderr or str(e)
                    logging.getLogger(__name__).debug("CatPred km stderr:\n%s", stderr)
                except Exception as e:
                    results["km_error"] = str(e)

        # 3. Cleanup storage (keep workdir on error for debugging)
        had_error = any(k in results for k in ("kcat_error", "km_error"))
        if had_error:
            logging.getLogger(__name__).warning(
                "CatPred run had errors; keeping work directory for inspection: %s",
                catpred_work_dir,
            )
        elif os.path.exists(catpred_work_dir):
            try:
                shutil.rmtree(catpred_work_dir)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to remove CatPred work directory %s: %s",
                    catpred_work_dir,
                    exc,
                )

        source = "catpred"
        errors = [results.get(f"{p}_error") for p in ["km", "kcat"] if f"{p}_error" in results]
        if errors:
            source = f"catpred(errors: {'; '.join(e for e in errors if e)})"[:255]

        raw = EstimatedKinetics(
            km=results.get("km"),
            km_per_substrate=results.get("km_per_substrate"),
            km_sd=results.get("km_sd"),
            km_sd_per_substrate=results.get("km_sd_per_substrate"),
            kcat=results.get("kcat"),
            ki=results.get("ki"),
            kcat_sd=results.get("kcat_sd"),
            ki_sd=results.get("ki_sd"),
            source=source
        )
        if memo_key is not None:
            try:
                self._memo[memo_key] = raw
                self._save_persistent_cache()
            except Exception as exc:
                logging.getLogger(__name__).debug("Failed to update CatPred memo cache for key %s: %s", memo_key, exc)
        return self._apply_kcat_scaling(raw, ec_number)


def build_estimator(
    name: Optional[str],
    include_sd: bool = False,
    ec_kcat_scale: Optional[Dict[str, float]] = None,
) -> Optional[BaseKineticsEstimator]:
    """Return a kinetics estimator (`catpred`), or None."""
    if not name:
        return None
    if name == "catpred":
        return CatPredEstimator(include_sd=include_sd, ec_kcat_scale=ec_kcat_scale)
    raise ValueError(f"Unknown kinetics_estimator: {name}")
