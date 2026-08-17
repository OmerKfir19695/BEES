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
    """Adaptor - base class for integrate with external kinetics estimators."""

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
        # In-process memoization: repeated CatPred calls are very expensive.
        # Keyed by (enzyme sequence, reactant set, inhibitor, include_sd).
        # ec_number is NOT in the key — the cached entry is the unscaled CatPred
        # prediction; scaling is applied after retrieval based on the caller's EC.
        self._memo: Dict[tuple, EstimatedKinetics] = {}
        # PERSISTENT prediction cache — ON BY DEFAULT for every project. CatPred is
        # by far the slowest step; persisting the prediction memo lets re-runs of any
        # network skip the subprocess. Keyed by (sequence, reactants, inhibitor,
        # include_sd, model checkpoint), so reuse is always correct (predictions are
        # deterministic in those inputs) and a model/checkpoint change invalidates
        # stale entries. Shared across projects (identical sequence+substrate ⇒
        # identical prediction), so any project benefits from another's cached calls.
        #   BEES_CATPRED_CACHE unset        -> default ~/.cache/bees/catpred_predictions.pkl
        #   BEES_CATPRED_CACHE=<path>       -> use that file
        #   BEES_CATPRED_CACHE=off|0|none   -> disable
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
        """Atomically write the memo to BEES_CATPRED_CACHE (best-effort, opt-in)."""
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

    # Paths to CatPred installation and models.
    # Set env vars (CATPRED_DIR, CATPRED_CHECKPOINT_BASE, CATPRED_CONDA_ENV) or
    # edit these defaults to match your installation.
    CATPRED_DIR = os.environ.get("CATPRED_DIR", "/path/to/CatPred")
    CHECKPOINT_BASE = os.environ.get(
        "CATPRED_CHECKPOINT_BASE",
        "/path/to/pretrained/production"
    )
    CONDA_ENV = os.environ.get("CATPRED_CONDA_ENV", "catpred")
    # Optional: full path to conda binary (needed when multiple conda installs exist)
    CONDA_BIN = os.environ.get("CATPRED_CONDA_BIN", "conda")
    # Optional: direct path to python binary in catpred env; if set, bypasses conda run entirely
    CATPRED_PYTHON = os.environ.get("CATPRED_PYTHON", "")

    def _apply_kcat_scaling(
        self,
        raw: EstimatedKinetics,
        ec_number: Optional[str],
    ) -> EstimatedKinetics:
        """Return raw if no scaling applies for this EC; otherwise return a copy
        with kcat multiplied by the EC-specific factor and `source` annotated.
        kcat_sd is intentionally not scaled — see Settings.kinetics_ec_kcat_scale."""
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
                        # Best-effort: continue, but keep an observable breadcrumb.
                        logging.getLogger(__name__).warning(
                            "Failed to stage CatPred item %r "
                            "(symlink error: %s; copy error: %s)",
                            item, exc, copy_exc,
                        )
        
        os.makedirs(os.path.join(catpred_work_dir, "output"), exist_ok=True)
        os.makedirs(os.path.join(catpred_work_dir, "demo"), exist_ok=True)
        # CatPred's demo_run.py writes a local ./predict.sh script. If we staged a
        # symlinked predict.sh from the CatPred repo, it will be read-only and the
        # run will fail with PermissionError. Ensure it's absent so demo_run.py can
        # create it.
        staged_predict_sh = os.path.join(catpred_work_dir, "predict.sh")
        try:
            if os.path.exists(staged_predict_sh):
                os.remove(staged_predict_sh)
        except Exception as exc:
            logging.getLogger(__name__).debug("Best-effort cleanup: could not remove %s: %s", staged_predict_sh, exc)

        results = {}

        # CatPred caches the per-protein ESM embedding keyed by `pdbpath`
        # (catpred/data/cache_utils.py: key = name -> ~/.cache.esm2_embeddings/.../{key}.pt).
        # A constant pdbpath would collide across enzymes: every query would reuse
        # whichever sequence first populated that cache slot, so the protein signal
        # would be frozen and predictions wrong. Key the cache by the sequence itself
        # so identical sequences share the embedding and different ones never collide.
        pdb_id = "bees_" + hashlib.md5(enzyme_sequence.encode("utf-8")).hexdigest()[:16]

        # --- kcat: one row with concatenated SMILES of reactants ---
        # Default: all reactants. Opt-in BEES_CATPRED_KCAT_OMIT_COFACTORS=1 drops
        # GENERAL_COFACTORS from the kcat SMILES only (Km queries unchanged).
        # Diagnose / ablation switch — not a production default.
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
                            # Backward compat: single km = first substrate
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
                # Cache the unscaled prediction so different EC contexts (e.g.
                # multi-EC enzymes like FabA) share the CatPred result.
                self._memo[memo_key] = raw
                self._save_persistent_cache()  # no-op unless BEES_CATPRED_CACHE set
            except Exception as exc:
                logging.getLogger(__name__).debug("Failed to update CatPred memo cache for key %s: %s", memo_key, exc)
        return self._apply_kcat_scaling(raw, ec_number)


class MeasuredTableEstimator(BaseKineticsEstimator):
    """Use measured kinetics from a CSV, keyed by EC + acyl-chain length.

    A reusable "use measured kinetics where available" backend. For any reaction
    whose EC (and, for chain-resolved enzymes, acyl-chain length) is in the table,
    it returns the measured kcat/Km; for anything not in the table it falls back to
    an internal CatPred delegate. This lets a model run on measured data where it
    exists (e.g. the Ruppe 2020 acyl-ACP FAS ladders) without losing coverage of
    reactions that only CatPred can estimate.

    CSV columns (parsed with comment='#', so '#'-prefixed lines are citations):
        ec            EC number, prefix form without "EC " (e.g. "3.1.2.14")
        chain         acyl-chain carbon count (even int 4..20) for chain-resolved
                      enzymes, or "*" for a single value covering all chains
        kcat_s        kcat in s^-1
        km_acyl_mM    Km of the acyl (chain-bearing) substrate, in mM
        km_malonyl_mM Km of malonyl substrate, in mM (blank if N/A)
        km_cofactor_mM Km of the redox cofactor, in mM (blank if N/A)
        cofactor      which cofactor km_cofactor_mM applies to: nadph | nadh | ""
        source        free-text citation (ignored by the parser)

    Units are mM (Km) and s^-1 (kcat) to match EstimatedKinetics; any µM→mM
    conversion (e.g. TesA Km = koff/kon in µM) is done when writing the CSV.

    SCALING: this estimator owns the final `ec_kcat_scale` application — for BOTH
    measured rows and CatPred-fallback rows. Its internal CatPred delegate is built
    UNSCALED so fallback rows are never scaled twice.
    """

    name = "measured_table"

    def __init__(
        self,
        table_path: str,
        include_sd: bool = False,
        ec_kcat_scale: Optional[Dict[str, float]] = None,
    ):
        self.include_sd = include_sd
        self.ec_kcat_scale: Dict[str, float] = dict(ec_kcat_scale or {})
        # Unscaled CatPred delegate — this class applies scaling once, at the end.
        self._fallback = CatPredEstimator(include_sd=include_sd, ec_kcat_scale=None)
        self._table = self._load_table(table_path)
        logging.getLogger(__name__).info(
            "MeasuredTableEstimator loaded %d EC entries from %s",
            len(self._table), table_path,
        )

    @staticmethod
    def _ec_prefix(ec_number: Optional[str]) -> Optional[str]:
        if not ec_number:
            return None
        s = str(ec_number).strip().upper()
        if s.startswith("EC "):
            s = s[3:].strip()
        parts = s.split(".")
        return ".".join(parts) if len(parts) == 4 else None

    @staticmethod
    def _nearest_chain(n: int) -> int:
        n = max(4, min(20, int(n)))
        return n if n % 2 == 0 else n - 1

    def _load_table(self, table_path: str) -> Dict[str, Dict[object, dict]]:
        df = pd.read_csv(table_path, comment="#", skip_blank_lines=True)
        df.columns = [c.strip() for c in df.columns]
        table: Dict[str, Dict[object, dict]] = {}
        for _, r in df.iterrows():
            ec = self._ec_prefix(str(r["ec"]))
            if ec is None:
                continue
            ch = str(r["chain"]).strip()
            key: object = "*" if ch == "*" else int(float(ch))
            def _f(col):
                v = r.get(col)
                try:
                    return float(v) if v is not None and str(v).strip() != "" and not pd.isna(v) else None
                except (TypeError, ValueError):
                    return None
            table.setdefault(ec, {})[key] = {
                "kcat_s": _f("kcat_s"),
                "km_acyl_mM": _f("km_acyl_mM"),
                "km_malonyl_mM": _f("km_malonyl_mM"),
                "km_cofactor_mM": _f("km_cofactor_mM"),
                "cofactor": (str(r.get("cofactor")).strip().lower()
                             if r.get("cofactor") is not None and str(r.get("cofactor")).strip() != "" else None),
            }
        return table

    @staticmethod
    def _is_malonyl(label: str) -> bool:
        return "malonyl" in label.lower()

    @staticmethod
    def _is_nadph(label: str) -> bool:
        lo = label.lower()
        return "nadph" in lo or "nadp" in lo

    @staticmethod
    def _is_nadh(label: str) -> bool:
        lo = label.lower()
        return ("nadh" in lo or lo in ("nad", "nad(+)", "nad+")) and "nadp" not in lo

    def _apply_scale(self, raw: EstimatedKinetics, ec_number: Optional[str]) -> EstimatedKinetics:
        """Apply ec_kcat_scale once (kcat only). Mirrors CatPredEstimator semantics."""
        if raw.kcat is None or not ec_number:
            return raw
        factor = self.ec_kcat_scale.get(ec_number)
        if factor is None:
            return raw
        return dataclasses.replace(
            raw, kcat=raw.kcat * factor,
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
        from bees.rules.physics_rules import detect_acyl_chain_length

        prefix = self._ec_prefix(ec_number)
        ec_rows = self._table.get(prefix) if prefix else None

        if not ec_rows:
            # Not in the measured table -> unscaled CatPred, then apply our scaling.
            raw = self._fallback.estimate(
                enzyme_sequence=enzyme_sequence,
                reactant_smiles=reactant_smiles,
                inhibitor_smiles=inhibitor_smiles,
                ec_number=ec_number,
            )
            return self._apply_scale(raw, ec_number)

        # Chain of the acyl substrate being transformed = longest detectable acyl
        # substrate (excludes malonyl C3 / acetyl C2 primers).
        chain_by_label = {lab: detect_acyl_chain_length(lab, reactant_smiles.get(lab))
                          for lab in reactant_smiles}
        main_n = None
        for lab, n in chain_by_label.items():
            if n is not None and n >= 4 and not self._is_malonyl(lab):
                main_n = n if main_n is None else max(main_n, n)

        # Pick the row: chain-resolved if present, else the "*" single-value row.
        row = None
        if main_n is not None:
            row = ec_rows.get(self._nearest_chain(main_n))
        if row is None:
            row = ec_rows.get("*")
        if row is None:
            # EC present but no matching chain and no "*" row -> fall back.
            raw = self._fallback.estimate(
                enzyme_sequence=enzyme_sequence, reactant_smiles=reactant_smiles,
                inhibitor_smiles=inhibitor_smiles, ec_number=ec_number,
            )
            return self._apply_scale(raw, ec_number)

        # Build km_per_substrate by classifying each reactant label.
        km_per: Dict[str, float] = {}
        for lab in reactant_smiles:
            n = chain_by_label.get(lab)
            if self._is_malonyl(lab) and row.get("km_malonyl_mM") is not None:
                km_per[lab] = row["km_malonyl_mM"]
            elif self._is_nadph(lab) and row.get("cofactor") == "nadph" and row.get("km_cofactor_mM") is not None:
                km_per[lab] = row["km_cofactor_mM"]
            elif self._is_nadh(lab) and row.get("cofactor") == "nadh" and row.get("km_cofactor_mM") is not None:
                km_per[lab] = row["km_cofactor_mM"]
            elif n is not None and n >= 2 and not self._is_malonyl(lab) and row.get("km_acyl_mM") is not None:
                km_per[lab] = row["km_acyl_mM"]

        raw = EstimatedKinetics(
            kcat=row.get("kcat_s"),
            km_per_substrate=km_per or None,
            km=(next(iter(km_per.values())) if km_per else None),
            source=f"measured_table(EC {prefix}, chain {self._nearest_chain(main_n) if main_n else '*'})",
        )
        return self._apply_scale(raw, ec_number)


def build_estimator(
    name: Optional[str],
    include_sd: bool = False,
    ec_kcat_scale: Optional[Dict[str, float]] = None,
    measured_kinetics_file: Optional[str] = None,
) -> Optional[BaseKineticsEstimator]:
    """
    Return the appropriate kinetics estimator based on the name.

    Args:
        name: Estimator backend name ('catpred' or 'measured_table')
        include_sd: If True, estimator will include standard deviation
        ec_kcat_scale: Optional per-EC kcat multiplier dict; see CatPredEstimator.
        measured_kinetics_file: Path to the measured-kinetics CSV (required for
            name == 'measured_table').
    """
    if not name:
        return None
    if name == "catpred":
        return CatPredEstimator(include_sd=include_sd, ec_kcat_scale=ec_kcat_scale)
    if name == "measured_table":
        if not measured_kinetics_file:
            raise ValueError(
                "kinetics_estimator='measured_table' requires settings.measured_kinetics_file"
            )
        return MeasuredTableEstimator(
            table_path=measured_kinetics_file,
            include_sd=include_sd,
            ec_kcat_scale=ec_kcat_scale,
        )
    raise ValueError(f"Unknown kinetics_estimator: {name}")

