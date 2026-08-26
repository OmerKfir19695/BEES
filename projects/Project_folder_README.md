# BEES Project Examples Summary

This document lists the example projects under `projects/` and how to run them.

## How to Run Any Example

From the repo root:

```bash
python BEES.py -i projects/<PROJECT_PATH>/input.yml
```

Replace `<PROJECT_PATH>` with the folder path to the project (e.g. `minimal`, `Glycolysis`, `fattyAcidSynthesis/fattyAcidSynthesis_ecoli`).

Available example projects:

| Project path | Description |
|---|---|
| `minimal` | Minimal single-enzyme example |
| `commented` | Heavily commented reference input |
| `Glycolysis/general` | Glycolysis (generic) |
| `Glycolysis/ecoli` | Glycolysis (E. coli database) |
| `KrebsCycle` | TCA / Krebs cycle |
| `PentosePhosphatePathway` | Pentose phosphate pathway |
| `fattyAcidSynthesis/FattyAcidSynthesisDemo` | FAS II generic demo |
| `fattyAcidSynthesis/fattyAcidSynthesis_ecoli` | FAS II E. coli database |
| `fattyAcidSynthesis/fattyAcidSynthesis_ecoli_branched` | FAS II E. coli (branched) |
| `FattyAcidElongation` | Fatty acid elongation |
| `CholesterolSynthesis` | Cholesterol synthesis |
| `PhospholipidSynthesis` | Phospholipid synthesis |
| `SphingolipidSynthesis` | Sphingolipid synthesis |

---


## Output Files

Each run generates output files in an `output/` folder inside the project directory (or in `settings.output_directory` if specified):

1. `output.log` - main execution log
2. `output_errors.log` - error log (if any)
3. `reactions_summary.txt` - summary of generated reactions
4. `input.yml` - validated input

For iterative mode (when `end_time` and `toleranceMoveToCore > 0`):

5. `ode_equations_iterN.txt` - ODE equations and parameters (including SD when available) per iteration
6. `simulation_profiles.csv` - concentration time-series (if `save_simulation_profiles: true`)
7. `flux_analysis.csv`
8. `flux_analysis_reactions.csv` - per-reaction classification history (first seen / core-enter iteration)
9. `core_reactions_species.csv` / `edge_reactions_species.csv` - final core and edge reaction tables
10. `model.xml` - SBML Level 3 export for COPASI / any SBML tool (requires `python-libsbml`)
11. `simulation_plot_iterN.png` - concentration vs time plots per iteration (if `save_simulation_plots: true`)

---

## Database Information

BEES uses the database currently containing **37,119 reactions** imported from the **BKMS** dataset (BRENDA, KEGG, MetaCyc, SABIO-RK) and passed the filtration process.

**Features:**
- **Full Stoichiometry:** Parsed from reaction templates (e.g., `ATP + (R)-pantoate + beta-alanine`).
- **SMILES Mapping:** Includes primary and candidate SMILES for reactants and products.
- **Trigger Logic:** Smart search ignores general cofactors (ATP, H2O, etc.) unless they are specifically matched in the database.

**In input files:**
- Use `database.name: db`
- `settings.kinetics_estimator` supports `catpred` only

---

## Important Notes

1. **Cofactor Handling:** BEES can automatically add some general cofactors required by the database stoichiometry. It also adds standard products for cofactors (e.g., ATP consumed → ADP + Pi produced).

2. **Kinetics Estimation:** `kinetics_estimator: catpred` is implemented. `amino_acid_sequence` is optional for enzymes, but CatPred requires it for estimation. If missing, kinetics estimation will fail for that enzyme.

3. **Simulation Parameters:** `end_time` and `time_step` are used by the enlarger and simulator when iterative or ODE simulation is run. Both must be present in the input. Elsewhere, the simulation change to "batch" mode. 

4. **SMILES:** SMILES are required for all non-enzyme species to ensure chemical identity. The initial SMILES must be provided by the user.

5. **Iterative enlargement stall fallback:** In some mechanisms you can see `R_char > 0` while `max_rr` stays near 0 for long periods, so no edge species get promoted and iterations run for a long time. BEES supports an optional reaction-level fallback (inspired by RMG’s “dynamics number” (dlnaccum)) to trigger promotion even when edge **net species rates** are uninformative.
   - Enable by setting `settings.toleranceMoveEdgeReactionToCore` to a finite value (it is **disabled by default** when unset).
   - Optional knobs:
     - `settings.dynamics_time_scale`: don’t evaluate dynamics before this simulation time (seconds). Default `0.0`.
     - `settings.dynamics_outer_step_interval`: evaluate dynamics every N stepwise outer-steps (performance knob). Default `1`.

Example:

```yaml
settings:
  toleranceMoveToCore: 1e-5
  toleranceInterruptSimulation: 1e-3
  toleranceMoveEdgeReactionToCore: 1e-5
  dynamics_time_scale: 0.0
  dynamics_outer_step_interval: 1
```

6. **Thermodynamics (equilibrator-api):** BEES uses `equilibrator-api` to compute ΔG°’ for each reaction and derives reversibility and `kcat_rev` via the Haldane relationship. Settings:

   | Setting | Where | Default | Description |
   |---|---|---|---|
   | `ionic_strength_M` | `environment` | `0.25` | Ionic strength in M (0–2) |
   | `pMg` | `environment` | `3.0` | pMg (0–10) |
   | `thermo_irreversible_cutoff_kJmol` | `settings` | `30.0` | \|ΔG°’\| above this (kJ/mol) marks reaction irreversible |
   | `thermo_smiles_substitutions` | `settings` | `null` | Label → SMILES overrides for compounds equilibrator cannot resolve |

   Set `BEES_DISABLE_THERMO=1` in the environment to skip thermodynamic calculations entirely (every reaction will be treated as irreversible).

   Example:

   ```yaml
   environment:
     temperature: 310.15
     pH: 7.4
     ionic_strength_M: 0.15
     pMg: 3.0

   settings:
     thermo_irreversible_cutoff_kJmol: 25.0
     thermo_smiles_substitutions:
       “Acyl-ACP”: “CC(=O)SCCNC(=O)CCNC(=O)[C@@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc23)[C@H](O)[C@@H]1OP(=O)(O)O”
   ```
