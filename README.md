# BEES

**Biochemical Engine for Enzymatic Systems** — build kinetic models of enzymatic reaction networks from a list of species and enzymes.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Contents

- [What BEES does](#what-bees-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input YAML](#input-yaml)
- [Output](#output)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## What BEES does

- Builds biochemical reaction networks from species + enzymes using a local database.
- Estimates missing kinetics (kcat, Km) with the deep-learning model CatPred.
- Computes thermodynamics (ΔG°′, Keq) with equilibrator-api.
- Applies a rules layer after kinetics and thermodynamics: physics corrections always run, system-specific calibrations are optional.
- Grows the network with a flux-based iterative (core–edge) algorithm driven by ODE simulation.
- Exports a reaction summary, tables, plots, flux analysis, and an SBML Level 3 model (`model.xml`).

## Installation

**Prerequisites:** Python 3.12+, Conda. For CatPred: `git` and `wget` (or `curl`) on PATH.

```bash
git clone https://github.com/DanaResearchGroup/BEES.git
cd BEES
make install          # BEES + CatPred
# make install-bees   # BEES only, no CatPred
```

`make install` creates the `bees_env` conda env, clones CatPred into the **parent** directory of BEES (`../CatPred`, `../catpred_pipeline`), downloads pretrained data (~1 GB), and writes `.env.bees` in the BEES root. BEES loads `.env.bees` automatically (it is gitignored).

## Quick start

```bash
conda activate bees_env
python BEES.py -i projects/minimal/input.yml
```

The examples need the database files in `db/` (not in git). Build them first, see [Database](#database).

| Flag | Meaning |
|------|---------|
| `-i <file>` | Input YAML |
| `-o <dir>` | Output directory |
| `-p <name>` | Project name |
| `-v <level>` | Log level (10/20/30/40/50) |

To build your own model, copy an example (`cp projects/minimal/input.yml my_input.yml`), edit it, and run `python BEES.py -i my_input.yml`.

Examples in `projects/`:

- `minimal/` — first three glycolysis steps, general database.
- `fattyAcidSynthesis/fattyAcidSynthesis_ecoli/` — *E. coli* fatty acid synthesis (FAS II), `ecoli` database.
- `commented/` — annotated template listing every allowed key.
- `example_output/` — sample output files.

## Input YAML

Required top-level blocks: `project`, `species`, `enzymes`, `environment`, `settings`, `database`. Unknown keys are rejected; see `projects/commented/input.yml` for the full key list.

- Each enzyme needs an EC number. Enzymes that use CatPred also need `amino_acid_sequence` (uppercase).
- `settings.simulation_mode`:
  - `iterative` — core–edge enlargement + ODE simulation. Needs at least one of `end_time`, `termination_conversion`, or `termination_rate_ratio`.
  - `batch` — reaction discovery only, no kinetics-driven pruning.

## Output

Written to `<project_directory>/output/` (or `settings.output_directory`):

| File | Content |
|------|---------|
| `reactions_summary.txt` | All generated reactions (core and edge) |
| `output.log` | Run log |
| `simulation_profiles.csv` + plots | Concentration time courses |
| `flux_analysis.csv` | Reaction fluxes |
| `core_reactions_species.csv`, `edge_reactions_species.csv` | Final core and edge sets |
| `model.xml` | SBML Level 3 model |

## Configuration

### Database

Set `database.name` to a file in `db/` without `.csv` (`db` = general BKMS-derived set, `ecoli` = *E. coli*). If the file is missing, `missing_policy: error` (default) stops the run; `fallback_with_warning` uses `db/db.csv` instead.

The two database files are generated and not stored in git (they are gitignored). Build them once after cloning:

| File | Content | Raw input (not shipped) | Full instructions |
|---|---|---|---|
| `db/db.csv` | general reaction database, 37,119 rows (BKMS-react reactions with an EC number; BKMS-only by default) | BKMS-react 2023.1, `Reactions_BKMS.tar.gz` | [db_csv/REBUILD_INSTRUCTIONS.md](db/db_build/db_csv/REBUILD_INSTRUCTIONS.md) |
| `db/ecoli.csv` | *E. coli* K-12 reaction database, 2,327 rows | UniProt UP000000625, 2026-04-20 snapshot | [ecoli_csv/REBUILD_INSTRUCTIONS.md](db/db_build/ecoli_csv/REBUILD_INSTRUCTIONS.md) |

Short version (download the raw input first, as described in each instruction file):

```bash
cd db/db_build/db_csv && python build_db_csv.py --bkms Reactions_BKMS.tar.gz --out ../../db.csv
cd ../ecoli_csv && python build_ecoli_csv.py --uniprot uniprot_UP000000625.tsv.gz --out ../../ecoli.csv
```

Both scripts need only Python 3.10+ (standard library). The E. coli build needs the 2026-04-20 UniProt file; a newer download gives different rows (see its instruction file).

### Kinetics (CatPred)

```yaml
settings:
  estimate_kinetics: true
  kinetics_estimator: catpred
```

Predictions are cached in `~/.cache/bees/catpred_predictions.pkl`: the first run of a network is slow, later runs skip CatPred. Disable with `BEES_CATPRED_CACHE=off`.

### Thermodynamics (equilibrator-api)

ΔG°′ and Keq are computed at the model's pH, ionic strength, pMg, and temperature. The first run downloads data (hundreds of MB) into `~/.cache/equilibrator`. Set `BEES_DISABLE_THERMO=1` to skip this step (all reactions become irreversible).

If a reaction's thermodynamics can't be computed, `settings.thermo_missing_data_policy` decides: `error` (default) stops the run, `warn_assume_irreversible` logs a warning, `silent_assume_irreversible` does not. Affected reactions are listed at the end of the run.

### Rules and calibrations

```yaml
settings:
  calibrations:
    - fabI_enoyl_reductase_measured_kcat
    - tesa_long_chain_preference
```

## Troubleshooting

**CatPred checkpoints not found (`kcat/`, `km/`).** Set `CATPRED_CHECKPOINT_BASE` to the folder that contains them (usually `../catpred_pipeline/data/pretrained/production`), in `.env.bees` or as an environment variable.

**Installing CatPred by hand (without `make install`).** Clone CatPred next to BEES (`../CatPred`, `../catpred_pipeline`), download and extract the pretrained archive into `catpred_pipeline`, and create the `catpred` conda env. Then set `CATPRED_DIR`, `CATPRED_CHECKPOINT_BASE`, and `CATPRED_CONDA_ENV=catpred` in `.env.bees`.

## Development

```bash
conda activate bees_env
pytest tests/ -v    # or: make test
```

## License

MIT — see [LICENSE](LICENSE).
