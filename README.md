# BEES
Biochemical Engine for Enzymatic kinetic modelS

## What BEES does

- Builds biochemical reaction networks from enzyme-substrate pairs (EC-based stoichiometry).
- Pulls kinetic parameters (Km, kcat, Vmax, delta-G) from a local database (37K+ reactions from BKMS).
- Can estimate missing kinetics via [CatPred](https://github.com/DanaResearchGroup/CatPred) when the database has no match.

## Installation

**Prerequisites:** Python 3.12+, Conda. For CatPred: `git` and `wget` (or `curl`) on PATH. Use a normal directory (not e.g. inside Trash).

### Clone and install this branch (from zero)

1. **Clone and enter the repo** 

2. **One-shot install (BEES + CatPred):**
   ``
   ./install.sh
   ```
   This creates the `bees_env` conda env, clones CatPred into the **parent** of BEES (`../CatPred`, `../catpred_pipeline`), downloads and extracts the pretrained data (~1 GB) and writes `.env.bees` in the BEES root. If the download or extraction fails, the script exits with a clear error. unlass this is happen don't need to do any more installtion. 

3. **try to Run BEES:**
   ```
   conda activate bees_env
   python BEES.py -i projects/Glycolysis/input.yml
   ```
   

4. **possable bug - If the install script reports that it could not find `kcat/` and `km/`:** set `CATPRED_CHECKPOINT_BASE` manually to the directory that contains those folders (often `.../catpred_pipeline/data/pretrained/production`). Put it in `.env.bees` or export it before running; see [Manual CatPred setup](#kinetics-estimation-catpred) below.


## Quick Start

1. Copy and edit an input file: `cp projects/minimal/input.yml my_input.yml` — set project name, at least one species and one enzyme (with EC number), environment (temperature, pH), `database.name: db`, and settings (end_time, time_step).
2. Run: `python BEES.py --input_file my_input.yml`
3. Output appears in `output/` in your project folder: `reactions_summary.txt`, logs.

## Examples

Example configs in `projects/`: `minimal/`, `Glycolysis/`, `fattyAcidSynthesis/FattyAcidSynthesisDemo/`, `ComprehensiveDemo/`, `commented/` (commented template). See `projects/Project_folder_README.md` for descriptions.



Kinetics are read from `db/db.csv`. In your input YAML set `database.name: db`.

## Kinetics estimation (CatPred)

CatPred runs in a separate conda env. If you used `./install.sh`, BEES auto-loads `.env.bees` when present, so just `conda activate bees_env` and run.

Enable in your input:

```yaml
settings:
  estimate_kinetics: true
  kinetics_estimator: catpred
  kinetics_include_sd: true
  smiles_mode: auto   # or 'interactive' to prompt for missing SMILES
```

Enzymes that need estimation must have `amino_acid_sequence` (or BEES will try to resolve it by EC). When the database has no kinetics, BEES calls CatPred as a subprocess; predictions go into the reaction summary.

**Manual CatPred setup (without install.sh):** clone CatPred into a sibling of BEES (e.g. `CatPred` and `catpred_pipeline`), download the pretrained archive into `catpred_pipeline`, extract it, then create the `catpred` conda env. Set `CATPRED_DIR` to the CatPred clone, `CATPRED_CHECKPOINT_BASE` to the directory that contains `kcat/` and `km/` (often `.../catpred_pipeline/data/pretrained/production`), and `CATPRED_CONDA_ENV=catpred`. BEES auto-loads `.env.bees` when present; otherwise export these in your shell before running.


## Development

Run tests: `pytest tests/ -v` (or `make test`).

## License

See LICENSE.
