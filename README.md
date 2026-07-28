# BEES
Biochemical Engine for Enzymatic kinetic modelS

## What BEES does

- Builds biochemical reaction networks (EC-based stoichiometry).
- Pulls reaction templates and stoichiometry from a local database (37K+ reactions).
- Can estimate missing kinetics via [CatPred](https://github.com/DanaResearchGroup/CatPred) when the database has no match.
- Implements the core–edge iterative enlargement algorithm.
- Exports models to SBML (Level 3).

## Installation

**Prerequisites:** Python 3.12+, Conda. For CatPred: `git` and `wget` (or `curl`) on PATH.

### Installation

1. **Clone and enter the repo**

2. **In your terminal run:**
   ```bash
   ./install.sh
   ```
   This creates the `bees_env` conda env, clones CatPred into the **parent** of BEES (`../CatPred`, `../catpred_pipeline`), downloads and extracts the pretrained data (~1 GB), and writes `.env.bees` in the BEES root. If the download or extraction fails, the script exits with a clear error.

3. **Try to run BEES:**
   ```bash
   conda activate bees_env
   python BEES.py -i projects/Glycolysis/input.yml
   ```

4. **Common issue — if the install script reports that it could not find `kcat/` and `km/`:** manually set `CATPRED_CHECKPOINT_BASE` to the directory that contains those folders (often `.../catpred_pipeline/data/pretrained/production`). Put it in `.env.bees` or export it before running; see [Manual CatPred setup](#kinetics-estimation-catpred) below.


## Projects - Quick Start

1. Copy and edit an input file: `cp projects/minimal/input.yml my_input.yml` — set project name, at least one species and one enzyme (with EC number), environment (temperature, pH), and settings (end_time, time_step).
2. Run: `python BEES.py --input_file my_input.yml`
3. Output appears in `output/` in your project folder: `reactions_summary.txt`, logs, simulation plots, and `model.xml`.

Example configs in `projects/`: `minimal/`, `Glycolysis/`, `fattyAcidSynthesis/`, and more. See `projects/Project_folder_README.md` for descriptions.


## SBML Export

After every iterative enlargement run, BEES automatically generates an SBML Level 3 file (`model.xml`) in the project's output directory ready to run.


**Manual CatPred setup (without install.sh):** clone CatPred into a sibling of BEES (e.g. `CatPred` and `catpred_pipeline`), download the pretrained archive into `catpred_pipeline`, extract it, then create the `catpred` conda env. Set `CATPRED_DIR` to the CatPred clone, `CATPRED_CHECKPOINT_BASE` to the directory that contains `kcat/` and `km/` (often `.../catpred_pipeline/data/pretrained/production`), and `CATPRED_CONDA_ENV=catpred`. BEES auto-loads `.env.bees` when present; otherwise export these in your shell before running.


## Thermodynamics (equilibrator-api)

BEES uses [equilibrator-api](https://gitlab.com/equilibrator/equilibrator-api) to compute ΔG°′ for each reaction at the model's pH / ionic strength / pMg, then derives reverse turnover numbers via the Haldane relationship for reversible Michaelis–Menten kinetics. The first import downloads the Component-Contribution training data (~hundreds of MB) into `~/.cache/equilibrator`. Set `BEES_DISABLE_THERMO=1` to skip the thermodynamics layer entirely (every reaction is then treated as irreversible, matching the legacy behavior).

## Development

Run tests: `pytest tests/ -v` (or `make test`).

## License

See LICENSE.
