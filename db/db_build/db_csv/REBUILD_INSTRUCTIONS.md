# Rebuild `db/db.csv` (BEES general reaction database)

`build_db_csv.py` rebuilds `db/db.csv` from the BKMS-react download. **Default
is BKMS-only** (37,119 rows with an EC number; SHA-256
`942cb60c...51c6c4bf`). Standard-library Python 3.10+, no internet.

The BEES v1.0.0 paper file also appended 56 hand-curated FAS rows (SHA-256
`4ab531f5...481586e`). Those rows are **not** in the default build; recover them
with `--include-curated-fas` (archive only).

## 1. Download the raw input (not shipped)
- BKMS-react release 2023.1, file `Reactions_BKMS.tar.gz`, from
  https://bkms.brenda-enzymes.org/download.php
- Licence: https://bkms.brenda-enzymes.org/license.php (no redistribution, so the file is not in this repo).
- SHA-256 of the file the release was built from:
  `4f13d9e9bba0457a88cbc27da35ce37d3cae149e545832cee75ceaf5d285a186`
  (check with `sha256sum Reactions_BKMS.tar.gz`). The `.csv` inside the archive works too.

## 2. Build and check

Run from this folder (`db/db_build/db_csv/`):

```bash
# Default: BKMS-only (current general DB)
python build_db_csv.py --bkms Reactions_BKMS.tar.gz --out ../../db.csv

# Optional: BEES v1.0.0 byte-identical (BKMS + 56 curated FAS)
python build_db_csv.py --bkms Reactions_BKMS.tar.gz --out db_v1.0.0.csv --include-curated-fas
```

With `--include-curated-fas`, the script prints `Identical to BEES v1.0.0 db.csv`
when the SHA-256 matches `4ab531f5...481586e`.

## What the build does
- Reads the BKMS reaction table (42,539 reactions).
- Drops the 5,420 reactions without an EC number; keeps 37,119.
- Parses each equation into a stoichiometry (reactants negative, products positive).
- Adds SMILES from `data/compound_smiles.csv`.
- Does **not** append curated FAS by default.

## Files
| File | Content |
|---|---|
| `build_db_csv.py` | the build |
| `data/compound_smiles.csv` | 18,050 compound names with SMILES: 17,995 from PubChem name search, 54 from a second PubChem pass with cleaned names, 1 manual (CO) |
| `data/curated_fas_reactions.csv` | **Archived.** 56 hand-curated fatty-acid-synthesis reactions used only in BEES v1.0.0 (`--include-curated-fas`). Not BKMS-sourced; not used by the FAS paper (that run uses `ecoli.csv`). First added 2026-05-05 (commit `9f9b5aae`). 36 rows have index `fas_ppant_9000xx` and meta `curated-fas-ppant`; 20 have empty index and meta |
| `fetch_smiles.py` | how the SMILES table was made (PubChem lookup). Not needed for the build; PubChem changes over time, so a rerun gives a close but not identical table (`--compare` reports agreement). A full run is ~18,000 web requests (hours) |
| `extract_tables.py` | one-time script that cut the two data tables out of the released `db.csv` |

## Data sources and licences
| Source | Used for | Terms |
|---|---|---|
| BKMS-react (Lang et al. 2011, Bioinformatics) | all 37,119 reactions | BKMS licence (https://bkms.brenda-enzymes.org/license.php): the data may not be redistributed without permission, so it is not shipped here; each user downloads it |
| PubChem (Kim et al., NAR; https://pubchem.ncbi.nlm.nih.gov) | the 18,049 SMILES labelled `pubchem` / `pubchem_second_pass` | NCBI public data, free to reuse; please cite PubChem |

## Notes
- SMILES were taken as returned by PubChem name search and not curated. About 37,000 of
  169,000 compound mentions have none (generic names, proteins, polymers).
- Some second-pass SMILES are only partial structures. For carrier-protein compounds
  PubChem returns the acyl thioacid stub, for example (3R)-3-hydroxyacyl-[acyl-carrier
  protein] -> `C(CO)C(=O)S`. They are kept because the paper version used them.
- The FAS case study uses `database.name: ecoli` (`db/ecoli.csv`), not this general file.
  Removing the 56 curated rows does not change FAS paper figures.

## Recover the v1.0.0 file without BKMS

`db/db.csv` is gitignored. The v1.0.0 file (with the 56 curated rows) is the blob at
`7eb568f^`. From the repo root:

```bash
git show 7eb568f^:db/db.csv > db/db.csv
sha256sum db/db.csv
```

This overwrites the local file. The hash must start `4ab531f5`.
