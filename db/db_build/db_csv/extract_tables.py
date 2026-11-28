#!/usr/bin/env python3
"""Create the two frozen data tables used by build_db_csv.py.

Run once (already done); kept to show where the tables came from. It reads the
released db/db.csv and writes:

  data/compound_smiles.csv        one row per compound name in the BKMS rows that has
                                  a SMILES: name, smiles, source
  data/curated_fas_reactions.csv  archived: the 56 rows that are not from BKMS
                                  (hand-curated FAS); opt-in via --include-curated-fas
                                  in build_db_csv.py; not in the default build

`source` is "pubchem" when the SMILES equals the first PubChem name lookup
(compounds_with_smiles.csv, 2026-02-08), "manual" for the carbon monoxide fix of
2026-11-03, and "pubchem_second_pass" for the rest (PubChem with cleaned/variant names,
including carrier-protein names the first pass skipped; see fetch_smiles.py).

Usage:
  python extract_tables.py --db ../../db.csv \
      --first-pass compounds_with_smiles.csv

  --first-pass is the first PubChem name-lookup table (columns compound_name, smiles).
  It is not shipped. It came from PubChem name search (https://pubchem.ncbi.nlm.nih.gov,
  PUG REST); fetch_smiles.py uses the same service.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(10**9)

HERE = Path(__file__).resolve().parent
MANUAL = {"CO": "manual"}


def is_bkms(row: dict) -> bool:
    if not row["meta"]:
        return False
    return json.loads(row["meta"]).get("source") == "BKMS-react"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--first-pass", type=Path, required=True)
    args = ap.parse_args()

    with args.db.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    with args.first_pass.open(newline="", encoding="utf-8", errors="replace") as f:
        first_pass = {r["compound_name"]: r["smiles"] for r in csv.DictReader(f)}

    smiles: dict[str, str] = {}
    for row in rows:
        if is_bkms(row) and row["compound_smiles"]:
            smiles.update(json.loads(row["compound_smiles"]))

    data = HERE / "data"
    data.mkdir(exist_ok=True)

    with (data / "compound_smiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "smiles", "source"])
        for name in sorted(smiles):
            if name in MANUAL:
                source = MANUAL[name]
            elif first_pass.get(name) == smiles[name]:
                source = "pubchem"
            else:
                source = "pubchem_second_pass"
            w.writerow([name, smiles[name], source])

    curated = [r for r in rows if not is_bkms(r)]
    with (data / "curated_fas_reactions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(curated)

    print(f"compound_smiles.csv: {len(smiles)} names")
    print(f"curated_fas_reactions.csv: {len(curated)} rows")


if __name__ == "__main__":
    main()
