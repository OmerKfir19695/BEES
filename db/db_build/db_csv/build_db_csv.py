#!/usr/bin/env python3
"""Build the BEES general reaction database (db/db.csv) from the BKMS-react download.

Source data (not included; download it yourself, see the BKMS-react license):
  Reactions_BKMS.tar.gz, BKMS-react release 2023.1,
  https://bkms.brenda-enzymes.org/download.php

Data files shipped with this script:
  data/compound_smiles.csv        compound name -> SMILES lookup (frozen, see REBUILD_INSTRUCTIONS.md)
  data/curated_fas_reactions.csv  archived: 56 hand-curated FAS rows used only in BEES v1.0.0
                                  (opt in with --include-curated-fas; not in the default build)

Steps:
  1. Read the BKMS reaction table (tab-separated, inside the tar.gz).
  2. Keep reactions that have an EC number and an equation of the form "A + B = C + D".
  3. Turn each equation into a stoichiometry: reactants negative, products positive,
     a leading integer is a coefficient ("2 NADPH"), species on both sides cancel.
  4. Attach SMILES for every species found in the lookup table.
  5. Write db.csv (BKMS-only by default).

Usage:
  python build_db_csv.py --bkms Reactions_BKMS.tar.gz --out db.csv
  python build_db_csv.py --bkms Reactions_BKMS.tar.gz --out db.csv --include-curated-fas  # v1.0.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

csv.field_size_limit(10**9)

HERE = Path(__file__).resolve().parent
SMILES_TABLE = HERE / "data" / "compound_smiles.csv"
CURATED_TABLE = HERE / "data" / "curated_fas_reactions.csv"

# SHA-256 of db/db.csv in BEES v1.0.0 (paper tag; BKMS + 56 curated FAS).
V1_0_0_SHA256 = "4ab531f56d2528e50fa9e735c0b052e86ce17a611c592f108635167fb481586e"

COLUMNS = ["index", "enzyme", "ec_number", "reaction", "stoichiometry", "meta", "compound_smiles"]

PLUS = re.compile(r"\s+\+\s+")
EQUALS = re.compile(r"\s*=\s*")
COEFFICIENT = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def read_bkms(path: Path) -> list[dict]:
    """Return the BKMS reaction table as a list of dicts (accepts .tar.gz or the .csv inside)."""
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("Reactions_BKMS.csv"))
            text = tar.extractfile(member).read().decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter="\t"))


def parse_species(token: str) -> tuple[int, str]:
    """'2 NADPH' -> (2, 'NADPH'); 'NADPH' -> (1, 'NADPH')."""
    match = COEFFICIENT.match(token)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return 1, token.strip()


def parse_side(side: str) -> list[tuple[int, str]]:
    species = [parse_species(t) for t in PLUS.split(side.strip()) if t.strip()]
    return [(n, name) for n, name in species if name]


def parse_reaction(equation: str) -> dict[str, int] | None:
    """Equation text -> {species: coefficient}, or None if it cannot be parsed."""
    sides = EQUALS.split(equation.strip(), maxsplit=1)
    if len(sides) != 2:
        return None
    reactants, products = parse_side(sides[0]), parse_side(sides[1])
    if not reactants or not products:
        return None

    stoichiometry: dict[str, int] = {}
    for n, name in reactants:
        stoichiometry[name] = stoichiometry.get(name, 0) - n
    for n, name in products:
        stoichiometry[name] = stoichiometry.get(name, 0) + n
    return {name: n for name, n in stoichiometry.items() if n != 0}


def bkms_to_db_row(bkms: dict, smiles: dict[str, str]) -> dict | None:
    ec = bkms["EC_Number"].strip()
    equation = (bkms["Reaction"] or "").strip()
    if not ec or not equation:
        return None
    stoichiometry = parse_reaction(equation)
    if stoichiometry is None:
        return None

    meta = {
        "source": "BKMS-react",
        "id": bkms["ID"],
        "ec": bkms["EC_Number"],
        "rxn_ids": {
            "BRENDA": bkms["Reaction_ID_BRENDA"],
            "KEGG": bkms["Reaction_ID_KEGG"],
            "MetaCyc": bkms["Reaction_ID_MetaCyc"],
            "SABIO_RK": bkms["Reaction_ID_SABIO_RK"],
        },
        "stoichiometry_check": bkms["Stoichiometry_Check"],
        "missing_substrate": bkms["Missing_Substrate"],
        "missing_product": bkms["Missing_Product"],
        "reaction": equation,
    }
    compound_smiles = {name: smiles[name] for name in sorted(stoichiometry) if name in smiles}

    return {
        "index": bkms["ID"],
        "enzyme": bkms["Recommended_Name"].strip(),
        "ec_number": f"EC {ec}",
        "reaction": equation,
        "stoichiometry": json.dumps(stoichiometry, ensure_ascii=False, separators=(",", ":")),
        "meta": json.dumps(meta, ensure_ascii=False),
        "compound_smiles": json.dumps(compound_smiles, ensure_ascii=False) if compound_smiles else "",
    }


def load_smiles(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row["smiles"] for row in csv.DictReader(f)}


def load_curated(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BEES db.csv from BKMS-react.")
    ap.add_argument("--bkms", type=Path, required=True, help="Reactions_BKMS.tar.gz (or the extracted .csv)")
    ap.add_argument("--out", type=Path, default=Path("db.csv"))
    ap.add_argument(
        "--include-curated-fas",
        action="store_true",
        help="Append archived data/curated_fas_reactions.csv (BEES v1.0.0 only; not the default)",
    )
    args = ap.parse_args()

    bkms = read_bkms(args.bkms)
    smiles = load_smiles(SMILES_TABLE)
    rows = [r for r in (bkms_to_db_row(b, smiles) for b in bkms) if r is not None]
    curated = load_curated(CURATED_TABLE) if args.include_curated_fas else []

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows + curated)
    data = buffer.getvalue().encode("utf-8")
    args.out.write_bytes(data)

    digest = hashlib.sha256(data).hexdigest()
    print(f"BKMS reactions read:     {len(bkms)}")
    print(f"BKMS reactions kept:     {len(rows)}")
    print(f"Curated FAS reactions:   {len(curated)}" + (" (opt-in)" if curated else " (skipped; default)"))
    print(f"Wrote {args.out} ({len(rows) + len(curated)} rows)")
    print(f"SHA-256 {digest}")
    if args.include_curated_fas:
        print(
            "Identical to BEES v1.0.0 db.csv"
            if digest == V1_0_0_SHA256
            else "DIFFERS from BEES v1.0.0 db.csv"
        )
    else:
        print("BKMS-only build (default). Use --include-curated-fas to match BEES v1.0.0.")


if __name__ == "__main__":
    main()
