#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------
# INSTALLATION OF BEES and optional external dependencies (CatPred)
#
# Options:
#   --no-catpred   Skip CatPred (BEES only, no kinetics estimation)
#   --help, -h     Show this help 
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEVTOOLS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

run_devtool() { bash "$DEVTOOLS_DIR/$1" "${@:2}"; }

SKIP_CATPRED=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-catpred) SKIP_CATPRED=true ;;
    --help|-h)
      echo "Usage: $0 [--no-catpred]"
      echo "  --no-catpred   Skip CatPred (BEES env only)."
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

cd "$REPO_ROOT"
echo "BEES repo root: $REPO_ROOT"

# 1) BEES conda environment
echo "=== Installing BEES ==="
run_devtool install_bees.sh

# 2) CatPred (optional, needed for kinetics estimation)
if [[ "$SKIP_CATPRED" == false ]]; then
  echo "=== Installing CatPred ==="
  run_devtool install_catpred.sh
else
  echo "Skipping CatPred (--no-catpred)."
fi

echo "Done installing BEES and dependencies."
