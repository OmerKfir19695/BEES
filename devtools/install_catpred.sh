#!/usr/bin/env bash
set -euo pipefail

# Clone CatPred, download pretrained data, create catpred env, write .env.bees.


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLONE_ROOT="$(dirname "$REPO_ROOT")"
CATPRED_REPO="${CLONE_ROOT}/CatPred"
CATPRED_PIPELINE="${CLONE_ROOT}/catpred_pipeline"
CATPRED_DATA_URL="${CATPRED_DATA_URL:-https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz}"
CATPRED_GIT_URL="${CATPRED_FORK_URL:-https://github.com/DanaResearchGroup/CatPred.git}"

echo "CatPred clone: $CATPRED_GIT_URL"
echo "Clone root:    $CLONE_ROOT"
echo "CatPred path:  $CATPRED_REPO"

# Avoid interactive prompts (some mamba versions reject -y for env commands)
export CONDA_ALWAYS_YES=true

# Use same package manager as install_bees.sh (micromamba / mamba / conda)
if command -v micromamba &>/dev/null; then
  COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
  COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
  COMMAND_PKG=conda
else
  echo "Conda, Mamba, or Micromamba is required for CatPred env."
  exit 1
fi
echo "Using $COMMAND_PKG for CatPred env"

# Clone
if [[ ! -d "${CATPRED_REPO}/.git" ]]; then
  echo "Cloning CatPred into ${CATPRED_REPO}..."
  git clone "${CATPRED_GIT_URL}" "${CATPRED_REPO}"
else
  echo "CatPred already cloned at ${CATPRED_REPO}."
fi

# Pretrained data
mkdir -p "${CATPRED_PIPELINE}"
TAR_FILE="${CATPRED_PIPELINE}/capsule_data_update.tar.gz"
if [[ ! -f "${TAR_FILE}" ]]; then
  echo "Downloading pretrained data (~1 GB, this may take a few minutes)..."
  if command -v wget &>/dev/null; then
    wget --show-progress -O "${TAR_FILE}" "${CATPRED_DATA_URL}" || { rm -f "${TAR_FILE}"; echo "ERROR: Download failed (wget)."; exit 1; }
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${TAR_FILE}" "${CATPRED_DATA_URL}" || { rm -f "${TAR_FILE}"; echo "ERROR: Download failed (curl)."; exit 1; }
  else
    echo "ERROR: Neither wget nor curl found. Install one and retry."
    exit 1
  fi
fi
# Verify the file is non-empty
if [[ ! -s "${TAR_FILE}" ]]; then
  echo "ERROR: Downloaded archive is empty: ${TAR_FILE}"
  echo "       Delete it and re-run to try again."
  exit 1
fi

if [[ ! -d "${CATPRED_PIPELINE}/kcat" ]] && [[ ! -d "${CATPRED_PIPELINE}/production" ]]; then
  echo "Extracting pretrained data..."
  (cd "${CATPRED_PIPELINE}" && tar -xzf capsule_data_update.tar.gz) || {
    echo "ERROR: Extraction failed. The archive may be corrupt."
    echo "       Delete ${TAR_FILE} and re-run to download again."
    exit 1
  }
fi

# Checkpoint base — find the directory that actually contains kcat/ and km/
CHECKPOINT_BASE=""
for candidate in \
    "${CATPRED_PIPELINE}/data/pretrained/production" \
    "${CATPRED_PIPELINE}/production" \
    "${CATPRED_PIPELINE}/capsule_data_update/production" \
    "${CATPRED_PIPELINE}/capsule_data_update" \
    "${CATPRED_PIPELINE}"; do
  if [[ -d "${candidate}/kcat" ]] && [[ -d "${candidate}/km" ]]; then
    CHECKPOINT_BASE="${candidate}"
    break
  fi
done

if [[ -z "${CHECKPOINT_BASE}" ]]; then
  echo "ERROR: Could not find kcat/ and km/ checkpoint directories under ${CATPRED_PIPELINE}."
  echo "       Archive contents:"
  find "${CATPRED_PIPELINE}" -maxdepth 5 -type d | head -50
  echo ""
  echo "       Set CATPRED_CHECKPOINT_BASE manually to the directory containing kcat/ and km/"
  echo "       (e.g. .../catpred_pipeline/data/pretrained/production), then run BEES."
  exit 1
fi

# CatPred conda env
if ! $COMMAND_PKG run -n catpred python -c "import sys; sys.exit(0)" 2>/dev/null; then
  echo "Creating conda environment: catpred"
  $COMMAND_PKG env create -f "${CATPRED_REPO}/environment.yml" -n catpred
fi
echo "Installing CatPred package in env catpred..."
(cd "${CATPRED_REPO}" && $COMMAND_PKG run -n catpred pip install -e . -q)

# Resolve the catpred env's Python path and conda binary for .env.bees
CATPRED_PYTHON_PATH=$($COMMAND_PKG run -n catpred python -c "import sys; print(sys.executable)" 2>/dev/null || true)
CONDA_BIN_PATH=$(command -v "$COMMAND_PKG")

# Env file for BEES
ENV_FILE="${REPO_ROOT}/.env.bees"
cat > "${ENV_FILE}" << EOF
# Auto-loaded by BEES.py when present; no need to source manually.
export CATPRED_DIR="${CATPRED_REPO}"
export CATPRED_CHECKPOINT_BASE="${CHECKPOINT_BASE}"
export CATPRED_CONDA_ENV=catpred
export CATPRED_CONDA_BIN="${CONDA_BIN_PATH}"
export CATPRED_PYTHON="${CATPRED_PYTHON_PATH}"
EOF
echo "Wrote ${ENV_FILE}"
echo "CatPred install complete. CATPRED_CHECKPOINT_BASE=${CHECKPOINT_BASE}"
