#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="attpc-latent"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${REPO_DIR}/environment.yml"

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [ -x /opt/conda/bin/conda ]; then
  CONDA_EXE="/opt/conda/bin/conda"
else
  echo "Could not find conda. Add conda to PATH or install conda before running this script." >&2
  exit 1
fi

if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists." >&2
  echo "This script only creates a fresh repo environment and will not update an existing one." >&2
  echo "To recreate it, remove the existing environment first:" >&2
  echo "  conda env remove -n ${ENV_NAME}" >&2
  exit 1
fi

echo "Creating conda environment '${ENV_NAME}' from ${ENV_FILE}"
"${CONDA_EXE}" env create --file "${ENV_FILE}"

echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
