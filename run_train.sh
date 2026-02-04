#!/bin/bash
#SBATCH -J MYGPUJOB
#SBATCH -o MYGPUJOB.o%j
#SBATCH -e MYGPUJOB.e%j
#SBATCH -p gpu
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH -t 5:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-nn}

cd -P "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"

CONDA_EXE="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# optional CUDA module (ignore failures)
if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg  # headless matplotlib

# Avoid JAX grabbing all GPU memory by default (optional, but usually desired on shared nodes)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# -----------------------------
# Install required Python deps
# -----------------------------
python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Install GPU-enabled JAX (fail if GPU not visible)
# -----------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This job step does not see NVIDIA tools/GPU." >&2
  exit 2
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi cannot list GPUs. GPU likely not visible to this job step." >&2
  nvidia-smi || true
  exit 2
fi

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"

# Using pip-provided CUDA/cuDNN wheels is the recommended path; LD_LIBRARY_PATH can shadow them.
unset LD_LIBRARY_PATH

if [ "$DRIVER_MAJOR" -ge 580 ]; then
  JAX_EXTRA="cuda13"
elif [ "$DRIVER_MAJOR" -ge 525 ]; then
  JAX_EXTRA="cuda12"
else
  echo "ERROR: NVIDIA driver ${DRIVER_VERSION} is too old for JAX CUDA wheels (need >=525 for CUDA 12, >=580 for CUDA 13)." >&2
  exit 2
fi

python -m pip install --upgrade "jax[${JAX_EXTRA}]"

# Remaining direct deps used by your scripts (+ Optuna)
python -m pip install --upgrade numpy tqdm matplotlib optax diffrax optuna

# Quick sanity print (must show at least one GPU device) + optuna version
python - <<'PY'
import jax, optuna
print("jax:", jax.__version__)
print("devices:", jax.devices())
print("optuna:", optuna.__version__)
if not any(d.platform in ("gpu", "cuda") for d in jax.devices()):
    raise SystemExit("ERROR: JAX did not initialize a GPU backend (still CPU-only).")
PY

srun python -u train_robertson.py
#srun python -u eval.py
