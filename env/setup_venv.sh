#!/usr/bin/env bash
# ==========================================================================
# autofm - virtual-environment bootstrap
# ==========================================================================
# Creates a Python virtual environment and installs every dependency autofm
# needs, choosing the right PyTorch build for the target:
#
#   * local / CPU  (default)  - the CPU PyTorch wheel; for development and
#                               small smoke runs off-cluster.
#   * Leonardo GPU (--cuda)   - a CUDA 12.1 (cu121) PyTorch build, which runs
#                               against Leonardo's 12.2 driver. The default
#                               "pip install torch" pulls a newer CUDA runtime
#                               (e.g. cu126) that the 12.2 driver cannot run and
#                               that crashes NCCL under DDP - so we pin cu121.
#
# IMPORTANT (Leonardo): run this on the LOGIN node. Compute nodes have no
# internet, so pip cannot download there.
#
# Usage:
#   bash env/setup_venv.sh                 # local CPU venv in ./.venv
#   bash env/setup_venv.sh --cuda          # cu121 GPU venv (Leonardo login node)
#   bash env/setup_venv.sh --venv .venv311 # custom venv path
#   bash env/setup_venv.sh --python python3.11
#   bash env/setup_venv.sh --cuda --venv /leonardo_work/.../env/venv-cu121
#
# Flags:
#   --cuda            install the cu121 GPU torch build (default: CPU build)
#   --venv PATH       venv location (default: .venv)
#   --python BIN      python interpreter to build the venv with (default: python3)
#   --cuda-index URL  override the torch CUDA wheel index (default: cu121)
#   -h | --help       show this help
# ==========================================================================

set -euo pipefail

# --- Defaults --------------------------------------------------------------
VENV_PATH=".venv"
PYTHON_BIN="python3"
USE_CUDA=0
CUDA_INDEX="https://download.pytorch.org/whl/cu121"

# Resolve repo root from this script's location so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_IN="$SCRIPT_DIR/requirements.in"

# --- Parse args ------------------------------------------------------------
usage() { sed -n '2,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)        USE_CUDA=1; shift ;;
    --venv)        VENV_PATH="${2:?--venv needs a path}"; shift 2 ;;
    --python)      PYTHON_BIN="${2:?--python needs an interpreter}"; shift 2 ;;
    --cuda-index)  CUDA_INDEX="${2:?--cuda-index needs a URL}"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "Unknown argument: $1" >&2; echo "Try: bash env/setup_venv.sh --help" >&2; exit 2 ;;
  esac
done

echo "== autofm venv setup =="
echo "  repo root  : $REPO_ROOT"
echo "  venv path  : $VENV_PATH"
echo "  python     : $PYTHON_BIN"
echo "  torch build: $([[ $USE_CUDA -eq 1 ]] && echo "CUDA (cu121)" || echo "CPU")"
echo

# --- Preconditions ---------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: interpreter '$PYTHON_BIN' not found." >&2
  echo "       On Leonardo load one first, e.g.: module load python/3.11" >&2
  exit 1
fi

if [[ ! -f "$REQ_IN" ]]; then
  echo "ERROR: dependency file not found: $REQ_IN" >&2
  exit 1
fi

# Warn on Python versions with poor CUDA-wheel availability (e.g. 3.14).
PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="${PY_VER%%.*}"; PY_MINOR="${PY_VER##*.}"
echo "  python ver : $PY_VER"
if [[ $USE_CUDA -eq 1 && ( $PY_MINOR -ge 13 || $PY_MAJOR -gt 3 ) ]]; then
  echo
  echo "WARNING: Python $PY_VER may have no cu121 PyTorch wheel. If the torch"
  echo "         install fails below, rebuild on Python 3.11 or 3.12:"
  echo "           module load python/3.11"
  echo "           bash env/setup_venv.sh --cuda --python python3.11 --venv .venv311"
  echo
fi

# --- Create the venv -------------------------------------------------------
if [[ -d "$VENV_PATH" ]]; then
  echo "venv already exists at $VENV_PATH - reusing it (installing into it)."
else
  echo "Creating venv at $VENV_PATH ..."
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

VENV_PY="$VENV_PATH/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv python not found at $VENV_PY (venv creation failed?)" >&2
  exit 1
fi

# --- Install -------------------------------------------------------------
echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip >/dev/null

if [[ $USE_CUDA -eq 1 ]]; then
  echo "Installing PyTorch (CUDA build) from $CUDA_INDEX ..."
  # torch + torchvision/torchaudio companions from the CUDA index.
  "$VENV_PY" -m pip install torch --index-url "$CUDA_INDEX"
else
  echo "Installing PyTorch (CPU build) ..."
  "$VENV_PY" -m pip install torch
fi

echo "Installing the remaining dependencies from requirements.in ..."
# torch is already installed above (with the right build); install the rest.
# Filter out torch so we never overwrite the chosen build with the default one.
REST="$(grep -vE '^\s*#|^\s*$' "$REQ_IN" | grep -viE '^\s*torch\s*($|[#])' || true)"
if [[ -n "$REST" ]]; then
  # shellcheck disable=SC2086
  echo "$REST" | sed 's/#.*//' | "$VENV_PY" -m pip install -r /dev/stdin
fi

# Install autofm itself in editable mode if a package config exists, else rely
# on running from the repo root (python -m autofm.*).
if [[ -f "$REPO_ROOT/pyproject.toml" || -f "$REPO_ROOT/setup.py" ]]; then
  echo "Installing autofm (editable) ..."
  "$VENV_PY" -m pip install -e "$REPO_ROOT"
fi

# --- Verify ----------------------------------------------------------------
echo
echo "== Verifying install =="
"$VENV_PY" - <<'PYCHECK'
import importlib
mods = ["numpy", "pandas", "scipy", "sklearn", "yaml", "torch"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok  {m:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  MISSING {m}: {e}")
try:
    import torch
    print(f"  torch cuda built-for : {torch.version.cuda}")
    print(f"  torch cuda available : {torch.cuda.is_available()} "
          "(False off a GPU node / on login node - expected)")
except Exception:
    pass
PYCHECK

echo
echo "== Done =="
echo "Activate with:   source $VENV_PATH/bin/activate"
echo "Or run directly: $VENV_PY -m autofm.pipeline --help"
if [[ $USE_CUDA -eq 1 ]]; then
  echo
  echo "On Leonardo, verify the GPU inside a compute allocation:"
  echo "  srun --gres=gpu:1 --time=00:05:00 --pty bash"
  echo "  $VENV_PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
fi
