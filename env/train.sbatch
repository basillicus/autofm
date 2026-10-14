#!/bin/bash
# ==========================================================================
# autofm - Leonardo (CINECA) SLURM job template
# ==========================================================================
# See docs/ENVIRONMENT_SETUP.md §4.
#
# Before submitting:
#   1. Replace <ACCOUNT>, <SHARED_ENV_PATH>, <PROJECT_PATH> with your values
#      (or source .env and substitute).
#   2. Choose debug vs full run (see QoS note below).
#   3. Adjust --gres for GPU count.
#
# Submit:   sbatch env/train.sbatch
# Monitor:  squeue --me
# Cancel:   scancel <jobid>
# Logs:     logs/autofm-train-<jobid>.out / .err
# ==========================================================================

# --- SLURM directives ------------------------------------------------------
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg       # debug queue: max 30 min. REMOVE for real runs (max 24 h).
#SBATCH --job-name=autofm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 1 A100. Set gpu:4 for whole node.
#SBATCH --time=00:30:00           # keep within QoS limit; raise for real runs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Setup ------------------------------------------------------------------
mkdir -p logs

# Activate the shared team environment
source <SHARED_ENV_PATH>/activate.sh

# Sanity: confirm the GPU is visible
echo "-------- GPU visibility --------"
python -c "
import torch
print('torch      ', torch.__version__)
print('CUDA avail ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device     ', torch.cuda.get_device_name(0))
    print('GPU memory ', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB')
"
echo "---------------------------"

# --- Run --------------------------------------------------------------------
cd <PROJECT_PATH>/<your-project-dir>

# Single-GPU training (default)
python -m autofm.pipeline.train --config config.yaml

# --- Multi-GPU (whole node, 4x A100) --------------------------------------
# Uncomment below and set --gres=gpu:4 above:
#   torchrun --nproc_per_node=4 -m autofm.pipeline.train --config config.yaml
# Get single-GPU correct first.
