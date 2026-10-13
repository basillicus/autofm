# autofm - Environment Setup Guide

Step-by-step setup for the environments we use:

1. **Leonardo (CINECA) - GPU / HPC** - real training on A100s, via the team's
   shared, pre-built environment.

> **Golden rule - match the team's versions.** The shared Leonardo environment
> defines the versions of record. Your local machine
> should match them so a model behaves the same in both places. The only difference
> is the PyTorch build: a CPU build locally, a CUDA build on Leonardo. Everything
> local runs inside a **virtual environment (venv)** - never the system Python.

---

## Table of contents

1. [Version matrix](#1-version-matrix)
2. [Local setup (CPU)](#2-local-setup-cpu)
3. [Leonardo (CINECA) - use the team's shared environment](#3-leonardo-cineca---use-the-teams-shared-environment)
4. [Running training on Leonardo (SLURM)](#4-running-training-on-leonardo-slurm)
5. [Leonardo gotchas](#5-leonardo-gotchas-read-before-your-first-job)
6. [Keeping the two environments aligned](#6-keeping-the-two-environments-aligned)
7. [Quick reference](#7-quick-reference)

---

## 1. Version matrix

These mirror the shared Leonardo environment (`env/requirements.lock` on the
cluster is the authoritative pinned list).

| Component | Version | Notes |
|-----------|---------|-------|
| Python | per shared env | Matches the shared Leonardo env. |
| PyTorch (`torch`) | per shared env | Leonardo: CUDA build. Local: CPU build. |
| PyTorch Geometric | matching torch | Pure-Python core; included in the shared env. |
| numpy, pandas, scipy | per `requirements.lock` | |
| scikit-learn | per `requirements.lock` | |
| xgboost | per `requirements.lock` | **CPU only on Leonardo** (see gotchas). |
| mapie | per `requirements.lock` | Conformal prediction. |
| matplotlib, networkx | per `requirements.lock` | |
| optuna, tensorboard | per `requirements.lock` | HPO + training monitoring. |

> **Authoritative list:** on Leonardo, the exact pinned versions live in
> `<SHARED_ENV_PATH>/requirements.lock`. When in doubt, that file wins over this
> table.

---

## 2. Local setup (CPU)

If your local machine has **no NVIDIA GPU / CUDA**, install the **CPU build** of
PyTorch. Local is for development, data preparation, and small smoke-test runs.
Real GPU training happens on Leonardo.

### 2.1 Install Python

```bash
python3 --version          # check what you have
# install the Python version used by the shared Leonardo env (your OS package manager)
```

### 2.2 Create and activate the venv

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
which python               # should end in .venv/bin/python
python --version           # should match the shared Leonardo env
python -m pip install --upgrade pip setuptools wheel
```

### 2.3 Install PyTorch (CPU build) + libraries

```bash
pip install torch          # CPU build (use the official index URL if your OS needs it)
pip install -r requirements.txt
```

> **Keep local in step with Leonardo.** To match the cluster exactly, copy
> `env/requirements.lock` from Leonardo and `pip install -r env/requirements.lock`
> locally (dropping any CUDA-only pins). The closer the versions, the fewer
> "works on my machine" surprises.

> **Note on device selection.** The code selects the device at runtime
> (`cuda` → `cpu`, and other backends where available), so the same code runs
> unchanged locally and on Leonardo.

### 2.4 Verify

```bash
python - <<'PY'
import torch, torch_geometric, sklearn, xgboost
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())      # False locally - expected
print("pyg", torch_geometric.__version__)
PY
```

Expected locally: `cuda available: False`.

---

## 3. Leonardo (CINECA) - use the team's shared environment

**Do not build your own environment on Leonardo.** The team already has a shared,
tested environment (Python, PyTorch / CUDA, PyG, and all our
libraries), validated on an A100 node. Use it.

> **Placeholders - replace with your own details.** This guide uses placeholders for
> anything project-specific; substitute your actual values (get them from your team
> lead / cluster allocation):
> - `<ACCOUNT>` → your SLURM project account.
> - `<SHARED_ENV_PATH>` → the absolute path to the team's shared env directory.
> - `<PROJECT_PATH>` → your permanent project/work directory on the cluster.

| Fact | Value |
|------|-------|
| Project account | `<ACCOUNT>` |
| Shared env path | `<SHARED_ENV_PATH>` |
| Activate script | `<SHARED_ENV_PATH>/activate.sh` |
| Pinned package list | `<SHARED_ENV_PATH>/requirements.lock` |
| GPU partition | `boost_usr_prod` (A100, up to 4 GPUs/node) |
| Debug QoS | `boost_qos_dbg` (max 30 min) |
| Full-run QoS | default (max 24 h) |
| Permanent storage | `<PROJECT_PATH>` |
| Scratch storage | `$CINECA_SCRATCH` (auto-deleted after 40 days) |

### 3.1 Log in

```bash
ssh <your-username>@login.leonardo.cineca.it
```

### 3.2 Activate the shared environment

In an interactive shell or inside a job script:

```bash
source <SHARED_ENV_PATH>/activate.sh
```

That's the whole setup - no venv to create, no modules to load, no pip installs.

### 3.3 Quick interactive GPU session (debug queue, 30 min)

```bash
srun -A <ACCOUNT> -p boost_usr_prod -q boost_qos_dbg \
     --gres=gpu:1 -c 8 --time=00:30:00 --pty bash

# then, on the GPU node:
source <SHARED_ENV_PATH>/activate.sh
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())   # True here
print("device:", torch.cuda.get_device_name(0))       # NVIDIA A100
PY
exit
```

> The **first `import torch` can take ~1 minute** (shared filesystem). That's normal.

### 3.4 Need another package?

Don't `pip install` ad hoc into the shared env. Instead:

1. Add the package to `env/requirements.in` (one per line, e.g. `seaborn` or `seaborn==0.14.0`).
2. On a **login node**, rebuild the env (takes a few minutes):
   ```bash
   bash <SHARED_ENV_PATH>/setup_env.sh
   ```

---

## 4. Running training on Leonardo (SLURM)

Use the repo's `env/train.sbatch` as a template - it already has the right
account and partition. A training job looks like:

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg       # debug: max 30 min. REMOVE this line for real runs (max 24 h).
#SBATCH --job-name=autofm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # up to 4 A100s per node
#SBATCH --time=00:30:00           # within the debug limit; raise for real runs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Activate the shared team environment
source <SHARED_ENV_PATH>/activate.sh

# Sanity line in the log
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- training command ---
cd <PROJECT_PATH>/<your-project-dir>
python -m autofm.pipeline.train --config config.yaml
```

Submit and manage:

```bash
sbatch myjob.sbatch        # submit
squeue --me                # check status
scancel <jobid>            # cancel
```

> **Multi-GPU (whole node, 4× A100):** set `--gres=gpu:4` and launch with
> `torchrun --nproc_per_node=4 -m autofm.pipeline.train ...`. Get a single-GPU run
> correct first.

---

## 5. Leonardo gotchas (read before your first job)

These are real constraints of this specific cluster/environment - ignoring them
causes most first-time failures.

- **Compute nodes have NO internet.** Download any data/models on the **login node**
  first. Never put `pip install` or a download inside a job script.
- **Don't train on the login node.** Always use `sbatch` or `srun`.
- **The GPU driver is old (CUDA 12.2).** Plain PyTorch (eager mode) works fine.
  - `torch.compile` may not work → **use eager mode.**
  - **XGBoost on GPU may not work → use CPU XGBoost.**
- **`torch.jit.script` may not be supported on the shared env's Python.** Ignore the warning, or use
  `torch.compile` where appropriate.
- **`$FAST` is full.** Put data and outputs in `<PROJECT_PATH>` (permanent) or
  `$CINECA_SCRATCH` (auto-deleted after 40 days). Not `$FAST`.
- **First `import torch` ~1 min** (shared filesystem) - expected, not a hang.

---

## 6. Keeping the two environments aligned

The shared Leonardo env is the source of truth (`env/requirements.lock`). Keep the
local machine close to it - the only legitimate difference is the torch build (CPU
locally vs. CUDA 12.6 on Leonardo). If a model behaves differently across the two, a
version drift is the first thing to check.

---

## 7. Quick reference

| Task | Local (CPU) | Leonardo (GPU) |
|------|-------------|----------------|
| Set up | `python3 -m venv .venv` + pip install | nothing - use the shared env |
| Activate | `source .venv/bin/activate` | `source <SHARED_ENV_PATH>/activate.sh` |
| Install torch | `pip install torch` (CPU) | already in the shared env |
| Run training | `python -m autofm.pipeline.train` | `sbatch myjob.sbatch` |
| Interactive GPU | n/a (CPU only) | `srun -A <ACCOUNT> -p boost_usr_prod -q boost_qos_dbg --gres=gpu:1 -c 8 --time=00:30:00 --pty bash` |
| Add a package | `pip install <pkg>` | add to `requirements.in`, run `setup_env.sh` on login node |
| Where to put data | project folder | `<PROJECT_PATH>` or `$CINECA_SCRATCH` |

---

*Team AFM_Ericsson - European AI Hackathon 2026*
