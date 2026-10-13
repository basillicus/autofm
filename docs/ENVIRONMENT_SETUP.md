# autofm — Environment Setup Guide

Step-by-step setup for the two environments we use:

1. **Local — macOS (Apple Silicon, CPU)** — development, data generation, small runs.
2. **Leonardo (CINECA) — GPU / HPC** — real training on A100s, via the team's
   shared, pre-built environment.

> **Golden rule — match the team's versions.** The shared Leonardo environment
> defines the versions of record (Python 3.14, PyTorch 2.14). Your local Mac should
> match them so a model behaves the same in both places. The only difference is the
> PyTorch build: a CPU build on the Mac, a CUDA build on Leonardo. Everything local
> runs inside a **virtual environment (venv)** — never the system Python.

---

## 0. Version matrix

These mirror the shared Leonardo environment (`env/requirements.lock` on the
cluster is the authoritative pinned list).

| Component | Version | Notes |
|-----------|---------|-------|
| Python | **3.14** | Matches the shared Leonardo env. |
| PyTorch (`torch`) | **2.14** | Leonardo: CUDA 12.6 build. Mac: CPU build. |
| PyTorch Geometric | latest for torch 2.14 | Pure-Python core; included in the shared env. |
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

## 1. Local setup — macOS (Apple Silicon, CPU)

Your Mac has **no NVIDIA GPU and no CUDA** (Apple Silicon is arm64). Locally we
install the **CPU build** of PyTorch. The Mac is for development, generating data,
and small smoke-test runs. Real GPU training happens on Leonardo.

### 1.1 Install Python 3.14

```bash
python3 --version          # check what you have
brew install python@3.14   # if you don't have 3.14
```

### 1.2 Create and activate the venv

From the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
which python               # should end in .venv/bin/python
python --version           # Python 3.14.x
python -m pip install --upgrade pip setuptools wheel
```

### 1.3 Install PyTorch (CPU build) + libraries

On macOS the default PyPI wheel is the CPU/MPS build — no special index URL:

```bash
pip install torch
pip install \
  torch_geometric \
  numpy pandas scipy scikit-learn xgboost mapie \
  matplotlib networkx pyyaml tabulate tensorboard optuna
```

> **Keep local in step with Leonardo.** To match the cluster exactly, copy
> `env/requirements.lock` from Leonardo and `pip install -r requirements.lock`
> locally (dropping any CUDA-only pins). The closer the versions, the fewer
> "works on my machine" surprises.

> **Note on MPS:** PyTorch on Apple Silicon can use Apple's **MPS** (Metal) backend
> for light local acceleration (`device = "mps"`). This is **not** CUDA and not a
> substitute for Leonardo. Our code selects the device at runtime
> (`cuda` → `mps` → `cpu`) so the same code runs unchanged everywhere.

### 1.4 Verify

```bash
python - <<'PY'
import torch, torch_geometric, sklearn, xgboost
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())      # False on Mac — expected
print("mps available:", torch.backends.mps.is_available())
print("pyg", torch_geometric.__version__)
PY
```

Expected on the Mac: `cuda available: False`, `mps available: True`.

---

## 2. Leonardo (CINECA) — use the team's shared environment

**Do not build your own environment on Leonardo.** The team already has a shared,
tested environment (Python 3.14, PyTorch 2.14 / CUDA 12.6, PyG, and all our
libraries), validated on an A100 node. Use it.

> **Placeholders — replace with your own details.** This guide uses placeholders for
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

### 2.1 Log in

```bash
ssh <your-username>@login.leonardo.cineca.it
```

### 2.2 Activate the shared environment

In an interactive shell or inside a job script:

```bash
source <SHARED_ENV_PATH>/activate.sh
```

That's the whole setup — no venv to create, no modules to load, no pip installs.

### 2.3 Quick interactive GPU session (debug queue, 30 min)

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

### 2.4 Need another package?

Don't `pip install` ad hoc into the shared env. Instead:

1. Add the package to `env/requirements.in` (one per line, e.g. `seaborn` or `seaborn==0.14.0`).
2. On a **login node**, rebuild the env (takes a few minutes):
   ```bash
   bash <SHARED_ENV_PATH>/setup_env.sh
   ```

---

## 3. Running training on Leonardo (SLURM)

Use the shared env's `env/gpu_check.sbatch` as a template — it already has the right
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

## 4. Leonardo gotchas (read before your first job)

These are real constraints of this specific cluster/environment — ignoring them
causes most first-time failures.

- **Compute nodes have NO internet.** Download any data/models on the **login node**
  first. Never put `pip install` or a download inside a job script.
- **Don't train on the login node.** Always use `sbatch` or `srun`.
- **The GPU driver is old (CUDA 12.2).** Plain PyTorch (eager mode) works fine.
  - `torch.compile` may not work → **use eager mode.**
  - **XGBoost on GPU may not work → use CPU XGBoost.**
- **`torch.jit.script` is not supported on Python 3.14.** Ignore the warning, or use
  `torch.compile` where appropriate.
- **`$FAST` is full.** Put data and outputs in `<PROJECT_PATH>` (permanent) or
  `$CINECA_SCRATCH` (auto-deleted after 40 days). Not `$FAST`.
- **First `import torch` ~1 min** (shared filesystem) — expected, not a hang.

---

## 5. Keeping the two environments aligned

The shared Leonardo env is the source of truth (`env/requirements.lock`). Keep the
local Mac close to it — the only legitimate difference is the torch build (CPU on Mac
vs. CUDA 12.6 on Leonardo). If a model behaves differently across the two, a version
drift is the first thing to check.

---

## 6. Quick reference

| Task | Local (Mac CPU) | Leonardo (GPU) |
|------|-----------------|----------------|
| Set up | `python3.14 -m venv .venv` + pip install | nothing — use the shared env |
| Activate | `source .venv/bin/activate` | `source <SHARED_ENV_PATH>/activate.sh` |
| Install torch | `pip install torch` (CPU) | already in the shared env |
| Run training | `python -m autofm.pipeline.train` | `sbatch myjob.sbatch` |
| Interactive GPU | n/a (CPU/MPS only) | `srun -A <ACCOUNT> -p boost_usr_prod -q boost_qos_dbg --gres=gpu:1 -c 8 --time=00:30:00 --pty bash` |
| Add a package | `pip install <pkg>` | add to `requirements.in`, run `setup_env.sh` on login node |
| Where to put data | project folder | `<PROJECT_PATH>` or `$CINECA_SCRATCH` |

---

*Team AFM_Ericsson — European AI Hackathon 2026*
