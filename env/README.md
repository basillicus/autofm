# `env/` - Environment-Specific Files

This directory is the single home for the environment-specific, version-controlled
files for `autofm`, kept out of the repo root to avoid clutter. Anything that
**pins** the stack or **differs between** environments lives here.

| Environment | Role | Hardware |
|-------------|------|----------|
| **Leonardo (CINECA)** | Real training, HPO, scale-out | NVIDIA A100 (CUDA 12.6) |

---

## Table of contents

1. [Contents](#1-contents)
2. [Dependencies](#2-dependencies)
3. [Cluster values (`.env`)](#3-cluster-values-env)
4. [What is *not* here](#4-what-is-not-here)

---

## 1. Contents

| File | Purpose | Source of truth |
|------|---------|-----------------|
| `requirements.in` | Top-level (direct) dependencies, loosely pinned. Edit when adding a package. | This repo |
| `requirements.lock` | Fully-resolved exact pins - the reproducible install. Generated on Leonardo. | Leonardo shared env |
| `train.sbatch` | SLURM training-job template (account / partition / QoS as placeholders). | This repo (template) |
| `.env.example` | Per-user / per-cluster values (SLURM account, cluster paths). Copy to `.env`. | This repo (template) |

The repo-root `requirements.txt` installs the direct deps (`env/requirements.in`)
so `pip install -r requirements.txt` works today; switch it to
`env/requirements.lock` once the lock is populated for a reproducible install.

---

## 2. Dependencies

`requirements.in` holds the **direct** dependencies (edit here). `requirements.lock`
holds the **exact resolved pins** and is the source of truth for versions.
Keep both committed and in sync.

```bash
# Add or change a package
echo "seaborn" >> env/requirements.in                 # 1. add the direct dep
bash <SHARED_ENV_PATH>/setup_env.sh                    # 2. regenerate the lock (on Leonardo login node)
# 3. commit BOTH requirements.in and requirements.lock
```

Never hand-edit `requirements.lock`, and never `pip install` ad hoc into the
shared Leonardo env - both cause silent version drift. Full setup steps are in
[`../docs/ENVIRONMENT_SETUP.md`](../docs/ENVIRONMENT_SETUP.md).

---

## 3. Cluster values (`.env`)

Per-user / per-cluster values (SLURM account, cluster paths) must not be
hard-coded in tracked files. `.env.example` documents them; copy it and fill in
your own:

```bash
cp env/.env.example .env     # .env is git-ignored; only .env.example is committed
```

Model and data settings do **not** go here - they live in `config.yaml` at the
repo root.

---

## 4. What is *not* here

- `activate.sh`, `setup_env.sh` - cluster-side scripts inside the team's shared
  Leonardo env (`<SHARED_ENV_PATH>/`), not project source.
- `config.yaml` - model/data hyperparameters live at the repo root; `env/` is for
  *environment* concerns (dependencies, cluster, paths) only.
- `.venv/` - local virtual environments are git-ignored; never commit them.
