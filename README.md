# SoftWhere — Reproduction

`reproduce.sh` rebuilds the **SoftWhere** preliminary results (reported in
`SoftWhere_Proposal.pdf`) from scratch: it clones the two project forks at branch
`project-e-2`, builds one shared environment, downloads the data, and runs the full
de-risking spike — gradient-flow proof, multi-foveal visualizations, distillation
sweeps, the teacher-agreement proxy, the map-diversity sweep, and the ADE20K
multi-object coverage eval.

---

## 1. Prerequisites

### Install `uv` (required)
The script builds its environment with [`uv`](https://docs.astral.sh/uv/). **Install it
before running** — the script exits immediately if `uv` is not on your `PATH`.

```bash
# official installer (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
# …or via pip / pipx
pip install uv            # or: pipx install uv

uv --version              # verify it is on PATH
```

### Other tools
`git`, `curl`, `unzip`, `tar` (all standard). The script checks for these and stops
with a clear message if one is missing.

### GPU
A CUDA GPU is expected (the pinned stack is torch 2.12 / CUDA 13). Defaults to GPU 0;
override with `CUDA_VISIBLE_DEVICES` (e.g. a MIG UUID) — see below.

### Repo access
SSH access to the two GitHub forks, on branch `project-e-2`:
`engichang1467/lookwhere` and `engichang1467/Open-TokenLearner`. **Both forks must
already contain the spike code** (the modified `modeling.py` / `tokenlearner/modules.py`
and all the `lookwhere/*.py` scripts). If you maintain those forks, push your latest
code to `project-e-2` first.

---

## 2. Run it

```bash
cd softwhere
./reproduce.sh
```

Common variations (any `UPPER_CASE` config var can be overridden via the environment):

```bash
SKIP_ADE20K=1 ./reproduce.sh                  # skip the ~923MB ADE20K download + coverage
CUDA_VISIBLE_DEVICES=MIG-<uuid> ./reproduce.sh # pin a specific GPU / MIG slice
SOFTWHERE_BASE=/data/se2 ./reproduce.sh        # clone + work somewhere other than ./
DIST_STEPS=300 ./reproduce.sh                  # shorter (less faithful) distillation
```

First run downloads several GB (checkpoint ~440 MB, imagenette ~325 MB, ADE20K
~923 MB) and takes a while; re-runs skip clones/downloads/venv that already exist.

---

## 3. What it outputs

### Console (the numbers)
As it runs you'll see, in order:

| Step | What prints | Expected (≈, ±0.01–0.03) |
|---|---|---|
| OpenTokenLearner tests | pytest summary | `25 passed` |
| Gradient-flow proof | per-param grad norms + verdict | `PASS` (6 head params, no leakage) |
| Teacher-agreement proxy | recall / IoU vs. teacher | recall ~0.42 (random ~0.10) |
| Map-diversity sweep | table over diversity weights | div=1: overlap ~0.02, fidelity ~0.094 |
| v1.0 vs v1.1 | fidelity KL | v1.0 ~0.094 < v1.1 ~0.144 |
| ADE20K coverage | object-recall per selector | LookWhere 0.70 > random 0.61 > SoftWhere 0.47/0.36 (a deliberate negative result — see proposal §5.5) |

### Files (written into the cloned `lookwhere/` repo)
- **Figures**
  - `softwhere_v10_untrained.png`, `softwhere_v11_untrained.png` — multi-foveal maps, untrained
  - `softwhere_v10_distilled.png`, `softwhere_v11_distilled.png` — after distillation (the "money" figure)
- **Distilled selector heads** (`.pt`)
  - `softwhere_head_v10_div{0,0.5,1,2}.pt`, `softwhere_head_v11_div{0,0.5,1,2}.pt` — the diversity sweep
  - `softwhere_head_v10_div1_ade.pt` — in-domain head for the ADE20K eval

The run ends with a `DONE` summary listing these artifacts and the expected headline numbers.

---

## 4. Notes


- **Environment.** One shared `uv` venv (`.venv/`) is built from `requirements.txt`
  with `--no-deps` (the file is a complete pinned freeze of both repos' environments,
  so it installs exactly as listed). It runs both the OpenTokenLearner tests and the
  LookWhere scripts.
- **Reproducibility.** Distillation runs are short and seed/order-sensitive; exact
  decimals vary by ±0.01–0.03 but orderings and conclusions are stable.
