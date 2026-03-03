# memory-vortex
Symbolic replay scheduler for continual learning — discover compact closed-form replay policies from training traces (GCA + DIFE hybrid).

# Memory Vortex v1.0
**Symbolic replay scheduler for continual / multi-task learning — discover compact closed-form replay policies from training traces (GCA + DIFE hybrid).**

Memory Vortex is a **lightweight controller** that outputs a replay fraction `r ∈ [0,1]` (how much old data to rehearse) as a **compact closed-form operator**.  
It also includes a **discovery engine** that can fit such operators from training traces using a blocked train/val/test protocol.

> ✅ Research prototype: useful for experimentation, ablation baselines, and fast iteration.  
> ❌ Not a guarantee of “zero forgetting,” not a full continual-learning algorithm, and not yet modality-aware (the default scheduler is modality-agnostic).

---

## Authorship / Credits
Created by **Adem Vessell** with collaborative assistance from:
- **ChatGPT 4.2 Thinking**
- **Grok 4.20 Beta 2**

(Implementation and evaluation are intended to be reproducible and reviewable.)

---

## What it does
- Evaluates a **replay schedule** in constant time:
  - `r = 0.0` → no replay
  - `r = 0.3` → mix ~30% replay examples into the training pool
- Provides a **fit-from-logs** pipeline:
  - fixed symbolic basis (sin/cos/exp/log/time)
  - ridge regression with **blocked train/val/test split**
  - exports an operator JSON you can load later
- Includes a **reproducible toy continual-learning benchmark**:
  - sequential Gaussian tasks
  - incremental logistic regression training
  - replay sampling from previous tasks
  - compares **no replay / constant replay / random replay / vortex**

---

## What it is (and isn’t)

### ✅ Is
- A compact, inspectable **replay controller**.
- A small “symbolic regression” pipeline to fit schedules from traces.
- A baseline harness to compare schedules honestly.

### ❌ Is not
- A memory store, retrieval system, or buffer replacement by itself.
- A proof of catastrophic-forgetting elimination.
- Production-ready for large-scale VLM/LLM training.

---

## Core concept
Instead of hand-tuning replay ratios, we can often **compress** an observed “replay need” signal into a small analytic operator.

A typical operator looks like:

\[
r(t) = \text{clip}_{[0,1]}\Big(b + \sum_i c_i \phi_i(t)\Big)
\]

with `t = task_n / t_scale` and basis terms such as:
- `sin(t)`, `cos(t)`, `sin(2t)`, `cos(2t)`
- `exp(-0.1t)`
- `t`
- `log(1+|sin(t)|)`

If `memory_vortex_operator.json` is missing, the scheduler uses a safe fallback operator (documented in the code).

---

## Install
```bash
pip install numpy sympy

## Run

```bash
pip install numpy sympy
python memory_vortex.py


