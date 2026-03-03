
---

### 2) `memory_vortex.py` (paste this into the file **as-is**)
```python
import numpy as np
import sympy as sp
import json
import time
import os
from typing import Dict, Tuple

# ====================== GLOBALS ======================
master_rng = np.random.default_rng(42)  # master seed for reproducibility

BASIS_ORDER = [
    "sin(t)", "cos(t)", "sin(2t)", "cos(2t)", "exp(-0.1t)", "t", "log(1+|sin(t)|)"
]

def eval_basis_numeric(t: float) -> np.ndarray:
    return np.array([
        np.sin(t),
        np.cos(t),
        np.sin(2.0 * t),
        np.cos(2.0 * t),
        np.exp(-0.1 * t),
        t,
        np.log(1.0 + np.abs(np.sin(t))),
    ], dtype=float)

# ====================== DISCOVERY ENGINE ======================
class GCADiscoveryEngineV1:
    """
    Fixed-basis symbolic regression via ridge (blocked train/val/test).
    Exports a RAW operator: strength = intercept_raw + dot(coef_raw, basis_raw(t)).
    """
    def __init__(self):
        self.t_sym = sp.symbols('t')
        self.basis_sympy = [
            sp.sin(self.t_sym),
            sp.cos(self.t_sym),
            sp.sin(2*self.t_sym),
            sp.cos(2*self.t_sym),
            sp.exp(-sp.Rational(1,10)*self.t_sym),
            self.t_sym,
            sp.log(1 + sp.Abs(sp.sin(self.t_sym))),
        ]

    def discover(self, task_n_raw: np.ndarray, y_data: np.ndarray, name="memory_vortex_v1", scale: float = 100.0):
        t = task_n_raw.astype(float) / float(scale)
        X_raw = np.vstack([eval_basis_numeric(float(tt)) for tt in t])  # (n, p)

        n = len(y_data)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        # Standardize using TRAIN stats only
        X_train = X_raw[:train_end]
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_stdized = (X_raw - X_mean) / X_std

        # Augment for intercept in standardized space
        X_aug = np.c_[X_stdized, np.ones(n)]

        X_train_aug, y_train = X_aug[:train_end], y_data[:train_end]
        X_val_aug, y_val = X_aug[train_end:val_end], y_data[train_end:val_end]
        X_test_aug, y_test = X_aug[val_end:], y_data[val_end:]

        # Select lambda on VAL
        lambdas = np.logspace(-4, 2, 20)
        best_lam, best_val_mae = None, np.inf
        for lam in lambdas:
            I = np.eye(X_train_aug.shape[1])
            coef = np.linalg.solve(X_train_aug.T @ X_train_aug + lam * I, X_train_aug.T @ y_train)
            y_pred = X_val_aug @ coef
            mae = float(np.mean(np.abs(y_val - y_pred)))
            if mae < best_val_mae:
                best_val_mae, best_lam = mae, float(lam)

        # Refit on TRAIN+VAL
        X_refit = np.vstack((X_train_aug, X_val_aug))
        y_refit = np.concatenate((y_train, y_val))
        I = np.eye(X_refit.shape[1])
        coef_aug = np.linalg.solve(X_refit.T @ X_refit + best_lam * I, X_refit.T @ y_refit)
        coef_aug[np.abs(coef_aug) < 1e-5] = 0

        # Convert standardized model -> RAW operator
        beta_std = coef_aug[:-1]
        b_std = coef_aug[-1]
        coef_raw = beta_std / X_std
        intercept_raw = float(b_std - np.dot(beta_std, X_mean / X_std))

        # TEST metric computed in standardized space
        test_mae = float(np.mean(np.abs(y_test - (b_std + X_test_aug[:, :-1] @ beta_std))))

        expr = intercept_raw + sum(float(coef_raw[i]) * self.basis_sympy[i] for i in range(len(coef_raw)))
        latex = sp.latex(sp.simplify(expr))
        sympy_str = str(sp.simplify(expr))

        result = {
            "schema": "memory-vortex/operator-v1",
            "name": name,
            "basis_order": BASIS_ORDER,
            "coefficients_raw": coef_raw.tolist(),
            "intercept_raw": intercept_raw,
            "t_scale": float(scale),
            "fit": {
                "method": "ridge",
                "lambda": best_lam,
                "val_mae": float(best_val_mae),
                "test_mae": float(test_mae),
                "split": {"train_frac": 0.70, "val_frac": 0.15, "blocked": True},
            },
            "symbolic": {"latex": latex, "sympy": sympy_str},
        }
        print(f"✅ Discovered {name} (val MAE: {best_val_mae:.4f}, test MAE: {test_mae:.4f})")
        return result

def save_operator(op: dict, path: str = "memory_vortex_operator.json") -> None:
    with open(path, "w") as f:
        json.dump(op, f, indent=2)

# ====================== SCHEDULER ======================
class MemoryVortexScheduler:
    """
    Loads a discovered operator JSON if present, otherwise uses a safe fallback.
    Returns a modality-keyed dict (currently modality-agnostic schedule).
    """
    def __init__(self, operator_file: str = "memory_vortex_operator.json", verbose: bool = True):
        self.verbose = verbose
        if os.path.exists(operator_file):
            with open(operator_file) as f:
                data = json.load(f)
            if data.get("schema") != "memory-vortex/operator-v1":
                raise ValueError(f"Unsupported operator schema: {data.get('schema')}")
            self.coef = np.array(data["coefficients_raw"], dtype=float)
            self.intercept = float(data["intercept_raw"])
            self.t_scale = float(data["t_scale"])
            if self.verbose:
                print("Loaded discovered operator from memory_vortex_operator.json")
        else:
            # Safe fallback: 0.01375 cos(2t) + 0.798 exp(-0.1t), with t = task_n / t_scale
            self.coef = np.array([0.0, 0.0, 0.0, 0.01375, 0.798, 0.0, 0.0], dtype=float)
            self.intercept = 0.0
            self.t_scale = 100.0
            if self.verbose:
                print("Using fallback operator (no JSON found)")

        self.modalities = ["vision", "text", "audio"]

    def strength(self, task_n: int) -> float:
        t = float(task_n) / self.t_scale
        x = eval_basis_numeric(t)
        s = self.intercept + float(np.dot(self.coef, x))
        return float(np.clip(s, 0.0, 1.0))

    def __call__(self, task_n: int) -> Dict[str, float]:
        s = self.strength(task_n)
        return {m: s for m in self.modalities}

# ====================== TOY CONTINUAL-LEARNING BENCHMARK ======================
def generate_task(task_id: int, rng: np.random.Generator, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learnable Gaussian blobs (binary classification) with drifting means.
    """
    mean1 = task_id * 1.2
    mean2 = task_id * 1.2 + 3.0
    X1 = rng.normal(size=(n_samples//2, 2)) + mean1
    X2 = rng.normal(size=(n_samples//2, 2)) + mean2
    X = np.vstack((X1, X2))
    y = np.concatenate((np.zeros(n_samples//2), np.ones(n_samples//2)))
    return X, y

def train_logistic(X: np.ndarray, y: np.ndarray, w_init=None, lr: float = 0.05, epochs: int = 30) -> np.ndarray:
    w = w_init.copy() if w_init is not None else np.zeros(X.shape[1])
    for _ in range(epochs):
        preds = 1.0 / (1.0 + np.exp(-X @ w))
        grad = X.T @ (preds - y) / len(y)
        w -= lr * grad
    return w

def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    preds = (X @ w > 0).astype(float)
    return float(np.mean(preds == y))

def run_real_benchmark(n_trials: int = 20, n_tasks: int = 5, verbose: bool = False) -> None:
    results = {"no": [], "const": [], "rand": [], "vortex": []}

    for trial in range(n_trials):
        # per-trial RNG for independence + reproducibility
        trial_seed = int(master_rng.integers(0, 2**32 - 1))
        rng_trial = np.random.default_rng(trial_seed)

        # one scheduler per trial (quiet during trials)
        scheduler = MemoryVortexScheduler(verbose=verbose)

        # weights per mode (carry forward)
        w = {m: None for m in results}
        old_datasets: list[Tuple[np.ndarray, np.ndarray]] = []

        for task in range(n_tasks):
            X_new, y_new = generate_task(task, rng_trial)

            for mode in results:
                if mode == "no":
                    replay_frac = 0.0
                elif mode == "const":
                    replay_frac = 0.3
                elif mode == "rand":
                    replay_frac = float(rng_trial.uniform(0.1, 0.5))
                else:
                    replay_frac = float(scheduler(task)["vision"])

                # replay sampling (row-level)
                if task > 0 and replay_frac > 0.0:
                    all_old_X = np.vstack([ds[0] for ds in old_datasets])
                    all_old_y = np.concatenate([ds[1] for ds in old_datasets])
                    n_replay = int(replay_frac * len(X_new))
                    replace = n_replay > len(all_old_X)
                    idx = rng_trial.choice(len(all_old_X), n_replay, replace=replace)
                    X_rep = all_old_X[idx]
                    y_rep = all_old_y[idx]
                    X_train = np.vstack((X_new, X_rep))
                    y_train = np.concatenate((y_new, y_rep))
                else:
                    X_train, y_train = X_new, y_new

                # incremental training
                w[mode] = train_logistic(X_train, y_train, w_init=w[mode])

                # evaluate avg accuracy on previous tasks only
                if old_datasets:
                    old_accs = [accuracy(w[mode], Xo, yo) for (Xo, yo) in old_datasets]
                    avg_old = float(np.mean(old_accs))
                else:
                    avg_old = accuracy(w[mode], X_new, y_new)

                results[mode].append(avg_old)

            old_datasets.append((X_new, y_new))

    print(f"Real Benchmark — avg OLD-task accuracy after final task ({n_trials} trials, mean ± std):")
    for mode in results:
        final_accs = np.array(results[mode]).reshape(n_trials, n_tasks)[:, -1]
        print(f"  {mode:8s} : {final_accs.mean():.3f} ± {final_accs.std():.3f}")

# ====================== OVERHEAD ======================
def benchmark_overhead(iters: int = 100_000) -> None:
    scheduler = MemoryVortexScheduler(verbose=False)
    # warmup
    for _ in range(10_000):
        scheduler(42)

    start = time.perf_counter_ns()
    for _ in range(iters):
        scheduler(42)
    ns = (time.perf_counter_ns() - start) / iters
    print(f"Scheduler overhead: {ns:.0f} ns ≈ {ns/1000:.2f} µs (mean)")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("Memory Vortex v1.0 — Research Prototype\n")
    run_real_benchmark()
    benchmark_overhead()
