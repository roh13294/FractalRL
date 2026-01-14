# fractal_rl_discrete/experiments/plot_gridworld_results.py
# Reads results/results_gridworld.json and writes diagnostic PNGs to figures/

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _repo_root() -> str:
    # This file is <repo_root>/experiments/plot_gridworld_results.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pad_to(arr: List[float], L: int) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(L, dtype=float)
    a = np.asarray(arr, dtype=float)
    if len(a) >= L:
        return a[:L]
    pad = np.full(L - len(a), a[-1], dtype=float)
    return np.concatenate([a, pad], axis=0)


def _collect(d: Dict[str, Any], mode: str) -> Tuple[List[int], List[float], List[List[float]]]:
    iters, times, deltas = [], [], []
    for run in d["runs"]:
        iters.append(int(run[mode]["hist"]["iters"]))
        times.append(float(run[mode].get("wall_time_sec", float("nan"))))
        deltas.append(run[mode]["hist"].get("deltas", []))
    return iters, times, deltas


def main() -> None:
    root = _repo_root()
    results_path = os.path.join(root, "results", "results_gridworld.json")

    # IMPORTANT: in your repo, "plots/" is a code folder. Use figures/ for image outputs.
    out_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("[plot] repo root:", root)
    print("[plot] reading:", results_path)
    print("[plot] writing to:", out_dir)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results file: {results_path}")

    with open(results_path, "r") as f:
        d = json.load(f)

    if "runs" not in d or len(d["runs"]) == 0:
        raise ValueError("results_gridworld.json has no 'runs'")

    seeds = d.get("seeds", list(range(len(d["runs"]))))

    base_iters, base_times, base_deltas = _collect(d, "baseline")
    frac_iters, frac_times, frac_deltas = _collect(d, "fractional")

    x = np.arange(len(seeds))
    width = 0.35

    # ---- 1) iterations per seed ----
    plt.figure()
    plt.bar(x - width/2, base_iters, width, label="baseline")
    plt.bar(x + width/2, frac_iters, width, label="fractional")
    plt.xticks(x, [str(s) for s in seeds])
    plt.xlabel("seed")
    plt.ylabel("iterations to tol")
    plt.title("Gridworld VI iterations (baseline vs fractional)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "diag_gridworld_iters.png")
    plt.savefig(p1)
    plt.close()
    print("[plot] wrote:", p1)

    # ---- 2) wall time per seed ----
    plt.figure()
    plt.bar(x - width/2, base_times, width, label="baseline")
    plt.bar(x + width/2, frac_times, width, label="fractional")
    plt.xticks(x, [str(s) for s in seeds])
    plt.xlabel("seed")
    plt.ylabel("wall time (sec)")
    plt.title("Gridworld VI wall time (baseline vs fractional)")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "diag_gridworld_time.png")
    plt.savefig(p2)
    plt.close()
    print("[plot] wrote:", p2)

    # ---- 3) delta curves mean ± std ----
    L = int(max(max(base_iters), max(frac_iters)))
    base_mat = np.stack([_pad_to(ds, L) for ds in base_deltas], axis=0)
    frac_mat = np.stack([_pad_to(ds, L) for ds in frac_deltas], axis=0)

    base_mean, base_std = base_mat.mean(axis=0), base_mat.std(axis=0)
    frac_mean, frac_std = frac_mat.mean(axis=0), frac_mat.std(axis=0)

    it = np.arange(1, L + 1)

    plt.figure()
    plt.plot(it, base_mean, label="baseline")
    plt.fill_between(it, base_mean - base_std, base_mean + base_std, alpha=0.2)
    plt.plot(it, frac_mean, label="fractional")
    plt.fill_between(it, frac_mean - frac_std, frac_mean + frac_std, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("max |V_{k+1} - V_k| (log)")
    plt.title("Gridworld convergence (mean ± std over seeds)")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "diag_gridworld_deltas_mean.png")
    plt.savefig(p3)
    plt.close()
    print("[plot] wrote:", p3)

    # ---- 4) value difference per seed ----
    diffs = []
    for run in d["runs"]:
        vb = np.array(run["baseline"]["V_end"], dtype=float)
        vf = np.array(run["fractional"]["V_end"], dtype=float)
        diffs.append(float(np.max(np.abs(vf - vb))))

    plt.figure()
    plt.bar(x, diffs)
    plt.xticks(x, [str(s) for s in seeds])
    plt.xlabel("seed")
    plt.ylabel("max |V_frac - V_base|")
    plt.title("Gridworld value difference (fractional vs baseline)")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "diag_gridworld_value_diff.png")
    plt.savefig(p4)
    plt.close()
    print("[plot] wrote:", p4)

    print("[plot] done")


if __name__ == "__main__":
    main()
