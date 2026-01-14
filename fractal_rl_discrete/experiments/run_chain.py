# fractal_rl_discrete/experiments/run_chain.py
# Chain MDP experiment runner with gridworld-style logging.

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

from core import value_iteration, value_iteration_fractional


@dataclass
class ChainSpec:
    N: int = 200
    gamma: float = 0.99
    slip: float = 0.00          # probability of random action instead of intended
    step_cost: float = -1.0
    terminal_reward: float = 0.0
    max_iters: int = 5000
    tol: float = 1e-8
    clip_V: float = 1e6


def build_chain_mdp(spec: ChainSpec, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    States: 0..N-1
    Actions: 0=left, 1=right
    Terminal at N-1 (absorbing)
    Reward: step_cost everywhere except terminal has terminal_reward.
    Returns:
      P: (A,S,S)
      R: (A,S)
      A_adj: (S,S) chain adjacency
      meta
    """
    rng = np.random.RandomState(seed)
    N = spec.N
    S = prove_S = N
    A = 2

    P = np.zeros((A, N, N), dtype=np.float64)
    R = np.zeros((A, N), dtype=np.float64)

    term = N - 1

    for s in range(N):
        if s == term:
            for a in range(A):
                P[a, s, s] = 1.0
                R[a, s] = spec.terminal_reward
            continue

        left = max(0, s - 1)
        right = min(N - 1, s + 1)

        for a in range(A):
            intended = left if a == 0 else right

            # intended transition
            P[a, s, intended] += (1.0 - spec.slip)

            # slip: pick random action uniformly
            slip_mass = spec.slip / A
            P[a, s, left] += slip_mass
            P[a, s, right] += slip_mass

            R[a, s] = spec.step_cost

    # normalize rows
    P = P / (P.sum(axis=2, keepdims=True) + 1e-12)

    # adjacency for chain graph (undirected)
    A_adj = np.zeros((N, N), dtype=np.float64)
    for s in range(N):
        if s - 1 >= 0:
            A_adj[s, s - 1] = 1.0
            A_adj[s - 1, s] = 1.0
        if s + 1 < N:
            A_adj[s, s + 1] = 1.0
            A_adj[s + 1, s] = 1.0

    meta = {
        "N": N,
        "A": A,
        "terminal_state": int(term),
        "seed": int(seed),
    }
    return P, R, A_adj, meta


def _vec_stats(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "l2": float(np.linalg.norm(v)),
        "mean": float(np.mean(v)),
    }


def _ensure_hist_fields(hist: Dict[str, Any], V_end: np.ndarray) -> Dict[str, Any]:
    out = dict(hist)
    if "deltas" not in out:
        out["deltas"] = []
    if "iters" not in out:
        out["iters"] = len(out["deltas"])
    if "converged" not in out:
        out["converged"] = False
    if "v_start" not in out:
        out["v_start"] = None
    if "v_end" not in out:
        out["v_end"] = V_end.tolist()
    return out


def main() -> None:
    spec = ChainSpec(
        N=200,
        gamma=0.99,
        slip=0.00,
        max_iters=5000,
        tol=1e-8,
        clip_V=1e6,
    )

    seeds = [0, 1, 2]

    frac_params = {
        "alpha": 0.50,
        "eta": 1.0,          # accepted for compat (unused in core)
        "shift": 1.0,
        "eps_eig": 1e-8,
        "cap_weights": 1e6,
        "normalize_gain": True,
    }

    results: Dict[str, Any] = {
        "env": "chain",
        "N": spec.N,
        "gamma": spec.gamma,
        "slip": spec.slip,
        "seeds": seeds,
        "frac_params": frac_params,
        "runs": [],
    }

    for seed in seeds:
        P, R, A_adj, meta = build_chain_mdp(spec, seed)

        run: Dict[str, Any] = {
            "seed": seed,
            "meta": meta,
        }

        # baseline
        t0 = time.time()
        Vb, hb = value_iteration(
            P, R, spec.gamma,
            max_iters=spec.max_iters,
            tol=spec.tol,
            clip_V=spec.clip_V,
            fail_fast=True,
        )
        t1 = time.time()
        hb = _ensure_hist_fields(hb, Vb)
        run["baseline"] = {
            "wall_time_sec": float(t1 - t0),
            "V_end": Vb.tolist(),
            "V_stats": _vec_stats(Vb),
            "hist": hb,
        }

        # fractional
        t0 = time.time()
        Vf, hf = value_iteration_fractional(
            P, R, spec.gamma,
            A_adj=A_adj,
            alpha=frac_params["alpha"],
            eta=frac_params["eta"],
            shift=frac_params["shift"],
            eps_eig=frac_params["eps_eig"],
            cap_weights=frac_params["cap_weights"],
            normalize_gain=frac_params["normalize_gain"],
            max_iters=spec.max_iters,
            tol=spec.tol,
            clip_V=spec.clip_V,
            fail_fast=True,
        )
        t1 = time.time()
        hf = _ensure_hist_fields(hf, Vf)
        run["fractional"] = {
            "wall_time_sec": float(t1 - t0),
            "V_end": Vf.tolist(),
            "V_stats": _vec_stats(Vf),
            "hist": hf,
        }

        run["compare"] = {
            "max_abs_V_diff": float(np.max(np.abs(Vf - Vb))),
        }

        results["runs"].append(run)

    out_path = "results/results_chain.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
