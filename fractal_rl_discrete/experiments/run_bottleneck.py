# fractal_rl_discrete/experiments/run_bottleneck.py
# Bottleneck MDP experiment runner with gridworld-style logging + EDGE LIST stored in JSON.

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

from core import value_iteration, value_iteration_fractional


@dataclass
class BottleneckSpec:
    n1: int = 60          # size of cluster 1
    n2: int = 60          # size of cluster 2
    p_in: float = 0.20    # intra-cluster edge probability
    goal: str = "cluster2"  # where the terminal goal lives: "cluster1" or "cluster2"
    gamma: float = 0.99
    step_cost: float = -1.0
    terminal_reward: float = 0.0
    max_iters: int = 5000
    tol: float = 1e-8
    clip_V: float = 1e6


def _edge_list_from_adj(A: np.ndarray) -> List[List[int]]:
    """
    Return undirected edge list as [[i,j], ...] with i<j for all A[i,j]>0.
    JSON-friendly.
    """
    A = np.asarray(A)
    n = A.shape[0]
    edges: List[List[int]] = []
    # upper triangle only
    rows, cols = np.nonzero(np.triu(A, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        edges.append([int(i), int(j)])
    return edges


def build_bottleneck_mdp(spec: BottleneckSpec, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build a two-community graph with a single bridge edge.

    We build an adjacency A_adj (undirected), then create a random-walk transition matrix P (S,S).
    This gives an MRP (single-action MDP) where value iteration is Bellman evaluation:
      V <- R + gamma P V

    Terminal is one node (absorbing).
    Reward: step_cost everywhere except terminal has terminal_reward.

    Returns:
      P: (S,S)
      R: (S,)
      A_adj: (S,S)
      meta
    """
    rng = np.random.RandomState(seed)

    n1, n2 = spec.n1, spec.n2
    S = n1 + n2

    # adjacency
    A = np.zeros((S, S), dtype=np.float64)

    # community 1 edges
    for i in range(n1):
        for j in range(i + 1, n1):
            if rng.rand() < spec.p_in:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # community 2 edges
    off = n1
    for i in range(n2):
        for j in range(i + 1, n2):
            if rng.rand() < spec.p_in:
                A[off + i, off + j] = 1.0
                A[off + j, off + i] = 1.0

    # Ensure each community is connected-ish: add a chain backbone
    for i in range(n1 - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    for i in range(n2 - 1):
        A[off + i, off + i + 1] = 1.0
        A[off + i + 1, off + i] = 1.0

    # Single bridge edge between a random node in cluster1 and a random node in cluster2
    b1 = int(rng.randint(0, n1))
    b2 = int(off + rng.randint(0, n2))
    A[b1, b2] = 1.0
    A[b2, b1] = 1.0

    # choose terminal goal node
    if spec.goal == "cluster1":
        term = int(rng.randint(0, n1))
    else:
        term = int(off + rng.randint(0, n2))

    # build random-walk P from adjacency
    deg = A.sum(axis=1)
    P = np.zeros((S, S), dtype=np.float64)

    for s in range(S):
        if s == term:
            P[s, s] = 1.0
            continue
        if deg[s] <= 0:
            P[s, s] = 1.0
        else:
            P[s, :] = A[s, :] / deg[s]

    # reward vector
    R = np.full(S, spec.step_cost, dtype=np.float64)
    R[term] = spec.terminal_reward

    edges = _edge_list_from_adj(A)

    meta = {
        "S": S,
        "n1": n1,
        "n2": n2,
        "p_in": spec.p_in,
        "bridge_nodes": [b1, b2],
        "terminal_state": term,
        "goal": spec.goal,
        "seed": seed,
        "edges": edges,  # <--- NEW: store graph structure for BFS distance plots
    }
    return P, R, A, meta


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
    spec = BottleneckSpec(
        n1=60,
        n2=60,
        p_in=0.20,
        goal="cluster2",
        gamma=0.99,
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
        "env": "bottleneck",
        "gamma": spec.gamma,
        "n1": spec.n1,
        "n2": spec.n2,
        "p_in": spec.p_in,
        "goal": spec.goal,
        "seeds": seeds,
        "frac_params": frac_params,
        "runs": [],
    }

    for seed in seeds:
        P, R, A_adj, meta = build_bottleneck_mdp(spec, seed)

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

    out_path = "results/results_bottleneck.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
