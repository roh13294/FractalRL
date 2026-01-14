# fractal_rl_discrete/experiments/run_gridworld.py
# Self-contained gridworld experiment runner.
# Writes results/results_gridworld.json with per-run histories and value vectors.

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

from core import value_iteration, value_iteration_fractional


# ----------------------------
# Gridworld construction
# ----------------------------

@dataclass
class GridSpec:
    H: int = 15
    W: int = 15
    wall_prob: float = 0.15
    slip: float = 0.10
    gamma: float = 0.99
    step_cost: float = -1.0
    terminal_reward: float = 0.0
    max_iters: int = 2000
    tol: float = 1e-8
    clip_V: float = 1e6


def _idx(r: int, c: int, W: int) -> int:
    return r * W + c


def _neighbors(r: int, c: int) -> List[Tuple[int, int, int]]:
    # action ordering: 0=up, 1=right, 2=down, 3=left
    return [
        (r - 1, c, 0),
        (r, c + 1, 1),
        (r + 1, c, 2),
        (r, c - 1, 3),
    ]


def build_gridworld(spec: GridSpec, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      P: (A,S,S) transition tensor
      R: (A,S) reward tensor
      A_adj: (S,S) adjacency for graph propagation
      meta: dict with mapping/walls/terminal info
    """
    rng = np.random.RandomState(seed)

    H, W = spec.H, spec.W
    S = H * W
    A = 4

    # walls: random but keep start and terminal open
    walls = rng.rand(H, W) < spec.wall_prob
    start = (0, 0)
    terminal = (H - 1, W - 1)
    walls[start] = False
    walls[terminal] = False

    # If walls block almost everything, clear them (rare but possible at high wall_prob)
    if walls.sum() > 0.6 * S:
        walls[:] = False
        walls[start] = False
        walls[terminal] = False

    # Transition tensor
    P = np.zeros((A, S, S), dtype=np.float64)
    R = np.zeros((A, S), dtype=np.float64)

    term_s = _idx(terminal[0], terminal[1], W)

    for r in range(H):
        for c in range(W):
            s = _idx(r, c, W)

            if walls[r, c]:
                # wall states are absorbing with zero reward (you should never be in them)
                for a in range(A):
                    P[a, s, s] = 1.0
                    R[a, s] = 0.0
                continue

            if s == term_s:
                # terminal absorbing
                for a in range(A):
                    P[a, s, s] = 1.0
                    R[a, s] = spec.terminal_reward
                continue

            # Determine deterministic next states for each intended action
            next_for_action = [s] * A
            for (nr, nc, a) in _neighbors(r, c):
                if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                    next_for_action[a] = _idx(nr, nc, W)
                else:
                    next_for_action[a] = s  # bump into wall/border => stay

            # Slip model: with prob (1-slip) do intended action,
            # with prob slip do a random action uniformly
            for a in range(A):
                # intended
                P[a, s, next_for_action[a]] += (1.0 - spec.slip)

                # slipped
                slip_mass = spec.slip / A
                for a2 in range(A):
                    P[a, s, next_for_action[a2]] += slip_mass

                # reward: step cost always (terminal handled above)
                R[a, s] = spec.step_cost

    # Renormalize rows (safety)
    row = P.sum(axis=2, keepdims=True)
    P = P / (row + 1e-12)

    # Graph adjacency based on grid neighbors (undirected), excluding walls
    A_adj = np.zeros((S, S), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            if walls[r, c]:
                continue
            s = _idx(r, c, W)
            for (nr, nc, _) in _neighbors(r, c):
                if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                    t = _idx(nr, nc, W)
                    A_adj[s, t] = 1.0
                    A_adj[t, s] = 1.0

    meta = {
        "H": H,
        "W": W,
        "S": S,
        "A": A,
        "start_state": int(_idx(start[0], start[1], W)),
        "terminal_state": int(term_s),
        "num_walls": int(walls.sum()),
        "walls_flat": walls.astype(int).reshape(-1).tolist(),
    }
    return P, R, A_adj, meta


# ----------------------------
# Helpers for JSON logging
# ----------------------------

def _vec_stats(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "l2": float(np.linalg.norm(v)),
        "mean": float(np.mean(v)),
    }


def _ensure_hist_fields(hist: Dict[str, Any], V_end: np.ndarray) -> Dict[str, Any]:
    """
    Make sure hist has the keys the analysis scripts expect, even if the core implementation changes.
    """
    out = dict(hist)

    if "deltas" not in out:
        out["deltas"] = []
    if "iters" not in out:
        out["iters"] = len(out["deltas"])
    if "converged" not in out:
        out["converged"] = False

    # v_start / v_end as JSON-safe lists
    if "v_start" not in out:
        out["v_start"] = None
    if "v_end" not in out:
        out["v_end"] = V_end.tolist()

    return out


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    # Keep these aligned with what your old JSON already exposed
    spec = GridSpec(
        H=15,
        W=15,
        wall_prob=0.15,
        slip=0.10,
        gamma=0.99,
        max_iters=2000,
        tol=1e-8,
        clip_V=1e6,
    )

    seeds = [0, 1, 2, 3, 4]

    # Fractional params (tune later)
    frac_params = {
        "alpha": 0.50,
        "eta": 1.0,          # accepted for compatibility (currently unused in core)
        "shift": 1.0,
        "eps_eig": 1e-8,
        "cap_weights": 1e6,
        "normalize_gain": True,
    }

    results: Dict[str, Any] = {
        "env": "gridworld",
        "gamma": spec.gamma,
        "H": spec.H,
        "W": spec.W,
        "wall_prob": spec.wall_prob,
        "slip": spec.slip,
        "seeds": seeds,
        "frac_params": frac_params,
        "runs": [],
    }

    for seed in seeds:
        P, R, A_adj, meta = build_gridworld(spec, seed=seed)

        run: Dict[str, Any] = {
            "seed": seed,
            "meta": meta,
        }

        # --- baseline ---
        t0 = time.time()
        V_base, hist_base = value_iteration(
            P, R, spec.gamma,
            max_iters=spec.max_iters,
            tol=spec.tol,
            clip_V=spec.clip_V,
            fail_fast=True,
        )
        t1 = time.time()

        hist_base = _ensure_hist_fields(hist_base, V_base)

        run["baseline"] = {
            "wall_time_sec": float(t1 - t0),
            "V_end": V_base.tolist(),
            "V_stats": _vec_stats(V_base),
            "hist": hist_base,
        }

        # --- fractional ---
        t0 = time.time()
        V_frac, hist_frac = value_iteration_fractional(
            P, R, spec.gamma,
            A_adj=A_adj,                   # triggers auto-eigendecomp in core
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

        hist_frac = _ensure_hist_fields(hist_frac, V_frac)

        run["fractional"] = {
            "wall_time_sec": float(t1 - t0),
            "V_end": V_frac.tolist(),
            "V_stats": _vec_stats(V_frac),
            "hist": hist_frac,
        }

        # diff summary
        diff = np.max(np.abs(V_frac - V_base))
        run["compare"] = {
            "max_abs_V_diff": float(diff),
        }

        results["runs"].append(run)

    # Write JSON
    out_path = "results/results_gridworld.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
