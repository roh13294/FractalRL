# fractal_rl_discrete/core.py
# Numerically-stable core for discrete fractal / fractional value propagation.
# Full file with:
# - robust graph sanitization + rescaled eigendecomposition
# - stable spectral filter
# - backward-compatible value_iteration_fractional API (eta, A_adj, L, etc.)
# - adds hist["v_start"] and hist["v_end"] for experiment compatibility

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any


# ----------------------------
# Utilities
# ----------------------------

def _as_float64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.float64:
        return x.astype(np.float64, copy=False)
    return x


def _finite_or_raise(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        raise FloatingPointError(f"{name} has non-finite values. min={mn}, max={mx}")


def _stats(name: str, x: np.ndarray) -> str:
    x = np.asarray(x)
    if x.size == 0:
        return f"{name}: empty"
    if not np.all(np.isfinite(x)):
        return f"{name}: non-finite present; min={np.nanmin(x)}, max={np.nanmax(x)}"
    return f"{name}: shape={x.shape}, min={x.min():.3e}, max={x.max():.3e}, norm={np.linalg.norm(x):.3e}"


def _sanitize_matrix(M: np.ndarray, *, clip_abs: float = 1e6) -> np.ndarray:
    """
    Replace NaN/inf with 0 and clip extreme magnitudes.
    Prevents eigendecomposition / matmul from producing garbage.
    """
    M = _as_float64(M)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None:
        M = np.clip(M, -float(clip_abs), float(clip_abs))
    return M


def ensure_stochastic(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure P is a valid row-stochastic transition matrix.
    Works for (S,S) or (A,S,S). Clips negatives, renormalizes rows.
    """
    P = _sanitize_matrix(P, clip_abs=1e6)

    if P.ndim == 2:
        P = np.clip(P, 0.0, None)
        P = P / (P.sum(axis=1, keepdims=True) + eps)
        _finite_or_raise("P(stochastic)", P)
        return P

    if P.ndim == 3:
        P = np.clip(P, 0.0, None)
        P = P / (P.sum(axis=2, keepdims=True) + eps)
        _finite_or_raise("P(stochastic)", P)
        return P

    raise ValueError(f"P must have ndim 2 or 3, got shape {P.shape}")


# ----------------------------
# Graph spectral helpers
# ----------------------------

def _symmetrize(A: np.ndarray) -> np.ndarray:
    A = _as_float64(A)
    return 0.5 * (A + A.T)


def build_laplacian_from_adjacency(A_adj: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build a symmetric unnormalized Laplacian L = D - A.
    Includes sanitization + symmetrization + jitter.
    """
    A = _sanitize_matrix(A_adj, clip_abs=1e6)
    A = _symmetrize(A)
    A = np.clip(A, 0.0, None)

    deg = A.sum(axis=1)
    D = np.diag(deg)
    L = D - A

    # jitter improves conditioning slightly
    L = L + float(eps) * np.eye(L.shape[0], dtype=np.float64)
    L = _symmetrize(L)
    L = _sanitize_matrix(L, clip_abs=1e12)
    return L


def eigendecompose_symmetric(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust eigen-decomposition for symmetric matrices.

    Rescales the matrix to avoid huge-magnitude eigensolvers producing
    numerically catastrophic eigenvectors. Eigenvectors are invariant to scalar scaling.
    """
    M = _sanitize_matrix(M, clip_abs=1e12)
    M = _symmetrize(M)

    max_abs = float(np.max(np.abs(M))) if M.size else 0.0
    if max_abs == 0.0:
        n = M.shape[0]
        evals = np.zeros(n, dtype=np.float64)
        evecs = np.eye(n, dtype=np.float64)
        return evals, evecs

    scale = max_abs
    M_scaled = M / scale

    evals_s, evecs = np.linalg.eigh(M_scaled)
    evals = _as_float64(evals_s) * scale
    evecs = _as_float64(evecs)

    # force column normalization
    col_norms = np.linalg.norm(evecs, axis=0) + 1e-12
    evecs = evecs / col_norms

    _finite_or_raise("evals", evals)
    _finite_or_raise("evecs", evecs)

    # final safety clip + renorm if something is still absurd
    emax = float(np.max(np.abs(evecs)))
    if emax > 1e3:
        evecs = np.clip(evecs, -1e3, 1e3)
        col_norms = np.linalg.norm(evecs, axis=0) + 1e-12
        evecs = evecs / col_norms

    return evals, evecs


# ----------------------------
# Spectral / Fractional operator (robust)
# ----------------------------

def apply_spectral_filter(
    v: np.ndarray,
    evecs: np.ndarray,
    evals: np.ndarray,
    alpha: float,
    *,
    shift: float = 1.0,
    eps_eig: float = 1e-8,
    cap_weights: float = 1e6,
    normalize_gain: bool = True,
) -> np.ndarray:
    """
    out = U diag(w) U^T v where:
      w = (max(evals + shift, eps_eig))^(-alpha)

    Robustness:
    - sanitize inputs
    - ignore BLAS spurious warnings during matmul
    - hard finiteness checks after matmul
    """
    v = _sanitize_matrix(v, clip_abs=1e9)
    evecs = _sanitize_matrix(evecs, clip_abs=1e6)
    evals = _sanitize_matrix(evals, clip_abs=1e12)

    _finite_or_raise("v", v)
    _finite_or_raise("evecs", evecs)
    _finite_or_raise("evals", evals)

    # normalize eigenvector columns (should already be orthonormal)
    col_norms = np.linalg.norm(evecs, axis=0) + 1e-12
    evecs = evecs / col_norms

    lam = np.maximum(evals + float(shift), float(eps_eig))
    w = lam ** (-float(alpha))
    w = np.clip(w, 0.0, float(cap_weights))

    if normalize_gain:
        m = float(np.max(np.abs(w)))
        if m > 0:
            w = w / m

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        tmp = evecs.T @ v
        out = evecs @ (w * tmp)

    if not np.all(np.isfinite(tmp)):
        raise FloatingPointError(
            "Non-finite tmp = evecs.T @ v.\n"
            + _stats("v", v) + "\n"
            + _stats("evecs", evecs) + "\n"
            + f"max|evecs|={float(np.max(np.abs(evecs))):.3e}"
        )
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            "Non-finite out = evecs @ (w*tmp).\n"
            + _stats("tmp", tmp) + "\n"
            + _stats("w", w) + "\n"
            + _stats("evecs", evecs)
        )

    return _as_float64(out)


fractional_operator = apply_spectral_filter
spectral_filter = apply_spectral_filter


# ----------------------------
# Value Iteration
# ----------------------------

def _compute_PV(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    If P is (S,S): returns (S,)
    If P is (A,S,S): returns (A,S)
    """
    if P.ndim == 2:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            return P @ V
    if P.ndim == 3:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            return np.einsum("ass,s->as", P, V)
    raise ValueError(f"Invalid P shape: {P.shape}")


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    *,
    max_iters: int = 500,
    tol: float = 1e-8,
    V0: Optional[np.ndarray] = None,
    clip_V: float = 1e6,
    fail_fast: bool = True,
    # optional fractal propagation
    use_fractional: bool = False,
    evecs: Optional[np.ndarray] = None,
    evals: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    frac_shift: float = 1.0,
    frac_eps_eig: float = 1e-8,
    frac_cap_weights: float = 1e6,
    frac_normalize_gain: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stable discrete value iteration.
    Adds hist["v_start"] and hist["v_end"] for experiment compatibility.
    """
    P = ensure_stochastic(P)
    R = _sanitize_matrix(R, clip_abs=1e9)
    gamma = float(gamma)

    if V0 is None:
        V = np.zeros(P.shape[-1], dtype=np.float64)
    else:
        V = _sanitize_matrix(V0, clip_abs=1e9).copy()

    _finite_or_raise("init V", V)
    _finite_or_raise("R", R)

    hist: Dict[str, Any] = {
        "deltas": [],
        "iters": 0,
        "converged": False,
        # JSON-safe copies
        "v_start": V.tolist(),
        "v_end": None,
    }

    for it in range(int(max_iters)):
        V_in = V

        if use_fractional:
            if evecs is None or evals is None:
                raise ValueError("use_fractional=True but evecs/evals not provided")
            V_in = apply_spectral_filter(
                V,
                evecs,
                evals,
                alpha,
                shift=frac_shift,
                eps_eig=frac_eps_eig,
                cap_weights=frac_cap_weights,
                normalize_gain=frac_normalize_gain,
            )

        PV = _compute_PV(P, V_in)

        if P.ndim == 2:
            if R.ndim != 1:
                raise ValueError(f"For P (S,S), R must be (S,), got {R.shape}")
            V_new = R + gamma * PV
        else:
            if R.ndim != 2:
                raise ValueError(f"For P (A,S,S), R must be (A,S), got {R.shape}")
            Q = R + gamma * PV
            V_new = np.max(Q, axis=0)

        if clip_V is not None:
            V_new = np.clip(V_new, -float(clip_V), float(clip_V))

        if fail_fast and not np.all(np.isfinite(V_new)):
            raise FloatingPointError(
                "V_new became non-finite.\n" + _stats("V", V) + "\n" + _stats("V_new", V_new)
            )

        delta = float(np.max(np.abs(V_new - V)))
        hist["deltas"].append(delta)
        hist["iters"] = it + 1

        V = V_new

        if delta < float(tol):
            hist["converged"] = True
            break

    hist["v_end"] = V.tolist()
    return V, hist


run_value_iteration = value_iteration
vi = value_iteration


# ----------------------------
# Backward-compatible wrapper (auto-eigendecomp)
# ----------------------------

def value_iteration_fractional(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    *,
    # preferred (new)
    evecs: Optional[np.ndarray] = None,
    evals: Optional[np.ndarray] = None,
    # legacy graph inputs (if evecs/evals not passed)
    A_adj: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    # method params
    alpha: float = 0.5,
    eta: float = 1.0,  # accepted for legacy experiments (currently unused)
    max_iters: int = 500,
    tol: float = 1e-8,
    V0: Optional[np.ndarray] = None,
    clip_V: float = 1e6,
    shift: float = 1.0,
    eps_eig: float = 1e-8,
    cap_weights: float = 1e6,
    normalize_gain: bool = True,
    fail_fast: bool = True,
    **_legacy_kwargs: Any,  # accepts additional legacy kwargs, ignored
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Backward-compatible fractional VI.

    Behavior:
    - If (evecs, evals) provided: sanitize + use them.
    - Else if L provided: sanitize + eigendecompose.
    - Else if A_adj provided: build Laplacian from A_adj + eigendecompose.
    - Else: raise a clear error.
    """
    if evecs is None or evals is None:
        if L is not None:
            Lm = _sanitize_matrix(L, clip_abs=1e12)
            evals, evecs = eigendecompose_symmetric(Lm)
        elif A_adj is not None:
            A = _sanitize_matrix(A_adj, clip_abs=1e6)
            Lm = build_laplacian_from_adjacency(A)
            evals, evecs = eigendecompose_symmetric(Lm)
        else:
            raise TypeError(
                "value_iteration_fractional requires either (evecs, evals) or one of (L, A_adj). "
                "Your experiment passed neither."
            )
    else:
        evecs = _sanitize_matrix(evecs, clip_abs=1e6)
        evals = _sanitize_matrix(evals, clip_abs=1e12)
        col_norms = np.linalg.norm(evecs, axis=0) + 1e-12
        evecs = evecs / col_norms

    return value_iteration(
        P,
        R,
        gamma,
        max_iters=max_iters,
        tol=tol,
        V0=V0,
        clip_V=clip_V,
        fail_fast=fail_fast,
        use_fractional=True,
        evecs=_as_float64(evecs),
        evals=_as_float64(evals),
        alpha=float(alpha),
        frac_shift=float(shift),
        frac_eps_eig=float(eps_eig),
        frac_cap_weights=float(cap_weights),
        frac_normalize_gain=bool(normalize_gain),
    )
