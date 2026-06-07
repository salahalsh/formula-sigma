"""Design-generation wrappers.

Three generators per family:
  - Our reference (numpy-only, used as canonical numerical target)
  - pyDOE3 (third-party cross-check)
  - Optional FORMULA-Sigma adapter (filled in when an API client exists)

Each returns a numpy.ndarray of shape (n_runs, n_factors).
"""
from __future__ import annotations

import numpy as np


# ---------- Full factorial ----------
def full_factorial(levels: list[int]) -> np.ndarray:
    """levels = [3, 3, 3] -> all 27 combinations of {0,1,2} x {0,1,2} x {0,1,2}."""
    grids = np.meshgrid(*[np.arange(l) for l in levels], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


# ---------- 2-level full and fractional factorial ----------
def ff_2level(n_factors: int) -> np.ndarray:
    """All 2^n combinations coded as +/-1."""
    return _twolevel(n_factors)


def _twolevel(n: int) -> np.ndarray:
    rows = 2 ** n
    out = np.empty((rows, n), dtype=int)
    for i in range(n):
        out[:, i] = np.tile(np.repeat([-1, 1], 2 ** i), rows // (2 ** (i + 1)))
    return out


# ---------- Central composite design (rotatable) ----------
def ccd_rotatable(n_factors: int, n_center: int = 0) -> np.ndarray:
    """Rotatable CCD with alpha = 2^(k/4) per Box-Wilson 1951.

    Returns coded factor matrix (-alpha, -1, 0, +1, +alpha).
    """
    alpha = float(2 ** (n_factors / 4.0))
    factorial = _twolevel(n_factors).astype(float)
    # Axial points: one per factor, both directions
    axial = np.zeros((2 * n_factors, n_factors))
    for i in range(n_factors):
        axial[2 * i,     i] = -alpha
        axial[2 * i + 1, i] = +alpha
    center = np.zeros((n_center, n_factors))
    return np.vstack([factorial, axial, center])


# ---------- Box-Behnken design (3-factor canonical) ----------
def bbd_3factor(n_center: int = 3) -> np.ndarray:
    """Canonical 3-factor BBD per Box-Behnken 1960. 12 edge + n_center."""
    edges = np.array([
        [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
        [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
        [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1],
    ], dtype=float)
    center = np.zeros((n_center, 3))
    return np.vstack([edges, center])


# ---------- Simplex-lattice mixture design ----------
def simplex_lattice(n_components: int, degree: int) -> np.ndarray:
    """Simplex-lattice {q, m} per Scheffe 1958.

    For q components and degree m, generates all proportion combinations
    where each x_i in {0, 1/m, 2/m, ..., 1} and sum(x) = 1.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    grid = [i / degree for i in range(degree + 1)]

    def recurse(remaining: float, n: int):
        if n == 1:
            if any(abs(remaining - g) < 1e-9 for g in grid):
                yield [round(remaining, 10)]
            return
        for g in grid:
            if g - 1e-9 <= remaining:
                for tail in recurse(round(remaining - g, 10), n - 1):
                    yield [g] + tail

    return np.array([row for row in recurse(1.0, n_components)])


# ---------- Latin hypercube ----------
def latin_hypercube(n_samples: int, n_factors: int, seed: int = 42) -> np.ndarray:
    """Latin hypercube on [0, 1]^n_factors with random in-bin placement."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, n_factors))
    cuts = np.linspace(0, 1, n_samples + 1)
    for j in range(n_factors):
        order = rng.permutation(n_samples)
        for i in range(n_samples):
            lo, hi = cuts[order[i]], cuts[order[i] + 1]
            out[i, j] = rng.uniform(lo, hi)
    return out


# ---------- 2-level fractional factorial ----------
def fractional_factorial(generators: str) -> np.ndarray:
    """Fractional factorial design from Box-Hunter generator string.

    `generators` example: 'a b c ab' (for 2^(4-1)_IV: 4 factors, 3 base + 1
    confounded as ab). Use `pyDOE3.fracfact` for the heavy lifting.
    """
    import pyDOE3
    return pyDOE3.fracfact(generators)


# ---------- D-optimal design (Fedorov-style exchange) ----------
def d_optimal(candidate_set: np.ndarray, n_runs: int, *,
              model_terms: callable | None = None,
              seed: int = 42,
              max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """D-optimal subset selection from a candidate set via Fedorov exchange.

    candidate_set : (N, k) candidate factor combinations
    n_runs        : how many rows to select (>= number of model terms)
    model_terms   : callable mapping (n, k) -> (n, p) expanded feature matrix
                    (default: linear with intercept)
    seed          : RNG seed
    max_iter      : Fedorov exchange iterations

    Returns (selected_design, selected_idx).
    """
    rng = np.random.default_rng(seed)
    N, k = candidate_set.shape
    if model_terms is None:
        model_terms = lambda X: np.column_stack([np.ones(len(X)), X])

    F_all = model_terms(candidate_set)
    if n_runs < F_all.shape[1]:
        raise ValueError(f"n_runs ({n_runs}) must be >= n model terms ({F_all.shape[1]})")

    # Initial random sample
    idx = rng.choice(N, size=n_runs, replace=False)
    F = F_all[idx]
    try:
        det = float(np.linalg.det(F.T @ F))
    except np.linalg.LinAlgError:
        det = 0.0

    for _ in range(max_iter):
        improved = False
        # For each currently-selected row, try swapping with each candidate
        for j, sel_row in enumerate(list(idx)):
            best_swap_det = det
            best_swap_cand = None
            for c in range(N):
                if c in idx:
                    continue
                new_idx = idx.copy()
                new_idx[j] = c
                F_new = F_all[new_idx]
                try:
                    new_det = float(np.linalg.det(F_new.T @ F_new))
                except np.linalg.LinAlgError:
                    new_det = 0.0
                if new_det > best_swap_det * 1.000001:  # require real improvement
                    best_swap_det = new_det
                    best_swap_cand = c
            if best_swap_cand is not None:
                idx[j] = best_swap_cand
                F = F_all[idx]
                det = best_swap_det
                improved = True
        if not improved:
            break

    return candidate_set[idx], np.asarray(idx)
