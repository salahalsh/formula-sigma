"""Unit tests for common.designs."""
import sys, math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import designs


def test_ccd_rotatable_3_factors():
    D = designs.ccd_rotatable(3, n_center=6)
    assert D.shape == (8 + 6 + 6, 3)        # 8 corners + 6 axial + 6 center
    alpha = float(np.max(np.abs(D)))
    assert abs(alpha - 2 ** (3 / 4.0)) < 1e-9


def test_bbd_3_factor_canonical():
    D = designs.bbd_3factor(n_center=3)
    assert D.shape == (12 + 3, 3)
    # No corner points (no [+/-1, +/-1, +/-1])
    assert not np.any(np.all(np.abs(D) == 1, axis=1))


def test_simplex_lattice_counts():
    # {q, m} = C(q+m-1, m)
    for q, m in [(3, 3), (4, 3), (3, 4)]:
        D = designs.simplex_lattice(q, m)
        assert D.shape == (math.comb(q + m - 1, m), q)
        assert np.allclose(D.sum(axis=1), 1.0)


def test_lhs_strata():
    D = designs.latin_hypercube(20, 3, seed=0)
    assert D.shape == (20, 3)
    # Each column: exactly one sample per stratum
    bins = np.floor(D * 20).astype(int)
    for j in range(D.shape[1]):
        counts = np.bincount(np.clip(bins[:, j], 0, 19), minlength=20)
        assert np.all(counts == 1)


def test_ff_2level():
    D = designs.ff_2level(3)
    assert D.shape == (8, 3)
    assert set(np.unique(D)) == {-1, 1}


def test_full_factorial_mixed_levels():
    D = designs.full_factorial([3, 3, 3])
    assert D.shape == (27, 3)
    for j in range(3):
        assert len(np.unique(D[:, j])) == 3


def test_fractional_factorial_2_to_4_minus_1():
    D = designs.fractional_factorial("a b c abc")
    assert D.shape == (8, 4)
    assert set(np.unique(D)) <= {-1.0, 1.0}
    # d = a*b*c aliasing check
    assert np.array_equal(D[:, 3], D[:, 0] * D[:, 1] * D[:, 2])


def test_d_optimal_beats_random():
    levels = np.linspace(-1, 1, 5)
    g1, g2 = np.meshgrid(levels, levels, indexing="ij")
    cands = np.column_stack([g1.ravel(), g2.ravel()])

    def terms(X):
        return np.column_stack([np.ones(len(X)), X, X ** 2, X[:, 0] * X[:, 1]])

    D, idx = designs.d_optimal(cands, n_runs=8, model_terms=terms,
                               seed=0, max_iter=30)
    assert D.shape == (8, 2)
    F = terms(D)
    det_opt = float(np.linalg.det(F.T @ F))
    # Compare to median random
    rng = np.random.default_rng(0)
    dets = []
    for _ in range(50):
        ri = rng.choice(len(cands), 8, replace=False)
        try:
            dets.append(float(np.linalg.det(terms(cands[ri]).T @ terms(cands[ri]))))
        except np.linalg.LinAlgError:
            dets.append(0.0)
    median_rand = float(np.median(dets))
    assert det_opt > median_rand * 2.0  # D-opt at least 2x median random
