"""
Tests for mlx_expm matrix functions.

Validates against scipy.linalg where available, and checks algebraic identities.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_expm import expm, expm_frechet, logm, sqrtm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x)


def _random_matrix(n: int, rng, dtype=np.float64) -> np.ndarray:
    A = rng.standard_normal((n, n)).astype(dtype)
    return A / n  # scale to avoid overflow


def _random_hermitian(n: int, rng) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (A + A.conj().T) / 2.0
    return H / n


def _random_positive_definite(n: int, rng) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return (A @ A.T) / n + np.eye(n)


# ---------------------------------------------------------------------------
# expm tests
# ---------------------------------------------------------------------------

class TestExpm:
    """Tests for the matrix exponential."""

    def test_identity(self):
        """exp(0) = I"""
        for n in [1, 2, 4, 8]:
            Z = mx.zeros((n, n))
            result = _to_numpy(expm(Z))
            np.testing.assert_allclose(result, np.eye(n), atol=1e-12)

    def test_diagonal(self):
        """exp(diag(d)) = diag(exp(d))"""
        d = np.array([0.0, 1.0, -1.0, 2.0, 0.5])
        D = np.diag(d)
        result = _to_numpy(expm(mx.array(D)))
        expected = np.diag(np.exp(d))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_1x1(self):
        """1x1 matrix."""
        A = mx.array([[3.0]])
        result = _to_numpy(expm(A))
        np.testing.assert_allclose(result, [[np.exp(3.0)]], rtol=1e-6)

    def test_nilpotent(self):
        """exp of a nilpotent matrix: exp([[0,1],[0,0]]) = [[1,1],[0,1]]"""
        A = mx.array([[0.0, 1.0], [0.0, 0.0]])
        result = _to_numpy(expm(A))
        expected = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_complex_hermitian(self):
        """exp(iH) should be unitary for Hermitian H."""
        rng = np.random.default_rng(42)
        for n in [2, 4, 8]:
            H = _random_hermitian(n, rng)
            U = _to_numpy(expm(mx.array(1j * H)))
            # U U^dag should be I
            UUd = U @ U.conj().T
            np.testing.assert_allclose(UUd, np.eye(n), atol=1e-5)

    def test_vs_scipy(self):
        """Compare with scipy.linalg.expm on random matrices."""
        scipy_expm = pytest.importorskip("scipy.linalg").expm
        rng = np.random.default_rng(123)

        for n in [2, 4, 8, 16, 32]:
            A = _random_matrix(n, rng)
            ref = scipy_expm(A)
            result = _to_numpy(expm(mx.array(A.astype(np.float32))))
            np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_vs_scipy_complex(self):
        """Compare with scipy on complex random matrices."""
        scipy_expm = pytest.importorskip("scipy.linalg").expm
        rng = np.random.default_rng(456)

        for n in [2, 4, 8, 16]:
            A = _random_matrix(n, rng) + 1j * _random_matrix(n, rng)
            ref = scipy_expm(A)
            result = _to_numpy(expm(mx.array(A.astype(np.complex64))))
            np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-4)

    def test_large_norm(self):
        """Test with a matrix of large norm (tests the scaling step).

        Uses norm ~40 (not 100) to stay within float32 dynamic range:
        exp(40) ~ 2.35e17, well within float32 max ~ 3.4e38.
        """
        scipy_expm = pytest.importorskip("scipy.linalg").expm
        A = np.array([[40.0, 1.0], [0.0, -5.0]])
        ref = scipy_expm(A)
        result = _to_numpy(expm(mx.array(A.astype(np.float32))))
        np.testing.assert_allclose(result, ref, rtol=1e-2)

    def test_integer_input(self):
        """Integer matrices should be promoted to float."""
        A = mx.array([[0, 1], [0, 0]], dtype=mx.int32)
        result = _to_numpy(expm(A))
        expected = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shape_error(self):
        """Non-square matrix should raise ValueError."""
        with pytest.raises(ValueError):
            expm(mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_quantum_evolution(self):
        """U = exp(-i H t) is unitary for Hermitian H, real t."""
        H = mx.array([[0.0, 1.0], [1.0, 0.0]])  # Pauli X
        t = 0.5
        U = expm(-1j * H * t)
        U_np = _to_numpy(U)
        # Check unitarity
        UUd = U_np @ U_np.conj().T
        np.testing.assert_allclose(UUd, np.eye(2), atol=1e-6)


# ---------------------------------------------------------------------------
# expm_frechet tests
# ---------------------------------------------------------------------------

class TestExpmFrechet:
    """Tests for the Frechet derivative of the matrix exponential."""

    def test_basic(self):
        """Frechet derivative via finite difference check."""
        rng = np.random.default_rng(42)
        n = 4
        A = _random_matrix(n, rng).astype(np.float32)
        E = _random_matrix(n, rng).astype(np.float32)

        A_mx = mx.array(A)
        E_mx = mx.array(E)

        expm_A, L = expm_frechet(A_mx, E_mx)

        # Finite difference: L ~ (expm(A + eps*E) - expm(A)) / eps
        eps = 1e-4
        expm_pert = _to_numpy(expm(mx.array((A + eps * E).astype(np.float32))))
        expm_A_np = _to_numpy(expm_A)
        L_fd = (expm_pert - expm_A_np) / eps
        L_np = _to_numpy(L)

        np.testing.assert_allclose(L_np, L_fd, rtol=0.05, atol=1e-3)

    def test_returns_correct_expm(self):
        """The expm returned by expm_frechet should match expm(A)."""
        rng = np.random.default_rng(99)
        A = _random_matrix(4, rng).astype(np.float32)
        E = _random_matrix(4, rng).astype(np.float32)

        expm_A_from_frechet, _ = expm_frechet(mx.array(A), mx.array(E))
        expm_A_direct = expm(mx.array(A))

        np.testing.assert_allclose(
            _to_numpy(expm_A_from_frechet),
            _to_numpy(expm_A_direct),
            rtol=1e-5,
        )

    def test_compute_expm_false(self):
        """When compute_expm=False, only L is returned."""
        A = mx.array([[1.0, 0.0], [0.0, 1.0]])
        E = mx.array([[0.0, 1.0], [1.0, 0.0]])
        result = expm_frechet(A, E, compute_expm=False)
        assert isinstance(result, mx.array)
        assert result.shape == (2, 2)

    def test_shape_error(self):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError):
            expm_frechet(mx.zeros((2, 2)), mx.zeros((3, 3)))


# ---------------------------------------------------------------------------
# logm tests
# ---------------------------------------------------------------------------

class TestLogm:
    """Tests for the matrix logarithm."""

    def test_identity(self):
        """log(I) = 0"""
        for n in [1, 2, 4]:
            I = mx.eye(n)
            result = _to_numpy(logm(I))
            np.testing.assert_allclose(result, np.zeros((n, n)), atol=1e-6)

    def test_exp_then_log(self):
        """log(exp(A)) ~ A for small A."""
        rng = np.random.default_rng(42)
        A = _random_matrix(4, rng).astype(np.float32)
        A = A * 0.1  # keep small

        eA = expm(mx.array(A))
        logA = logm(eA)

        np.testing.assert_allclose(_to_numpy(logA), A, atol=1e-3)

    def test_diagonal(self):
        """log(diag(exp(d))) = diag(d)"""
        d = np.array([0.5, 1.0, 2.0, 0.1])
        D = np.diag(np.exp(d)).astype(np.float32)
        result = _to_numpy(logm(mx.array(D)))
        np.testing.assert_allclose(result, np.diag(d), atol=1e-3)

    def test_vs_scipy(self):
        """Compare with scipy.linalg.logm on a positive definite matrix."""
        scipy_logm = pytest.importorskip("scipy.linalg").logm
        rng = np.random.default_rng(77)
        A = _random_positive_definite(4, rng).astype(np.float64)

        ref = scipy_logm(A)
        result = _to_numpy(logm(mx.array(A.astype(np.float32))))

        np.testing.assert_allclose(result, ref, atol=0.05)


# ---------------------------------------------------------------------------
# sqrtm tests
# ---------------------------------------------------------------------------

class TestSqrtm:
    """Tests for the matrix square root."""

    def test_identity(self):
        """sqrt(I) = I"""
        for n in [1, 2, 4]:
            I = mx.eye(n)
            result = _to_numpy(sqrtm(I))
            np.testing.assert_allclose(result, np.eye(n), atol=1e-8)

    def test_diagonal(self):
        """sqrt(diag(d^2)) = diag(d) for d > 0"""
        d = np.array([1.0, 4.0, 9.0, 16.0])
        D = np.diag(d).astype(np.float32)
        result = _to_numpy(sqrtm(mx.array(D)))
        expected = np.diag(np.sqrt(d))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_squaring_roundtrip(self):
        """S = sqrtm(A); S @ S ~ A for positive definite A."""
        rng = np.random.default_rng(42)
        A = _random_positive_definite(4, rng).astype(np.float32)
        S = sqrtm(mx.array(A))
        S_np = _to_numpy(S)
        recovered = S_np @ S_np
        np.testing.assert_allclose(recovered, A, rtol=1e-4, atol=1e-4)

    def test_vs_scipy(self):
        """Compare with scipy.linalg.sqrtm on a positive definite matrix."""
        scipy_sqrtm = pytest.importorskip("scipy.linalg").sqrtm
        rng = np.random.default_rng(88)
        A = _random_positive_definite(4, rng).astype(np.float64)

        ref = scipy_sqrtm(A)
        result = _to_numpy(sqrtm(mx.array(A.astype(np.float32))))

        np.testing.assert_allclose(result, ref, atol=1e-3)

    def test_1x1(self):
        """1x1 matrix."""
        A = mx.array([[4.0]])
        result = _to_numpy(sqrtm(A))
        np.testing.assert_allclose(result, [[2.0]], atol=1e-8)

    def test_shape_error(self):
        """Non-square matrix should raise ValueError."""
        with pytest.raises(ValueError):
            sqrtm(mx.array([[1.0, 2.0]]))
