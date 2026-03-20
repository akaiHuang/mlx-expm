"""
GPU-accelerated matrix exponential and related functions for Apple MLX.

Implements:
  - expm(A):              Matrix exponential via scaling-and-squaring with [13/13] Pade approximant
  - expm_frechet(A, E):   Frechet derivative of the matrix exponential
  - logm(A):              Matrix logarithm via inverse scaling-and-squaring
  - sqrtm(A):             Matrix square root via Denman-Beavers iteration

All matrix multiplications run on GPU; linalg.solve/inv route through CPU
(MLX linalg decompositions are currently CPU-only, but the data stays in
unified memory so the transfer cost is minimal on Apple Silicon).

Reference:
  Higham, "Functions of Matrices", SIAM, 2008.
  Al-Mohy & Higham, "A New Scaling and Squaring Algorithm for the Matrix Exponential",
    SIAM J. Matrix Anal. Appl. 31(3), 2009.
  Al-Mohy & Higham, "Computing the Frechet Derivative of the Matrix Exponential,
    with an application to condition number estimation",
    SIAM J. Matrix Anal. Appl. 30(4), 2009.
  Kenney & Laub, "A Schur-Frechet Algorithm for Computing the Logarithm and
    Exponential of a Matrix", SIAM J. Matrix Anal. Appl. 19(3), 1998.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pade coefficients for the [13/13] approximant (Higham 2005, Table 10.2)
# b_j = (2m-j)! m! / ((2m)! j! (m-j)!)  with m = 13
# ---------------------------------------------------------------------------

_PADE_COEFFS_13 = [
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
]

# Theta values for choosing the Pade order (Table 10.3 in Higham 2008)
# We only use order 13.
_THETA_13 = 5.371920351148152


def _eye_like(n: int, dtype: mx.Dtype) -> mx.array:
    """Create an identity matrix of given size and dtype.

    Works around MLX limitation where mx.eye does not support complex dtypes.
    """
    I = mx.eye(n)
    if dtype != I.dtype:
        I = I.astype(dtype)
    return I


def _onenorm(A: mx.array) -> float:
    """Compute the 1-norm of a matrix (max absolute column sum)."""
    return float(mx.max(mx.sum(mx.abs(A), axis=-2)))


def _solve_complex(P: mx.array, Q: mx.array) -> mx.array:
    """Solve P @ X = Q for complex P, Q.

    MLX linalg.solve does not support complex types, so we convert to
    an equivalent 2n x 2n real system:
        [[Pr, -Pi], [Pi, Pr]] @ [[Xr], [Xi]] = [[Qr], [Qi]]

    Then extract X = Xr + i*Xi.
    """
    n = P.shape[0]
    Pr = P.real.astype(mx.float32)
    Pi = P.imag.astype(mx.float32)
    Qr = Q.real.astype(mx.float32)
    Qi = Q.imag.astype(mx.float32)

    # Build 2n x 2n real system
    top = mx.concatenate([Pr, -Pi], axis=1)
    bottom = mx.concatenate([Pi, Pr], axis=1)
    P_real = mx.concatenate([top, bottom], axis=0)

    # Build 2n x n right-hand side
    Q_real = mx.concatenate([Qr, Qi], axis=0)

    # Solve on CPU
    X_real = mx.linalg.solve(P_real, Q_real, stream=mx.cpu)

    Xr = X_real[:n, :]
    Xi = X_real[n:, :]
    return Xr.astype(mx.complex64) + 1j * Xi.astype(mx.complex64)


def _solve_P_Q(U: mx.array, V: mx.array) -> mx.array:
    """Solve (V - U) X = (V + U) for X = R_{13}(A), the Pade approximant.

    P = V - U (denominator), Q = V + U (numerator).
    Returns X = P^{-1} Q via mx.linalg.solve on CPU stream.
    Handles both real and complex matrices.
    """
    P = V - U
    Q = V + U

    if mx.issubdtype(P.dtype, mx.complexfloating):
        return _solve_complex(P, Q)
    else:
        # linalg.solve is CPU-only in current MLX; unified memory keeps it fast
        return mx.linalg.solve(P, Q, stream=mx.cpu)


def _pade13(A: mx.array) -> mx.array:
    """Evaluate the [13/13] Pade approximant of exp(A).

    Computes U and V such that exp(A) ~ (V - U)^{-1} (V + U).
    Uses the standard Horner-like evaluation (Higham 2005, Algorithm 2.3).
    """
    b = _PADE_COEFFS_13
    n = A.shape[-1]
    ident = _eye_like(n, A.dtype)

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4

    # W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    # W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I
    W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident

    # Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
    Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
    # Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident

    W = A6 @ W1 + W2
    U = A @ W
    V = A6 @ Z1 + Z2

    return U, V


def expm(A: mx.array) -> mx.array:
    """Compute the matrix exponential of A using scaling and squaring
    with the [13/13] Pade approximant.

    This is the same algorithm used by scipy.linalg.expm (Higham 2005/2009).

    Parameters
    ----------
    A : mx.array, shape (..., n, n)
        Square matrix (real or complex). Batched inputs supported.

    Returns
    -------
    mx.array, shape (..., n, n)
        The matrix exponential exp(A).

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx_expm import expm
    >>> H = mx.array([[0.0, 1.0], [1.0, 0.0]])
    >>> U = expm(-1j * H * 0.5)  # quantum time evolution
    """
    A = mx.array(A)

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected a square matrix, got shape {A.shape}")

    # Ensure at least float32 or complex64
    if A.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64,
                   mx.int16, mx.int8, mx.uint16, mx.uint8):
        A = A.astype(mx.float32)

    n = A.shape[-1]

    # Handle 1x1 trivially
    if n == 1:
        return mx.exp(A)

    # Scaling step: find s such that ||A / 2^s||_1 <= theta_13
    mx.eval(A)
    norm_A = _onenorm(A)

    if norm_A == 0.0:
        return _eye_like(n, A.dtype)

    s = max(0, int(np.ceil(np.log2(norm_A / _THETA_13))))
    if s > 0:
        A_scaled = A * (2.0 ** (-s))
    else:
        A_scaled = A

    # Pade [13/13] approximant
    U, V = _pade13(A_scaled)
    R = _solve_P_Q(U, V)

    # Squaring phase: R = R^{2^s}
    for _ in range(s):
        R = R @ R

    mx.eval(R)
    return R


def expm_frechet(
    A: mx.array,
    E: mx.array,
    compute_expm: bool = True,
) -> Tuple[mx.array, mx.array] | mx.array:
    """Compute the Frechet derivative of the matrix exponential.

    The Frechet derivative L(A, E) is the unique linear map satisfying:
        expm(A + t*E) = expm(A) + t * L(A, E) + O(t^2)

    Uses the block-triangular trick (Kenney & Laub 1998, Van Loan 1978):
        expm([[A, E], [0, A]]) = [[expm(A), L(A,E)], [0, expm(A)]]

    Parameters
    ----------
    A : mx.array, shape (n, n)
        Square matrix.
    E : mx.array, shape (n, n)
        Direction matrix.
    compute_expm : bool, default True
        If True, return (expm(A), L(A,E)). If False, return only L(A,E).

    Returns
    -------
    (expm_A, L) if compute_expm is True, else L.
    """
    A = mx.array(A)
    E = mx.array(E)

    if A.shape != E.shape:
        raise ValueError(f"A and E must have the same shape, got {A.shape} and {E.shape}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrices, got shape {A.shape}")

    n = A.shape[0]

    # Promote to common dtype
    dtype = A.dtype
    if E.dtype != dtype:
        if mx.issubdtype(E.dtype, mx.complexfloating) or mx.issubdtype(dtype, mx.complexfloating):
            dtype = mx.complex64
        else:
            dtype = mx.float32
        A = A.astype(dtype)
        E = E.astype(dtype)

    # Build the 2n x 2n block matrix [[A, E], [0, A]]
    zero_block = mx.zeros((n, n), dtype=dtype)
    top = mx.concatenate([A, E], axis=1)       # (n, 2n)
    bottom = mx.concatenate([zero_block, A], axis=1)  # (n, 2n)
    M = mx.concatenate([top, bottom], axis=0)  # (2n, 2n)

    result = expm(M)

    expm_A = result[:n, :n]
    L = result[:n, n:]

    if compute_expm:
        return expm_A, L
    else:
        return L


def _pade_log_approx(E: mx.array) -> mx.array:
    """Compute log(I + E) via [4/4] Pade approximant.

    Uses the diagonal Pade approximant for log(1+x) centered at x=0.
    The [4/4] approximant is:
        log(I+E) ~ E * p(E) * q(E)^{-1}
    where p and q are matrix polynomials chosen so that the approximation
    is accurate to O(E^9).

    For ||E|| < 0.25 this gives ~7 digits of accuracy (float32 sufficient).
    """
    n = E.shape[-1]
    ident = _eye_like(n, E.dtype)

    E2 = E @ E
    E3 = E2 @ E
    E4 = E2 @ E2

    # Coefficients for [4/4] Pade of log(1+x)/x:
    # Numerator: 1 + (35/12)x + (35/24)x^2 + (7/24)x^3 + (1/60)x^4  (... approximately)
    # We use the standard continued-fraction form instead.
    #
    # Actually, for better numerical stability, use the identity:
    # log(I+E) = E (I + E/2)^{-1}  is the [1/1] Pade (too low order).
    #
    # Better: use the recurrence from Higham "Functions of Matrices" Sec 11.5.
    # For float32 with ||E|| < 0.5, a degree-16 Taylor series is more than enough.
    # We use degree-16 Horner evaluation for stability:

    # log(I+E) = E - E^2/2 + E^3/3 - E^4/4 + ... via Horner in E:
    # Group as: E * (I - E*(1/2 - E*(1/3 - E*(1/4 - ...))))
    # This is more numerically stable than computing high powers.

    # Degree-16 Horner scheme:
    # c_k = (-1)^{k+1} / k, i.e. coefficients of E^k in log(1+E)
    # Horner: start from the innermost coefficient
    # S = c_{16}*I
    # S = c_{15}*I + E*S
    # ...
    # S = c_1*I + E*S = I + E*S  (since c_1=1, and then multiply by E at end)
    # Actually log(I+E) = sum_{k=1}^{16} (-1)^{k+1}/k * E^k
    # = E * sum_{k=0}^{15} (-1)^k/(k+1) * E^k
    # = E * [I - E/2 + E^2/3 - E^3/4 + ...]
    # Horner on the inner sum:

    S = (1.0 / 16.0) * ident   # coefficient for E^15 in inner sum: (-1)^15/16 = -1/16...
    # Actually let me just be explicit. Inner sum coefficients a_k = (-1)^k / (k+1)

    S = ident * (-1.0 / 16.0)  # a_15 = -1/16
    S = ident * (1.0 / 15.0) + E @ S    # a_14 = 1/15
    S = ident * (-1.0 / 14.0) + E @ S   # a_13
    S = ident * (1.0 / 13.0) + E @ S    # a_12
    S = ident * (-1.0 / 12.0) + E @ S   # a_11
    S = ident * (1.0 / 11.0) + E @ S    # a_10
    S = ident * (-1.0 / 10.0) + E @ S   # a_9
    S = ident * (1.0 / 9.0) + E @ S     # a_8
    S = ident * (-1.0 / 8.0) + E @ S    # a_7
    S = ident * (1.0 / 7.0) + E @ S     # a_6
    S = ident * (-1.0 / 6.0) + E @ S    # a_5
    S = ident * (1.0 / 5.0) + E @ S     # a_4
    S = ident * (-1.0 / 4.0) + E @ S    # a_3
    S = ident * (1.0 / 3.0) + E @ S     # a_2
    S = ident * (-1.0 / 2.0) + E @ S    # a_1
    S = ident + E @ S                     # a_0 = 1

    return E @ S


def _inverse_scaling_and_squaring_logm(
    A: mx.array,
    max_sqrt_iters: int = 32,
) -> mx.array:
    """Matrix logarithm via inverse scaling and squaring.

    Algorithm:
    1. Repeatedly take square roots: X <- sqrtm(X) until ||X - I|| < 0.5
       (count s square roots taken)
    2. Compute log(X) via degree-16 Horner evaluation of log(I + (X-I))
    3. Scale back: log(A) = 2^s * log(X)

    Uses tolerance 0.5 to limit the number of square root iterations,
    preserving float32 precision in the final Taylor/Horner evaluation.
    """
    n = A.shape[-1]
    ident = _eye_like(n, A.dtype)

    X = A
    s = 0

    # Repeated square roots to bring X close to I
    # Use tolerance 0.5 — the degree-16 series converges well for ||E|| < 0.5
    for _ in range(max_sqrt_iters):
        mx.eval(X)
        diff_norm = _onenorm(X - ident)
        if diff_norm < 0.5:
            break
        X = _denman_beavers_sqrt(X, max_iters=32, tol=1e-10)
        s += 1

    # Now X ~ I + E where ||E|| < 0.5
    E = X - ident
    log_X = _pade_log_approx(E)

    # Scale back
    result = log_X * (2.0 ** s)
    mx.eval(result)
    return result


def logm(A: mx.array) -> mx.array:
    """Compute the principal matrix logarithm of A.

    Uses inverse scaling and squaring with Denman-Beavers square roots.

    Parameters
    ----------
    A : mx.array, shape (..., n, n)
        Square matrix. Must have no eigenvalues on the closed negative real axis.

    Returns
    -------
    mx.array, shape (..., n, n)
        The principal matrix logarithm log(A).

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx_expm import expm, logm
    >>> A = mx.array([[1.0, 2.0], [0.0, 3.0]])
    >>> logm(expm(A))  # should recover A
    """
    A = mx.array(A)

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected a square matrix, got shape {A.shape}")

    if A.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64,
                   mx.int16, mx.int8, mx.uint16, mx.uint8):
        A = A.astype(mx.float32)

    n = A.shape[-1]
    if n == 1:
        return mx.log(A)

    return _inverse_scaling_and_squaring_logm(A)


def _denman_beavers_sqrt(
    A: mx.array,
    max_iters: int = 50,
    tol: float = 1e-8,
) -> mx.array:
    """Matrix square root via Denman-Beavers iteration.

    Iteration:
        Y_{k+1} = (Y_k + Z_k^{-1}) / 2
        Z_{k+1} = (Z_k + Y_k^{-1}) / 2

    Converges quadratically: Y_k -> A^{1/2}, Z_k -> A^{-1/2}.

    Uses mx.linalg.inv on CPU stream (unified memory makes this fast on
    Apple Silicon).
    """
    n = A.shape[-1]
    ident = _eye_like(n, A.dtype)

    Y = A
    Z = ident

    for _ in range(max_iters):
        Z_inv = mx.linalg.inv(Z, stream=mx.cpu)
        Y_inv = mx.linalg.inv(Y, stream=mx.cpu)
        Y_new = (Y + Z_inv) / 2.0
        Z_new = (Z + Y_inv) / 2.0

        # Check convergence
        mx.eval(Y_new, Z_new)
        diff = _onenorm(Y_new - Y)
        if diff < tol * _onenorm(Y_new):
            return Y_new

        Y = Y_new
        Z = Z_new

    return Y


def sqrtm(A: mx.array) -> mx.array:
    """Compute the principal matrix square root of A.

    Uses the Denman-Beavers iteration, which converges quadratically
    for matrices with no eigenvalues on the closed negative real axis.

    Parameters
    ----------
    A : mx.array, shape (..., n, n)
        Square matrix with no eigenvalues on the closed negative real axis.

    Returns
    -------
    mx.array, shape (..., n, n)
        The principal matrix square root S such that S @ S = A.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx_expm import sqrtm
    >>> A = mx.array([[4.0, 0.0], [0.0, 9.0]])
    >>> S = sqrtm(A)  # [[2, 0], [0, 3]]
    """
    A = mx.array(A)

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected a square matrix, got shape {A.shape}")

    if A.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64,
                   mx.int16, mx.int8, mx.uint16, mx.uint8):
        A = A.astype(mx.float32)

    n = A.shape[-1]
    if n == 1:
        return mx.sqrt(A)

    return _denman_beavers_sqrt(A)
