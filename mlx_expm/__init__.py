"""
mlx-expm: GPU-accelerated matrix exponential and related functions for Apple MLX.

MLX has no built-in expm. This package fills the gap with:
  - expm(A):              Matrix exponential (scaling + squaring, [13/13] Pade)
  - expm_frechet(A, E):   Frechet derivative of the matrix exponential
  - logm(A):              Matrix logarithm (inverse scaling + squaring)
  - sqrtm(A):             Matrix square root (Denman-Beavers iteration)

All operations run on Apple GPU via MLX.

Usage:
    from mlx_expm import expm, logm, sqrtm
    U = expm(-1j * H * t)   # quantum time evolution
"""

from mlx_expm.matrix_functions import expm, expm_frechet, logm, sqrtm

__version__ = "0.1.0"
__all__ = ["expm", "expm_frechet", "logm", "sqrtm"]
