"""
Benchmark: mlx_expm.expm vs scipy.linalg.expm

Measures wall-clock time for matrix exponential on random matrices
of increasing size. Reports speedup factor.

Usage:
    python benchmark.py
    python benchmark.py --sizes 32 64 128 256 512 --trials 20
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from mlx_expm import expm


def benchmark_mlx(A_mx: mx.array, warmup: int = 3, trials: int = 10) -> float:
    """Benchmark mlx_expm.expm, return median time in seconds."""
    # Warmup
    for _ in range(warmup):
        R = expm(A_mx)
        mx.eval(R)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        R = expm(A_mx)
        mx.eval(R)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))


def benchmark_scipy(A_np: np.ndarray, warmup: int = 3, trials: int = 10) -> float:
    """Benchmark scipy.linalg.expm, return median time in seconds."""
    from scipy.linalg import expm as scipy_expm

    # Warmup
    for _ in range(warmup):
        scipy_expm(A_np)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        scipy_expm(A_np)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))


def check_accuracy(A_np: np.ndarray, A_mx: mx.array) -> float:
    """Return max absolute error between mlx_expm and scipy results."""
    from scipy.linalg import expm as scipy_expm

    ref = scipy_expm(A_np)
    res = np.array(expm(A_mx))

    return float(np.max(np.abs(res - ref)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark mlx_expm vs scipy")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Matrix sizes to benchmark",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--complex", action="store_true", help="Use complex matrices")
    parser.add_argument("--skip-scipy", action="store_true", help="Skip scipy comparison")
    args = parser.parse_args()

    has_scipy = True
    if not args.skip_scipy:
        try:
            import scipy.linalg  # noqa: F401
        except ImportError:
            print("scipy not found; skipping scipy comparison.\n")
            has_scipy = False

    print("=" * 72)
    print("mlx-expm benchmark")
    print("=" * 72)
    print(f"  Trials:  {args.trials}")
    print(f"  Warmup:  {args.warmup}")
    print(f"  Complex: {args.complex}")
    print(f"  Sizes:   {args.sizes}")
    print()

    header = f"{'n':>6s}  {'mlx (ms)':>10s}"
    if has_scipy and not args.skip_scipy:
        header += f"  {'scipy (ms)':>10s}  {'speedup':>8s}  {'max |err|':>12s}"
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((n, n)).astype(np.float64)
        if args.complex:
            A_np = A_np + 1j * rng.standard_normal((n, n))

        # Scale down so expm doesn't overflow
        A_np = A_np / n

        A_mx = mx.array(A_np.astype(np.complex64 if args.complex else np.float32))

        t_mlx = benchmark_mlx(A_mx, warmup=args.warmup, trials=args.trials)

        line = f"{n:>6d}  {t_mlx*1e3:>10.3f}"

        if has_scipy and not args.skip_scipy:
            t_scipy = benchmark_scipy(A_np, warmup=args.warmup, trials=args.trials)
            speedup = t_scipy / t_mlx if t_mlx > 0 else float("inf")
            err = check_accuracy(A_np, A_mx)
            line += f"  {t_scipy*1e3:>10.3f}  {speedup:>7.2f}x  {err:>12.2e}"

        print(line)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
