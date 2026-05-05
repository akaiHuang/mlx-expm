# Benchmark Results

Machine: Apple M1 Max, macOS 26.1
Python: 3.11.12
MLX: 0.31.1
SciPy reference: `scipy.linalg.expm` (LAPACK / Pade, CPU)
Method: 10 trials per configuration, median reported. 3 warm-up runs excluded. Random seed `42`.

`mlx-expm` uses scaling-and-squaring with [13/13] Pade (same algorithm family as SciPy), all matmuls + linear solve dispatched to MLX GPU.

## Real (float32 / float64) — `python benchmark.py`

| n    | mlx (ms) | scipy (ms) | speedup | max \|err\| |
|-----:|---------:|-----------:|--------:|-----------:|
|   16 |    1.356 |      0.012 |   0.01x |   2.46e-07 |
|   32 |    0.935 |      0.026 |   0.03x |   3.48e-07 |
|   64 |    1.634 |      0.066 |   0.04x |   4.97e-07 |
|  128 |    2.448 |      0.358 |   0.15x |   8.91e-07 |
|  256 |    2.820 |      1.496 |   0.53x |   1.68e-06 |
|  512 |   13.369 |      7.219 |   0.54x |   1.97e-06 |
| 1024 |   40.430 |     84.244 |   2.08x |   2.71e-06 |
| 2048 |  311.831 |    581.800 |   1.87x |   4.04e-06 |

mlx input is `float32`; scipy input is `float64`. Errors below are `|res_mlx - res_scipy_f64|`, dominated by float32 precision of the MLX path.

## Complex (complex64 / complex128) — `python benchmark.py --sizes 32 64 128 256 512 --complex`

| n   | mlx (ms) | scipy (ms) | speedup | max \|err\| |
|----:|---------:|-----------:|--------:|-----------:|
|  32 |    5.156 |      0.071 |   0.01x |   5.47e-07 |
|  64 |   14.673 |      0.597 |   0.04x |   1.01e-06 |
| 128 |    8.164 |      2.142 |   0.26x |   1.17e-06 |
| 256 |    7.874 |     15.460 |   1.96x |   1.79e-06 |
| 512 |   41.957 |     55.793 |   1.33x |   3.06e-06 |

## Key Observations

1. **Crossover is around n=256-1024**: below this, the GPU dispatch / kernel-launch overhead dominates and SciPy's CPU LAPACK wins handily. Above this, the GPU pulls ahead.

2. **Real (float32) crossover is ~n=1024**: SciPy is 60-100x faster at n<=128, ~2x slower at n>=1024. Peak speedup observed: **2.08x at n=1024**.

3. **Complex (complex64) crossover is earlier (~n=256)**: complex matmul has higher arithmetic intensity, so the GPU wins sooner. Peak speedup: **1.96x at n=256**, **1.33x at n=512**.

4. **Accuracy is float32-limited**: max absolute error scales with matrix size and stays at the `~1e-6 * n` level expected for float32. For float64-grade accuracy, use SciPy on CPU. (MLX has no float64 GPU support on Apple Silicon, so this is a hardware constraint, not an algorithm bug.)

5. **No accuracy regression vs SciPy reference**: errors match a clean Pade-13 + scaling-squaring implementation; no Denman-Beavers / logm code path is exercised by `expm` benchmark.

## When mlx-expm Wins

- `n >= 1024` real, or `n >= 256` complex.
- Inputs already on the GPU (skips the host->device copy that the benchmark does include in the warm-up).
- Batch / pipeline contexts where SciPy's GIL-bound CPU expm would block other GPU work.
- Quantum simulation `exp(-iHt)` where `H` is naturally complex and at the moderate-n sweet spot.

## When SciPy Wins

- Small matrices (`n <= 256` real, `n <= 128` complex). LAPACK's tuned `dgemm` + cache locality beat GPU dispatch for these sizes.
- Need float64 accuracy.
- One-shot CPU-side computation with no surrounding GPU pipeline.

## Reproduce

```bash
pip install -e .
python benchmark.py
python benchmark.py --sizes 32 64 128 256 512 --complex
python benchmark.py --sizes 1024 2048
```
