# Changelog

## 0.1.1 — 2026-05-05

### Added
- Measured benchmark results in `benchmark_results.md` (M1 Max, MLX
  0.31.1). 10 trials per configuration, median reported, 3 warm-up
  runs excluded. Reproduce with `python benchmark.py`.

### Changed
- `pyproject.toml` license metadata updated to PEP 639
  (`license = "MIT"`); the redundant `License :: OSI Approved :: MIT License`
  classifier has been removed to avoid the new setuptools warning.

### Performance (M1 Max, vs `scipy.linalg.expm`)
Real (float32 / float64):

| n    | mlx (ms) | scipy (ms) | speedup |
|-----:|---------:|-----------:|--------:|
| 1024 |    40.43 |      84.24 |   2.08x |
| 2048 |   311.83 |     581.80 |   1.87x |

Complex (complex64 / complex128):

| n   | mlx (ms) | scipy (ms) | speedup |
|----:|---------:|-----------:|--------:|
| 256 |     7.87 |      15.46 |   1.96x |
| 512 |    41.96 |      55.79 |   1.33x |

Crossover ~n=1024 real, ~n=256 complex. Accuracy is float32-limited
(max abs err scales as `~1e-6 * n`), matching a clean Pade-13 +
scaling-and-squaring implementation.

## 0.1.0 — 2026-03-20
- Initial release: GPU-accelerated matrix exponential / logarithm for MLX.
