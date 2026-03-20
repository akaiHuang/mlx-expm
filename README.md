# mlx-expm

GPU-accelerated matrix exponential and related functions for [Apple MLX](https://github.com/ml-explore/mlx).

MLX has no built-in `expm`. This package fills the gap.

## Functions

| Function | Description | Algorithm |
|----------|-------------|-----------|
| `expm(A)` | Matrix exponential | Scaling & squaring + [13/13] Pade |
| `expm_frechet(A, E)` | Frechet derivative of expm | Block-triangular (Van Loan) |
| `logm(A)` | Principal matrix logarithm | Inverse scaling & squaring |
| `sqrtm(A)` | Principal matrix square root | Denman-Beavers iteration |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
import mlx.core as mx
from mlx_expm import expm, logm, sqrtm

# Quantum time evolution: U = exp(-i H t)
H = mx.array([[0.0, 1.0], [1.0, 0.0]])  # Pauli X
t = 0.5
U = expm(-1j * H * t)

# Matrix logarithm (inverse of expm)
A = mx.array([[1.0, 2.0], [0.0, 3.0]])
L = logm(expm(A))  # recovers A

# Matrix square root
S = sqrtm(mx.array([[4.0, 0.0], [0.0, 9.0]]))  # [[2, 0], [0, 3]]

# Frechet derivative
A = mx.array([[1.0, 0.5], [0.0, 2.0]])
E = mx.array([[0.1, 0.0], [0.0, 0.1]])
expm_A, L = expm_frechet(A, E)
```

## Applications

- **Quantum mechanics**: Unitary evolution `U = exp(-iHt)`
- **Control theory**: State transition matrix `exp(At)`
- **Differential equations**: Matrix ODE solutions
- **Neural ODEs**: Continuous-time dynamics
- **Lie groups**: Exponential map on matrix Lie algebras

## Algorithm Details

### expm — Scaling and Squaring with [13/13] Pade

The same algorithm as `scipy.linalg.expm` (Higham 2005/2009):

1. **Scaling**: Find `s` such that `||A / 2^s||_1 <= theta_13 = 5.37`
2. **Pade**: Evaluate the [13/13] rational approximant `R_{13}(A/2^s)`
3. **Squaring**: Recover `exp(A) = R_{13}^{2^s}` via repeated squaring

This requires 13 matrix multiplications + 1 linear solve — all on MLX GPU.

### logm — Inverse Scaling and Squaring

1. Repeatedly compute square roots (via Denman-Beavers) until `||X - I||` is small
2. Evaluate `log(I + E)` via degree-8 Taylor series
3. Scale back by `2^s`

### sqrtm — Denman-Beavers Iteration

Quadratically convergent iteration:
- `Y_{k+1} = (Y_k + Z_k^{-1}) / 2`
- `Z_{k+1} = (Z_k + Y_k^{-1}) / 2`

Converges to `Y -> A^{1/2}`, `Z -> A^{-1/2}`.

## Benchmark

```bash
python benchmark.py
python benchmark.py --sizes 32 64 128 256 512 --complex
```

## Tests

```bash
pytest tests/ -v
```

## References

- Higham, *Functions of Matrices*, SIAM, 2008.
- Al-Mohy & Higham, "A New Scaling and Squaring Algorithm for the Matrix Exponential", SIAM J. Matrix Anal. Appl. 31(3), 2009.
- Al-Mohy & Higham, "Computing the Frechet Derivative of the Matrix Exponential", SIAM J. Matrix Anal. Appl. 30(4), 2009.

## License

MIT — Sheng-Kai Huang
