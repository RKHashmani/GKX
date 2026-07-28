"""Testbed for interior-eigenvalue extraction on gyrokinetic-like spectra.

GKX's linear growth rate comes from a dense eigendecomposition: every eigenvalue
and eigenvector of an n x n operator, n = n_laguerre * n_hermite * ntheta, to
keep one. Measured cost scales as n^2.19, reaching ~60 min and 17 GB at the
published ITG resolution (32, 16) with ntheta = 64, which is what puts a
converged multi-device campaign out of reach.

The obstacle is the spectrum, not the implementation. Measured on a QA boundary:

    rightmost eigenvalue   0.143 - 0.127i    (the ITG mode)
    spectral radius        80.15
    ratio                  ~560

The wanted eigenvalue is deep in the interior. Plain Arnoldi converges to
extremal |lambda| and returns the large-|Im| modes instead -- verified, and the
error does NOT shrink with krylov_dim, which distinguishes wrong-region
convergence from under-convergence.

The literature on this exact operator is unambiguous. Roman, Kammerer, Merz &
Jenko, *Parallel Computing* **36**, 339 (2010) evaluate shift-and-invert, the
Cayley transform, and harmonic projection for GENE's linear gyrokinetic operator
and conclude that harmonic projection beats the spectral transformations "with a
gain of one order of magnitude at least", is "always at least five times
faster", needs only ~10-12 basis vectors, and -- decisively for a matrix-free
code -- requires no large linear solves. They also report that shift-and-invert
is a poor fit here precisely because the operator is only available implicitly,
so no good preconditioner exists for the inner solves.

**What this testbed establishes, and it is a negative result worth keeping.**
Harmonic extraction applied to a single Arnoldi pass does NOT recover the
interior eigenvalue, on the real operator or on a synthetic spectrum built to
mimic it. On the synthetic case (spectral radius 234, target |lambda| = 0.19,
ratio ~1200) neither standard nor harmonic Ritz values get close at m = 20, 40 or
60:

    m     standard best |err|    harmonic best |err|
    20         2.4e+00                2.1e+01
    40         2.4e+00                1.0e+01
    60         1.9e-01                6.7e+00

So harmonic extraction is not a drop-in replacement for the Rayleigh-Ritz step.
The paper's method is harmonic extraction *inside Krylov-Schur with restarts*:
the restarting is what repeatedly filters the subspace toward the target, and
without it the extraction has no good subspace to extract from. Any
implementation has to include the restart loop -- that is the algorithm, not an
optimization of it.

The synthetic generator is here so that loop can be developed against a spectrum
whose answer is known exactly, before being trusted on an operator where it is
not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def synthetic_spectrum(
    n: int = 400,
    *,
    interior: complex = 0.143 - 0.127j,
    imaginary_extent: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A dense operator with a known, gyrokinetic-like spectrum.

    One rightmost interior eigenvalue, a subdominant one, and a cloud of stable
    modes spread far along the imaginary axis -- the structure that defeats plain
    Krylov. Returned with its exact eigenvalues so any extraction scheme can be
    scored without a reference eigensolve.
    """

    generator = np.random.default_rng(seed)
    bulk = (generator.normal(size=n - 2) * 0.01 - 0.05) + 1j * generator.normal(
        size=n - 2
    ) * imaginary_extent
    eigenvalues = np.concatenate([[interior, 0.10 - 0.30j], bulk])
    basis, _ = np.linalg.qr(
        generator.normal(size=(n, n)) + 1j * generator.normal(size=(n, n))
    )
    matrix = basis @ np.diag(eigenvalues) @ basis.conj().T
    return matrix, eigenvalues


def arnoldi(matvec, v0: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Arnoldi with full reorthogonalization; returns the basis and Hessenberg."""

    n = v0.size
    basis = np.zeros((m + 1, n), dtype=complex)
    hessenberg = np.zeros((m + 1, m), dtype=complex)
    basis[0] = v0 / np.linalg.norm(v0)
    for i in range(m):
        w = matvec(basis[i])
        for _ in range(2):  # two passes: classical Gram-Schmidt loses orthogonality
            for j in range(i + 1):
                overlap = np.vdot(basis[j], w)
                w = w - overlap * basis[j]
                hessenberg[j, i] += overlap
        norm = np.linalg.norm(w)
        hessenberg[i + 1, i] = norm
        if abs(norm) < 1.0e-14:
            return basis[: i + 2], hessenberg[: i + 2, : i + 1]
        basis[i + 1] = w / norm
    return basis, hessenberg


def standard_ritz(hessenberg: np.ndarray) -> np.ndarray:
    m = hessenberg.shape[1]
    return np.linalg.eigvals(hessenberg[:m, :m])


def harmonic_ritz(hessenberg: np.ndarray, target: complex) -> np.ndarray:
    """Harmonic Ritz values about ``target``.

    Roman et al. (2010) Eq. (21): with the Arnoldi relation
    ``A V = V B + v b^H``, ``g = (B - target I)^-H b``, the harmonic Ritz values
    are the eigenvalues of ``B + g b^H``. Everything is m x m, so this costs
    nothing next to the matrix-vector products -- which is the whole appeal
    relative to a spectral transformation.
    """

    m = hessenberg.shape[1]
    block = hessenberg[:m, :m]
    residual = np.zeros(m, dtype=complex)
    residual[-1] = np.conj(hessenberg[m, m - 1])
    g = np.linalg.solve((block - target * np.eye(m)).conj().T, residual)
    return np.linalg.eigvals(block + np.outer(g, residual.conj()))


def score(
    matrix: np.ndarray, eigenvalues: np.ndarray, dimensions: tuple[int, ...], seed: int = 0
) -> list[dict]:
    """Score both extractions against the known rightmost eigenvalue."""

    truth = eigenvalues[int(np.argmax(eigenvalues.real))]
    generator = np.random.default_rng(seed)
    start = generator.normal(size=matrix.shape[0]) + 1j * generator.normal(
        size=matrix.shape[0]
    )

    rows = []
    for m in dimensions:
        _, hessenberg = arnoldi(lambda x: matrix @ x, start, m)
        standard = standard_ritz(hessenberg)
        harmonic = harmonic_ritz(hessenberg, target=truth)
        rows.append(
            {
                "m": m,
                "standard_best_error": float(np.abs(standard - truth).min()),
                "harmonic_best_error": float(np.abs(harmonic - truth).min()),
                "standard_max_real": float(standard.real.max()),
                "harmonic_max_real": float(harmonic.real.max()),
                "true_max_real": float(truth.real),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/interior_eigenvalue_testbed.json"),
    )
    args = parser.parse_args()

    matrix, eigenvalues = synthetic_spectrum(args.n)
    truth = eigenvalues[int(np.argmax(eigenvalues.real))]
    print(
        f"synthetic spectrum: n={args.n}, rightmost {truth.real:+.6f}{truth.imag:+.6f}i, "
        f"spectral radius {np.abs(eigenvalues).max():.1f}, "
        f"ratio {np.abs(eigenvalues).max() / abs(truth):.0f}"
    )
    rows = score(matrix, eigenvalues, tuple(args.dimensions))
    print(f"\n{'m':>5}{'standard |err|':>17}{'harmonic |err|':>17}")
    for row in rows:
        print(
            f"{row['m']:>5}{row['standard_best_error']:>17.3e}"
            f"{row['harmonic_best_error']:>17.3e}"
        )
    print(
        "\nNeither converges without restarts: harmonic extraction needs the "
        "Krylov-Schur restart loop, which is the algorithm rather than a tuning knob."
    )

    summary = {
        "kind": "interior_eigenvalue_testbed",
        "claim_level": "negative_result_single_pass_extraction_insufficient",
        "n": args.n,
        "true_rightmost": [float(truth.real), float(truth.imag)],
        "spectral_radius": float(np.abs(eigenvalues).max()),
        "rows": rows,
        "reference": (
            "Roman, Kammerer, Merz & Jenko, Parallel Computing 36, 339 (2010): "
            "harmonic projection beats shift-and-invert and Cayley by >=5x on the "
            "GENE linear operator, needs ~10-12 basis vectors, and requires no "
            "large linear solves"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
