"""
The radiosity system and its solution (canestrad's hd_mv_mlt and hd_mgcr over meschach;
doc/legacy.rst). From the near-field counts (nearfield.counts), each face's row:

    opaque face i:   B_i − ρ_i Σ_j c_ij dFF B_j                                           = B0_i
    leaf, upper:     B_sup − Σ_j (c_ij > 0 ? ρ_sup : τ_inf) c_ij dFF B_j                   = B0_sup
    leaf, lower:     B_inf − Σ_j (c_ij > 0 ? τ_sup : ρ_inf) c_ij dFF B_j                   = B0_inf

(c_ij the receiver's net count of emitter face j: positive seen above, negative below; a sensor reflects all). The
system is solved by meschach's restarted MGCR (Leyk's modified generalised conjugate residuals), transcribed with its
order of operations: double precision, sequential sums. `solve` also offers scipy's GMRES on the same matrix-free
product (precision "full": its rounding is not canestrad's). Both converge in a few steps: the solve is a few
percent of a radiosity run, the near field's form factors the rest.
"""
import numpy as np
from .kernels import mgcr, product


SOLVERS = ("mgcr", "gmres")


def solve(name, rhs, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff,
          tolerance=1e-6, krylov=100, limit=1000):
    """The radiosities (x, steps) of the system, by meschach's MGCR (*name* "mgcr", canestrad's) or scipy's GMRES
    ("gmres", restarted every *krylov* steps), to a residual *tolerance* relative to the first one (x = 0: the
    right-hand side's), in *limit* steps at most."""
    if name == "mgcr":
        return mgcr(rhs, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff,
                    tolerance, krylov, limit)
    if name != "gmres":
        raise ValueError(f"the linear solver is one of {SOLVERS}, got '{name}'")
    from scipy.sparse.linalg import LinearOperator, gmres

    def apply(x):
        return product(np.ascontiguousarray(x, dtype=np.float64), np.zeros(rhs.shape[0]), indptr, faces, values,
                       receiver_faces, receiver_translucent, reflectance, transmittance, dff)
    steps = [0]

    def count(_):
        steps[0] += 1
    operator = LinearOperator((rhs.shape[0], rhs.shape[0]), matvec=apply, dtype=np.float64)
    x, info = gmres(operator, rhs, rtol=tolerance, atol=0., restart=krylov, maxiter=limit, callback=count,
                    callback_type="pr_norm")
    if info < 0:
        raise RuntimeError(f"GMRES failed ({info})")
    return x, steps[0]
