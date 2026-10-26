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
from numba import njit

MACHINE_EPSILON = 2.2204460492503130808e-16


@njit(cache=True, nogil=True, error_model="numpy")
def product(x, out, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff):
    """hd_mv_mlt: out = A x, row by row of the receivers (*receiver_faces*: each receiver's first face)."""
    for r in range(indptr.shape[0] - 1):
        face = receiver_faces[r]
        upper = x[face]
        rho_sup = -reflectance[face]
        lower, tau_sup, rho_inf, tau_inf = 0.0, 0.0, 0.0, 0.0
        if receiver_translucent[r]:
            lower = x[face + 1]
            tau_sup = -transmittance[face]
            rho_inf = reflectance[face + 1]
            tau_inf = transmittance[face + 1]
        for index in range(indptr[r], indptr[r + 1]):
            j, count = faces[index], values[index]
            upper += x[j] * count * dff * (rho_sup if count > 0 else tau_inf)
            if receiver_translucent[r]:
                lower += x[j] * count * dff * (tau_sup if count > 0 else rho_inf)
        out[face] = upper
        if receiver_translucent[r]:
            out[face + 1] = lower
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def incoming(x, indptr, faces, values, receiver_faces, receiver_translucent, dff, n_faces):
    """Each face's irradiance from the near field (n_faces,): Σ_j |c_ij| dFF B_j over the emitter faces its disc sees,
    the upper face's from the positive counts, a leaf's lower face's from the negative ones; receivers beyond
    *n_faces* (the sensors) left out."""
    out = np.zeros(n_faces)
    for r in range(indptr.shape[0] - 1):
        face = receiver_faces[r]
        if face >= n_faces:
            continue
        for index in range(indptr[r], indptr[r + 1]):
            j, count = faces[index], values[index]
            if count > 0:
                out[face] += x[j] * count * dff
            elif receiver_translucent[r]:
                out[face + 1] -= x[j] * count * dff
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def _norm(vector):
    total = 0.0
    for i in range(vector.shape[0]):
        total += vector[i] * vector[i]
    return np.sqrt(total)


@njit(cache=True, nogil=True, error_model="numpy")
def _inner(a, b):
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


@njit(cache=True, nogil=True, error_model="numpy")
def _add_scaled(out, vector, scale):
    """v_mltadd(out, vector, scale, out): out += scale · vector (a copy when scale is 0, a plain sum when 1)."""
    if scale == 0.0:
        return
    for i in range(out.shape[0]):
        out[i] = out[i] + vector[i] if scale == 1.0 else out[i] + scale * vector[i]


@njit(cache=True, nogil=True, error_model="numpy")
def _upper_solve(matrix, rhs, size):
    """Usolve on the leading size × size block of an upper triangular matrix."""
    out = np.zeros(size)
    last = size - 1
    while last >= 0 and rhs[last] == 0.0:
        out[last] = 0.0
        last -= 1
    for i in range(last, -1, -1):
        total = 0.0
        for j in range(i + 1, last + 1):
            total += matrix[i, j] * out[j]
        out[i] = (rhs[i] - total) / matrix[i, i]
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def mgcr(rhs, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff,
         tolerance=1e-6, krylov=100, limit=1000):
    """hd_mgcr from x = 0: (x, steps)."""
    dimension = rhs.shape[0]
    x = np.zeros(dimension)
    residual = np.zeros(dimension)
    directions = np.zeros((krylov, dimension))
    alpha = np.zeros(krylov)
    beta = np.zeros(krylov)
    upper = np.zeros((krylov, krylov))
    initial = 0.0
    steps = 0
    done = False
    while steps < limit:
        product(x, residual, indptr, faces, values, receiver_faces, receiver_translucent, reflectance,
                transmittance, dff)
        for k in range(dimension):
            residual[k] = rhs[k] - residual[k]
        norm = _norm(residual)
        dd = norm
        if steps == 0:
            initial = abs(norm)
        if norm == 0.0:
            break
        directions[0, :] = residual
        i = 0
        while i < krylov and steps < limit:
            steps += 1
            product(directions[i], residual, indptr, faces, values, receiver_faces, receiver_translucent,
                    reflectance, transmittance, dff)
            if i < krylov - 1:
                following = directions[i + 1]
                following[:] = residual
                for j in range(i):
                    beta[j] = _inner(directions[j + 1], following)
                    _add_scaled(following, directions[j + 1], -beta[j])
                beta[i] = norm = _norm(following)
                if norm <= MACHINE_EPSILON * initial:
                    i -= 1
                    done = True
                    break
                scale = 1.0 / norm
                if scale == 0.0:
                    following[:] = 0.0
                elif scale != 1.0:
                    for k in range(dimension):
                        following[k] = scale * following[k]
                alpha[i] = _inner(directions[0], following)
            else:
                for j in range(i):
                    beta[j] = _inner(directions[j + 1], residual)
                norm = _inner(residual, residual)
                for j in range(i):
                    norm -= beta[j] * beta[j]
                if np.sqrt(abs(norm)) <= MACHINE_EPSILON * initial:
                    i -= 1
                    done = True
                    break
                if norm < 0.0:
                    i -= 1
                    steps -= 1
                    break
                beta[i] = np.sqrt(norm)
                alpha[i] = _inner(directions[0], residual)
                for j in range(i):
                    alpha[i] -= beta[j] * alpha[j]
                alpha[i] /= beta[i]
            for j in range(krylov):
                upper[j, i] = beta[j]
            ratio = alpha[i] / dd
            if abs(ratio - 1.0) <= MACHINE_EPSILON * initial:
                dd = 0.0
            else:
                ratio = 1.0 - ratio * ratio
                if ratio < 0.0:
                    break
                dd *= np.sqrt(ratio)
            if dd <= initial * tolerance:
                done = True
                break
            i += 1
        if i >= krylov:
            i = krylov - 1
        coefficients = _upper_solve(upper, alpha, i + 1)
        for j in range(i + 1):
            _add_scaled(x, directions[j], coefficients[j])
        if done:
            break
    return x, steps


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
