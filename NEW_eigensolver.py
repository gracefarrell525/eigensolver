from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.linalg import eig
from scipy.integrate import solve_ivp

from NEW_disk_model import DiskModel, DiskParams


@dataclass # automatically creates an __init__ function and stores all the variables
class ModeSet:
    x: np.ndarray # radial grid (nx,)
    omegas: np.ndarray # eigenvals, (nmodes,)
    modes: np.ndarray # shape (nmodes, nx)
    A: np.ndarray # discretized differential operator plus the non-eigenvalue terms
    B: np.ndarray # the discretized weight matrix multiplying omega
    residual_norms: np.ndarray #for accuracy check
    ode_checks: list[dict] # list of dictionaries, for each mode, store diagnostic info


def log_grid(xin: float, xout: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(xin), np.log10(xout), n)

#finite differencing
def derivative_weights_first(x0: float, xpts: np.ndarray) -> np.ndarray:
    #returns weights w st the f'(x_0) = sum_j w_j f(x_j). Only use for boundary rows where we need robust one-sided derivative
    
    xpts = np.asarray(xpts, dtype=float) #where ftn is being sampled 
    m = len(xpts)
    A = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)
    #Aw = b

    dx = xpts - x0 #dx_j = x_j - x_0
    for k in range(m):
        A[k, :] = dx**k #1's, lin terms, quad terms, etc
    b[1] = 1.0
    return np.linalg.solve(A, b) #solves for finite diff weights


def find_turning_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # find r's where y(x) crosses 0. Lin. interp. (y = omega_p(x) - omega -> WKB turning pts)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.sign(y)
    s[s == 0.0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0.0)[0]
    #detecting turning pts, sets 0 to 1 so all -1 guarantee sign crossing

    roots = []
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        roots.append(x0 - y0 * (x1 - x0) / (y1 - y0)) # root estimation
    return np.array(roots, dtype=float)


def phase_integral(model: DiskModel, x: np.ndarray, omega: float) -> tuple[float, np.ndarray]:
    # computes I = int(k dR), bound states should satisfy I = (2n+1)pi

    f = model.omegap(x) - omega #function f, roots are tps
    tps = find_turning_points(x, f) # finding vals of x where f crosses 0
    if len(tps) < 2:
        print("No proper trapped region, phase integral not defined ):")
        return np.nan, tps

    left, right = tps[0], tps[-1]
    mask = (x >= left) & (x <= right)
    xx = x[mask] #subset domain within tps
    if xx.size < 2:
        print("Not enough points inside trapped region ):")
        return np.nan, tps
    
    lnx = np.log(xx)
    kR = model.kR(xx, omega)
    good = np.isfinite(kR)
    
    if good.sum() < 2:
        print("Found a candidate, but not enough points to compute integral ):")
        return np.nan, tps
    
    I = 2.0 * np.trapz(kR[good], lnx[good]) 
    print(I, tps)
    return I, tps

def boundary_elimination_coeffs(model: DiskModel, x: np.ndarray, side: str) -> tuple[float, float]:
    # E_0 = c_1 E_1 + c_2 E_2
    # E_(N-1) = c_1 E_(N-2) + c_2 E_(N-3)
    # computes c_1 and c_2 which replace boundary values.
    # sets E or E' to 0 for dirichlet or neumann

    if side == "inner":
        kind = model.par.inner_bc_kind
        x0 = x[0]
        pts = x[:3] #3 nearby points to improve convergence/approximation
    elif side == "outer":
        kind = model.par.outer_bc_kind
        x0 = x[-1]
        pts = x[-3:]
    else:
        raise ValueError("side must be inner or outer")

    if kind == "dirichlet":
        return 0.0, 0.0

    alpha = 0.0 if kind == "neumann" else model.boundary_alpha(side, x0)
    w = derivative_weights_first(x0, pts)

    if side == "inner":
        denom = w[0] + alpha
        return -w[1] / denom, -w[2] / denom
        #c_1 = - omega_1/omega_0 + alpha
        #c_2 = - omega_2/omega_0 + alpha

    denom = w[2] + alpha
    return -w[1] / denom, -w[0] / denom


def assemble_generalized_evp(model: DiskModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # makes A and B for AE = omegaBE

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        raise ValueError("Need at least 5 grid points.")
    
    #(pE')' + qE = omega r E
    p = model.p_flux(x)
    q = model.q_potential(x)
    r = model.r_weight(x)

    nint = n - 2 # number of unknowns
    A = np.zeros((nint, nint), dtype=float)
    B = np.zeros((nint, nint), dtype=float)

    cL1, cL2 = boundary_elimination_coeffs(model, x, "inner")
    cR1, cR2 = boundary_elimination_coeffs(model, x, "outer")

    #loop over interior grid pts
    for i in range(1, n - 1):
        row = i - 1
        dxm = x[i] - x[i - 1] # spacing to left
        dxp = x[i + 1] - x[i] # spacing to right
        dxc = 0.5 * (dxm + dxp) # avg cell width

        p_imh = 0.5 * (p[i - 1] + p[i])
        p_iph = 0.5 * (p[i] + p[i + 1])
        # approx p's at midpoints

        cim1 = p_imh / (dxm * dxc)
        ci = -p_imh / (dxm * dxc) - p_iph / (dxp * dxc) + q[i]
        cip1 = p_iph / (dxp * dxc)
        #coeffs for E_i-1, E_i, E_i+1

        if i - 1 == 0:
            A[row, 0] += cim1 * cL1
            if nint > 1:
                A[row, 1] += cim1 * cL2 # left coeffs
        else:
            A[row, row - 1] += cim1

        A[row, row] += ci # middle coeffs

        if i + 1 == n - 1:
            A[row, nint - 1] += cip1 * cR1
            if nint > 1:
                A[row, nint - 2] += cip1 * cR2 #right coeffs
        else:
            A[row, row + 1] += cip1

        B[row, row] = r[i]

    return A, B


def normalize_mode(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    # normalizing modes, returning normalized eigenvectors

    v = np.real_if_close(v).astype(float) #removes very small imaginary components
    norm2 = float(v @ (B @ v)) #v^TBv
    if np.isfinite(norm2) and abs(norm2) > 0.0:
        v = v / np.sqrt(abs(norm2)) # makes v^TBv = 1
    else:
        vmax = np.max(np.abs(v))
        if vmax > 0.0:
            v = v / vmax #max|v| = 1
    idx = np.argmax(np.abs(v))
    if v[idx] < 0.0:
        v = -v
    return v


def count_nodes(v: np.ndarray, eps: float = 1e-10) -> int:
    vv = np.asarray(v)
    mask = np.abs(vv) > eps * np.max(np.abs(vv))
    if mask.sum() < 3:
        return 0
    s = np.sign(vv[mask])
    s[s == 0.0] = 1.0
    return int(np.sum(s[:-1] * s[1:] < 0.0))


def solve_modes(
    model: DiskModel,
    x: np.ndarray,
    nmodes: int = 6,
    sort_by: str = "descending_omega",
    check_ode: bool = True,
) -> ModeSet:
 
    # solves eigenvalue problem, returns physically useful modes
    # filters out complex and outside of omega_p
    A, B = assemble_generalized_evp(model, x)
    evals, evecs = eig(A, B) # computes eigenvalues of A and B

    finite = np.isfinite(evals.real) & np.isfinite(evals.imag)
    realish = np.abs(evals.imag) < 1e-8 * np.maximum(1.0, np.abs(evals.real))
    keep = finite & realish #masks

    evals = evals[keep].real
    evecs = evecs[:, keep].real

    # Physical filter
    wpot = model.omegap(x)
    wmin = np.nanmin(wpot)
    wmax = np.nanmax(wpot)
    phys = (evals >= wmin - 1e-10 * max(1.0, abs(wmin))) & (evals <= wmax + 1e-10 * max(1.0, abs(wmax))) #boolean mask, selecting only valid e-vals
    evals = evals[phys]
    evecs = evecs[:, phys] #shape (nx, nmodes)

    interior_modes = np.array([normalize_mode(evecs[:, i], B) for i in range(evecs.shape[1])]) #applying normalization

    # reconstruct full modes including boundaries.
    cL1, cL2 = boundary_elimination_coeffs(model, x, "inner")
    cR1, cR2 = boundary_elimination_coeffs(model, x, "outer")
    modes = []
    for u in interior_modes:
        v = np.zeros(len(x), dtype=float)
        v[1:-1] = u
        v[0] = cL1 * v[1] + cL2 * v[2]
        v[-1] = cR1 * v[-2] + cR2 * v[-3]
        vmax = np.max(np.abs(v)) + 1e-30
        v /= vmax
        if v[np.argmax(np.abs(v))] < 0.0:
            v = -v
        modes.append(v)
    modes = np.array(modes)
    nodes = np.array([count_nodes(v) for v in modes])

    if sort_by == "descending_omega":
        order = np.argsort(evals)[::-1]
    elif sort_by == "ascending_omega":
        order = np.argsort(evals)
    elif sort_by == "nodes_then_omega":
        order = np.lexsort((-evals, nodes))
    else:
        raise ValueError("sort_by must be 'descending_omega', 'ascending_omega', or 'nodes_then_omega'")

    evals = evals[order]
    modes = modes[order]
    nodes = nodes[order]

    evals = evals[:nmodes]
    modes = modes[:nmodes]
    nodes = nodes[:nmodes]

    residual_norms = []
    for om, u in zip(evals, interior_modes[order][:nmodes] if len(interior_modes) else []): # checking for existence of interior modes
        res = A @ u - om * (B @ u)
        denom = np.linalg.norm(A @ u) + np.linalg.norm(om * (B @ u)) + 1e-30 # ||v|| = sqrt(sum(v_1^2))
        residual_norms.append(np.linalg.norm(res) / denom) # normalized residual
    residual_norms = np.array(residual_norms)

    ode_checks = []
    if check_ode:
        for om, v in zip(evals, modes):
            ode_checks.append(check_mode_with_ode(model, x, om, v))

    return ModeSet(
        x=x,
        omegas=evals,
        modes=modes,
        A=A,
        B=B,
        residual_norms=residual_norms,
        ode_checks=ode_checks,
    )


# ODE verification, plugging back in solved e-vals
# E'' + aE' + bE = wcE
def mode_rhs(model: DiskModel, omega: float):

    beta = model.par.beta
    Q = model.par.Qoverhsq

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        E, dE = y
        L = float(model.SprimeoverS(x))
        cterm = ((x * L * (1.0 + beta) + beta * (1.0 - beta)) / x**2 + 6.0 * Q * x ** (beta - 5.0))
        ddE = x ** (beta - 1.5) * omega * E - ((x * L + 3.0) / x) * dE - cterm * E
        return np.array([dE, ddE], dtype=float) #solving for E''

    return rhs #returns function for rhs of equation

# verification
def check_mode_with_ode(model: DiskModel, x: np.ndarray, omega: float, mode: np.ndarray) -> dict:

    x = np.asarray(x, dtype=float)
    v = np.asarray(mode, dtype=float)

    w = derivative_weights_first(x[0], x[:3])
    dv0 = float(np.dot(w, v[:3])) #w_0V_0 + w_1v_1 + w_2v_2
    
    #v[0] and dv0 -> init conds. for ODE integration

    rhs = mode_rhs(model, omega)
    sol = solve_ivp(rhs, (x[0], x[-1]), np.array([v[0], dv0]), t_eval=x, rtol=1e-9, atol=1e-11) # solve the rhs on the interval x[0] = x[1], with intitial conditions v[0] and dv0, return the solution evaluate on x, relative and absolute tolerances

    if not sol.success:
        return {
            "success": False,
            "message": sol.message,
            "shape_error": np.nan,
            "left_bc_residual": np.nan,
            "right_bc_residual": np.nan,
            "ode_mode": None,
        } # in case integration failed

    ode_mode = sol.y[0] #sol.y[0] is E(x), sol.y[1] is E'(x)

    #in order to compare shapes on same scale
    scale = float(np.dot(v, ode_mode) / (np.dot(ode_mode, ode_mode) + 1e-30))
    ode_mode *= scale

    vnorm = v / (np.max(np.abs(v)) + 1e-30)
    onorm = ode_mode / (np.max(np.abs(ode_mode)) + 1e-30)
    shape_error = np.linalg.norm(vnorm - onorm) / np.sqrt(len(x))

    left_alpha = 0.0 if model.par.inner_bc_kind == "neumann" else (model.boundary_alpha("inner", x[0]) if model.par.inner_bc_kind == "robin" else np.nan)
    right_alpha = 0.0 if model.par.outer_bc_kind == "neumann" else (model.boundary_alpha("outer", x[-1]) if model.par.outer_bc_kind == "robin" else np.nan)

    # approximating E'(x_in) and E'(x_out)
    dleft = float(np.dot(derivative_weights_first(x[0], x[:3]), ode_mode[:3]))
    dright = float(np.dot(derivative_weights_first(x[-1], x[-3:]), ode_mode[-3:]))

    if model.par.inner_bc_kind == "dirichlet":
        left_bc_residual = abs(ode_mode[0])
    else:
        left_bc_residual = abs(dleft + left_alpha * ode_mode[0])

    if model.par.outer_bc_kind == "dirichlet":
        right_bc_residual = abs(ode_mode[-1])
    else:
        right_bc_residual = abs(dright + right_alpha * ode_mode[-1])

    return {
        "success": True,
        "message": "ODE integration succeeded",
        "shape_error": shape_error,
        "left_bc_residual": left_bc_residual,
        "right_bc_residual": right_bc_residual,
        "ode_mode": ode_mode,
    }
