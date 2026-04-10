from __future__ import print_function

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import solve_bvp, odeint,quad,trapezoid, quad_vec
from scipy.optimize import root,fsolve,anderson,fmin,fminbound,root_scalar
from scipy.special import gamma, jv, yv, iv, kv
from tqdm import tqdm

import disk_model_fin as model

cycle= plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use('classic')

ATOL=1e-11
RTOL=1e-13

DISK_MODEL=6 # 5 = inf disk from Munoz 2020. Make mock inf disk with model 6

def run_solver(xspan=None, **kwargs):
    #method: auto, scan, or wkb
    #xin: xlim or fixed
    
    verbose = bool(kwargs.get("verbose", True))
    
    def log(*args):
        if verbose:
            print(*args)
    
    method = kwargs.get("method", "auto")
    nmodes = int(kwargs.get("nmodes", model.n))
    mode_index = int(kwargs.get("mode_index", 0))
    ngrid = int(kwargs.get("ngrid", 2000))
    
    if xspan is None:
        xin, x_lim, x_peak, omega_peak = choose_xin(xin_mode=kwargs.get("xin_mode", "xlim"),
            xin_fixed=float(kwargs.get("xin", 0.01)),
            xin_use_max=bool(kwargs.get("xin_use_max", False)),
            log=log) # maybe issue is here? 
            
        xspan = np.logspace(np.log10(xin), np.log10(model.xout), 3*model.GRID_POINTS)
        log(f"[grid] xspan: N={len(xspan)} from {xspan[0]:.4g} to {xspan[-1]:.4g}")
        
    global BC_IN, BC_OUT
    BC_IN = model.bc_function(xspan[0])
    BC_OUT = model.bc_function(xspan[-1])
    log(f"[BC] BC_IN={BC_IN:.6g} BC_OUT={BC_OUT:.6g}")

    # method selection
    if method == "auto":
        method = "wkb" if nmodes <= 2 else "scan"
    log(f"[method] {method} (nmodes={nmodes})")

    if method == "wkb":
        # return 1 or 2 modes using WKB guess + shooting
        modes = []
        for n in range(mode_index, mode_index + nmodes):
            log(f"[wkb] solving n={n}")
            omega, sol = solve_shooting(xspan, n)
            nodes = count_nodes(sol[:,0])
            modes.append((omega, sol, nodes))
            log(f"[wkb] n={n} omega={omega:+.6e} nodes(y)={nodes}")
        return modes, xspan

    # scan method
    xprof = np.logspace(np.log10(xspan[0]), np.log10(xspan[-1]), 4000)
    wpot = model.omegap(xprof)
    wmax = np.nanmax(wpot)
    wmin = np.nanmin(wpot)
    omega_min = max(wmin, -5.0*wmax)
    omega_max = 0.9999*wmax
    log(f"[scan] omega range [{omega_min:+.6g}, {omega_max:+.6g}] (wmin={wmin:+.6g}, wmax={wmax:+.6g})")

    modes = scan_spectrum(xspan, omega_min, omega_max, ngrid=ngrid, max_modes=nmodes, verbose=verbose)
    log(f"[scan] found {len(modes)} unique node modes")
    return modes, xspan
    
def choose_xin(xin_mode="xlim", xin_fixed=0.01, xin_use_max = False, log = lambda *a, **k: None):
    x_peak, omega_peak = get_maximum()
    x_lim = root(lambda x: model.omegap(x) + 220.0*np.abs(omega_peak), 0.8*x_peak).x[0]
    
    if xin_mode == "xlim":
        xin = max(0.75*model.Rcav, x_lim) if xin_use_max else x_lim
    else:
        xin = xin_fixed
    
    log(f"[xin] x_peak={x_peak:.6g} omega_peak={omega_peak:.6g} x_lim={x_lim:.6g} xin={xin:.6g}")
    return float(xin), float(x_lim), float(x_peak), float(omega_peak)

def get_maximum():
    # Find a local maximum
    x_peak = fminbound(lambda x: -model.omegap(x), model.xin, 5.0, disp=0)
    omega_peak = model.omegap(x_peak)
    return x_peak, omega_peak

def get_turning_points(om):
    x_peak,_ = get_maximum()
    turn1 = root(lambda z: (om-model.omegap(z)),x_peak*0.85,tol=1e-20).x
    turn2 = root(lambda z: (om-model.omegap(z)),x_peak*1.1,tol=1e-20,method='lm').x #lambda function is for compactness, a way to quickly define a function without it existing in memory
    return turn1, turn2

def solve_wkb_objective(om,area,x_peak):
    tp1, tp2 = get_turning_points(om) 
    return 2*quad(lambda z: np.sqrt((model.omegap(z)-om)*z**(model.beta-1.5)),
                  tp1,tp2,epsabs=1e-20)[0] - area

def solve_wkb_frequency(n):
    x_peak,_ = get_maximum()
    area = np.pi*(2 * n + 1)
    sol = root(solve_wkb_objective,0.5,args=(area,x_peak),tol=1e-20,method='lm',
               options={'ftol':1e-9,'xtol':1e-12,'gtol':0,'eps':1e-9,'maxiter': 1000000000}).x
    return sol[0]


def odefun_ecc(z,x,om):
    E = z[0]
    Eprime = z[1]

    dE_dx = Eprime
    L = model.SprimeoverS(x)
    dEprime_dx = x**(model.beta -1.5) * om * E \
                 - (x*L + 3)/x * Eprime \
                 - ((x*L * (1. + model.beta) + \
                    model.beta * (1. - model.beta)) /x**2 + \
                    x**(model.beta - 5.0) * 6. * model.Qoverhsq) * E 
 
    return [dE_dx,dEprime_dx]

def objective1(y):
    bc = [1.0,-BC_IN]
    sol = odeint(odefun_ecc,bc, xspan,args=(y[0],),rtol=RTOL,atol=ATOL)#,mxordn=15,mxstep=2000)
    return [sol[:,1][-1]+BC_OUT * sol[:,0][-1]]

def objective1b(y):
    bc = [1.0,-BC_OUT] #De/dx = 0, want to change to E = 0 (or ability to do both)
    sol = odeint(odefun_ecc,bc, xspan[::-1],args=(y[0],),rtol=RTOL,atol=ATOL,mxordn=15,mxstep=20000)
    return [(sol[:,1][-1]+BC_IN*sol[:,0][-1])]
    
def shoot_and_residual(omega,xspan):
    # returns value of boundary condition at the opposite boundary and solution
    if model.SHOOT_OUT:
        bc = [1.0, -BC_IN]
        sol = odeint(odefun_ecc, bc, xspan, args=(omega,),
                     rtol=RTOL, atol=ATOL, mxordn=15, mxstep=30000)
        resid = sol[-1,1] + BC_OUT * sol[-1,0]
        return resid, sol

    elif model.SHOOT_IN:
        bc = [1.0, -BC_OUT]
        sol = odeint(odefun_ecc, bc, xspan[::-1], args=(omega,),
                     rtol=RTOL, atol=ATOL, mxordn=15, mxstep=30000)[::-1,:]
        resid = sol[0,1] + BC_IN * sol[0,0]
        return resid, sol

def count_nodes(E, eps=1e-14):
    #counting sign changes in E
    E = np.asarray(E)
    mask = np.abs(E) > eps
    if mask.sum() < 3:
        return 0
    s = np.sign(E[mask])
    # treat zeros as +1
    s[s == 0] = 1.0
    return np.sum(s[:-1] * s[1:] < 0)
    
def scan_spectrum(xspan, omega_min, omega_max, ngrid=400, max_modes=20, verbose=False):
    # bracket roots of residual, refine list, then return list of omega, solution, and nodes
    omegas = np.linspace(omega_min, omega_max, ngrid)
    vals = np.empty_like(omegas)

    # Evaluate residuals on grid
    for i, om in enumerate(tqdm(omegas, desc="Residual grid", leave=False)):
        try:
            vals[i] = shoot_and_residual(om, xspan)[0]
        except Exception:
            vals[i] = np.nan

    modes = []
    for i in tqdm(range(len(omegas)-1)):
        f1, f2 = vals[i], vals[i+1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1 == 0.0:
            omega_root = omegas[i]
            try:
                resid, sol = shoot_and_residual(omega_root, xspan)
            except Exception:
                continue
        elif f1 * f2 > 0:
            continue
        else:
            a, b = omegas[i], omegas[i+1]
            try:
               r = root_scalar(lambda om: shoot_and_residual(om, xspan)[0], bracket=(a, b), method="brentq", xtol=1e-14, rtol=1e-12, maxiter=200)
               if not r.converged:
                   continue
               omega_root = r.root
               resid, sol = shoot_and_residual(omega_root, xspan)
            except Exception:
                continue 
            
        y = sol[:,0]
        nodes = count_nodes(y)
            
        if verbose:
            print(f"[scan] omega={omega_root:+.6e} resid={resid:+.3e} nodes(y)={nodes}")
            
        modes.append((omega_root, sol, nodes))
            
        if len(modes) >= max_modes*5:
            break
        
    modes.sort(key=lambda t: (t[2], -t[0]))
    
        # keeping only unique modes
    unique = []
    seen_nodes = set()
    for om, sol, nd in modes:
        if nd in seen_nodes:
            continue
        unique.append((om, sol, nd))
        seen_nodes.add(nd)
        if len(unique) >= max_modes:
            break
    
    return unique


def solve_shooting(xspan, n):
    
    omega_guess = solve_wkb_frequency(n) 
    guess = [omega_guess]
    if (model.SHOOT_OUT):
        yb = root(objective1,guess,tol=1e-19,method='lm',options={'ftol':1e-20})
        omega = yb.x[0] #first soln to shooting 
        bc = [1.0,-BC_IN]
        sol = odeint(odefun_ecc,bc,xspan,args=(omega,),atol=ATOL,rtol=RTOL,mxordn=15,mxstep=3000) #replace with solve_ivp - more modern
    elif (model.SHOOT_IN):
        yb = root(objective1b,guess,method='lm',options={'ftol':ATOL,'xtol':RTOL,'gtol':0,'eps':1e-9,'maxiter': 2100000000})
        omega = yb.x[0]
        bc = [1.0,-BC_OUT]
        sol = odeint(odefun_ecc,bc,xspan[::-1],args=(omega,),atol=ATOL,rtol=RTOL,mxordn=15,mxstep=30000)[::-1,:]

    return omega,sol
    
def debug_residual_samples(xspan):
    test_omegas = [-10, -1, -0.1, 0.0, 0.1, 1, 10]
    for om in test_omegas:
        try:
            r, _ = shoot_and_residual(om, xspan)
            print(f"F({om:+.3e}) = {r:+.3e}")
        except Exception as e:
            print(f"F({om:+.3e}) failed: {e}")

if __name__ == '__main__':
    
    model.h0 = float(sys.argv[1])
    model.qb = float(sys.argv[2])
    model.Rcav = float(sys.argv[3])
    model.l0 = float(sys.argv[4])
    model.Qoverhsq = 0.25 * model.qb/(1+model.qb)**2/model.h0**2
    model.Rout = float(sys.argv[5])
    model.xout = float(sys.argv[6])
    model.n = int(sys.argv[7])
    model.taper_power = float(sys.argv[8])
    
    output_file = f"spectrum_h{model.h0:.3f}_q{model.qb:.4f}_Rcav{model.Rcav:.3f}_l0{model.l0:.3f}_Rout{model.Rout:.2f}_xout{model.xout:.2f}_n{model.n:.1f}_taper{model.taper_power:.2f}.npz"
    
    if Path(output_file).exists():
        print("File already exists. Exiting.")
        exit() # comment out if changing something in disk model but not .sh file
    
    # Find a local maximum
    x_peak = fminbound(lambda x: -model.omegap(x), model.xin,5.0,disp=0)
    omega_peak = model.omegap(x_peak)
    x_lim = root(lambda x: (model.omegap(x)+220*np.abs(omega_peak)),0.8*x_peak).x[0]
    xin=max(0.75*model.Rcav,x_lim)
    xin=x_lim 
    xspan = np.logspace(np.log10(xin), np.log10(model.xout), 3*model.GRID_POINTS)

    # Boundary conditions
    BC_IN = (model.bc_function(xspan[0]))
    BC_OUT = (model.bc_function(xspan[-1]))
    
    #omega bounds 
    xprof = np.logspace(np.log10(xspan[0]), np.log10(xspan[-1]), 4000)
    wpot = model.omegap(xprof)
    wmax = np.nanmax(wpot)
    wmin = np.nanmin(wpot)
    
    # margin
    omega_min = max(wmin, -5.0*wmax)
    omega_max = 0.9999*wmax
    modes = scan_spectrum(xspan, omega_min, omega_max, ngrid=2000, max_modes=model.n)    
    
    debug_residual_samples(xspan)

    print("Found modes:", len(modes))
    for j,(om, sol, nodes) in enumerate(modes):
        print(j, "omega=", om, "nodes=", nodes)
    
    # pack spectrum
    omegas = np.array([m[0] for m in modes])
    nodes  = np.array([m[2] for m in modes])
    E_modes = np.array([m[1][:,0] / model.u(xspan) for m in modes])  # physical E(R)
    
    output_name = f"spectrum_h{model.h0:.3f}_q{model.qb:.4f}_Rcav{model.Rcav:.3f}_l0{model.l0:.3f}_Rout{model.Rout:.2f}_xout{model.xout:.2f}_n{model.n:.1f}_taper{model.taper_power:.2f}"
    
    cfg = dict(
    method="auto",          # "auto" -> wkb if nmodes<=2 else scan
    nmodes=int(model.n),
    mode_index=0,           # used only for wkb
    xin_mode="xlim",        # "xlim" or "fixed"
    xin=0.01,               # used only if xin_mode="fixed"
    xin_use_max=False,      # True -> max(0.75*Rcav, x_lim)
    ngrid=2000,
    verbose=True
)

    modes, xspan = run_solver(**cfg)

    debug_residual_samples(xspan)

    print("Found modes:", len(modes))
    for j, (om, sol, nodes) in enumerate(modes):
        print(j, "omega=", om, "nodes(y)=", nodes)

    # pack and save
    omegas = np.array([m[0] for m in modes])
    nodes  = np.array([m[2] for m in modes])

    # Save BOTH y and E_phys so you can debug nodes + plot physical eccentricity
    Ymodes = np.array([m[1][:,0] for m in modes])
    Emodes = np.array([m[1][:,0] / model.u(xspan) for m in modes])

    print("[save] writing npz...")
    np.savez(output_name + ".npz",
        x=xspan,
        omegas=omegas,
        nodes=nodes,
        Ymodes=Ymodes,
        Emodes=Emodes
    )
    print("[save] done:", output_name + ".npz")