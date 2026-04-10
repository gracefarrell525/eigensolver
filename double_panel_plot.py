import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import sys
from pathlib import Path

plt.style.use("classic")

import disk_model_fin as model #DISK_MODEL = 6

def sigma_profile_plot(ax, model, sigma, xprof):
    ax.plot(xprof, sigma, lw=1.3, label="density profile")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R/a_b$", size=16)
    ax.set_ylabel(r"$\Sigma/\Sigma_0$", size=16)
    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-3, 5)
    ax.legend(frameon=False, fontsize=11)
    
def omega_potential_plot(ax, model, omegas, xprof):
    wpot = model.omegap(xprof)

    ax.plot(xprof, wpot, lw=2.0)
    for om in omegas:
        ax.axhline(om, lw=1.0, color = "red", alpha=0.7) 

    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_xlim(1, 500)
    ax.set_ylim(-2, 12)
    ax.set_ylabel(r"$\omega_{\rm pot}/(\frac{1}{2}h_0^2\Omega_b)$")

    ax.axvline(model.Rcav, lw=1.0, color="k", alpha=0.5)
    
def eigenfunctions_plot(ax, model, Emodes, x, num_modes):
    nmodes = Emodes.shape[0]
    nplot= min(num_modes, nmodes)
    for i in range(nplot):
        ax.plot(x, np.abs(Emodes[i]) / np.max(np.abs(Emodes[i])), label=f"mode {i}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$|E|$")
    ax.legend()
    
def contour_plot(ax, model, xprof, omegas, kR_max=15.0, NkR=500, n_highlight = 6, scale = None):
    if scale is None:
        scale = 1.0
    
    R = xprof
    kR = np.linspace(-kR_max, kR_max, NkR)
    RR, KRR = np.meshgrid(R, kR, indexing = 'xy') #creates 2D phase space
    
    wpot = model.omegap(RR)
    Omega = model.Omega(RR)
    cs2 = model.c2(RR)

    k = KRR / RR
    omega_map = (wpot - (cs2 / (2.0 * Omega)) * k**2) / scale
    
    finite = np.isfinite(omega_map)
    vmin = np.nanpercentile(omega_map[finite], 5)
    vmax = np.nanpercentile(omega_map[finite], 95)
    levels = np.linspace(vmin, vmax, 30) #30 contour levels within 5th and 95th percentile

    ax.contour(RR, KRR, omega_map, levels=30, linewidths=0.6, alpha=0.6)
    
    idx = np.argsort(omegas)[::-1]
    omegas_sorted = np.asarray(omegas)[idx]

    for j, om in enumerate(omegas_sorted[:n_highlight]):
        lev = (om / scale)
        ls = "-" if om >= 0 else "--"
        col = "red" if j == 0 else "black"

        ax.contour(RR, KRR, omega_map, levels=[lev], colors=col, linestyles=ls, linewidths=2.0)
        
        I, tps = phase_integral(model, xprof, om)
        if np.isfinite(I):
            ax.text(R[5], (0.85*kR_max) - j*(0.12*kR_max),
                    rf"$\omega={lev:.3g}$,  $I/\pi={I/np.pi:.2f}$",
                    fontsize=8, color=col)
    
    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$kR$")
    ax.set_xlim(1, 8)
    ax.set_ylim(-kR_max, kR_max)

def find_turning_points(x, f):
    s = np.sign(f)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    tps = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        f0, f1 = f[i], f[i+1]
        tps.append(x0 - f0*(x1-x0)/(f1-f0))
    return tps
    
def phase_integral(model, xprof, omega):
    wpot = model.omegap(xprof)
    f = wpot - omega
    tps = find_turning_points(xprof, f)
    if len(tps) < 2: #not a closed cavity if less than 2 turning pts
        return np.nan, None

    tp1, tp2 = tps[0], tps[-1]
    mask = (xprof >= tp1) & (xprof <= tp2) #restricts region b/w tps

    k2 = (2.0 * model.Omega(xprof[mask]) / model.c2(xprof[mask])) * np.maximum(wpot[mask] - omega, 0.0) # k^2 = (2*Omega/c_s^2)(omega_pot - omega)

    k = np.sqrt(k2)

    I = 2.0 * np.trapz(k, xprof[mask])
    return I, (tp1, tp2)

def apply_params(model, h0, qb, Rcav, l0, n, xout, Rout=None, taper_power=None):
    model.h0 = h0
    model.qb = qb
    model.Rcav = Rcav
    model.l0 = l0
    model.Qoverhsq = 0.25 * qb/(1+qb)**2 / h0**2
    model.xout = xout
    model.n = n

    # for finite model
    if Rout is not None and hasattr(model, "Rout"):
        model.Rout = Rout
    if taper_power is not None and hasattr(model, "taper_power"):
        model.taper_power = taper_power


if __name__ == "__main__":

    h0   = float(sys.argv[1])
    qb   = float(sys.argv[2])
    Rcav = float(sys.argv[3])
    l0   = float(sys.argv[4])
    Rout = float(sys.argv[5]) if len(sys.argv) > 5 else None
    xout = float(sys.argv[6])
    n    = int(sys.argv[7])
    taper = float(sys.argv[8]) if len(sys.argv) > 8 else None

    print(h0,qb,Rcav,l0,Rout,xout,n,taper)
        
    input_file = f"spectrum_h{h0:.3f}_q{qb:.4f}_Rcav{Rcav:.3f}_l0{l0:.3f}_Rout{Rout:.2f}_xout{xout:.2f}_n{n:.1f}_taper{taper:.2f}.npz"
    
    if Path(input_file).exists():
        print("File " + input_file + " exists. Great!")
    else:
        print("File " + input_file + " does not exist. Not great!")
        exit()
    
    file = np.load(input_file)
    
    
    x = file["x"] #radial grid used when solving ODEs
    omegas = file["omegas"] #eigenmodes
    nodes = file["nodes"] #number of 0 crossings in corrsponding eigenfunction
    
    print("reading", len(nodes), "modes.")

    Emodes = file["Emodes"] #actual eigenfunctions for every mode
    
    
    index = np.argsort(nodes)
    omegas_s = omegas[index]
    nodes_s = nodes[index]
    
    try:
        Emodes_s = Emodes[index, :]
    except:
        print("No modes found, continuing...")
    
    print(x.shape,omegas.shape,nodes.shape,Emodes.shape)

    apply_params(model, h0, qb, Rcav, l0, n, xout, Rout=Rout, taper_power=taper)

    xprof = np.logspace(np.log10(max(0.8*model.Rcav, 1.2*model.l0**2)), np.log10(min(model.xout, 1e4)), 4000)


    Sigma = model.S(xprof)
    Sigma0 = model.S(model.Rcav)

    sig = Sigma / Sigma0

    # plot 
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.94, wspace=0.25)

    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    
    sigma_profile_plot(ax0, model, sig, xprof)
    omega_potential_plot(ax1, model, omegas_s, xprof)
    eigenfunctions_plot(ax2, model, Emodes_s, x, 5) #cap mode plotting at 20
    contour_plot(ax3, model, xprof, omegas)
    
    #output
    out = f"double_panel_h{model.h0:.3f}_q{model.qb:.4f}_Rcav{model.Rcav:.3f}_l0{model.l0:.3f}_Rout{model.Rout:.2f}_xout{model.xout:.02f}_n{model.n:.1f}_taper{model.taper_power:.2f}.png"
    plt.savefig(out, dpi=200)
    print("Saved figure:", out)
