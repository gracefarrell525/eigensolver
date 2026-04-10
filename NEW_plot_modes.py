from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from NEW_eigensolver import find_turning_points, phase_integral, count_nodes

plt.style.use("classic")


def plot_surface_density(ax, model, x):
    sigma = model.S(x)
    ax.plot(x, sigma, lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$\Sigma(R)$")
    ax.set_title("Surface density profile")
    
    ax.set_xlim(1e-1, 50) #PLT 1 DOMAIN
    ax.set_ylim(1e-6, 5) #PLT 1 RANGE

    if model.par.has_cavity_in_domain():
        ax.axvline(model.par.Rcav, ls="--", lw=1.0, color="k", alpha=0.6)
    if model.par.has_taper_in_domain():
        ax.axvline(model.par.Rout, ls=":", lw=1.0, color="k", alpha=0.6)


def plot_omega_potential(ax, model, x, omegas):
    wpot = model.omegap(x)
    ax.plot(x, wpot, lw=2.0, label=r"$\omega_{\rm pot}(R)$")

    for i, om in enumerate(omegas):
        ax.axhline(om, lw=1.0, alpha=0.8, color = "r", label=fr"$\omega_{i} = {om:.3f}$" if i < 6 else None)

    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(r"$\omega_{\rm pot}$ vs $R/a_b$")
    ax.legend(frameon=False, fontsize=8)
    
    ax.set_xlim(1e-1, 50) #PLT 2 DOMAIN
    ax.set_ylim(-15, 0) #PLT 2 RANGE


def plot_eccentricity_functions(ax, model, x, omegas, modes):
    wpot = model.omegap(x)

    for i, (om, mode) in enumerate(zip(omegas, modes)):
        y = np.abs(mode)
        y /= np.max(y) + 1e-30

        ax.plot(x, y, lw=1.8, label=f"mode {i}")

        #tps = find_turning_points(x, wpot - om)
        #for tp in tps:
            #ax.axvline(tp, ls="--", lw=0.8, color="0.6", alpha=0.5)

        nodes = count_nodes(mode)
        ynodes = np.interp(nodes, x, y)
        #ax.plot(nodes, ynodes, "o", ms=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$|E|/|E|_{\max}$")
    ax.set_title("Eccentricity functions")
    ax.legend(frameon=False, fontsize=8)
    
    ax.set_xlim(1, 50) #PLT 3 DOMAIN
    ax.set_ylim(1e-5, 2) #PLT 3 RANGE


def plot_kR_contour(ax, model, x, omegas, nmodes_to_show=None, nk=500):
    R = x
    
    if nmodes_to_show is None:
        nmodes_to_show = len(omegas)
    else:
        nmodes_to_show = min(nmodes_to_show, len(omegas))
    
    kr_vals = [] #to set right y_lim
    for om in omegas[:nmodes_to_show]:
        kr = model.kR(R, om)
        good = np.isfinite(kr)
        if np.any(good):
            kr_vals.append(np.nanmax(kr[good]))
        if kr_vals:
            kR_max = 1.15 * max(kr_vals)
        else:
            kR_max = 20.0
    
    kR = np.linspace(-kR_max, kR_max, nk)
    RR, KRR = np.meshgrid(R, kR, indexing="xy")

    omega_map = model.omegap(RR) - (model.c2(RR) / (2.0 * model.Omega(RR))) * (KRR / RR) ** 2

    finite = np.isfinite(omega_map)
    vmin = np.nanpercentile(omega_map[finite], 5)
    vmax = np.nanpercentile(omega_map[finite], 95)
    levels = np.linspace(vmin, vmax, 40)

    ax.contour(RR, KRR, omega_map, levels=levels, linewidths=0.6, alpha=0.7)

    for i, om in enumerate(omegas[:nmodes_to_show]):
        ax.contour(RR, KRR, omega_map, levels=[om], linewidths=2.0)

        I, _ = phase_integral(model, x, om)
#        if np.isfinite(I):
#            ax.text(
#                R[5],
#                0.9 * kR_max - 0.12 * i * kR_max,
#                rf"mode {i}: $I/\pi={I/np.pi:.3f}$",
#                fontsize=8,
#            )

    ax.set_xscale("log")
    ax.set_xlabel(r"$R/a_b$")
    ax.set_ylabel(r"$kR$")
    ax.set_ylim(-kR_max, kR_max)
    ax.set_title(r"$kR$ contour plot")
    
    ax.set_xlim(0.1, 20) #PLT 4 DOMAIN
    ax.set_ylim(-50, 50) #PLT 4 RANGE


def summarize_modes(model, x, omegas, modes):
    print("\nMode summary")
    print("=" * 90)

    wpot = model.omegap(x)
    print(f"omega_p min = {np.nanmin(wpot):+.6e}")
    print(f"omega_p max = {np.nanmax(wpot):+.6e}")
    print("-" * 90)

    for i, (om, mode) in enumerate(zip(omegas, modes)):
        tps = find_turning_points(x, wpot - om)
        node_count = count_nodes(mode)
        I, _ = phase_integral(model, x, om)

        print(
            f"mode {i:02d}   omega={om:+.8e}   "
            f"nodes={node_count:2d}   turning_points={len(tps):2d}   "
            f"I/pi={I/np.pi if np.isfinite(I) else np.nan:.5f}"
        )
