from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from NEW_disk_model import DiskParams, DiskModel
from NEW_eigensolver import log_grid, solve_modes, find_turning_points, phase_integral
from NEW_plot_modes import (
    plot_surface_density,
    plot_omega_potential,
    plot_eccentricity_functions,
    plot_kR_contour,
    summarize_modes,
)


def str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "y", "on") #catches "yes-like" values from .sh file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--h0", type=float, required=True)
    p.add_argument("--qb", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--l0", type=float, required=True)

    p.add_argument("--xin", type=float, required=True)
    p.add_argument("--xout", type=float, required=True)

    p.add_argument("--Rcav", type=float, default=None)
    p.add_argument("--use_cavity", type=str, required=True)

    p.add_argument("--Rout", type=float, default=None)
    p.add_argument("--taper_power", type=float, default=3.0)
    p.add_argument("--use_outer_taper", type=str, required=True)

    p.add_argument("--inner_bc_kind", type=str, default="combo")
    p.add_argument("--outer_bc_kind", type=str, default="combo")
    
    p.add_argument("--thermo", type=str, default="isothermal")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--pindex", type=float, default=None)
    p.add_argument("--use_boundary_factor", type=str, default="true")

    p.add_argument("--nmodes", type=int, default=6)
    p.add_argument("--ngrid", type=int, default=600)

    p.add_argument("--savefig", type=str, default="false")
    p.add_argument("--outfile", type=str, default="modes.png")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    par = DiskParams(
        h0=args.h0,
        qb=args.qb,
        beta=args.beta,
        l0=args.l0,
        xin=args.xin,
        xout=args.xout,
        Rcav=args.Rcav,
        use_cavity=str2bool(args.use_cavity),
        Rout=args.Rout,
        taper_power=args.taper_power,
        use_outer_taper=str2bool(args.use_outer_taper),
        inner_bc_kind=args.inner_bc_kind,
        outer_bc_kind=args.outer_bc_kind,
        thermo=args.thermo,
        gamma=args.gamma,
        p=args.pindex,
        use_boundary_factor=str2bool(args.use_boundary_factor)
        )

    model = DiskModel(par) 
    x = log_grid(par.xin, par.xout, args.ngrid) 

    result = solve_modes(model, x, nmodes=args.nmodes, sort_by="nodes_then_omega", check_ode=True) # ascending_omega, descending_omega, nodes_then_omega 

    summarize_modes(model, x, result.omegas, result.modes) 

    print("\nVerification checks") 
    print("=" * 90)
    for i, check in enumerate(result.ode_checks):
        print(
            f"mode {i:02d}   "
            f"shape_error={check['shape_error']:.3e}   "
            f"left_bc={check['left_bc_residual']:.3e}   "
            f"right_bc={check['right_bc_residual']:.3e}"
        )

    fig = plt.figure(figsize=(15, 11))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.28, wspace=0.25)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    plot_surface_density(ax1, model, x)
    plot_omega_potential(ax2, model, x, result.omegas)
    plot_eccentricity_functions(ax3, model, x, result.omegas, result.modes)
    plot_kR_contour(ax4, model, x, result.omegas, nk=500)

    if str2bool(args.savefig):
        plt.savefig(args.outfile, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to {args.outfile}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
