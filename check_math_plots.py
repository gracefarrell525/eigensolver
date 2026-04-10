import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")

import disk_model_fin as model  # disk_model_inf or disk_model_fin


def make_x_grid(model, n):
    xmin_safe = max(0.8 * model.Rcav, 1.2 * model.l0**2)
    xmax_safe = min(model.xout, 1e4)
    return np.logspace(np.log10(xmin_safe), np.log10(xmax_safe), n)


if __name__ == "__main__":
    model.h0 = 0.1
    model.qb = 0.0
    model.Rcav = 2.5
    model.l0 = 0.7
    model.Qoverhsq = 0.25 * model.qb/(1+model.qb)**2 / model.h0**2

    if hasattr(model, "Rout"):
        model.Rout = 25.0
    if hasattr(model, "taper_power"):
        model.taper_power = 12.0

#    x = make_x_grid(model, 500)
#    sparse_x = np.linspace(max(0.8 * model.Rcav, 1.2 * model.l0**2), min(model.xout, 1e4), 100)

    xmin = max(model.Rcav, 1.2*model.l0**2)
    xmax = min(model.xout, 5*model.Rout)
    sparse_x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    x = np.logspace(np.log10(xmin), np.log10(xmax), 500)

    S = model.S(sparse_x)
    
    floor = 1e-280 
    mask = np.isfinite(S) & (S > floor)
    
    xv = sparse_x[mask]
    Sv = S[mask]

    
    d0_num = Sv
    
    print("sparse_x:", sparse_x[:5], "...", sparse_x[-5:])
    print("S stats: finite", np.isfinite(S).sum(), "/", S.size,
          " >0", (S > 0).sum(), " !=0", (S != 0).sum(),
          " nan", np.isnan(S).sum())
    
    
    print("kept points:", xv.size)
    if xv.size:
        print("xv min/max:", xv.min(), xv.max())
        print("Sv min/max:", Sv.min(), Sv.max())
    
    # IMPORTANT: abort early if too few points
    if xv.size < 3:
        raise RuntimeError(f"Not enough valid points for gradient: {xv.size}")

    
    dS_dx = np.gradient(Sv, xv, edge_order=0)
    d2S_dx2 = np.gradient(dS_dx, xv, edge_order=0)
    d1_num = np.abs(dS_dx / Sv)
    d2_num = np.abs(d2S_dx2 / Sv)
 
    good = np.isfinite(d0_num) & np.isfinite(d1_num) & np.isfinite(d2_num)

    xv = xv[good]
    d0_num = d0_num[good]
    d1_num = d1_num[good]
    d2_num = d2_num[good]
    
    #S_dense_num = np.interp(x, sparse_x, S)  
    #d0_num = np.abs(S_dense_num) 

    d0_analytic = np.abs(model.S(x)) 
    d1_analytic = np.abs(model.SprimeoverS(x)) #dense      
    d2_analytic = np.abs(model.SprimeprimeoverS(x)) #dense
    #dS_dx = np.abs(np.gradient(S, sparse_x, edge_order=2)) #sparse x grid 
    #d2S_dx2 = np.abs(np.gradient(dS_dx, sparse_x, edge_order=2)) #sparse x grid

    #d1_num = dS_dx / S
    #d2_num = d2S_dx2 / S

    eps = 1e-30
#    frac1 = (d1_num - d1_analytic) / (np.abs(d1_analytic) + eps)
#    frac2 = (d2_num - d2_analytic) / (np.abs(d2_analytic) + eps)
#
    # plots 
#    fig = plt.figure(figsize=(12, 8))
#    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.28)
#
#    ax1 = fig.add_subplot(221)
#    ax2 = fig.add_subplot(222)
#    ax3 = fig.add_subplot(223)
#    ax4 = fig.add_subplot(224)

    fig = plt.figure(figsize=(15, 5))  # wider, shorter works better for 1x3
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, wspace=0.3)
    
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    #S
    ax1.plot(x, d0_analytic, lw=1.2, label="analytic $\Sigma$")
    ax1.plot(xv, d0_num, marker = "o", ls =  None, label=r"num $\Sigma$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$x = R/a_b$")
    ax1.set_ylabel(r"$\Sigma'/\Sigma$")
    ax1.set_ylim(1e-3, 1)
    ax1.legend(frameon=False, fontsize=10)
    
    #SprimeoverS
    ax2.plot(x, d1_analytic, lw=1.2, label="analytic $\Sigma '/\Sigma$")
    ax2.plot(xv, d1_num, marker = "o", ls =  None, label=r"num grad($\Sigma)/\Sigma$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$x = R/a_b$")
    ax2.set_ylabel(r"$\Sigma'/\Sigma$")
    ax2.set_ylim(1e-3, 1)
    ax2.legend(frameon=False, fontsize=10)
    
    #SprimeprimeoverS
    ax3.plot(x, d2_analytic, lw=1.2, label=r"analytic $\Sigma ''/\Sigma$")
    ax3.plot(xv, d2_num, marker = "o", ls =  None, label=r"num grad(grad($\Sigma))/\Sigma$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$x = R/a_b$")
    ax3.set_ylabel(r"$\Sigma''/\Sigma$")
    ax3.set_ylim(1e-3, 1)
    ax3.legend(frameon=False, fontsize=10)


#    # SprimeoverS
#    ax1.plot(x, d1_analytic, lw=1.2, label="analytic $\Sigma '/\Sigma$")
#    ax1.plot(x, d1_num, marker = "o", ls =  None, label=r"num grad($\Sigma)/\Sigma$")
#    ax1.set_xscale("log")
#    ax1.set_yscale("log")
#    ax1.set_xlabel(r"$x = R/a_b$")
#    ax1.set_ylabel(r"$\Sigma'/\Sigma$")
#    ax1.legend(frameon=False, fontsize=10)
#
#    # fraction
#    ax2.plot(x, frac1, lw=1.0)
#    ax2.set_xscale("log")
#    ax2.set_yscale("log")
#    ax2.set_xlabel(r"$x = R/a_b$")
#    ax2.set_ylabel("fractional diff (1st)")
#    ax2.axhline(0, lw=0.8)
#
#    #SprimeprimeoverS
#    ax3.plot(x, d2_analytic, lw=1.2, label=r"analytic $\Sigma ''/\Sigma$")
#    ax3.plot(x, d2_num, marker = "o", ls =  None, label=r"num grad(grad($\Sigma))/\Sigma$")
#    ax3.set_xscale("log")
#    ax3.set_yscale("log")
#    ax3.set_xlabel(r"$x = R/a_b$")
#    ax3.set_ylabel(r"$\Sigma''/\Sigma$")
#    ax3.legend(frameon=False, fontsize=10)
#
#    #fraction
#    ax4.plot(x, frac2, lw=1.0)
#    ax4.set_xscale("log")
#    ax4.set_yscale("log")
#    ax4.set_xlabel(r"$x = R/a_b$")
#    ax4.set_ylabel("fractional diff (2nd)")
#    ax4.axhline(0, lw=0.8)

    plt.show()
