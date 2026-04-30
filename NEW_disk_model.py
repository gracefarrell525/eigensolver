from __future__ import annotations # for using types before defining them

import numpy as np
from dataclasses import dataclass
from typing import Optional # input isn't optional, but could be "None" or some other option

ArrayLike = np.ndarray | float # "ArrayLike" can either be an np.ndarray or a float 


@dataclass
class DiskParams:
    # condenses into one disk model
    # optional cavity and outer taper
    # domain and profile shape are independent choices (not coupled)
    
    # Basic scalings
    R0: float = 1.0
    Omega0: float = 1.0
    h0: float = 0.1
    beta: float = 1.0 # beta = 1 means no flaring
    gamma: float = 1.0 # gamma = 1 means isothermal
    use_boundary_factor: bool = True #uses 1-l0*x^-1/2
    thermo: str = "isothermal" #munoz format, adiabatic is lee


    # Binary parameters
    qb: float = 0.9 # mass ratio (m_2/m_1)
    eb: float = 0.0 # binary eccentricity
    coeff3D: float = 0.0

    l0: float = 0.7 # net torque per unit accreted mass exerted by the CBD on the binary. 0.7 from Munoz Lithwick 2020
    p: float = None # if None, use p = 1.5 - beta, Optional[float] = None

    # Domain
    xin: float = 0.1
    xout: float = 1.0e3

    # Inner cavity controls
    Rcav: Optional[float] = 2.5 # Rcav can either be a float or None, currently 2.5
    cavity_steepness: float = 12.0
    use_cavity: bool = True

    # Outer taper controls
    Rout: Optional[float] = None # Rout can either be a float or None
    taper_power: float = 1.0
    use_outer_taper: bool = False

    # Boundary condition controls for the eigenproblem
    # "robin" (aE + b dE/dr = 0), "neumann" (dE/dr = 0), or "dirichlet" (E(R) = 0)
    # mess around with this and see how boundary conditions effect e-val solns
    inner_bc_kind: str = "combo"
    outer_bc_kind: str = "combo"

    # If alpha is None, use the Muñoz/Lithwick-style isothermal choice.
    inner_bc_alpha: Optional[float] = None
    outer_bc_alpha: Optional[float] = None

    @property # makes a method behave like a variable
    def isothermal(self) -> bool:
        return self.thermo.lower() == "isothermal"
    
    @property
    def is_adiabatic(self) -> bool:
        return self.thermo.lower() == "adiabatic"
    
    @property
    def power_law_index(self) -> float:
        return 1.5 - self.beta if self.p is None else self.p

    @property
    def Qoverhsq(self) -> float:
        return 0.25 * self.qb / (1.0 + self.qb) ** 2 / self.h0**2

    def has_cavity_in_domain(self) -> bool:
        # true only when I want a cavity and cavity radius lies inside domain
        return self.use_cavity and (self.Rcav is not None) and (self.Rcav > self.xin)

    def has_taper_in_domain(self) -> bool:
        # true only when I want a taper and taper radius is inside domain
        return self.use_outer_taper and (self.Rout is not None) and (self.Rout < self.xout)

class DiskModel:
    # disk physics 

    def __init__(self, params: DiskParams):
        self.par = params #store the input of params inside this object as self.par
        # __init__ runs automatically when I create an object, the constructor of the class.
        # self is the actual object I just created (made myself a disk :))
        # params: DiskParams means params should be an object of type DiskParams
         
    def c2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.par.h0**2 * self.par.Omega0**2 * self.par.R0**2 * x ** (-self.par.beta)

    def c2primeoverc2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return -self.par.beta / x

    def c2primeprimeoverc2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.par.beta * (self.par.beta + 1.0) / x**2

    def Omega(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.par.Omega0 * x ** (-1.5)
   
    def cavity_factor(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_cavity_in_domain():
            return np.ones_like(x)
        zeta = self.par.cavity_steepness
        Rcav = float(self.par.Rcav)
        return np.exp(-(Rcav / x) ** zeta)

    def dln_cavity_dx(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_cavity_in_domain():
            return np.zeros_like(x)
        zeta = self.par.cavity_steepness
        Rcav = float(self.par.Rcav)
        return zeta * (Rcav / x) ** zeta / x

    def d2ln_cavity_dx2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_cavity_in_domain():
            return np.zeros_like(x)
        zeta = self.par.cavity_steepness
        Rcav = float(self.par.Rcav)
        return -zeta * (zeta + 1.0) * (Rcav / x) ** zeta / x**2

#    def power_factor(self, x: ArrayLike) -> ArrayLike:
#        x = np.asarray(x, dtype=float)
#        p = self.par.power_law_index
#        return x ** (-p) * (1.0 - self.par.l0 / np.sqrt(x))

    def sigma_powerlaw_factor(self, x) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        p = self.par.power_law_index
        return (x**(-p))
        
    def dln_sigma_powerlaw_dx(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        p = self.par.power_law_index
        return -p / x
    
    def d2ln_sigma_powerlaw_dx2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        p = self.par.power_law_index
        return p / x**2
    
    def boundary_factor(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.use_boundary_factor:
            return np.ones_like(x)
        return 1.0 - self.par.l0 / np.sqrt(x)
        
    def dln_boundary_dx(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.use_boundary_factor:
            return np.zeros_like(x)
        l0 = self.par.l0
        return 0.5 * l0 * x**(-1.5) / (1.0 - l0 / np.sqrt(x))
        
    def d2ln_boundary_dx2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.use_boundary_factor:
            return np.zeros_like(x)
        l0 = self.par.l0
        f = 1.0 - l0 / np.sqrt(x)
        fp = 0.5 * l0 * x**(-1.5)
        fpp = -0.75 * l0 * x**(-2.5)
        return fpp / f - (fp / f)**2

#    def powerprime(self, x: ArrayLike) -> ArrayLike:
#        x = np.asarray(x, dtype=float)
#        p = self.par.power_law_index
#        return (-p) * x ** (-p - 1.0) + self.par.l0 * (p + 0.5) * x ** (-p - 1.5)

#    def powerprimeprime(self, x: ArrayLike) -> ArrayLike:
#        x = np.asarray(x, dtype=float)
#        p = self.par.power_law_index
#        return p * (p + 1.0) * x ** (-p - 2.0) - self.par.l0 * (p + 0.5) * (p + 1.5) * x ** (-p - 2.5)

#    def dln_power_dx(self, x: ArrayLike) -> ArrayLike:
#        x = np.asarray(x, dtype=float)
#        return self.powerprime(x) / self.power_factor(x)

#    def d2ln_power_dx2(self, x: ArrayLike) -> ArrayLike:
#        x = np.asarray(x, dtype=float)
#        p1 = self.powerprime(x)
#        p2 = self.powerprimeprime(x)
#        p0 = self.power_factor(x)
#        return p2 / p0 - (p1 / p0) ** 2

    def taper_factor(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_taper_in_domain():
            return np.ones_like(x)
        a = float(self.par.taper_power)
        Rout = float(self.par.Rout)
        return np.exp(-(x / Rout) ** a) 

    def dln_taper_dx(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_taper_in_domain():
            return np.zeros_like(x)
        a = float(self.par.taper_power)
        Rout = float(self.par.Rout)
        return -(a / Rout) * (x / Rout) ** (a - 1.0)

    def d2ln_taper_dx2(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if not self.par.has_taper_in_domain():
            return np.zeros_like(x)
        a = float(self.par.taper_power)
        Rout = float(self.par.Rout)
        return -(a * (a - 1.0) / Rout**2) * (x / Rout) ** (a - 2.0)

    def S(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.cavity_factor(x) * self.sigma_powerlaw_factor(x) * self.boundary_factor(x) * self.taper_factor(x)

    def SprimeoverS(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.dln_cavity_dx(x) + self.dln_sigma_powerlaw_dx(x) + self.dln_boundary_dx(x) + self.dln_taper_dx(x)

    def SprimeprimeoverS(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        d1 = self.SprimeoverS(x)
        d2ln = self.d2ln_cavity_dx2(x) + self.d2ln_sigma_powerlaw_dx2(x) + self.d2ln_boundary_dx2(x) + self.d2ln_taper_dx2(x)
        return d2ln + d1**2

    def P(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.S(x) * self.c2(x) / self.par.gamma

    def PprimeoverP(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.SprimeoverS(x) + self.c2primeoverc2(x)

    def PprimeprimeoverP(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.SprimeprimeoverS(x) + 2.0 * self.SprimeoverS(x) * self.c2primeoverc2(x) + self.c2primeprimeoverc2(x)
    
    def omegap_isothermal(self, x: ArrayLike) -> ArrayLike: # separated isothermal
        x = np.asarray(x, dtype=float)
        beta = self.par.beta
        return -x ** (-beta - 0.5) * (
            0.75
            - 0.25 * (x * self.SprimeoverS(x)) ** 2
            + (0.5 + x * self.c2primeoverc2(x)) * x * self.SprimeoverS(x)
            + 0.5 * x**2 * self.SprimeprimeoverS(x)
            + 2.0 * x * self.c2primeoverc2(x)
            + x**2 * self.c2primeprimeoverc2(x)
            - 6.0 * x ** (beta - 3.0) * self.par.Qoverhsq
            + self.par.coeff3D * (3.0 + 1.5 * x * self.c2primeoverc2(x))
        )
    
    def omegap_adiabatic(self, x: ArrayLike) -> ArrayLike: # separated adiabatic
        x = np.asarray(x, dtype=float)
        beta = self.par.beta
        return -x ** (-beta - 0.5) * (
            0.75
            - 0.25 * (x * self.PprimeoverP(x)) ** 2
            + (1.5 - 1.0 / self.par.gamma) * x * self.PprimeoverP(x)
            + 0.5 * x**2 * self.PprimeprimeoverP(x)
            - 6.0 * x ** (beta - 3.0) * self.par.Qoverhsq
        )
        
    def omegap(self, x: ArrayLike) -> ArrayLike: #choice between iso. and adia.
        x = np.asarray(x, dtype=float)
        if self.par.isothermal:
            return self.omegap_isothermal(x)
        return self.omegap_adiabatic(x)
        
#    def omegap(self, x: ArrayLike) -> ArrayLike:
#        #computing the precession-frequency profile depending on whether or not the disk is isothermal. If not iso, use adiabatic-like formula
#        x = np.asarray(x, dtype=float)
#        beta = self.par.beta
#        
#        if self.par.isothermal:
#            return -x ** (-beta - 0.5) * (
#                0.75
#                - 0.25 * (x * self.SprimeoverS(x)) ** 2
#                + (0.5 + x * self.c2primeoverc2(x)) * x * self.SprimeoverS(x)
#                + 0.5 * x**2 * self.SprimeprimeoverS(x)
#                + 2.0 * x * self.c2primeoverc2(x)
#                + x**2 * self.c2primeprimeoverc2(x)
#                - 6.0 * x ** (beta - 3.0) * self.par.Qoverhsq
#                + self.par.coeff3D * (3.0 + 1.5 * x * self.c2primeoverc2(x))
#            )
#
#        return -x ** (-beta - 0.5) * (
#            0.75
#            - 0.25 * (x * self.PprimeoverP(x)) ** 2
#            + (1.5 - 1.0 / self.par.gamma) * x * self.PprimeoverP(x)
#            + 0.5 * x**2 * self.PprimeprimeoverP(x)
#            - 6.0 * x ** (beta - 3.0) * self.par.Qoverhsq
#        )

    def default_robin_alpha(self, x: float) -> float:
        if self.par.isothermal:
            return self.c2primeoverc2(x)
        return 0.0

    def boundary_alpha(self, side: str, x: float) -> float:
        if side == "inner":
            alpha = self.par.inner_bc_alpha
        elif side == "outer":
            alpha = self.par.outer_bc_alpha
        else:
            raise ValueError("side must be 'inner' or 'outer'")
        return self.default_robin_alpha(x) if alpha is None else float(alpha)

    # separated p, q, and r for iso. and adia.
    def p_flux_isothermal(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return x**3 * np.maximum(self.S(x), 1e-300)
    
    def q_potential_isothermal(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        beta = self.par.beta
        L = self.SprimeoverS(x)
        c_term = (
            (x * L * (1.0 + beta) + beta * (1.0 - beta)) / x**2
            + 6.0 * self.par.Qoverhsq * x ** (beta - 5.0)
        )
        return self.p_flux_isothermal(x) * c_term
    
    def r_weight_isothermal(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return self.p_flux_isothermal(x) * x ** (self.par.beta - 1.5)
        
    def p_flux_adiabatic(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return np.maximum(self.S(x), 1e-300) * x**(3.0 - self.par.beta)
    
    def q_potential_adiabatic(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        sigc2 = np.maximum(self.S(x), 1e-300) * x**(-self.par.beta)
        dsigc2_dx = np.gradient(sigc2, x, edge_order=2)
        return (x**2 / self.par.gamma) * dsigc2_dx
    
    def r_weight_adiabatic(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return np.maximum(self.S(x), 1e-300) * x**1.5

    def p_flux(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if self.par.isothermal:
            return self.p_flux_isothermal(x)
        return self.p_flux_adiabatic(x)
        
    def q_potential(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if self.par.isothermal:
            return self.q_potential_isothermal(x)
        return self.q_potential_adiabatic(x)
    
    def r_weight(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        if self.par.isothermal:
            return self.r_weight_isothermal(x)
        return self.r_weight_adiabatic(x)

    # helper functions for contour map
    def k2(self, x: ArrayLike, omega: float) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return (2.0 * self.Omega(x) / self.c2(x)) * (self.omegap(x) - omega)

    def k(self, x: ArrayLike, omega: float) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        k2 = self.k2(x, omega)
        out = np.full_like(k2, np.nan, dtype=float)
        mask = k2 >= 0.0
        out[mask] = np.sqrt(k2[mask]) # leaves NaNs instead of writing complex numbers, since I am only concerned with where waves can propagate. 
        return out
    
    def kR2(self, x: ArrayLike, omega: float) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        return x**2 * self.k2(x, omega)

    def kR(self, x: ArrayLike, omega: float) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        kR2 = self.kR2(x, omega)
        out = np.full_like(kR2, np.nan, dtype=float)
        mask = kR2 >= 0.0
        out[mask] = np.sqrt(kR2[mask])
        return out
