import numpy as np




R0=1.0
h0=0.1
Omega0=1.0


# Central binary
qb=0.9
Qoverhsq = 0.25 * qb/(1+qb)**2 / h0**2
eb=0.0



# 3D effects
coeff3D = 0

DISK_MODEL=5 # Munoz 2020 eq 4
#DISK_MODEL=6 #trial finite disk modeling
#DISK_MODEL=7 #possible third version with cutoff rather than tapering?

if (DISK_MODEL == 5):
    NMODES =2 
    SHOOT_OUT = False
    SHOOT_IN = True
    ISOTHERMAL = True
    R0 = 1.0
    Rcav = 2.5
    l0 = 0.7
    beta = 1.0
    xin = 0.1
    xout = 470.0 * R0
    GRID_POINTS = 15000
    LOG_ECC = False

if (DISK_MODEL == 6):
    NMODES =2 
    SHOOT_OUT = False
    SHOOT_IN = True
    ISOTHERMAL = True
    R0 = 1.0
    Rcav = 2.5 
    l0 = 0.7
    beta = 1.0
    xin = 0.1
    xout = 470.0 * R0
    GRID_POINTS = 15000
    LOG_ECC = False
    Rout = 25 #R/a_b? same as other R's?
    taper_power = 12 #for steepness of tapering, similar to zeta in inf disk?
    Rcutoff = 25 #might need to add a third disk model for this? straight cutoff, no tapering

    
if (ISOTHERMAL):
    GAMMA = 1.0
else:
    GAMMA = 1.5
    



GRID_BREAK = GRID_POINTS//2

    
if (DISK_MODEL == 5):
    
    p= 1.5 - beta #beta is a random probability distribution ftn? 
    zeta = 12
    
    def f(x):
        return np.exp(-(Rcav/x)**zeta + 1.e-16) 

    def fprimeoverf(x):
        return (zeta*(Rcav/x)**zeta)/x


    def fprimeprimeoverf(x):
        return (zeta*(Rcav/x)**zeta)**2/x**2  -zeta*(zeta+1)*(Rcav/x)**zeta/x**2
    
    def S(x):
        return f(x) * x**(-p) * (1. - l0 / x**0.5)

    def SprimeoverS(x):
        return (fprimeoverf(x) - p/x + l0 / 2 / x**1.5/(1-l0/x**0.5))
    
    def SprimeprimeoverS(x):
        return (1.0/x**2/(1-l0/x**0.5) * (p * (p + 1) - l0/x**0.5 * (p + 0.5) * (p + 1.5))\
                +fprimeoverf(x) /x * (l0 / x**0.5/(1 - l0/x**0.5) - 2 * p) \
                +fprimeprimeoverf(x))

if (DISK_MODEL ==6):
    
    p= 1.5 - beta #beta is a random probability distribution ftn? 
    zeta = 12
    
    def f(x):
        return np.exp(-(Rcav/x)**zeta + 1.e-16)  #keep?

    def fprimeoverf(x):
        return (zeta*(Rcav/x)**zeta)/x #keep?


    def fprimeprimeoverf(x):
        return (zeta*(Rcav/x)**zeta)**2/x**2  -zeta*(zeta+1)*(Rcav/x)**zeta/x**2 #keep? 
    
    def tap(x):
        return np.exp(-(x/Rout)**taper_power + 1.e-16) #added
    
    def tapprimeovertap(x):
        return ((-taper_power) * ((x**(taper_power - 1))/(Rout**taper_power))) #added

    def tapprimeprimeovertap(x):
        return (tapprimeovertap(x))**2 - taper_power*(taper_power-1) * (x**(taper_power-2))/\
        (Rout**taper_power) #added
        
    def S(x):
        return (f(x) * x**(-p) * (1. - l0 / x**0.5)) * tap(x) #fixed?

    def SprimeoverS(x):
        return (fprimeoverf(x) - p/x + l0 / 2 / x**1.5/(1-l0/x**0.5)) + tapprimeovertap(x)\
        #fixed?
    
    def SprimeprimeoverS(x):
        return ((1.0/x**2/(1-l0/x**0.5) * (p * (p + 1) - l0/x**0.5 * (p\
        + 0.5) * (p + 1.5)) + fprimeoverf(x) /x * (l0 / x**0.5/(1 - l0/\
        x**0.5) - 2 * p) + fprimeprimeoverf(x))) + 2 * (SprimeoverS(x))\
        + tapprimeprimeovertap(x) + (tapprimeovertap(x)**2) #fixed?


    
def f0(x):
    return 0.75 * qb/(1+qb)**2 * x**(-2) # forcing term
    
def c2(x):
    return h0**2 * Omega0**2 * R0**2 * x**(-beta)

def c2primeoverc2(x):
    return -beta/x

def c2primeprimeoverc2(x):
    return beta * (beta + 1) / x**2


def Omega(x):
    return Omega0 * x**(-1.5)

def P(x):
    return S(x) * c2(x) /GAMMA

def PprimeoverP(x):
    return (SprimeoverS(x) + c2primeoverc2(x))

def PprimeprimeoverP(x):
    return (SprimeprimeoverS(x) + 2 * SprimeoverS(x) * c2primeoverc2(x)
            + c2primeprimeoverc2(x))

if ISOTHERMAL:
    def omegap(x): #omega potential
        return -x**(-beta-0.5) * (0.75 - 0.25 * (x * SprimeoverS(x))**2\
                                  + (0.5 + x * c2primeoverc2(x)) * x * SprimeoverS(x) \
                                  + 0.5 * x**2 * SprimeprimeoverS(x) \
                                  + 2 * x * c2primeoverc2(x) + x**2 * c2primeprimeoverc2(x)\
                                  - 6 * x**(beta-3.0) * Qoverhsq
                                  + coeff3D * (3 + 1.5 * x * c2primeoverc2(x))) #hard code functions for finite/different model

else:
                               
    def omegap(x):
        return -x**(-beta-0.5) * (0.75 - 0.25 * (x * PprimeoverP(x))**2\
                               + (1.5 - 1./GAMMA)* x * PprimeoverP(x) \
                               + 0.5 * x**2 * PprimeprimeoverP(x) \
                               - 6 * x**(beta-3.0) * Qovershsq)

def bc_function(x):
    if (ISOTHERMAL):
        return -c2primeoverc2(x)
    else:
        return 0




def xi_from_x(x):
    return 2.0/(beta+0.5)*x**(0.5*beta+0.25)

def x_from_xi(xi):
    return ((beta+0.5)*xi/2)**(4.0/(2*beta+1))

def V(xi):# Liouville-form potential
    x = ((beta+0.5)*xi/2)**(4.0/(2*beta+1))
    return omegap(x) + (beta-1.5)*(beta+2.5)/4/(beta+0.5)**2/xi**2

def bc_function_liouville(xi):
    x =((beta+0.5)*xi/2)**(4.0/(2*beta+1))
    return -(0.25*(15-2*beta) + x * SprimeoverS(x)) * 2.0/(2*beta+1)/xi


def u(x):
   if (ISOTHERMAL):
       return (x**3 * S(x))**(-0.5)
   else:
       return (x**3 * P(x))**(-0.5)

def uprime(x):
    if (ISOTHERMAL):
        return -0.5 * S(x) * (3*x**2 * S(x) + x**3 * SprimeoverS(x)) * u(x)**3
    else:
        return -0.5 * P(x)* (3*x**2  + x**3 * PprimeoverP(x)) * u(x)**3

def uprimeoveru(x):
    if (ISOTHERMAL):
        return -0.5 * (3/x  + SprimeoverS(x))
    else:
        return -0.5 * (3/x  + PprimeoverP(x))

def Phi(xi):
    if (ISOTHERMAL):
        x =((beta+0.5)*xi/2)**(4.0/(2*beta+1))
        return x**(3.0/8-0.25*beta)
    else:
        return None


