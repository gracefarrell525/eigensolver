import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import sys

plt.style.use('classic')

import disk_model_fin as model




global coeff0, coeff1, coeff2

def xi(x):
    return (4/(2*model.beta+1)) * x**(0.25*(2*model.beta+1))

def R(xi):
    return ((0.5*model.beta + 0.25)*xi)**(1.0/(0.5*model.beta + 0.25))

def V(xi):
    x = R(xi)
    V = model.omegap(x) + 1.0/16*(model.beta-1.5)*(model.beta+2.5)*x**(-0.5*(2*model.beta+1))

    return V


eigenfunction_data='./'

if __name__ == '__main__':
    
    h0 = float(sys.argv[1])
    qb= float(sys.argv[2])
    Rcav = float(sys.argv[3])
    l0 = float(sys.argv[4])
    
    filename=eigenfunction_data+'eigenfunction_h%.3f_q%.4f_Rcav%.3f_l0%.3f.txt'%(h0,qb,Rcav,l0)


    data=np.loadtxt(filename)


    x=data[0,1:]
    y=data[1,1:]
    omega=data[1,0]
    

    model.Rcav = Rcav
    model.h0=h0
    model.l0 = l0
    model.qb = qb
    model.Qoverhsq= 0.25 * qb/(1+qb)**2/h0**2
    E = model.u(x) * y
    YY = y * x**(-model.beta/4-3.0/8)
    xx = x/model.Rcav
    
    
    #############################################
    # Plot figure
    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(left=0.08,top=0.96,right=0.98,bottom=0.16,wspace=0.2)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)


    
    # First panel ##################

    func = E/model.c2(x)
    ax0.plot(xx,func/func.max(),lw=0.8)
    ax0.set_xscale('log')
    ax0.set_yscale('log')

    ax0.set_ylim(1e-5,1.4)
    ax0.set_xlim(xx.min(),xx.max())
    ax0.set_xlabel(r'$R/R_{\rm cav}$',size=20)
    ax0.set_ylabel(r'$E$',size=20)
    
    # Second panel ##################
    ax1.plot(xx,y/y.max(),'b.')#,lw=2.0)
    ax1.set_xscale('log')
    ax1.set_ylim(0,1.2)
    ax1.set_xlim(xx.min(),xx.max())
    ax1.text(0.98,0.95,r'$\omega=%.6f\Omega_{\rm cav}$' % (omega*model.h0**2/2/(model.Rcav)**(-1.5)),
             size=22,ha='right',va='top',transform=ax1.transAxes)
    ax1.set_xlabel(r'$R/R_{\rm cav}$',size=20)
    ax1.set_ylabel(r'$y$',size=20)
    
    plt.show()
    ############################################
