#Created by Breschine Cummins on July 19, 2012.

# Copyright (C) 2012 Breschine Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

import numpy as np
from scipy.integrate import ode
import os
from cPickle import Pickler
import lib_ExactSolns as lES
try:
    import StokesFlow.utilities.fileops as fo
    print("not Stokesflow")
except:
    import utilities.fileops as fo
try:
    import StokesFlow.RegOldroydB.lib_Gridding as mygrids
    print("not Stokesflow")
except:
    import RegOldroydB.lib_Gridding as mygrids

def checkXLine(freqlist=[10*k for k in range(1,31)]):
    a = 1.0
    mu = 1.0
    rho = 1.0
    U = 1.0
    x = np.linspace(1.0,10.0,100)
    y = np.zeros(x.shape)
    z = np.zeroes(x.shape)
    u_fourier = []
    v_fourier = []
    w_fourier = []
    u_quasi = []
    v_quasi = []
    w_quasi = []
    for freq in freqlist:
        alph = np.sqrt(1j*2*pi*freq / (mu/rho))
        T = 10.0/(2*np.pi*freq)
        dt = 1.0/(2*np.pi*freq) / 40.0
        tvec = np.arange(0,T+dt,dt)
        uf,vf,wf = calcFourier(x,y,z,a,alph,freq,mu,tvec)
        u_fourier.append(uf)
        v_fourier.append(vf)
        w_fourier.append(wf)

def calcFourier(x,y,z,a,alph,freq,mu,tvec):
    soln_u = np.zeros((len(x),len(tvec)))
    soln_v = np.zeros((len(x),len(tvec)))
    soln_w = np.zeros((len(x),len(tvec)))
    for k in range(len(tvec)):
        out = lES.sphere3DOscillating(x,y,z,a,alph,freq,mu,tvec[k])
        soln_u[:,k] = out[0]
        soln_v[:,k] = out[1]
        soln_w[:,k] = out[2]
    return soln_u, soln_v, soln_w

def quasiWrapper(t,y,pdict):
    N = len(y)/3
    y = np.zeros((len(y),))
    out = lES.sphere3D(y[:N],y[N:2*N],y[2*N:],pdict['a'],pdict['mu'])
    y[:N] = out[0]
    y[N:2*N] = out[1]
    y[2*N:] = out[2]
    return y

def calcQuasiSteady():
    #initialize integrator
    r = ode(myodefunc).set_integrator('vode',method=method,rtol=rtol,nsteps=4000).set_initial_value(y0,t0).set_f_params(wdict) 



