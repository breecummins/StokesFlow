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

def calcFourier(x,y,z,a,alph,freq,mu,c,U,tvec):
    soln_u = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    soln_v = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    soln_w = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    for k in range(len(tvec)):
        out = lES.sphere3DOscillating(x,y,z,a,alph,freq,mu,c,U,tvec[k])
        soln_u[:,k] = out[0]
        soln_v[:,k] = out[1]
        soln_w[:,k] = out[2]
    return soln_u, soln_v, soln_w

def checkXLine_multfreqs(basedir,basename,freqlist=[10*k for k in range(1,31)]):
    '''
    Calculates motion along the x-axis caused by a sphere centered at the origin oscillating as the sum of sinusoids of all the frequencies in freqlist simultaneously. Also handles lists of one element - single frequencies.
    Uses Stokeslet+dipole soln, Fourier and quasi-steady methods. 
    '''
    a = 1.0
    mu = 1.0
    rho = 1.0
    U = np.array([1.0/len(freqlist),0.0,0.0])
    c = np.array([0.0,0.0,0.0])
    x = np.linspace(a,a*10.0,100) 
    y = np.zeros(x.shape)
    z = np.zeros(x.shape)
    pdict = {'a':a,'mu':mu,'borderinit':np.array([x[0],y[0],z[0]]),'centerinit':c,'freqlist':freqlist,'U':U}
    period = 1.0/min(freqlist)
    T = 10.0*period
    dt = period / 40.0
    tvec = np.arange(0,T+dt,dt)
    mydict = {'u_fourier':0.0,'v_fourier':0.0,'w_fourier':0.0,'u_quasi':0.0,'v_quasi':0.0,'w_quasi':0.0,'dt':dt,'freqlist':freqlist,'x':x,'y':y,'z':z,'pdict':pdict}
    for freq in freqlist:
        alph = np.sqrt(1j*2*np.pi*freq / (mu/rho))
        uf,vf,wf = calcFourier(x,y,z,a,alph,freq,mu,c,U,tvec)
        mydict['u_fourier']+=uf
        mydict['v_fourier']+=vf
        mydict['w_fourier']+=wf
    uq,vq,wq = calcQuasiSteady_multfreqs(x,y,z,pdict,dt,tvec)
    mydict['u_quasi']=uq
    mydict['v_quasi']=vq
    mydict['w_quasi']=wq
    pdict['Umsg'] = 'lambda t: np.array([U[0]*np.cos(2*np.pi*freq*t),0.0,0.0]) summed over frequency'
    fname = os.path.join(basedir,basename)
    F = open( fname+'.pickle', 'w' )
    Pickler(F).dump(mydict)
    F.close()
    
def vel_multfreqs(U,t,freqlist):
    u = np.zeros((3,))
    for freq in freqlist:
        u += U*np.array([np.cos(2*np.pi*freq*t),0.0,0.0])
    return u
        
def quasiWrapper_multfreqs(t,Y,pdict):
    N = len(Y)/3
    x = Y[:N]
    y = Y[N:2*N]
    z = Y[2*N:]
    d = np.array([x[0],y[0],z[0]]) - pdict['borderinit']
    center = pdict['centerinit'] + d
    U = vel_multfreqs(pdict['U'],t,pdict['freqlist'])
    out = lES.sphere3D(x,y,z,pdict['a'],pdict['mu'],center,U)
    Y = np.zeros((len(Y),))
    Y[:N] = out[0]
    Y[N:2*N] = out[1]
    Y[2*N:] = out[2]
    return Y

def calcQuasiSteady_multfreqs(x,y,z,pdict,dt,tvec):
    t0 = tvec[0] - dt/2 # will want to center-difference in time
    N= len(x)
    y0 = np.zeros((3*N,))
    y0[:N] = x
    y0[N:2*N] = y
    y0[2*N:] = z   
    M = len(tvec) +1 
    soln_x = np.zeros((len(x),M))
    soln_y = np.zeros((len(x),M))
    soln_z = np.zeros((len(x),M))
    soln_x[:,0] = x
    soln_y[:,0] = y
    soln_z[:,0] = z
    r = ode(quasiWrapper_multfreqs).set_integrator('vode',method='bdf',rtol=1.e-6,nsteps=4000).set_initial_value(y0,t0).set_f_params(pdict) 
    k=1
    while r.successful() and r.t < tvec[-1]:  
        r.integrate(t0+k*dt)
        soln_x[:,k] = r.y[:N]
        soln_y[:,k] = r.y[N:2*N]
        soln_z[:,k] = r.y[2*N:]
        k += 1
    soln_u = (soln_x[:,1:] - soln_x[:,:-1]) / dt
    soln_v = (soln_y[:,1:] - soln_y[:,:-1]) / dt
    soln_w = (soln_z[:,1:] - soln_z[:,:-1]) / dt
    return soln_u, soln_v, soln_w



def checkSurface_multfreqs(basedir,basename,freqlist=[10*k for k in range(1,31)]):
    '''
    Calculates motion along the x-axis caused by a sphere centered at the origin oscillating as the sum of sinusoids of all the frequencies in freqlist simultaneously. Also handles lists of one element - single frequencies.
    Uses Stokeslets over surface, Fourier and quasi-steady methods.
    '''
    a = 1.0
    mu = 1.0
    rho = 1.0
    U = np.array([1.0/len(freqlist),0.0,0.0])
    x = np.linspace(a,a*10.0,100) 
    y = np.zeros(x.shape)
    z = np.zeros(x.shape)
    xline = np.column_stack([x,y,z])
    sphere_array = a*np.loadtxt('voronoivertices0600.dat')
    pdict = {'a':a,'mu':mu,'borderinit':sphere_array,'freqlist':freqlist,'U':U}
    period = 1.0/min(freqlist)
    T = 10.0*period
    dt = period / 40.0
    tvec = np.arange(0,T+dt,dt)
    mydict = {'u_fourier':0.0,'v_fourier':0.0,'w_fourier':0.0,'u_quasi':0.0,'v_quasi':0.0,'w_quasi':0.0,'dt':dt,'freqlist':freqlist,'x':x,'y':y,'z':z,'pdict':pdict}
    for freq in freqlist:
        alph = np.sqrt(1j*2*np.pi*freq / (mu/rho))
        uf,vf,wf = calcFourier_surface(xline,pdict['borderinit'],a,alph,freq,mu,U,tvec)
        mydict['u_fourier']+=uf
        mydict['v_fourier']+=vf
        mydict['w_fourier']+=wf

def calcFourier_surface(obspts,nodes,a,alph,freq,mu,U,tvec):
    soln_u = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    soln_v = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    soln_w = np.zeros((len(x),len(tvec)),dtype=np.complex128)
    for k in range(len(tvec)):
        f = FourierSurfaceTraction()
        out = lES.BrinkmanletsExact(obspts,nodes,f,alph)
        soln_u[:,k] = out[0]
        soln_v[:,k] = out[1]
        soln_w[:,k] = out[2]
    return soln_u, soln_v, soln_w
    
if __name__ == '__main__':
    basedir = os.path.expanduser('~/CricketProject/QuasiSteadyVSFourier/')
    if not os.path.exists(basedir):
        basedir = '/Volumes/PATRIOT32G/CricketProject/QuasiSteadyVSFourier/'   
        if not os.path.exists(basedir):        
            print('Choose a different directory for saving files')
            raise(SystemExit)
    freqlist = [100]
    basename = 'freq100'
    checkXLine_multfreqs(basedir,basename,freqlist)
    #freqlist = range(10,21)
    #basename = 'multfreqs_010to020'
    #checkXLine_multfreqs(basedir,basename,freqlist)

