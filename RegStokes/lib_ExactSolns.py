#Created by Breschine Cummins on May 11, 2012.

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
import scipy.special as ss
import matplotlib.pyplot as plt

def circularCylinder2D(x,y,a,mu=1.0):
    '''
    Exact solution for the velocity field and pressure
    at (x,y) caused by a circular cylinder 
    of radius 'a' that is infinitely long in 
    the z direction (modeled as a cross-section
    in two dimensions) and is moving with a 
    speed of 1.0 in the x direction through 
    a Stokes fluid of viscosity 'mu'. 
    
    x and y are ndarrays of identical size. 
    The outputs u,v,p are the same size as x and y.
    '''
    f0 = 8*np.pi*mu/(1-2*np.log(a))*np.array([1.0, 0])
    rsqrd = x**2 + y**2
    fdotx = f0[0]*x + f0[1]*y
    HD1 = -(np.log(rsqrd) - a**2/rsqrd)/(8*np.pi)
    HD2 = fdotx*(1- a**2/rsqrd)/(4*np.pi*rsqrd)
    u = (f0[0]*HD1 + x*HD2)/mu
    v = (f0[1]*HD1 + y*HD2)/mu
    p = fdotx/(2*np.pi*rsqrd)
    return u,v,p

def circularCylinder2DOscillating(x,y,a,nu,mu,freq,vh,vinf,t=[0]):
    '''
    Exact solution for the velocity field and pressure
    at (x,y) and times in the list 't'
    caused by a circular cylinder 
    of radius 'a' that is infinitely long in 
    the z direction (modeled as a cross-section
    in two dimensions) and is oscillating at a
    frequency of 'freq' with peak magnitude 'vh'
    in the x direction through a Stokes fluid 
    with a background oscillating flow of 
    frequency 'freq' and peak magnitude 'vinf', 
    kinematic viscosity 'nu' and dynamic viscosity 'mu'. 
    
    x and y are vectors of identical length. 
    The outputs u,v,p have the same number
    of rows as the length of x and y, and the
    same number of columns as the length of 't'. 
    '''
    #calculate constants and radial distance
    om = 2*np.pi*freq
    lam = np.sqrt(1j*om/nu)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)    
    #velocity terms, space only
    sboth = (a**2/r**2)*ss.kv(2,lam*a) - ss.kv(2,lam*r)
    su = vinf - ( (vinf - vh)/ss.kv(0,lam*a) ) * ( -(a**2/r**2)*ss.kv(2,lam*a) + ss.kv(2,lam*r) + ss.kv(0,lam*r) + 2*np.cos(phi)**2 * sboth )
    sv = -( (vinf -vh)*np.sin(2*phi)/ss.kv(0,lam*a) ) * sboth     
    sp = -lam**2 * mu * ((a**2/r)*(vinf - vh)*ss.kv(2,lam*a)/ss.kv(0,lam*a) + vinf*r)*np.cos(phi)
    #add time 
    u = np.zeros((len(x),len(t)),np.complex128)
    v = np.zeros((len(x),len(t)),np.complex128)        
    p = np.zeros((len(x),len(t)),np.complex128)        
    for k in range(len(t)):
        u[:,k] = np.exp(1j*om*t[k]) * su
        v[:,k] = np.exp(1j*om*t[k]) * sv
        p[:,k] = np.exp(1j*om*t[k]) * sp
    return u,v,p

def sphere3D(x,y,z,a,mu=1.0):
    '''
    Exact solution for the velocity field and pressure
    at (x,y,z) caused by a sphere of radius 
    'a' moving with a speed of 1.0 in the x 
    direction through a Stokes fluid of viscosity 'mu'. 
    
    x,y,z are ndarrays of identical size. 
    The outputs u,v,w,p are the same size as x,y,z.
    '''
    pass

def sphere3DOscillating(x,y,z,a,alph,freq,mu=1.0,t=0):
    '''
    Exact solution for the velocity field and pressure
    at (x,y,z) and time 't' caused by a sphere 
    of radius 'a' at the origin oscillating at a frequency of 
    'freq' with peak magnitude of 1.0 in the x 
    direction through a Stokes fluid with dynamic
    viscosity 'mu'. 'alph = sqrt(i*omega/nu)' is 
    a complex number that contains the ratio of 
    angular frequency to kinematic viscosity.
    
    x,y,z are ndarrays of identical size. 
    The outputs u,v,w,p are the same size as x,y,z.
    xdrag, ydrag, and zdrag are scalars representing the
    total drag on the sphere (ref Pozrikidis 1997 p. 310, white book,
    but using my sign convention for lambda). 
    
    '''
    H1a, H2a = Brinkmanlets3D(a,alph)
    D1a, D2a = BrinkmanletDipoles3D(a,alph)
    f1 = mu*D2a/(H1a*D2a - H2a*D1a)
    g1 = -f1*H2a/D2a
    r = np.sqrt(x**2 + y**2 + z**2)
    H1, H2 = Brinkmanlets3D(r,alph)
    D1, D2 = BrinkmanletDipoles3D(r,alph)
    fdotx = f1*x + 0*y + 0*z
    gdotx = g1*x + 0*y + 0*z
    HD1 = f1*H1(r,alph) + g1*D1(r,alph)
    HD2 = fdotx*H2(r,alph) + gdotx*D2(r,alph)
    u = ((HD1 + x*HD2)/mu)*np.exp(1j*2*np.pi*freq*t)
    v = ((0   + y*HD2)/mu)*np.exp(1j*2*np.pi*freq*t)
    w = ((0   + z*HD2)/mu)*np.exp(1j*2*np.pi*freq*t)
    p = fdotx/(4*np.pi*r**3)
    xdrag = -6*np.pi*mu*a*(1 + alph*a + (alph**2*a**2)/9)*np.exp(1j*2*np.pi*freq*t) 
    ydrag = 0
    zdrag = 0
    return u,v,w,p,xdrag,ydrag,zdrag

def sphere3DOscillatingSurfaceTraction(x,y,z,a,alph,freq,mu=1.0,t=0):
    '''
    Exact solution for the pointwise surface traction
    at (x,y,z) and time 't' caused by a sphere 
    of radius 'a' at the origin oscillating at a frequency of 
    'freq' with peak magnitude of 1.0 in the x 
    direction through a Stokes fluid with dynamic
    viscosity 'mu'. 'alph = sqrt(i*omega/nu)' is 
    a complex number that contains the ratio of 
    angular frequency to kinematic viscosity.
    
    x,y,z are ndarrays of identical size and must represent 
    points on a sphere of a radius a.
    The outputs t1,t2,t3 are the same size as x,y,z' and are 
    calculated according to Pozrikidis 1997 pps. 303, 306, and 310, 
    white book, but using my sign convention for lambda. 
    
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    lam = a*alph
    R = alph*r
    C = -2*np.exp(-R)*(1 + 3/R + 3/R**2) + 6/R**2
    ft1 = 6*np.pi*mu*a*(1 + lam + (lam**2)/3)
    gt1 = -np.pi*a**3 * (6/lam**2) *(np.exp(lam) - 1 - lam -(lam**2)/3)
    ftdotx = ft1*x + 0*y + 0*z
    gtdotx = gt1*x + 0*y + 0*z
    t11 = ( (C - np.exp(-R)*(R+1))*ft1 + mu*np.exp(-R)*(6 + 6*R + 3*R**2 + R**3 )*gt1/r**2 ) / (4*np.pi*r**2)
    t12 = ( (np.exp(-R)*(R+1) -1 - 3*C)*ftdotx - mu*np.exp(-R)*(18 + 18*R + 7*R**2 + R**3 )*gtdotx/r**2 ) / (4*np.pi*r**4)
    t1 = ( t11 + x*t12 ) * np.exp(1j*2*np.pi*freq*t) 
    t2 = y*t12 * np.exp(1j*2*np.pi*freq*t) 
    t3 = z*t12 * np.exp(1j*2*np.pi*freq*t) 
    return t1,t2,t3

def test_sphere3DOscillating():
    a = 0.25
    nu = 1.1
    freq=71
    alph = np.sqrt(1j*freq/nu)
    phi = np.linspace(0,2*np.pi,100)
    x1 = a*np.cos(phi)
    y1 = a*np.sin(phi)
    z1 = np.zeros(x1.shape)
    x2 = np.zeros(x1.shape)
    y2 = a*np.cos(phi)
    z2 = a*np.sin(phi)
    x = np.column_stack([x1,x2])
    y = np.column_stack([y1,y2])
    z = np.column_stack([z1,z2])
    u,v,w = sphere3DOscillating(x,y,z,a,alph,freq,mu=1.0,t=0)
    erru = np.max(np.abs(np.real(u)-1) )
    errv = np.max(np.abs(np.real(v))   )
    errw = np.max(np.abs(np.real(w))   )
    print("The following should all be zero:")
    print(erru,errv,errw)
    
def Brinkmanlets3D(r,alph):    
    H1 = (np.exp(-r*alph)*(1+r*alph+r**2*alph**2) - 1) / (4*np.pi*r**3*alph**2)
    H2 = (-np.exp(-r*alph)*(3+3*r*alph+r**2*alph**2) + 3) / (4*np.pi*r**5*alph**2)
    return H1, H2

def BrinkmanletDipoles3D(r,alph):    
    D1 = (np.exp(-r*alph)*(1+r*alph+r**2*alph**2)) / (4*np.pi*r**3)
    D2 = (-np.exp(-r*alph)*(3+3*r*alph+r**2*alph**2)) / (4*np.pi*r**5)
    return D1, D2
    
def BrinkmanletsExact(obspts,nodes,f,alph,mu=1.0):
    '''
    Calculates velocity at obspts due to forces f at nodes
    in three space. alph is a parameter relating to either 
    frequency or porosity. mu is fluid viscosity. mu = 1.0 
    for a non-dim'l problem.
    
    '''
    vel = np.zeros((obspts.shape[0],3),dtype=type(alph))
    for k in range(obspts.shape[0]):
        pt = obspts[k,:]
        dif = pt - nodes
        r = np.sqrt((dif**2).sum(1))
        H1, H2  = Brinkmanlets3D(r,alph)
        N = nodes.shape[0]
        rows = np.zeros((3,3*N),dtype=type(alph))
        ind = 3*np.arange(N) 
        for j in range(3):            
            for i in range(3):
                if j == i:
                    rows[j,ind+i] = H1 + (dif[:,j]**2)*H2
                elif j < i:
                    rows[j,ind+i] = dif[:,j]*dif[:,i]*H2
                elif j > i:
                    rows[j,ind+i] = rows[i,ind+j]
            vel[k,j] = (rows[j,:]*f.flat).sum()
    return vel

    
    
    
    
    
    
    
    
    