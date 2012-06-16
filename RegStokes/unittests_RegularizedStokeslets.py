#Created by Breschine Cummins on May 6, 2012.

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
import os
import engine_RegularizedStokeslets as RS
import lib_RegularizedStokeslets as lRS
import lib_ExactSolns as lES
import viz_RegularizedStokeslets as vRS

def regCircularCylinder2D(x,y,a,eps,mu=1.0):
    '''
    Regularized approximation to the circular
    cylinder exact solution.

    x and y are ndarrays of identical size. 
    The outputs u and v are the same size as x and y.
    '''
    rsd = RS.StokesletsAndDipoles2DMixedSquareCubicCircleBCs(eps,mu,0.0,a)
    obspts = np.column_stack([x,y])
    nodes = np.zeros((1,2))
    BCvel = np.array([1.0,0.0])
    f0 = rsd.fcst*BCvel
    U = rsd.calcVel(obspts,nodes,f0)
    u = U[:,0]
    v = U[:,1]
    p = rsd.calcPressure(obspts,nodes,f0)
    return u, v, p

def testCircularCylinder2D():
    '''
    Checks velocity outside of a circle using L2 error;
    checks pressure outside of a circle using L2 error.
    
    '''
    print('Test case: Steadily translating circular cylinder in 2D...')
    h = 0.02
    xg,yg = np.meshgrid(np.arange(-0.5+h/2,0.5,h), np.arange(-0.5+h/2,0.5,h))
    x = xg.flatten()
    y = yg.flatten()
    a = 0.25
    u,v,p = lES.circularCylinder2D(x,y,a)
#    ulevs=vRS.contourCircle(xg,yg,a,u.reshape(xg.shape),os.path.expanduser('~/scratch/unittestu.pdf'),None,  "u for steady circle")
#    vlevs=vRS.contourCircle(xg,yg,a,v.reshape(xg.shape),os.path.expanduser('~/scratch/unittestv.pdf'),None,  "v for steady circle")
#    plevs=vRS.contourCircle(xg,yg,a,p.reshape(xg.shape),os.path.expanduser('~/scratch/unittestp.pdf'),None,  "p for steady circle")
    elist = [0.25/(5*2**k) for k in range(2,5)]
    err = np.zeros((len(elist),))
    errp = np.zeros((len(elist),))
    for k in range(len(elist)):
        eps = elist[k]
        ureg, vreg, preg = regCircularCylinder2D(x,y,a,eps)
        ind = np.nonzero(x**2+y**2 > a**2)
        L2err = h*np.sqrt( ((u[ind]-ureg[ind])**2).sum() + ((v[ind]-vreg[ind])**2).sum() )
        err[k] = L2err
#        vRS.contourCircle(xg,yg,a,ureg.reshape(xg.shape),os.path.expanduser('~/scratch/unittestureg%02d.pdf' % k),ulevs,  "Reg u for steady circle")
#        vRS.contourCircle(xg,yg,a,vreg.reshape(xg.shape),os.path.expanduser('~/scratch/unittestvreg%02d.pdf' % k),vlevs,  "Reg v for steady circle")
#        vRS.contourCircle(xg,yg,a,preg.reshape(xg.shape),os.path.expanduser('~/scratch/unittestpreg%02d.pdf' % k),plevs,  "Reg p for steady circle")
        L2errp = h*np.sqrt( ((p[ind]-preg[ind])**2).sum() )
        errp[k] = L2errp
    conv = np.zeros((len(elist)-1,))
    convp = np.zeros((len(elist)-1,))
    for k in range(len(elist)-1):
        conv[k]=np.log2(err[k]/err[k+1])
        convp[k]=np.log2(errp[k]/errp[k+1])
    if np.all(err < 1.e-3) and np.all(np.abs(conv-2) < 0.01):
        print("Velocity passed")
    else:
        print("Velocity failed")
    if np.all(errp < 3.e-3) and np.all(np.abs(convp-2) < 0.01):
        print("Pressure passed")
    else:
        print("Pressure failed")
#    print("Vel L2 error should be small:")
#    print(err)
#    print("Vel convergence rates should be approximately 2.0: ")
#    print(conv)
#    print("Pressure L2 error should be small:")
#    print(errp)
#    print("Pressure convergence rates should be approximately 2.0: ")
#    print(convp)

def regOscillatingSphere3D(rsd,obspts,spts,a,freq,t=0):
    '''
    Regularized approximation to a sphere oscillating
    in the x direction in 3 space.

    '''
    nodes = np.zeros((1,3))
    BCvel = np.array([1.0,0.0,0.0])
    f0 = rsd.fcst*BCvel
    U = rsd.calcVel(obspts,nodes,f0)*np.exp(1j*2*np.pi*freq*t)
    u = U[:,0]
    v = U[:,1]
    w = U[:,2]
#    ub = rsd.calcVel(spts,nodes,f0)
#    print("vel on sphere, max(u), min(u) (should be 1), max(v), min(v), max(w), min(w), (should be 0)")
#    print(np.max(np.abs(ub[:,0])),np.min(np.abs(ub[:,0])),np.max(np.abs(ub[:,1])),np.min(np.abs(ub[:,1])),np.max(np.abs(ub[:,2])),np.min(np.abs(ub[:,2])))
    p = rsd.calcPressure(obspts,nodes,f0)*np.exp(1j*2*np.pi*freq*t)
    S = rsd.calcStressTensor(spts,nodes,f0)
    t1 = (S[:,0,0]*spts[:,0] + S[:,0,1]*spts[:,1] + S[:,0,2]*spts[:,2]) / a
    t2 = (S[:,1,0]*spts[:,0] + S[:,1,1]*spts[:,1] + S[:,1,2]*spts[:,2]) / a
    t3 = (S[:,2,0]*spts[:,0] + S[:,2,1]*spts[:,1] + S[:,2,2]*spts[:,2]) / a
    return u, v, w, p, t1, t2, t3

def testOscillatingSphere3D():
    '''
    Checks velocity outside of a sphere using L2 error;
    checks pressure outside of a sphere using L2 error;
    checks pointwise surface traction on the sphere using L-infinity error.
    
    '''
    print('Test case: Oscillating sphere in 3D (be patient, this may take a few minutes)...')
    a = 0.25
    h = a/5.
    X = np.mgrid[-2*a+h/2:2*a:h,-2*a+h/2:2*a:h,-2*a:2*a+h/2:h]
    M = X.shape[-1]
    N = np.floor(M/2.0)
    zlev = X[2,0,0,N]
    xg = X[0,:,:,N]
    yg = X[1,:,:,N]
    x = X[0,:,:,:].flatten()
    y = X[1,:,:,:].flatten()
    z = X[2,:,:,:].flatten()
    obspts = np.column_stack([x,y,z])
    freq = 71
    t=0
    mu = 1.0
    nu = 1.1
    alph = np.sqrt(1j*2*np.pi*freq/nu)
#    spts = a*np.loadtxt(os.path.join(os.path.split(__file__)[0], 'voronoivertices0300.dat'))
    tN = 50
    dtheta = 2*np.pi/tN
    theta = np.arange(dtheta/2,2*np.pi,dtheta)
    phi = np.arange(dtheta/2,np.pi,dtheta)
    xs = np.zeros((tN**2/2,))
    ys = np.zeros((tN**2/2,))
    zs = np.zeros((tN**2/2,))
    for i in range(tN):
        th = theta[i]
        xs[i*tN/2:(i+1)*tN/2] = a*np.cos(theta[i])*np.sin(phi)
        ys[i*tN/2:(i+1)*tN/2] = a*np.sin(theta[i])*np.sin(phi)
        zs[i*tN/2:(i+1)*tN/2] = a*np.cos(phi)
    spts = np.column_stack([xs,ys,zs])
    u,v,w,p,xdrag,ydrag,zdrag = lES.sphere3DOscillating(x,y,z,a,alph,freq,mu)
    t1,t2,t3 = lES.sphere3DOscillatingSurfaceTraction(spts[:,0],spts[:,1],spts[:,2],a,alph,freq,mu)
#    umlevs=vRS.contourCircle(xg,yg,a,np.abs(u.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestuabs.pdf' ),None, "|u| for sphere at z=%f" % zlev)
#    vmlevs=vRS.contourCircle(xg,yg,a,np.abs(v.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvabs.pdf'),None,  "|v| for sphere at z=%f" % zlev)
#    ualevs=vRS.contourCircle(xg,yg,a,np.angle(u.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestuang.pdf'),None,"angle(u) for sphere at z=%f" % zlev)
#    valevs=vRS.contourCircle(xg,yg,a,np.angle(v.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvang.pdf'),None,"angle(v) for sphere at z=%f" % zlev)
#    pmlevs=vRS.contourCircle(xg,yg,a,np.abs(p.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpabs.pdf'),None,  "|p| for sphere at z=%f" % zlev)
#    palevs=vRS.contourCircle(xg,yg,a,np.angle(p.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpang.pdf'),None,"angle(p) for sphere at z=%f" % zlev)
#    print("exact soln")
#    print(xdrag,ydrag,zdrag)
    myfactor = 1.4
    elist = [a/(4*(myfactor**k)) for k in range(0,3)]
#    elist = [a/(2*(myfactor**k)) for k in range(0,3)]
    errmagG = np.zeros((len(elist),))
    errangG = np.zeros((len(elist),))
    errmagE = np.zeros((len(elist),))
    errangE = np.zeros((len(elist),))
    errpmagG = np.zeros((len(elist),))
    errpangG = np.zeros((len(elist),))
    errpmagE = np.zeros((len(elist),))
    errpangE = np.zeros((len(elist),))
    errtmagG = np.zeros((len(elist),3))    
    errtangG = np.zeros((len(elist),3))
    errtmagE = np.zeros((len(elist),3))    
    errtangE = np.zeros((len(elist),3))
    for k in range(len(elist)):
        eps = elist[k]
        rsd = RS.Brinkman3DGaussianStokesletsAndDipolesSphericalBCs(eps,mu,alph,a)
        ureg, vreg, wreg, preg,t1reg,t2reg,t3reg = regOscillatingSphere3D(rsd,obspts,spts,a,freq)
#        vRS.contourCircle(xg,yg,a,np.abs(ureg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittesturegabs%02d.pdf' % k ),umlevs, "Reg |u| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.abs(vreg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvregabs%02d.pdf' % k),vmlevs,  "Reg |v| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.angle(ureg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittesturegang%02d.pdf' % k),ualevs,"Reg angle(u) for sphere at z=%f" % zlev) 
#        vRS.contourCircle(xg,yg,a,np.angle(vreg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvregang%02d.pdf' % k),valevs,"Reg angle(v) for sphere at z=%f" % zlev) 
#        vRS.contourCircle(xg,yg,a,np.abs(preg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpregabs%02d.pdf' % k),pmlevs,  "Reg |p| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.angle(preg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpregang%02d.pdf' % k),palevs,"Reg angle(p) for sphere at z=%f" % zlev) 
#        print("Gaussian approx -- drag")
#        print(xdragreg,ydragreg,zdragreg)
        ind = np.nonzero(x**2+y**2+z**2 > a**2)
        L2errmag = h**(1.5)*np.sqrt( ((np.abs(u[ind])-np.abs(ureg[ind]))**2).sum() + ((np.abs(v[ind])-np.abs(vreg[ind]))**2).sum() + ((np.abs(w[ind])-np.abs(wreg[ind]))**2).sum() )
        L2errang = h**(1.5)*np.sqrt( ((np.angle(u[ind])-np.angle(ureg[ind]))**2).sum() + ((np.angle(v[ind])-np.angle(vreg[ind]))**2).sum() + ((np.angle(w[ind])-np.angle(wreg[ind]))**2).sum() )
        errmagG[k] = L2errmag
        errangG[k] = L2errang
        L2errpmag = h**(1.5)*np.sqrt( ((np.abs(p[ind])-np.abs(preg[ind]))**2).sum() )
        L2errpang = h**(1.5)*np.sqrt( ((np.angle(p[ind])-np.angle(preg[ind]))**2).sum() )
        errpmagG[k] = L2errpmag
        errpangG[k] = L2errpang
        L2errtmag = [np.max(np.abs((np.abs(t1)-np.abs(t1reg)))/np.abs(t1)),np.max(np.abs((np.abs(t2)-np.abs(t2reg)))/np.abs(t2)),np.max(np.abs((np.abs(t3)-np.abs(t3reg)))/np.abs(t3))]
        L2errtang = [np.max(np.abs((np.angle(t1)-np.angle(t1reg))/np.angle(t1))),np.max(np.abs((np.angle(t2)-np.angle(t2reg))/np.angle(t2))),np.max(np.abs((np.angle(t3)-np.angle(t3reg))/np.angle(t3)))]
        errtmagG[k,:] = L2errtmag
        errtangG[k,:] = L2errtang
    convmagG = np.zeros((len(elist)-1,))
    convangG = np.zeros((len(elist)-1,))
    convpmagG = np.zeros((len(elist)-1,))
    convpangG = np.zeros((len(elist)-1,))
    for k in range(len(elist)-1):
        convmagG[k]=np.log(errmagG[k]/errmagG[k+1])/np.log(myfactor)
        convangG[k]=np.log(errangG[k]/errangG[k+1])/np.log(myfactor)
        convpmagG[k]=np.log(errpmagG[k]/errpmagG[k+1])/np.log(myfactor)
        convpangG[k]=np.log(errpangG[k]/errpangG[k+1])/np.log(myfactor)
    if np.all(errmagG < 2.e-4) and np.all(errangG < 2.e-4):
        print("Gaussian blob, velocity passed")
    else:
        print("Gaussian blob, velocity failed")
    if np.all(errpmagG < 2.e-4) and np.all(errpangG < 2.e-4):
        print("Gaussian blob, pressure passed")
    else:
        print("Gaussian blob, pressure failed")
    if np.all(errtmagG < 2.e-4) and np.all(errtangG < 2.e-4):
        print("Gaussian blob, surface traction passed")
    else:
        print("Gaussian blob, surface traction failed")
    print("Surface traction error should be small:")
    print(errtmagG)
    print(errtangG)
    print("Vel L2 error should be small:")
    print(errmagG)
    print(errangG)
    print("Vel convergence rates should be high: ")
    print(convmagG)
    print(convangG)
    print("Pressure L2 error should be small:")
    print(errpmagG)
    print(errpangG)
    print("Pressure convergence rates should be high: ")
    print(convpmagG)
    print(convpangG)
    myfactor = 1.4
    elist = [a/(12*(myfactor**k)) for k in range(2,5)]
#    elist = [a/(6*(myfactor**k)) for k in range(2,5)]
    for k in range(len(elist)):
        eps = elist[k]
        rsd = RS.Brinkman3DNegExpStokesletsAndDipolesSphericalBCs(eps,mu,alph,a)
        ureg, vreg, wreg, preg,t1reg,t2reg,t3reg = regOscillatingSphere3D(rsd,obspts,spts,a,freq)
#        vRS.contourCircle(xg,yg,a,np.abs(ureg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittesturegabs%02d.pdf' % k ),umlevs, "Reg |u| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.abs(vreg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvregabs%02d.pdf' % k),vmlevs,  "Reg |v| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.angle(ureg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittesturegang%02d.pdf' % k),ualevs,"Reg angle(u) for sphere at z=%f" % zlev) 
#        vRS.contourCircle(xg,yg,a,np.angle(vreg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestvregang%02d.pdf' % k),valevs,"Reg angle(v) for sphere at z=%f" % zlev) 
#        vRS.contourCircle(xg,yg,a,np.abs(preg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpregabs%02d.pdf' % k),pmlevs,  "Reg |p| for sphere at z=%f" % zlev)      
#        vRS.contourCircle(xg,yg,a,np.angle(preg.reshape((M-1,M-1,M))[:,:,N]),os.path.expanduser('~/scratch/unittestpregang%02d.pdf' % k),palevs,"Reg angle(p) for sphere at z=%f" % zlev) 
#        print("Neg exp approx")
#        print(xdragreg,ydragreg,zdragreg)
        ind = np.nonzero(x**2+y**2+z**2 > a**2)
        L2errmag = h**(1.5)*np.sqrt( ((np.abs(u[ind])-np.abs(ureg[ind]))**2).sum() + ((np.abs(v[ind])-np.abs(vreg[ind]))**2).sum() + ((np.abs(w[ind])-np.abs(wreg[ind]))**2).sum() )
        L2errang = h**(1.5)*np.sqrt( ((np.angle(u[ind])-np.angle(ureg[ind]))**2).sum() + ((np.angle(v[ind])-np.angle(vreg[ind]))**2).sum() + ((np.angle(w[ind])-np.angle(wreg[ind]))**2).sum() )
        errmagE[k] = L2errmag
        errangE[k] = L2errang
        L2errpmag = h**(1.5)*np.sqrt( ((np.abs(p[ind])-np.abs(preg[ind]))**2).sum() )
        L2errpang = h**(1.5)*np.sqrt( ((np.angle(p[ind])-np.angle(preg[ind]))**2).sum() )
        errpmagE[k] = L2errpmag
        errpangE[k] = L2errpang
        L2errtmag = [np.max(np.abs((np.abs(t1)-np.abs(t1reg)))/np.abs(t1)),np.max(np.abs((np.abs(t2)-np.abs(t2reg)))/np.abs(t2)),np.max(np.abs((np.abs(t3)-np.abs(t3reg)))/np.abs(t3))]
        L2errtang = [np.max(np.abs((np.angle(t1)-np.angle(t1reg))/np.angle(t1))),np.max(np.abs((np.angle(t2)-np.angle(t2reg))/np.angle(t2))),np.max(np.abs((np.angle(t3)-np.angle(t3reg))/np.angle(t3)))]
        errtmagE[k,:] = L2errtmag
        errtangE[k,:] = L2errtang
    convmagE = np.zeros((len(elist)-1,))
    convangE = np.zeros((len(elist)-1,))
    convpmagE = np.zeros((len(elist)-1,))
    convpangE = np.zeros((len(elist)-1,))
    for k in range(len(elist)-1):
        convmagE[k]=np.log(errmagE[k]/errmagE[k+1])/np.log(myfactor)
        convangE[k]=np.log(errangE[k]/errangE[k+1])/np.log(myfactor)
        convpmagE[k]=np.log(errpmagE[k]/errpmagE[k+1])/np.log(myfactor)
        convpangE[k]=np.log(errpangE[k]/errpangE[k+1])/np.log(myfactor)
    if np.all(errmagE < 2.e-4) and np.all(errangE < 2.e-4):
        print("Negative exponential blob, velocity passed")
    else:
        print("Negative exponential blob, velocity failed")
    if np.all(errpmagE < 2.e-4) and np.all(errpangE < 2.e-4):
        print("Negative exponential blob, pressure passed")
    else:
        print("Negative exponential blob, pressure failed")
    if np.all(errtmagE < 2.e-4) and np.all(errtangE < 2.e-4):
        print("Negative exponential blob, surface traction passed")
    else:
        print("Negative exponential blob, surface traction failed")
    print("Surface traction error should be small:")
    print(errtmagE)
    print(errtangE)
    print("Vel L2 error should be small:")
    print(errmagE)
    print(errangE)
    print("Vel convergence rates should be high: ")
    print(convmagE)
    print(convangE)
    print("Pressure L2 error should be small:")
    print(errpmagE)
    print(errpangE)
    print("Pressure convergence rates should be high: ")
    print(convpmagE)
    print(convpangE)

def regOscillatingSphere3D_Drag(rsd,spts,phi,dtheta,a,freq,t=0):
    '''
    Regularized approximation to a sphere oscillating
    in the x direction in 3 space, drag calculation only.

    '''
    nodes = np.zeros((1,3))
    BCvel = np.array([1.0,0.0,0.0])
    f0 = rsd.fcst*BCvel
    S = rsd.calcStressTensor(spts,nodes,f0)
    t1 = (S[:,0,0]*spts[:,0] + S[:,0,1]*spts[:,1] + S[:,0,2]*spts[:,2]) / a
    t2 = (S[:,1,0]*spts[:,0] + S[:,1,1]*spts[:,1] + S[:,1,2]*spts[:,2]) / a
    t3 = (S[:,2,0]*spts[:,0] + S[:,2,1]*spts[:,1] + S[:,2,2]*spts[:,2]) / a
    SinPhi = np.tile(np.sin(phi),(1,2*len(phi)))
    xdrag = dtheta**2*a**2*((t1*SinPhi).sum())*np.exp(1j*2*np.pi*freq*t)
    ydrag = dtheta**2*a**2*((t2*SinPhi).sum())*np.exp(1j*2*np.pi*freq*t)
    zdrag = dtheta**2*a**2*((t3*SinPhi).sum())*np.exp(1j*2*np.pi*freq*t)
    return xdrag, ydrag, zdrag

def testOscillatingSphere3D_Drag():
    '''
    Checks total drag on the sphere (scalar value).
    
    '''
    print('Test case: Oscillating sphere in 3D, testing drag (takes a very long time)...')
    a = 0.25
    freq = 71
    t=0
    mu = 1.0
    nu = 1.1
    alph = np.sqrt(1j*2*np.pi*freq/nu)
    tN = 400
    dtheta = 2*np.pi/tN
    theta = np.arange(dtheta/2,2*np.pi,dtheta)
    phi = np.arange(dtheta/2,np.pi,dtheta)
    xs = np.zeros((tN**2/2,))
    ys = np.zeros((tN**2/2,))
    zs = np.zeros((tN**2/2,))
    for i in range(tN):
        th = theta[i]
        xs[i*tN/2:(i+1)*tN/2] = a*np.cos(theta[i])*np.sin(phi)
        ys[i*tN/2:(i+1)*tN/2] = a*np.sin(theta[i])*np.sin(phi)
        zs[i*tN/2:(i+1)*tN/2] = a*np.cos(phi)
    spts = np.column_stack([xs,ys,zs])
    u,v,w,p,xdrag,ydrag,zdrag = lES.sphere3DOscillating(spts[:,0],spts[:,1],spts[:,2],a,alph,freq,mu)
    myfactor = 1.4
    elist = [a/(4*(myfactor**k)) for k in range(0,3)]
#    elist = [a/(2*(myfactor**k)) for k in range(0,3)]
    errdragG = np.zeros((len(elist),3))
    errdragE = np.zeros((len(elist),3))
    for k in range(len(elist)):
        eps = elist[k]
        rsd = RS.Brinkman3DGaussianStokesletsAndDipolesSphericalBCs(eps,mu,alph,a)
        xdragreg, ydragreg, zdragreg = regOscillatingSphere3D_Drag(rsd,spts,phi,dtheta,a,freq)
        errdragG[k,:] = [np.abs(xdrag-xdragreg)/np.abs(xdrag),np.abs(ydrag-ydragreg),np.abs(zdrag-zdragreg)]
    if np.all(errdragG < 2.e-4):
        print("Gaussian blob, drag passed")
    else:
        print("Gaussian blob, drag failed")
#    print("Drag error should be small:")
#    print(errdragG)
    myfactor = 1.4
    elist = [a/(12*(myfactor**k)) for k in range(2,5)]
#    elist = [a/(6*(myfactor**k)) for k in range(2,5)]
    for k in range(len(elist)):
        eps = elist[k]
        rsd = RS.Brinkman3DNegExpStokesletsAndDipolesSphericalBCs(eps,mu,alph,a)
        xdragreg, ydragreg, zdragreg = regOscillatingSphere3D_Drag(rsd,spts,phi,dtheta,a,freq)
        errdragE[k,:] = [np.abs(xdrag-xdragreg)/np.abs(xdrag),np.abs(ydrag-ydragreg),np.abs(zdrag-zdragreg)]
    if np.all(errdragE < 2.e-4):
        print("Negative exponential blob, drag passed")
    else:
        print("Negative exponential blob, drag failed")
#    print("Drag error should be small:")
#    print(errdragE)

def testFuncVals():
    '''
    Checks to be sure that the complicated formulae were 
    correctly copied by comparing to high precision
    pointwise values calculated in Mathematica.
    
    '''
    print('Function value tests from Mathematica.....')
    pdict={}
    pdict['eps'] = 0.03
    pdict['alph'] = np.sqrt(7.0*1j)
    pdict['sig'] = pdict['eps']*pdict['alph']
    pdict['thresh'] = pdict['eps']/100.
    r = np.array([0,0.25])
    # Gaussian blob, exact values come from Mathematica
    H1exact = np.array([1.8963117661339852216 - 0.0931211960687448326*1j, 0.067908212764266076555 - 0.055432933653040201982*1j])
    H2exact = np.array([443.35980806439800972 - 1.32741823628967426*1j, 2.4404898604743916479 - 0.2069860590271001555*1j])
    H1primeexact = np.array([0.0, -0.60329371256711228355 + 0.14113746065668841823*1j])
    H2primeexact = np.array([0.0, -29.395138366516494699 + 1.053577573927327649*1j])
    D1exact = np.array([-4433.5980806439783919 + 13.2741823629378928*1j, 5.4809887145119535745 + 0.4753574893498642151*1j])
    D2exact = np.array([-2.9561573274166509509*10**6 + 3103.5186564507881*1j, -243.01309017596247486 + 17.08342902332080726*1j])
    D1primeexact = np.array([0.0, -62.103460371883727476 - 4.223055987969729586*1j])
    D2primeexact = np.array([0.0, 4881.8648087655546988 - 205.7659685656169586*1j])
    err = []
    err.append(np.abs(H1exact - lRS.Brinkman3DGaussianStokesletsH1(r,pdict))/np.abs(H1exact))
    err.append(np.abs(H2exact - lRS.Brinkman3DGaussianStokesletsH2(r,pdict))/np.abs(H2exact))
    err.append(np.abs(H1primeexact - lRS.Brinkman3DGaussianStokesletsH1prime(r,pdict))/np.array([1.0,np.abs(H1primeexact[1])]))
    err.append(np.abs(H2primeexact - lRS.Brinkman3DGaussianStokesletsH2prime(r,pdict))/np.array([1.0,np.abs(H2primeexact[1])]))
    err.append(np.abs(D1exact - lRS.Brinkman3DGaussianDipolesD1(r,pdict))/np.abs(D1exact))
    err.append(np.abs(D2exact - lRS.Brinkman3DGaussianDipolesD2(r,pdict))/np.abs(D2exact))
    err.append(np.abs(D1primeexact - lRS.Brinkman3DGaussianDipolesD1prime(r,pdict))/np.array([1.0,np.abs(D1primeexact[1])]))
    err.append(np.abs(D2primeexact - lRS.Brinkman3DGaussianDipolesD2prime(r,pdict))/np.array([1.0,np.abs(D2primeexact[1])]))
    if np.max(err) < 1.e-12:
        print('Gaussian blob passed.')
    else:
        print('Gaussian blob failed.')
#    print(np.asarray(err))
    # Neg exp blob, exact values come from Mathematica
    H1exact = np.array([0.47116982833562753852 - 0.07775746287614701546*1j, 0.083024119015813968581 - 0.053653287738405212870*1j])
    H2exact = np.array([13.9803972091344324014 - 0.3298188798349392381*1j, 1.7062131615749160929 - 0.1626763107839912237*1j])
    H1primeexact = np.array([0.0, -0.72570276119787324731 + 0.12528555871648225373*1j])
    H2primeexact = np.array([0.0, -15.688166406032580724 + 0.598252033080565737*1j])
    D1exact = np.array([-139.803972091344292039 + 3.298188798349392048*1j, 3.8571458785220054111 + 0.5811688331106981131*1j])
    D2exact = np.array([-23389.070323087380530 + 97.862780463940751*1j, -209.20310866007693562 + 11.94349213102443308*1j])
    D1primeexact = np.array([0.0, -17.150005531690332816 - 5.079919328385114063*1j])
    D2primeexact = np.array([324880.26465619757073, 3621.6498270682741349 - 109.8171648422285216*1j])
    err = []
    err.append(np.abs(H1exact - lRS.Brinkman3DNegExpStokesletsH1(r,pdict))/np.abs(H1exact))
    err.append(np.abs(H2exact - lRS.Brinkman3DNegExpStokesletsH2(r,pdict))/np.abs(H2exact))
    err.append(np.abs(H1primeexact - lRS.Brinkman3DNegExpStokesletsH1prime(r,pdict))/np.array([1.0,np.abs(H1primeexact[1])]))
    err.append(np.abs(H2primeexact - lRS.Brinkman3DNegExpStokesletsH2prime(r,pdict))/np.array([1.0,np.abs(H2primeexact[1])]))
    err.append(np.abs(D1exact - lRS.Brinkman3DNegExpDipolesD1(r,pdict))/np.abs(D1exact))
    err.append(np.abs(D2exact - lRS.Brinkman3DNegExpDipolesD2(r,pdict))/np.abs(D2exact))
    err.append(np.abs(D1primeexact - lRS.Brinkman3DNegExpDipolesD1prime(r,pdict))/np.array([1.0,np.abs(D1primeexact[1])]))
    err.append(np.abs(D2primeexact - lRS.Brinkman3DNegExpDipolesD2prime(r,pdict))/np.abs(D2primeexact))
    if np.max(err) < 1.e-12:
        print('Neg exp blob passed.')
    else:
        print('Neg exp blob failed.')
#    print(np.asarray(err))
    # Compact blob, exact values come from Mathematica
    r = np.array([0,0.02,0.25])
    H1exact = np.array([9627.076974767855063 + 12058.117477905690976*1j, 2.605618500000000000*10**6 - 1.8061312000000000000*10**7 *1j, 0.066674906585973553463 - 0.055538917881437423196*1j])
    H2exact = np.array([21670.352667372244468 - 6630.315469410513288*1j, 3.1163431936000000000*10**10 + 4.8050478576250000000*10**10 *1j, 2.4951647561886174387 - 0.2108728878412049479*1j])
    D1exact = np.array([-216703.52667372245924 + 66303.15469410512014*1j, 6.7715072000000000000*10**7 + 1.10514024000000000000*10**8 *1j, 5.4817306041107123349 + 0.4667243461018151240*1j])
    D2exact = np.array([-4.9638787847965842485*10**8 + 152291.26877511491*1j, -1.3030351231800000000*10**11 - 2.2853257625600000000*10**11 *1j, -242.98588237426278624 + 17.46615329332031408*1j])
    H1reg=lRS.Brinkman3DCompactStokesletsH1(r,pdict)
    H2reg=lRS.Brinkman3DCompactStokesletsH2(r,pdict)
    D1reg=lRS.Brinkman3DCompactDipolesD1(r,pdict)   
    D2reg=lRS.Brinkman3DCompactDipolesD2(r,pdict)   
    err = []
    err.append( np.abs(H1exact - H1reg)/np.abs(H1exact) )
    err.append( np.abs(H2exact - H2reg)/np.abs(H2exact) )
    err.append( np.abs(D1exact - D1reg)/np.abs(D1exact) )
    err.append( np.abs(D2exact - D2reg)/np.abs(D2exact) )
    if np.max(err) < 1.e-12:
        print('Compact blob passed.')
    else:
        print('Compact blob failed.')
#    print(np.asarray(err))
    print(H1reg[0])
#    print(H2reg)
#    print(D1reg)
#    print(D2reg)
    


    
if __name__ == '__main__':
    testFuncVals()
#    testCircularCylinder2D()   
#    testOscillatingSphere3D()
#    testOscillatingSphere3D_Drag()