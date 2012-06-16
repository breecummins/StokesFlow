#Created by Breschine Cummins on May 8, 2012.

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
import os
from cPickle import Pickler
import engine_RegularizedStokeslets as RS
import viz_RegularizedStokeslets as vRS
import CompareRegExact as CRE
import lib_ExactSolns as lES

def exactDragForces(a,v,alph,mu):
    ''' 
    Exact solution for drag force on an infinite (in z) cylinder
    of radius a oscillating in the x-direction with peak speed v.
    alph = sqrt(i*om*rho/mu), where om is angular frequency, rho is
    fluid density, and mu is fluid viscosity.
    The output is the complex number representing magnitude and 
    phase of cylinder motion at a given frequency.
    
    '''
    b = alph*a
    xdrag = -( (mu*np.pi*v*b**2)/ss.kv(0,b) ) * (ss.kv(2,b) + 2*ss.kv(1,b)/b)
    ydrag = 0 
    return xdrag, ydrag

def regDragForces(pdict,rb,f):
    '''
    Approximate drag force on the cylinder (in a plane) with regularized forces.
        
    '''
    S = rb.calcStressTensor(pdict['obsptscirc'],pdict['nodes'],f)
    h = pdict['circh']
    theta = pdict['theta']
#    print("Difference in off-diagonal elements in the stress tensor. Should be zero:")
#    print(np.max(np.abs(S[:,0,1]-S[:,1,0])))
    integrandx = S[:,0,0]*np.cos(theta) + S[:,0,1]*np.sin(theta)
    integrandy = S[:,1,0]*np.cos(theta) + S[:,1,1]*np.sin(theta)
    xdrag = h*(integrandx.sum())
    ydrag = h*(integrandy.sum())
    return xdrag, ydrag

def regVel(pdict,rb,f):    
    uaxis = rb.calcVel(pdict['obsptszline'],pdict['nodes'],f)
    udom = rb.calcVel(pdict['obspts'],pdict['nodes'],f)
    return uaxis, udom
    
def varyFreqEps_Drag(pdict,freqlist,epslist):
    xdrag_exact = []
    ydrag_exact = []
    xdrag_negex = [[] for j in range(len(freqlist))]
    ydrag_negex = [[] for j in range(len(freqlist))]
    xdrag_gauss = [[] for j in range(len(freqlist))]
    ydrag_gauss = [[] for j in range(len(freqlist))]
#    xdrag_ncofs = [[] for j in range(len(freqlist))]
#    ydrag_ncofs = [[] for j in range(len(freqlist))]
#    xdrag_gcofs = [[] for j in range(len(freqlist))]
#    ydrag_gcofs = [[] for j in range(len(freqlist))]
    for j in range(len(freqlist)):
        freq = freqlist[j]
        print("Freq=%d" % freq)
        alph=np.sqrt(1j*2*np.pi*freq/pdict['nu'])
        xdrag, ydrag = exactDragForces(pdict['circrad'],pdict['vh'],alph,pdict['mu'])
        xdrag_exact.append(xdrag)
        ydrag_exact.append(ydrag)
        for k in range(len(epslist)):
            eps = epslist[k]
            print("Eps=%f" % eps)
            rb, f =   CRE.regSolnNegExpStokesletsOnly(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
            xd,yd = regDragForces(pdict,rb,f)
            xdrag_negex[j].append(xd)
            ydrag_negex[j].append(yd)
            rb, f = CRE.regSolnGaussianStokesletsOnly(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
            xd, yd = regDragForces(pdict,rb,f)
            xdrag_gauss[j].append(xd)
            ydrag_gauss[j].append(yd)
#            rb, f =   CRE.regSolnChainofSpheresNegExp(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
#            xd,yd = calcRegSoln(pdict,rb,f)
#            xdrag_ncofs[j].append(xd)
#            ydrag_ncofs[j].append(yd)
#            rb, f = CRE.regSolnChainofSpheresGaussian(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
#            xd, yd = calcRegSoln(pdict,rb,f)
#            xdrag_gcofs[j].append(xd)
#            ydrag_gcofs[j].append(yd)
        earr = np.asarray(epslist)
#        xarrg = np.asarray([np.abs(xdrag_exact[j])*np.ones((len(epslist,))),np.abs(xdrag_gauss[j]),np.abs(xdrag_gcofs[j])]).transpose()
#        yarrg = np.asarray([np.abs(ydrag_exact[j])*np.ones((len(epslist,))),np.abs(ydrag_gauss[j]),np.abs(ydrag_gcofs[j])]).transpose()
#        xarrn = np.asarray([np.abs(xdrag_exact[j])*np.ones((len(epslist,))),np.abs(xdrag_negex[j]),np.abs(xdrag_ncofs[j])]).transpose()
#        yarrn = np.asarray([np.abs(ydrag_exact[j])*np.ones((len(epslist,))),np.abs(ydrag_negex[j]),np.abs(ydrag_ncofs[j])]).transpose()
        xarr = np.asarray([np.abs(xdrag_exact[j])*np.ones((len(epslist,))),np.abs(xdrag_gauss[j]),np.abs(xdrag_negex[j])]).transpose()
        yarr = np.asarray([np.abs(ydrag_exact[j])*np.ones((len(epslist,))),np.abs(ydrag_gauss[j]),np.abs(xdrag_negex[j])]).transpose()
#        vRS.plainPlots(earr,xarrg,'Magnitude x-drag','Epsilon','Drag',['exact','gauss','gauss spheres'],os.path.expanduser('~/scratch/xdrag_zhdiameter_gauss%03d.pdf' % freq))
#        vRS.plainPlots(earr,yarrg,'Magnitude y-drag','Epsilon','Drag',['exact','gauss','gauss spheres'],os.path.expanduser('~/scratch/ydrag_zhdiameter_gauss%03d.pdf' % freq))
#        vRS.plainPlots(earr,xarrn,'Magnitude x-drag','Epsilon','Drag',['exact','negex','negex spheres'],os.path.expanduser('~/scratch/xdrag_zhdiameter_negex%03d.pdf' % freq))
#        vRS.plainPlots(earr,yarrn,'Magnitude y-drag','Epsilon','Drag',['exact','negex','negex spheres'],os.path.expanduser('~/scratch/ydrag_zhdiameter_negex%03d.pdf' % freq))
        vRS.plainPlots(earr,xarr,'Magnitude x-drag','Epsilon','Drag',['exact','gauss','negex'],os.path.expanduser('~/CricketProject/ChooseEpsilon/xdrag_zhdiameter_freq%03d.pdf' % freq))
        vRS.plainPlots(earr,yarr,'Magnitude y-drag','Epsilon','Drag',['exact','gauss','negex'],os.path.expanduser('~/CricketProject/ChooseEpsilon/ydrag_zhdiameter_freq%03d.pdf' % freq))
    return xdrag_exact, ydrag_exact, xdrag_negex, ydrag_negex, xdrag_gauss, ydrag_gauss       

def varyFreqEps_zVel(pdict,freqlist,epslist):
    u_exact = []
    v_exact = []
    u_negex = [[] for j in range(len(freqlist))]
    v_negex = [[] for j in range(len(freqlist))]
    w_negex = [[] for j in range(len(freqlist))]
    uz_negex= [[] for j in range(len(freqlist))]
    vz_negex= [[] for j in range(len(freqlist))]
    wz_negex= [[] for j in range(len(freqlist))]
    u_gauss = [[] for j in range(len(freqlist))]
    v_gauss = [[] for j in range(len(freqlist))]
    w_gauss = [[] for j in range(len(freqlist))]
    uz_gauss= [[] for j in range(len(freqlist))]
    vz_gauss= [[] for j in range(len(freqlist))]
    wz_gauss= [[] for j in range(len(freqlist))]
    for j in range(len(freqlist)):
        freq = freqlist[j]
        print("Freq=%d" % freq)
        alph=np.sqrt(1j*2*np.pi*freq/pdict['nu'])
        u,v,p=lES.circularCylinder2DOscillating(pdict['obspts'][:,0],pdict['obspts'][:,1],pdict['circrad'],pdict['nu'],pdict['mu'],freq,pdict['vh'],0)
        u_exact.append(u)
        v_exact.append(v)
        for k in range(len(epslist)):
            eps = epslist[k]
            print("Eps=%f" % eps)
            rb, f =   CRE.regSolnNegExpStokesletsOnly(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
            uaxis,udom = regVel(pdict,rb,f)
            u_negex[j].append(udom[:,0])
            v_negex[j].append(udom[:,1])
            w_negex[j].append(udom[:,2])
            uz_negex[j].append(uaxis[:,0])
            vz_negex[j].append(uaxis[:,1])
            wz_negex[j].append(uaxis[:,2])
            rb, f = CRE.regSolnGaussianStokesletsOnly(pdict['nodes'],eps,pdict['mu'],alph,pdict['circrad'],pdict['vh'])
            uaxis,udom = regVel(pdict,rb,f)
            u_gauss[j].append(udom[:,0])
            v_gauss[j].append(udom[:,1])
            w_gauss[j].append(udom[:,2])
            uz_gauss[j].append(uaxis[:,0])
            vz_gauss[j].append(uaxis[:,1])
            wz_gauss[j].append(uaxis[:,2])
    return u_exact, v_exact, u_negex, v_negex, w_negex, uz_negex, vz_negex, wz_negex, u_gauss, v_gauss, w_gauss, uz_gauss, vz_gauss, wz_gauss       
        
def optimizeEps():
    pdict = setParams()
    freqlist = range(10,310,25)
    epslist = [k*pdict['circrad'] for k in np.arange(0.1,1.5,0.1)]
    u_exact, v_exact, u_negex, v_negex, w_negex, uz_negex, vz_negex, wz_negex, u_gauss, v_gauss, w_gauss, uz_gauss, vz_gauss, wz_gauss = varyFreqEps_zVel(pdict,freqlist,epslist)
    umag_err_negex = np.zeros((len(freqlist),len(epslist),3))
    uang_err_negex = np.zeros((len(freqlist),len(epslist),3))
    umag_err_gauss = np.zeros((len(freqlist),len(epslist),3))
    uang_err_gauss = np.zeros((len(freqlist),len(epslist),3))
    umag_axiserr_negex = np.zeros((len(freqlist),len(epslist),3))
    uang_axiserr_negex = np.zeros((len(freqlist),len(epslist),3))
    umag_axiserr_gauss = np.zeros((len(freqlist),len(epslist),3))
    uang_axiserr_gauss = np.zeros((len(freqlist),len(epslist),3))
    ind = np.nonzero(pdict['obspts'][:,0]**2 + pdict['obspts'][:,1]**2 > 4*pdict['circrad']**2)
    print("Calculating error....")
    for j in range(len(freqlist)):
#        umlevs = vRS.contourCircle(pdict['obspts'][:,0],pdict['obspts'][:,1],pdict['circrad'],np.abs(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_exact_freq%03d.pdf' % freq))
#        ualevs = vRS.contourCircle(pdict['obspts'][:,0],pdict['obspts'][:,1],pdict['circrad'],np.angle(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_uang_exact_freq%03d.pdf' % freq))
#        vmlevs = vRS.contourCircle(pdict['obspts'][:,0],pdict['obspts'][:,1],pdict['circrad'],np.abs(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vmag_exact_freq%03d.pdf' % freq))
#        valevs = vRS.contourCircle(pdict['obspts'][:,0],pdict['obspts'][:,1],pdict['circrad'],np.angle(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vang_exact_freq%03d.pdf' % freq))
        for k in range(len(epslist)):
            umag_err_negex, uang_err_negex = calcErr(umag_err_negex, uang_err_negex,u_exact[j][ind],v_exact[j][ind],0.0,u_negex[j][k][ind],v_negex[j][k][ind],w_negex[j][k][ind],j,k,pdict['h'])
            umag_err_gauss, uang_err_gauss = calcErr(umag_err_gauss, uang_err_gauss,u_exact[j][ind],v_exact[j][ind],0.0,u_gauss[j][k][ind],v_gauss[j][k][ind],w_gauss[j][k][ind],j,k,pdict['h'])
            umag_axiserr_negex, uang_axiserr_negex = calcErr(umag_axiserr_negex, uang_axiserr_negex,np.median(uz_negex[j][k]),np.median(vz_negex[j][k]),np.median(wz_negex[j][k]),uz_negex[j][k],vz_negex[j][k],wz_negex[j][k],j,k,pdict['h'])
            umag_axiserr_gauss, uang_axiserr_gauss = calcErr(umag_axiserr_gauss, uang_axiserr_gauss,np.median(uz_gauss[j][k]),np.median(vz_gauss[j][k]),np.median(wz_gauss[j][k]),uz_gauss[j][k],vz_gauss[j][k],wz_gauss[j][k],j,k,pdict['h'])
    earr = np.asarray(epslist)
#    vRS.plainPlots(earr,umag_err_negex[:,:,0].transpose(),'Magnitude u, neg exp','Epsilon','L2 error',[str(f) for f in freqlist],os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_err_negex.pdf'))
#    vRS.plainPlots(earr,umag_err_gauss[:,:,0].transpose(),'Magnitude u, gaussian','Epsilon','L2 error',[str(f) for f in freqlist],os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_err_gauss.pdf'))
#    vRS.plainPlots(earr,umag_axiserr_negex[:,:,0].transpose(),'Difference in u from median on z-axis, neg exp','Epsilon','L2 error',[str(f) for f in freqlist],os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_axiserr_negex.pdf'))
#    vRS.plainPlots(earr,umag_axiserr_gauss[:,:,0].transpose(),'Difference in u from median on z-axis, gaussian','Epsilon','L2 error',[str(f) for f in freqlist],os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_axiserr_gauss.pdf'))
    mydict = {}
    mydict['umag_err_negex'] = umag_err_negex
    mydict['umag_err_gauss'] = umag_err_gauss
    mydict['uang_err_negex'] = uang_err_negex
    mydict['uang_err_gauss'] = uang_err_gauss
    mydict['umag_axiserr_negex'] = umag_axiserr_negex
    mydict['umag_axiserr_gauss'] = umag_axiserr_gauss
    mydict['uang_axiserr_negex'] = uang_axiserr_negex
    mydict['uang_axiserr_gauss'] = uang_axiserr_gauss
    mydict['u_exact'] = u_exact
    mydict['v_exact'] = v_exact
    mydict['u_negex'] = u_negex
    mydict['v_negex'] = v_negex
    mydict['w_negex'] = w_negex
    mydict['u_gauss'] = u_gauss
    mydict['v_gauss'] = v_gauss
    mydict['w_gauss'] = w_gauss
    mydict['uz_negex'] = uz_negex
    mydict['vz_negex'] = vz_negex
    mydict['wz_negex'] = wz_negex
    mydict['uz_gauss'] = uz_gauss
    mydict['vz_gauss'] = vz_gauss
    mydict['wz_gauss'] = wz_gauss
    mydict['pdict'] = pdict
    mydict['freqlist'] = freqlist
    mydict['epslist'] = epslist
    F = open( os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_farfield_BConaxis_hairrad05.pickle'), 'w' )
    Pickler(F).dump(mydict)
    F.close()

        
def calcErr(umerr,uaerr,ue,ve,we,ur,vr,wr,j,k,h):
    umerr[j,k,0] = h*np.sqrt( ((np.abs(ue)-np.abs(ur))**2).sum() ) 
    umerr[j,k,1] = h*np.sqrt( ((np.abs(ve)-np.abs(vr))**2).sum() )
    umerr[j,k,2] = h*np.sqrt( ((np.abs(we)-np.abs(wr))**2).sum() )
    uaerr[j,k,0] = h*np.sqrt( ((np.angle(ue)-np.angle(ur))**2).sum() ) 
    uaerr[j,k,1] = h*np.sqrt( ((np.angle(ve)-np.angle(vr))**2).sum() )
    uaerr[j,k,2] = h*np.sqrt( ((np.angle(we)-np.angle(wr))**2).sum() )
    return umerr, uaerr


def setParams():
    pdict ={}
    circrad = 0.005 #millimeters
    pdict['circrad'] = circrad
    zh = circrad
    halfzpts = 200
    pdict['N'] = 400
    th = 2*np.pi/pdict['N']
    ch = circrad*th
    pdict['circh'] = ch
    theta = np.arange(th/2.,2*np.pi,th)
    pdict['theta'] = theta
    xcirc = circrad*np.cos(theta)
    ycirc = circrad*np.sin(theta)
    Xcirc = np.column_stack([xcirc,ycirc])
    pdict['obsptscirc'] = np.column_stack([Xcirc,np.zeros(Xcirc[:,0].shape)])
    #spatial parameters
    hl = 20*circrad
    hextent = hl - hl/2
    Npts = 5
    size = (Npts,4*Npts)
    pdict['h'] = hextent/size[0]
    X1 = CRE.makeGridCenter(size,pdict['h'],(-hl,-hl))
    X1 = np.column_stack([X1,np.zeros(X1[:,0].shape)])
    X2 = CRE.makeGridCenter(size,pdict['h'],(hl/2,-hl))
    X2 = np.column_stack([X2,np.zeros(X2[:,0].shape)])
    newsize = (2*Npts,Npts)
    X3 = CRE.makeGridCenter(newsize,pdict['h'],(-hl/2,-hl))
    X3 = np.column_stack([X3,np.zeros(X3[:,0].shape)])
    X4 = CRE.makeGridCenter(newsize,pdict['h'],(-hl/2,hl/2))
    X4 = np.column_stack([X4,np.zeros(X4[:,0].shape)])
    pdict['obspts'] = np.row_stack([X1,X2,X3,X4])
    nodes = np.zeros((2*halfzpts+1,3))
    nodes[:,2] = np.arange(-halfzpts*zh,(halfzpts+1)*zh,zh)
    pdict['nodes'] = nodes
    Nz = (halfzpts*2)*10
    zline = np.zeros((Nz,3))
    zline[:,2] = np.linspace(-halfzpts*zh,halfzpts*zh,Nz)
    pdict['obsptszline'] = zline
    pdict['vh']=1.0 #mm/s
    pdict['mu']=1.85e-8 #kg/(mm s)
    pdict['nu']=15.7 #mm^2/s
    return pdict

if __name__ == '__main__':
    optimizeEps()
#    setParams()
    