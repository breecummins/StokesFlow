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
import batchscript_CompareRegExact as CRE
import lib_ExactSolns as lES
try:
    import StokesFlow.utilities.fileops as fo
except:
    import utilities.fileops as fo

def regVel(pdict,rb,f):    
    uaxis = rb.calcVel(pdict['obsptszline'],pdict['nodes'],f)
    udom = rb.calcVel(pdict['obspts'],pdict['nodes'],f)
    return uaxis, udom
    
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
    pdict, fname = setParams()
    freqlist = [5,10,15,20,25] 
    freqlist.extend(range(35,310,25))
    epslist = [k*pdict['circrad'] for k in np.arange(0.05,1.8,0.05)]
    u_exact, v_exact, u_negex, v_negex, w_negex, uz_negex, vz_negex, wz_negex, u_gauss, v_gauss, w_gauss, uz_gauss, vz_gauss, wz_gauss = varyFreqEps_zVel(pdict,freqlist,epslist)
    mydict = {}
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
    F = open( fname+'.pickle', 'w' )
    Pickler(F).dump(mydict)
    F.close()
    adderrs(fname,mydict)
    
def adderrs(fname,mydict=None):
    if mydict == None:
        basedir,basename = os.path.split(fname)
        mydict = fo.loadPickle(basename,basedir,'n')
    d = fo.ExtractDict(mydict)
    umag_err_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_err_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_err_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_err_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_axiserr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_axiserr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_axiserr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_axiserr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_relerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_relerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_relerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_relerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_axisrelerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_axisrelerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_axisrelerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_axisrelerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    ind = np.nonzero(d.pdict['obspts'][:,0]**2 + d.pdict['obspts'][:,1]**2 > 4*d.pdict['circrad']**2)
    zh = d.pdict['obsptszline'][1,2] - d.pdict['obsptszline'][0,2]
    print(zh) 
    print("Calculating error....")
    for j in range(len(d.freqlist)):
#        umlevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.abs(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_exact_freq%03d.pdf' % freq))
#        ualevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.angle(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_uang_exact_freq%03d.pdf' % freq))
#        vmlevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.abs(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vmag_exact_freq%03d.pdf' % freq))
#        valevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.angle(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vang_exact_freq%03d.pdf' % freq))
        for k in range(len(d.epslist)):
            umag_err_negex, uang_err_negex = calcErr(umag_err_negex, uang_err_negex,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_negex[j][k][ind],d.v_negex[j][k][ind],d.w_negex[j][k][ind],j,k,d.pdict['h']**2)
            umag_err_gauss, uang_err_gauss = calcErr(umag_err_gauss, uang_err_gauss,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_gauss[j][k][ind],d.v_gauss[j][k][ind],d.w_gauss[j][k][ind],j,k,d.pdict['h']**2)
            umag_axiserr_negex, uang_axiserr_negex = calcErr(umag_axiserr_negex, uang_axiserr_negex,np.median(d.uz_negex[j][k]),np.median(d.vz_negex[j][k]),np.median(d.wz_negex[j][k]),d.uz_negex[j][k],d.vz_negex[j][k],d.wz_negex[j][k],j,k,zh)
            umag_axiserr_gauss, uang_axiserr_gauss = calcErr(umag_axiserr_gauss, uang_axiserr_gauss,np.median(d.uz_gauss[j][k]),np.median(d.vz_gauss[j][k]),np.median(d.wz_gauss[j][k]),d.uz_gauss[j][k],d.vz_gauss[j][k],d.wz_gauss[j][k],j,k,zh)
            umag_relerr_negex, uang_relerr_negex = calcRelErr(umag_relerr_negex, uang_relerr_negex,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_negex[j][k][ind],d.v_negex[j][k][ind],d.w_negex[j][k][ind],j,k)
            umag_relerr_gauss, uang_relerr_gauss = calcRelErr(umag_relerr_gauss, uang_relerr_gauss,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_gauss[j][k][ind],d.v_gauss[j][k][ind],d.w_gauss[j][k][ind],j,k)
            umag_axisrelerr_negex, uang_axisrelerr_negex = calcRelErr(umag_axisrelerr_negex, uang_axisrelerr_negex,np.median(d.uz_negex[j][k]),np.median(d.vz_negex[j][k]),np.median(d.wz_negex[j][k]),d.uz_negex[j][k],d.vz_negex[j][k],d.wz_negex[j][k],j,k)
            umag_axisrelerr_gauss, uang_axisrelerr_gauss = calcRelErr(umag_axisrelerr_gauss, uang_axisrelerr_gauss,np.median(d.uz_gauss[j][k]),np.median(d.vz_gauss[j][k]),np.median(d.wz_gauss[j][k]),d.uz_gauss[j][k],d.vz_gauss[j][k],d.wz_gauss[j][k],j,k)
    #add entries to dict and save
    mydict['umag_err_negex'] = umag_err_negex
    mydict['umag_err_gauss'] = umag_err_gauss
    mydict['uang_err_negex'] = uang_err_negex
    mydict['uang_err_gauss'] = uang_err_gauss
    mydict['umag_axiserr_negex'] = umag_axiserr_negex
    mydict['umag_axiserr_gauss'] = umag_axiserr_gauss
    mydict['uang_axiserr_negex'] = uang_axiserr_negex
    mydict['uang_axiserr_gauss'] = uang_axiserr_gauss
    mydict['umag_relerr_negex'] = umag_relerr_negex
    mydict['umag_relerr_gauss'] = umag_relerr_gauss
    mydict['uang_relerr_negex'] = uang_relerr_negex
    mydict['uang_relerr_gauss'] = uang_relerr_gauss
    mydict['umag_axisrelerr_negex'] = umag_axisrelerr_negex
    mydict['umag_axisrelerr_gauss'] = umag_axisrelerr_gauss
    mydict['uang_axisrelerr_negex'] = uang_axisrelerr_negex
    mydict['uang_axisrelerr_gauss'] = uang_axisrelerr_gauss
    F = open( fname+'.pickle', 'w' )
    Pickler(F).dump(mydict)
    F.close()

def adderrs_Linf(fname,mydict=None):
    if mydict == None:
        basedir,basename = os.path.split(fname)
        mydict = fo.loadPickle(basename,basedir,'n')
    d = fo.ExtractDict(mydict)
    umag_Linf_err_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_err_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_err_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_err_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_axiserr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_axiserr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_axiserr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_axiserr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_relerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_relerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_relerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_relerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_axisrelerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_axisrelerr_negex = np.zeros((len(d.freqlist),len(d.epslist),3))
    umag_Linf_axisrelerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    uang_Linf_axisrelerr_gauss = np.zeros((len(d.freqlist),len(d.epslist),3))
    ind = np.nonzero(d.pdict['obspts'][:,0]**2 + d.pdict['obspts'][:,1]**2 > 4*d.pdict['circrad']**2)
    zh = d.pdict['obsptszline'][1,2] - d.pdict['obsptszline'][0,2]
    print(zh) 
    print("Calculating error....")
    for j in range(len(d.freqlist)):
#        umlevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.abs(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_umag_exact_freq%03d.pdf' % freq))
#        ualevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.angle(u_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_uang_exact_freq%03d.pdf' % freq))
#        vmlevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.abs(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vmag_exact_freq%03d.pdf' % freq))
#        valevs = vRS.contourCircle(d.pdict['obspts'][:,0],d.pdict['obspts'][:,1],d.pdict['circrad'],np.angle(v_exact[j]),os.path.expanduser('~/CricketProject/ChooseEpsilon/zradius_vang_exact_freq%03d.pdf' % freq))
        for k in range(len(d.epslist)):
            umag_Linf_err_negex, uang_Linf_err_negex = calcErrLinf(umag_Linf_err_negex, uang_Linf_err_negex,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_negex[j][k][ind],d.v_negex[j][k][ind],d.w_negex[j][k][ind],j,k)
            umag_Linf_err_gauss, uang_Linf_err_gauss = calcErrLinf(umag_Linf_err_gauss, uang_Linf_err_gauss,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_gauss[j][k][ind],d.v_gauss[j][k][ind],d.w_gauss[j][k][ind],j,k)
            umag_Linf_axiserr_negex, uang_Linf_axiserr_negex = calcErrLinf(umag_Linf_axiserr_negex, uang_Linf_axiserr_negex,np.median(d.uz_negex[j][k]),np.median(d.vz_negex[j][k]),np.median(d.wz_negex[j][k]),d.uz_negex[j][k],d.vz_negex[j][k],d.wz_negex[j][k],j,k)
            umag_Linf_axiserr_gauss, uang_Linf_axiserr_gauss = calcErrLinf(umag_Linf_axiserr_gauss, uang_Linf_axiserr_gauss,np.median(d.uz_gauss[j][k]),np.median(d.vz_gauss[j][k]),np.median(d.wz_gauss[j][k]),d.uz_gauss[j][k],d.vz_gauss[j][k],d.wz_gauss[j][k],j,k)
            umag_Linf_relerr_negex, uang_Linf_relerr_negex = calcRelErrLinf(umag_Linf_relerr_negex, uang_Linf_relerr_negex,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_negex[j][k][ind],d.v_negex[j][k][ind],d.w_negex[j][k][ind],j,k)
            umag_Linf_relerr_gauss, uang_Linf_relerr_gauss = calcRelErrLinf(umag_Linf_relerr_gauss, uang_Linf_relerr_gauss,d.u_exact[j][ind],d.v_exact[j][ind],0.0,d.u_gauss[j][k][ind],d.v_gauss[j][k][ind],d.w_gauss[j][k][ind],j,k)
            umag_Linf_axisrelerr_negex, uang_Linf_axisrelerr_negex = calcRelErrLinf(umag_Linf_axisrelerr_negex, uang_Linf_axisrelerr_negex,np.median(d.uz_negex[j][k]),np.median(d.vz_negex[j][k]),np.median(d.wz_negex[j][k]),d.uz_negex[j][k],d.vz_negex[j][k],d.wz_negex[j][k],j,k)
            umag_Linf_axisrelerr_gauss, uang_Linf_axisrelerr_gauss = calcRelErrLinf(umag_Linf_axisrelerr_gauss, uang_Linf_axisrelerr_gauss,np.median(d.uz_gauss[j][k]),np.median(d.vz_gauss[j][k]),np.median(d.wz_gauss[j][k]),d.uz_gauss[j][k],d.vz_gauss[j][k],d.wz_gauss[j][k],j,k)
    #add entries to dict and save
    mydict['umag_Linf_err_negex'] = umag_Linf_err_negex
    mydict['umag_Linf_err_gauss'] = umag_Linf_err_gauss
    mydict['uang_Linf_err_negex'] = uang_Linf_err_negex
    mydict['uang_Linf_err_gauss'] = uang_Linf_err_gauss
    mydict['umag_Linf_axiserr_negex'] = umag_Linf_axiserr_negex
    mydict['umag_Linf_axiserr_gauss'] = umag_Linf_axiserr_gauss
    mydict['uang_Linf_axiserr_negex'] = uang_Linf_axiserr_negex
    mydict['uang_Linf_axiserr_gauss'] = uang_Linf_axiserr_gauss
    mydict['umag_Linf_relerr_negex'] = umag_Linf_relerr_negex
    mydict['umag_Linf_relerr_gauss'] = umag_Linf_relerr_gauss
    mydict['uang_Linf_relerr_negex'] = uang_Linf_relerr_negex
    mydict['uang_Linf_relerr_gauss'] = uang_Linf_relerr_gauss
    mydict['umag_Linf_axisrelerr_negex'] = umag_Linf_axisrelerr_negex
    mydict['umag_Linf_axisrelerr_gauss'] = umag_Linf_axisrelerr_gauss
    mydict['uang_Linf_axisrelerr_negex'] = uang_Linf_axisrelerr_negex
    mydict['uang_Linf_axisrelerr_gauss'] = uang_Linf_axisrelerr_gauss
    F = open( fname+'.pickle', 'w' )
    Pickler(F).dump(mydict)
    F.close()
        
def calcErr(umerr,uaerr,ue,ve,we,ur,vr,wr,j,k,dx):
    umerr[j,k,0] = np.sqrt(dx)*np.sqrt( ((np.abs(ue)-np.abs(ur))**2).sum() ) 
    umerr[j,k,1] = np.sqrt(dx)*np.sqrt( ((np.abs(ve)-np.abs(vr))**2).sum() )
    umerr[j,k,2] = np.sqrt(dx)*np.sqrt( ((np.abs(we)-np.abs(wr))**2).sum() )
    uaerr[j,k,0] = np.sqrt(dx)*np.sqrt( ((np.angle(ue)-np.angle(ur))**2).sum() ) 
    uaerr[j,k,1] = np.sqrt(dx)*np.sqrt( ((np.angle(ve)-np.angle(vr))**2).sum() )
    uaerr[j,k,2] = np.sqrt(dx)*np.sqrt( ((np.angle(we)-np.angle(wr))**2).sum() )
    return umerr, uaerr

def calcRelErr(umerr,uaerr,ue,ve,we,ur,vr,wr,j,k):
    N = len(ue.flat)
    ueabsdenom, ueangledenom = getRelErrDenom(ue)
    veabsdenom, veangledenom = getRelErrDenom(ve)
    weabsdenom, weangledenom = getRelErrDenom(we)
    umerr[j,k,0] = (1.0/N)*np.sqrt( ( ( (np.abs(ue)-np.abs(ur))/ueabsdenom )**2).sum() ) 
    umerr[j,k,1] = (1.0/N)*np.sqrt( ( ( (np.abs(ve)-np.abs(vr))/veabsdenom )**2).sum() ) 
    umerr[j,k,2] = (1.0/N)*np.sqrt( ( ( (np.abs(we)-np.abs(wr))/weabsdenom )**2).sum() ) 
    uaerr[j,k,0] = (1.0/N)*np.sqrt( ( ( (np.angle(ue)-np.angle(ur))/ueangledenom )**2).sum() ) 
    uaerr[j,k,1] = (1.0/N)*np.sqrt( ( ( (np.angle(ve)-np.angle(vr))/veangledenom )**2).sum() ) 
    uaerr[j,k,2] = (1.0/N)*np.sqrt( ( ( (np.angle(we)-np.angle(wr))/weangledenom )**2).sum() ) 
    return umerr, uaerr

def calcErrLinf(umerr,uaerr,ue,ve,we,ur,vr,wr,j,k):
    umerr[j,k,0] = np.max( np.abs(np.abs(ue)  -np.abs(ur))   )
    umerr[j,k,1] = np.max( np.abs(np.abs(ve)  -np.abs(vr))   )
    umerr[j,k,2] = np.max( np.abs(np.abs(we)  -np.abs(wr))   )
    uaerr[j,k,0] = np.max( np.abs(np.angle(ue)-np.angle(ur)) )
    uaerr[j,k,1] = np.max( np.abs(np.angle(ve)-np.angle(vr)) )
    uaerr[j,k,2] = np.max( np.abs(np.angle(we)-np.angle(wr)) )
    return umerr, uaerr

def calcRelErrLinf(umerr,uaerr,ue,ve,we,ur,vr,wr,j,k):
    ueabsdenom, ueangledenom = getRelErrDenom(ue)
    veabsdenom, veangledenom = getRelErrDenom(ve)
    weabsdenom, weangledenom = getRelErrDenom(we)
    umerr[j,k,0] = np.max( np.abs( (np.abs(ue)  -np.abs(ur))/ueabsdenom )     )
    umerr[j,k,1] = np.max( np.abs( (np.abs(ve)  -np.abs(vr))/veabsdenom )     )
    umerr[j,k,2] = np.max( np.abs( (np.abs(we)  -np.abs(wr))/weabsdenom )     )
    uaerr[j,k,0] = np.max( np.abs( (np.angle(ue)-np.angle(ur))/ueangledenom ) )
    uaerr[j,k,1] = np.max( np.abs( (np.angle(ve)-np.angle(vr))/veangledenom ) )
    uaerr[j,k,2] = np.max( np.abs( (np.angle(we)-np.angle(wr))/weangledenom ) )
    return umerr, uaerr

def getRelErrDenom(we):
    weabsdenom = np.abs(we)
    weangledenom = np.angle(we)
    if type(we) is np.ndarray:
        wind = np.nonzero(weabsdenom < 1.e-10)
        weabsdenom[wind] = 1.0
        wind = np.nonzero(weangledenom < 1.e-10)
        weangledenom[wind] = 1.0
    else:
        if weabsdenom < 1.e-10:
            weabsdenom = 1.0
        if weangledenom < 1.e-10:
            weangledenom = 1.0
    return weabsdenom, weangledenom


def setParams():
    basename = 'zquarterradius_farfield_BConaxis_hairrad05'
    if os.path.exists('/Volumes/PATRIOT32G'):
        basedir = '/Volumes/PATRIOT32G/CricketProject/ChooseEpsilon/'
    else:
        basedir = os.path.expanduser('~/CricketProject/ChooseEpsilon/')
    fname = os.path.join(basedir,basename)
    pdict ={}
    circrad = 0.005 #millimeters
    pdict['circrad'] = circrad
    zh = circrad/4
    halfzpts = np.round(200*circrad/zh)
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
    pdict['obspts'] = np.rod.w_stack([X1,X2,X3,X4])
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
    return pdict, fname

if __name__ == '__main__':
#    optimizeEps()
    print('z radius....')
    basename = 'zradius_farfield_BConaxis_hairrad05'
    if os.path.exists('/Volumes/PATRIOT32G'):
        basedir = '/Volumes/PATRIOT32G/CricketProject/ChooseEpsilon/'
    else:
        basedir = os.path.expanduser('~/CricketProject/ChooseEpsilon/')
    fname = os.path.join(basedir,basename)
    adderrs_Linf(fname)
    print('z half radius...')
    basename = 'zhalfradius_farfield_BConaxis_hairrad05'
    fname = os.path.join(basedir,basename)
    adderrs_Linf(fname)
#    setParams()
    
