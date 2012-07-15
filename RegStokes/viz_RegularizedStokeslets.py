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
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import StokesFlow.utilities.fileops as fo
except:
    import utilities.fileops as fo
import lib_RegularizedStokeslets as lRS

def contourCircle(x,y,a,u,fname=None,ulevs=None,titlestr=None):
    ind = np.nonzero(x**2 + y**2 > a**2)
    if ulevs == None:
        umin = np.min(u[ind])
        if umin <0:
            umin = 1.05*umin
        else:
            umin = 0.95*umin
        umax = np.max(u[ind])
        if umax <0:
            umax = 0.95*umax
        else:
            umax = 1.05*umax
        ulevs = np.linspace(umin,umax,30)
    plt.clf()
    ph2=plt.contourf(x,y,u,ulevs,cmap=cm.RdGy)
    plt.colorbar(ph2)
    phi = 2*np.pi*np.arange(0,1.01,0.01)
    plt.plot(a*np.cos(phi),a*np.sin(phi),'k',linewidth=1.0)
    plt.title(titlestr)
    if fname != None:
        plt.savefig(fname,format='pdf')
    return ulevs

def plainPlots(xvals,yvals,titlestr,xstr,ystr,leglabels=None,fname=None,clearme=True,stylestr='-'):
    if clearme:
        plt.clf()
    if len(yvals.shape) == 2 and len(xvals.shape)==1:
        for k in range(yvals.shape[1]):
            if leglabels != None:
                plt.plot(xvals,yvals[:,k],label=leglabels[k])
            else:
                plt.plot(xvals,yvals[:,k])
    elif len(yvals.shape) == 2 and len(xvals.shape)==2:
        for k in range(yvals.shape[1]):
            if leglabels != None:
                plt.plot(xvals[:,k],yvals[:,k],stylestr,label=leglabels[k])
            else:
                plt.plot(xvals[:,k],yvals[:,k],stylestr)
    else:
        plt.plot(xvals,yvals) 
    plt.title(titlestr)
    plt.xlabel(xstr)
    plt.ylabel(ystr)
    if leglabels != None:
        plt.legend()
    if fname != None:
        plt.savefig(fname,format='pdf')

def plotchooseepserr(basedir,basename):
    mydict = fo.loadPickle(basename,basedir)
    earr = np.asarray(mydict['epslist'])
    freqlist = mydict['freqlist']
    plainPlots(earr,mydict['umag_err_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_err_negex.pdf')
    plainPlots(earr,mydict['umag_err_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_err_gauss.pdf')
    plainPlots(earr,mydict['umag_axiserr_negex'][:,:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','L2 error',[str(f) for f in freqlist],basedir+basename+ '/umag_axiserr_negex.pdf')
    plainPlots(earr,mydict['umag_axiserr_gauss'][:,:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_axiserr_gauss.pdf')
    plainPlots(earr,mydict['umag_relerr_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','Relative L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_relerr_negex.pdf')
    plainPlots(earr,mydict['umag_relerr_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','Relative L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_relerr_gauss.pdf')
    plainPlots(earr,mydict['umag_axisrelerr_negex'][:,:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','Relative L2 error',[str(f) for f in freqlist],basedir+basename+ '/umag_axisrelerr_negex.pdf')
    plainPlots(earr,mydict['umag_axisrelerr_gauss'][:,:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','Relative L2 error',[str(f) for f in freqlist],basedir+basename+'/umag_axisrelerr_gauss.pdf')
    
    eind = 6
    plainPlots(mydict['pdict']['obsptszline'][:,2],np.abs(np.asarray(mydict['uz_negex'])[:,eind,:]),'Magnitude u along axis, $\epsilon = $ %.04f, neg exp' % mydict['epslist'][eind],'z','|u|',[str(f) for f in freqlist],basedir+basename+'/umag_zline_negex.pdf')
    plainPlots(mydict['pdict']['obsptszline'][:,2],np.abs(np.asarray(mydict['uz_gauss'])[:,eind,:]),'Magnitude u along axis, $\epsilon = $ %.04f, gaussian' % mydict['epslist'][eind],'z','|u|',[str(f) for f in freqlist],basedir+basename+'/umag_zline_gauss.pdf')
    
    
def plotblobs(basedir,basename,epslist):
    r = np.linspace(0,1.e-2)
    negex=[]
    gauss=[]
    for eps in epslist:
        negex.append(lRS.Brinkman3DNegExpBlob(r,eps))
        gauss.append(lRS.Brinkman3DGaussianBlob(r,eps))
    leg = ['$\epsilon$ = ' + str(s) for s in epslist]
    plainPlots(r,np.asarray(negex).transpose(),'Negative Exponential Blob','distance (mm) from blob location','blob strength',leg,basedir+basename+'/negexpblob'+ str(epslist[0]*10000) +'.pdf')
    plainPlots(r,np.asarray(gauss).transpose(),'Gaussian Exponential Blob','distance (mm) from blob location','blob strength',leg,basedir+basename+'/gaussblob'+ str(epslist[0]*10000) +'.pdf')

if __name__ == '__main__':
    basedir = os.path.expanduser('/Volumes/LCD/CricketProject/ChooseEpsilon/')
    basename = 'zradius_farfield_BConaxis_hairrad05'
    plotchooseepserr(basedir,basename)
    basename = 'zhalfradius_farfield_BConaxis_hairrad05'
    plotchooseepserr(basedir,basename)
#    epslist = [k*0.005 for k in np.arange(0.05,1.8,0.05)]
#    plotblobs(basedir,basename,epslist)
#    plotblobs(basedir,basename,[0.0015])
#    plotblobs(basedir,basename,[0.004])
#    plotblobs(basedir,basename,[0.006])
