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
import matplotlib as mpl
try:
    import StokesFlow.utilities.fileops as fo
    print("not Stokesflow")
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
    ph2=plt.contourf(x,y,u,ulevs,cmap=mpl.cm.RdGy)
    plt.colorbar(ph2)
    phi = 2*np.pi*np.arange(0,1.01,0.01)
    plt.plot(a*np.cos(phi),a*np.sin(phi),'k',linewidth=1.0)
    plt.title(titlestr)
    if fname:
        plt.savefig(fname,format='pdf')
    return ulevs

def myPlots(xvals,yvals,titlestr,xstr,ystr,leglabels=None,plottype=plt.plot,fname=None,clearme=True,stylestr='k-'):
    #works with plt.loglog too
    if clearme:
        plt.clf()
    if len(yvals.shape) == 2 and len(xvals.shape)==1:
        for k in range(yvals.shape[1]):
            if leglabels != None:
                plottype(xvals,yvals[:,k],stylestr,linewidth=2.0,label=leglabels[k])
            else:
                plottype(xvals,yvals[:,k],stylestr,linewidth=2.0)
    elif len(yvals.shape) == 2 and len(xvals.shape)==2:
        for k in range(yvals.shape[1]):
            if leglabels != None:
                plottype(xvals[:,k],yvals[:,k],stylestr,linewidth=2.0,label=leglabels[k])
            else:
                plottype(xvals[:,k],yvals[:,k],stylestr,linewidth=2.0)
    else:
        plottype(xvals,yvals,stylestr,linewidth=2.0,label=leglabels) 
    plt.xlim([np.min(xvals),np.max(xvals)])
    if titlestr != None:  
        plt.title(titlestr)
    if xstr != None: 
        plt.xlabel(xstr)
    if ystr != None:
        plt.ylabel(ystr)
    if leglabels != None:
        plt.legend()
    mpl.rc('font',size=22)
    if fname != None:
        plt.savefig(fname,format='pdf', bbox_inches="tight")

def plotchooseepserr_farfield(mydict,basedir,basename):
    earr = np.asarray(mydict['epslist'])
    freqlist = mydict['freqlist']
    leg = [str(f) for f in freqlist]
    plainPlots(earr,mydict['umag_err_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','L2 error',leg,basedir+basename+'/umag_err_negex.pdf')
    plainPlots(earr,mydict['umag_err_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','L2 error',leg,basedir+basename+'/umag_err_gauss.pdf')
    plainPlots(earr,mydict['umag_relerr_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','Relative L2 error',leg,basedir+basename+'/umag_relerr_negex.pdf')
    plainPlots(earr,mydict['umag_relerr_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','Relative L2 error',leg,basedir+basename+'/umag_relerr_gauss.pdf')

def plotchooseepserr_zline(mydict,basedir,basename):
    earr = np.asarray(mydict['epslist'])
    freqlist = mydict['freqlist']
    leg = [str(f) for f in freqlist]
    plainPlots(earr[7:],mydict['umag_axiserr_negex'][:,7:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','L2 error',leg,basedir+basename+ '/umag_axiserr_trunc2_negex.pdf')
    plainPlots(earr[7:],mydict['umag_axiserr_gauss'][:,7:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','L2 error',leg,basedir+basename+'/umag_axiserr_trunc2_gauss.pdf')
    plainPlots(earr[7:],mydict['umag_axisrelerr_negex'][:,7:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','Relative L2 error',leg,basedir+basename+ '/umag_axisrelerr_trunc2_negex.pdf')
    plainPlots(earr[7:],mydict['umag_axisrelerr_gauss'][:,7:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','Relative L2 error',leg,basedir+basename+'/umag_axisrelerr_trunc2_gauss.pdf')

def plotchooseepserr_Linf_farfield(mydict,basedir,basename):
    earr = np.asarray(mydict['epslist'])
    freqlist = mydict['freqlist']
    leg = [str(f) for f in freqlist]
    plainPlots(earr,mydict['umag_Linf_err_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','$L_/infty$ error',leg,basedir+basename+'/umag_Linf_err_negex.pdf')
    plainPlots(earr,mydict['umag_Linf_err_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','$L_/infty$ error',leg,basedir+basename+'/umag_Linf_err_gauss.pdf')
    plainPlots(earr,mydict['umag_Linf_relerr_negex'][:,:,0].transpose(),'Magnitude u, neg exp','$\epsilon$','Relative $L_/infty$ error',leg,basedir+basename+'/umag_Linf_relerr_negex.pdf')
    plainPlots(earr,mydict['umag_Linf_relerr_gauss'][:,:,0].transpose(),'Magnitude u, gaussian','$\epsilon$','Relative $L_/infty$ error',leg,basedir+basename+'/umag_Linf_relerr_gauss.pdf')

def plotchooseepserr_Linf_zline(mydict,basedir,basename):
    earr = np.asarray(mydict['epslist'])
    freqlist = mydict['freqlist']
    leg = [str(f) for f in freqlist]
    plainPlots(earr[7:],mydict['umag_Linf_axiserr_negex'][:,7:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','$L_/infty$ error',leg,basedir+basename+ '/umag_Linf_axiserr_trunc2_negex.pdf')
    plainPlots(earr[7:],mydict['umag_Linf_axiserr_gauss'][:,7:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','$L_/infty$ error',leg,basedir+basename+'/umag_Linf_axiserr_trunc2_gauss.pdf')
    plainPlots(earr[7:],mydict['umag_Linf_axisrelerr_negex'][:,7:,0].transpose(),'Difference in u from median on z-axis, neg exp','$\epsilon$','Relative $L_/infty$ error',leg,basedir+basename+ '/umag_Linf_axisrelerr_trunc2_negex.pdf')
    plainPlots(earr[7:],mydict['umag_Linf_axisrelerr_gauss'][:,7:,0].transpose(),'Difference in u from median on z-axis, gaussian','$\epsilon$','Relative $L_/infty$ error',leg,basedir+basename+'/umag_Linf_axisrelerr_trunc2_gauss.pdf')

def plotzline(mydict,basedir,basename,eind,find):    
    eps = mydict['epslist'][eind]
    freq = mydict['freqlist'][find]
    plainPlots(mydict['pdict']['obsptszline'][:,2],np.abs(np.asarray(mydict['uz_negex'])[find,eind,:].transpose()),'Magnitude u along axis, $\epsilon = $ %.04f, freq = %d, neg exp'  % (eps,freq),'z','|u|',None,basedir+basename+'/umag_zline_eps%05d_freq%03d_negex.pdf' % (eps*100000,freq))
    plainPlots(mydict['pdict']['obsptszline'][:,2],np.abs(np.asarray(mydict['uz_gauss'])[find,eind,:].transpose()),'Magnitude u along axis, $\epsilon = $ %.04f, freq = %d, gaussian' % (eps,freq),'z','|u|',None,basedir+basename+'/umag_zline_eps%05d_freq%03d_gauss.pdf' % (eps*100000,freq))
    
    
def plotblobs(basedir,basename,epslist):
    r = np.linspace(0,1.e-2)
    negex=[]
    gauss=[]
    for eps in epslist:
        negex.append(lRS.Brinkman3DNegExpBlob(r,eps))
        gauss.append(lRS.Brinkman3DGaussianBlob(r,eps))
    leg = ['$\epsilon$ = ' + str(s) for s in epslist]
    plainPlots(r,np.asarray(negex).transpose(),'Negative Exponential Blob','distance (mm) from blob location','blob strength',leg,basedir+basename+'/negexpblob%05d.pdf' % int(np.round(epslist[0]*100000)))
    plainPlots(r,np.asarray(gauss).transpose(),'Gaussian Exponential Blob','distance (mm) from blob location','blob strength',leg,basedir+basename+'/gaussblob%05d.pdf' % int(np.round(epslist[0]*100000)))

if __name__ == '__main__':
#    basedir = os.path.expanduser('/Volumes/PATRIOT32G/CricketProject/QuasiSteadyVSFourier/')
#    basename = 'freq185'
#    print('loading file...')
#    mydict = fo.loadPickle(basename,basedir)
#    tvec = mydict['dt'][0]*np.arange(mydict['u_fourier'][0].shape[1])
#    ptind = 0#len(mydict['x'])-1
#    freqind = 0
#    freq = mydict['freqlist'][freqind]
#    plainPlots(tvec,np.real(mydict['u_fourier'][freqind][ptind,:]),'u fourier, x loc = %0.2f' % mydict['x'][ptind],'time','x velocity',None,fname=os.path.join(os.path.join(basedir,basename),'u_fourier_freq%03d_point%02d.pdf' % (freq,ptind)))
#    plainPlots(tvec,np.real(mydict['u_quasi'][freqind][ptind,:]),'u quasi, x loc = %0.2f' % mydict['x'][ptind],'time','x velocity',None,fname=os.path.join(os.path.join(basedir,basename),'u_quasi_freq%03d_point%02d.pdf' % (freq,ptind)))

    basedir = os.path.expanduser('~/CricketProject/QuasiSteadyVSFourier/')
    if not os.path.exists(basedir):
        basedir = '/Volumes/PATRIOT32G/CricketProject/QuasiSteadyVSFourier/'   
        if not os.path.exists(basedir):        
            print('Choose a different directory for saving files')
            raise(SystemExit)
    basename = 'freq100'
    print('loading file...')
    mydict = fo.loadPickle(basename,basedir)
    N = mydict['u_fourier'][0].shape[1]
    tvec = mydict['dt']*np.arange(0,N)
    for ptind in [0,len(mydict['x'])/2,len(mydict['x'])-1]:
        myPlots(tvec,np.real(mydict['u_fourier'][0][ptind,:]),'location = %0.2f' % mydict['x'][ptind],'time','velocity',fname=os.path.join(os.path.join(basedir,basename),'u_fourier_point%02d.pdf' % ptind),stylestr='k-')
        myPlots(tvec,np.real(mydict['u_quasi'][0][ptind,:]),'location = %0.2f' % mydict['x'][ptind],'time','velocity',fname=os.path.join(os.path.join(basedir,basename),'u_quasi_point%02d.pdf' % ptind),stylestr='k-')
    maxvec_fouri = np.zeros(mydict['x'].shape)
    maxvec_quasi = np.zeros(mydict['x'].shape)
    for ptind in range(len(mydict['x'])):
        maxvec_fouri[ptind] = np.max(np.real(mydict['u_fourier'][0][ptind,int(N/2):]))
        maxvec_quasi[ptind] = np.max(np.real(mydict['u_quasi'][0][ptind,int(N/2):]))
    #print(maxvec_quasi)
    #print(maxvec_fouri)
    #plot fourier results
    myPlots(mydict['x'],maxvec_fouri,None,'initial location','max velocity','fourier simulation',plt.loglog,os.path.join(os.path.join(basedir,basename),'u_fourier_xdecay.pdf'),stylestr='k-')
    #plot slope of -3
    myPlots(mydict['x'],(mydict['x'])**(-3)*np.exp(0.5),None,'initial location','max velocity','slope of -3',plt.loglog,os.path.join(os.path.join(basedir,basename),'u_fourier_xdecay.pdf'),False,'k--')
    #plot quasi-steady results
    myPlots(mydict['x'],maxvec_quasi,None,'initial location','max velocity','quasi-steady simulation',plt.loglog,os.path.join(os.path.join(basedir,basename),'u_quasi_xdecay.pdf'),stylestr='k-')
    #plot slope of -1
    myPlots(mydict['x'],(mydict['x'])**(-1)*np.exp(0.5),None,'initial location','max velocity','slope of -1',plt.loglog,os.path.join(os.path.join(basedir,basename),'u_quasi_xdecay.pdf'),False,'k--')


##    epslist = [k*0.005 for k in np.arange(0.05,1.8,0.05)]
##    eind = 10
##    eps = epslist[eind]
##    freqlist = [5,10,15,20,25] 
##    freqlist.extend(range(35,310,25))
##    find = int(len(freqlist)/2)
##    freq = freqlist[find]
##    print('freq',freq,'eps',eps)
#    basedir = os.path.expanduser('/Volumes/PATRIOT32G/CricketProject/ChooseEpsilon/')
#    basename = 'zradius_farfield_BConaxis_hairrad05'
#    print('loading file...')
#    mydict = fo.loadPickle(basename,basedir)
##    print('z radius, zline...')
##    plotzline(mydict,basedir,basename,eind,find)
##    plotblobs(basedir,basename,[eps])
##    print('z radius, L2 far field error...')
##    plotchooseepserr_farfield(mydict,basedir,basename)
##    print('z radius, L2 axis error...')
##    plotchooseepserr_zline(mydict,basedir,basename)
#    print('z radius, Linf far field error...')
#    plotchooseepserr_Linf_farfield(mydict,basedir,basename)
#    print('z radius, Linf axis error...')
#    plotchooseepserr_Linf_zline(mydict,basedir,basename)
#
#    basename = 'zhalfradius_farfield_BConaxis_hairrad05'
#    print('loading file...')
#    mydict = fo.loadPickle(basename,basedir)
##    print('z half radius, zline...')
##    plotzline(mydict,basedir,basename,eind,find)
##    plotblobs(basedir,basename,[eps])
##    print('z half radius, L2 far field error...')
##    plotchooseepserr_farfield(mydict,basedir,basename)
##    print('z half radius, L2 axis error...')
##    plotchooseepserr_zline(mydict,basedir,basename)
#    print('z half radius, Linf far field error...')
#    plotchooseepserr_Linf_farfield(mydict,basedir,basename)
#    print('z half radius, Linf axis error...')
#    plotchooseepserr_Linf_zline(mydict,basedir,basename)
#
##    plotblobs(basedir,basename,epslist)
##    plotblobs(basedir,basename,[0.0015])
##    plotblobs(basedir,basename,[0.004])
##    plotblobs(basedir,basename,[0.006])
