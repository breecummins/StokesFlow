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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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



