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

def checkXLine():
    a = 1.0
    mu = 1.0
    rho = 1.0
    U = 1.0
    x = np.linspace(1.0,10.0,250)
    y = np.zeros(x.shape)
    z = np.zeroes(x.shape)
    for freq in [10*k for k in range(1,31)]:
        T = 10.0/(2*np.pi*freq)
        

def calcFourier():
    lES.sphere3DOscillating(x,y,z,a,alph,freq,mu,t)
    
def calcQuasiSteady():
    lES.sphere3D(x,y,z,a,mu)

