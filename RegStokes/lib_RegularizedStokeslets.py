#Created by Breschine Cummins on May 5, 2012.

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

def Brinkman3DNegExpStokesletsH1(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=(2*pdict['sig']**2 + 8*pdict['sig'] + 9) / (168*pdict['eps']*np.pi*pdict['sig']**4 + 672*pdict['eps']*np.pi*pdict['sig']**3 + 1008*pdict['eps']*np.pi*pdict['sig']**2 + 672*pdict['eps']*np.pi*pdict['sig'] + 168*pdict['eps']*np.pi) #analytic limit of H1 at r=0
    H1 = lim*np.ones(r.shape)    
    term0 = -1.0/(4*pdict['alph']**2*np.pi*r[ind]**3) + np.exp(-pdict['alph']*r[ind]) * ((7-pdict['sig']**2)/(28*np.pi*(pdict['sig']**2 - 1)**4)) * ( 1.0/r[ind] + 1.0/ (pdict['alph']*r[ind]**2) + 1.0 / (pdict['alph']**2*r[ind]**3) )
    term1 = r[ind]**2 / (224*pdict['eps']**3*np.pi*(pdict['sig']**2 - 1)) + (5*pdict['sig']**2 - 11) * r[ind] / (224*pdict['eps']**2*np.pi*(pdict['sig']**2 - 1)**2)
    term2 = (3*pdict['sig']**4 - 10*pdict['sig']**2 + 13) / (56*np.pi*pdict['eps']*(pdict['sig']**2 - 1)**3) + (7*pdict['sig']**6 - 28*pdict['sig']**4 + 43*pdict['sig']**2 - 34) / (56*np.pi*(pdict['sig']**2 - 1)**4 * r[ind])
    term3 = pdict['eps']*(7*pdict['sig']**6 + 42*pdict['sig']**2 - 27 - 28*pdict['sig']**4) / (28*np.pi*(pdict['sig']**2 - 1)**4 * r[ind]**2) 
    term4 = pdict['eps']**2*(7*pdict['sig']**6 + 42*pdict['sig']**2 - 27 - 28*pdict['sig']**4) / (28*np.pi*(pdict['sig']**2 - 1)**4 * r[ind]**3)   
    H1[ind] = term0 + np.exp(-r[ind]/pdict['eps']) * (  term1 +  term2 + term3 + term4 )
    return H1

    
def Brinkman3DNegExpStokesletsH2(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=(3*pdict['sig']**2 + 8*pdict['sig'] +2) / (1680*pdict['eps']**3*np.pi*pdict['sig']**4 + 6720*pdict['eps']**3*np.pi*pdict['sig']**3 + 10080*pdict['eps']**3*np.pi*pdict['sig']**2 + 6720*pdict['eps']**3*np.pi*pdict['sig'] + 1680*pdict['eps']**3*np.pi) #analytic limit of H2 
    H2 = lim*np.ones(r.shape)
    term0 = 3.0/(4*r[ind]**5*pdict['alph']**2*np.pi) - (7 - pdict['sig']**2)*np.exp(-pdict['alph']*r[ind]) * (pdict['alph']**2*r[ind]**2 + 3*pdict['alph']*r[ind] + 3) / (28*r[ind]**5*pdict['alph']**2*np.pi*(pdict['sig']**2 - 1)**4)
    term1 = -1.0/ (224*np.pi*pdict['eps']**3*(pdict['sig']**2 - 1)) - (7*pdict['sig']**2 - 13) / (224*np.pi*pdict['eps']**2*r[ind]*(pdict['sig']**2 - 1)**2)
    term2 = - (7*pdict['sig']**4 - 21*pdict['sig']**2 + 20) / (56*np.pi*pdict['eps']*(pdict['sig']**2 - 1)**3 * r[ind]**2) - (-84*pdict['sig']**4 + 125*pdict['sig']**2 - 74 + 21*pdict['sig']**6) / (56*np.pi*(pdict['sig']**2 - 1)**4 * r[ind]**3)
    term3 = - 3*pdict['eps']*(7*pdict['sig']**6 + 42*pdict['sig']**2 - 27 - 28*pdict['sig']**4) / (28*np.pi*(pdict['sig']**2 - 1)**4 * r[ind]**4) 
    term4 = - 3*pdict['eps']**2*(7*pdict['sig']**6 + 42*pdict['sig']**2 - 27 - 28*pdict['sig']**4) / (28*np.pi*(pdict['sig']**2 - 1)**4 * r[ind]**5)    
    H2[ind] = term0 + np.exp(-r[ind]/pdict['eps']) *( term1 + term2 + term3 + term4 )
    return H2

def Brinkman3DNegExpStokesletsH1prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    H1p = lim*np.ones(r.shape)    
    term0 = 3.0/(4*np.pi*r[ind]**2 * z**2) 
    term1 = np.exp(-z)*(3+3*z+2*z**2+z**3)*(-7+pdict['sig']**2) / (28*np.pi*r[ind]**2 * z**2 * (-1 + pdict['sig']**2)**4)
    term2 = 3*r[ind]**5 * pdict['eps'] * (-3 + pdict['sig']**2) * (-1 + pdict['sig']**2)**2 + r[ind]**6 * (-1 + pdict['sig']**2)**3
    term3 = r[ind]**4 * pdict['eps']**2 * (-41 + 65*pdict['sig']**2 - 31*pdict['sig']**4 + 7*pdict['sig']**6)  
    term4 = 24 * pdict['eps']**5 * (r[ind] + pdict['eps']) * (-27 + 42*pdict['sig']**2 - 28*pdict['sig']**4 + 7*pdict['sig']**6) 
    term5 = 4*r[ind]**3 * pdict['eps']**3 * (-34 + 43*pdict['sig']**2 - 28*pdict['sig']**4 + 7*pdict['sig']**6) 
    term6 = 4*r[ind]**2 * pdict['eps']**4 * (-88 + 127*pdict['sig']**2 - 84*pdict['sig']**4 + 21*pdict['sig']**6) 
    H1p[ind] = term0 + term1 - np.exp(-r[ind]/pdict['eps'])*(term2+term3+term4+term5+term6) / (224*np.pi*r[ind]**4 * pdict['eps']**4 * (-1 + pdict['sig']**2)**4) 
    return H1p
   
def Brinkman3DNegExpStokesletsH2prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    H2p = lim*np.ones(r.shape)    
    term0 = -15.0 / (4*np.pi*r[ind]**4 * z**2)
    term1 = -np.exp(-z) * (15 + 15*z + 6*z**2 + z**3) * (-7 + pdict['sig']**2) / (28*np.pi*r[ind]**4 * z**2 * (-1 + pdict['sig']**2)**4)
    term2 = r[ind]**6 * (-1 + pdict['sig']**2)**3 + r[ind]**5 * pdict['eps'] * (-1 + pdict['sig']**2)**2 * (-13 + 7*pdict['sig']**2)
    term3 = 120*pdict['eps']**5 * (r[ind] + pdict['eps']) * (-27 + 42*pdict['sig']**2 - 28*pdict['sig']**4 + 7*pdict['sig']**6)
    term4 = 4*r[ind]**3 * pdict['eps']**3 * (-114 + 207*pdict['sig']**2 - 140*pdict['sig']**4 + 35*pdict['sig']**6)
    term5 = 12*r[ind]**2 * pdict['eps']**4 * (-128 + 209*pdict['sig']**2 - 140*pdict['sig']**4 + 35*pdict['sig']**6)
    term6 = r[ind]**4 * pdict['eps']**2 * (-93 + 197*pdict['sig']**2 - 139*pdict['sig']**4 + 35*pdict['sig']**6)  
    H2p[ind] = term0 + term1 + np.exp(-r[ind]/pdict['eps'])*(term2+term3+term4+term5+term6) / (224*np.pi*r[ind]**6 * pdict['eps']**4 * (-1 + pdict['sig']**2)**4)
    return H2p

def Brinkman3DNegExpStokesletsPressure(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=1.0/(168*np.pi*pdict['eps']**3)
    P = lim*np.ones(r.shape)
    P[ind] = 1.0/(4*np.pi*r[ind]**3) - np.exp(-r[ind]/pdict['eps'])*(r[ind]**4+ 8*pdict['eps']*r[ind]**3 + 28*r[ind]**2*pdict['eps']**2 + 56*r[ind]*pdict['eps']**3 + 56*pdict['eps']**4) / (224*np.pi*pdict['eps']**4 * r[ind]**3)
    return P

def Brinkman3DNegExpDipolesD1(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= -(2 + 8*pdict['sig'] + 3*pdict['sig']**2) / (168*np.pi*pdict['eps']**3 * (1+pdict['sig'])**4)
    D1 = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 = -np.exp(-z) * (1+z+z**2) * (-7+pdict['sig']**2) / (28*np.pi*r[ind]**3 * (-1+pdict['sig']**2)**4)
    term11 = 8*pdict['eps']**4 * (-7+pdict['sig']**2) * (r[ind]+pdict['eps']) 
    term12 = r[ind]**4 * (-1+pdict['sig']**2)**2*( r[ind]*(-1+pdict['sig']**2) - pdict['eps']*(5+pdict['sig']**2) ) 
    term13 = 4*r[ind]**2 * pdict['eps']**2 * ( pdict['eps']*(-7 - 6*pdict['sig']**2 + pdict['sig']**4) - r[ind]*(3 + pdict['sig']**2 - 5*pdict['sig']**4 + pdict['sig']**6))
    term1 = np.exp(-r[ind]/pdict['eps']) * (term11 + term12 + term13) / (224*np.pi*r[ind]**3 * pdict['eps']**5 * (-1+pdict['sig']**2)**4)
    D1[ind] = term0 + term1
    return D1

def Brinkman3DNegExpDipolesD2(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= -(3 + 12*pdict['sig'] + 16*pdict['sig']**2 + 4*pdict['sig']**3) / (1680*np.pi*pdict['eps']**5 * (1+pdict['sig'])**4)
    D2 = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 = np.exp(-z) * (3+3*z+z**2) * (-7+pdict['sig']**2) / (28*np.pi*r[ind]**5 * (-1+pdict['sig']**2)**4)
    term11 = -24*pdict['eps']**4 * (-7+pdict['sig']**2) * (r[ind]+pdict['eps']) 
    term12 = r[ind]**4 * (-1+pdict['sig']**2)**2*( -r[ind]*(-1+pdict['sig']**2) - pdict['eps']*(-7+pdict['sig']**2) ) 
    term13 = 4*r[ind]**2 * pdict['eps']**2 * ( pdict['eps']*(21- 10*pdict['sig']**2 + pdict['sig']**4) + r[ind]*(7 - 8*pdict['sig']**2 + pdict['sig']**4))
    term1 = np.exp(-r[ind]/pdict['eps']) * (term11 + term12 + term13) / (224*np.pi*r[ind]**5 * pdict['eps']**5 * (-1+pdict['sig']**2)**4)
    D2[ind] = term0 + term1
    return D2

def Brinkman3DNegExpDipolesD1prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= 0*pdict['alph']
    D1p = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 = np.exp(-z)*(3+3*z+2*z**2+z**3)*(-7+pdict['sig']**2) / (28*np.pi*r[ind]**4 * (-1+pdict['sig']**2)**4)
    term1 = -24*pdict['eps']**5 * (r[ind] + pdict['eps']) * (-7+pdict['sig']**2) - r[ind]**6 * (-1+pdict['sig']**2)**3
    term2 = 3*r[ind]**5 * pdict['eps'] * (-1+pdict['sig']**2)**2 * (1+pdict['sig']**2) - 4*r[ind]**3 * pdict['eps']**3 * (-7 -6*pdict['sig']**2 + pdict['sig']**4)
    term3 = r[ind]**4 * pdict['eps']**2 * (7 + 13*pdict['sig']**2 - 23*pdict['sig']**4 + 3*pdict['sig']**6)
    term4 = r[ind]**2 * pdict['eps']**4 * (84 + 16*pdict['sig']**2 - 4*pdict['sig']**4)
    D1p[ind] = term0 + np.exp(-r[ind]/pdict['eps'])*(term1+term2+term3+term4) / (224*np.pi*r[ind]**4 * pdict['eps']**6 * (-1+pdict['sig']**2)**4)
    return D1p

def Brinkman3DNegExpDipolesD2prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= 1.0 / (1344*np.pi*pdict['eps']**6)
    D2p = lim*np.ones(r.shape,dtype=type(pdict['alph']))
    z = r[ind]*pdict['alph']
    term0 = -np.exp(-z)*(15+15*z+6*z**2+z**3)*(-7+pdict['sig']**2) / (28*np.pi*r[ind]**6 * (-1+pdict['sig']**2)**4)
    term1 = 120*pdict['eps']**5 * (r[ind] + pdict['eps']) * (-7+pdict['sig']**2) + r[ind]**6 * (-1+pdict['sig']**2)**3
    term2 = r[ind]**5 * pdict['eps'] * (-1+pdict['sig']**2)**2 * (-7+pdict['sig']**2) - 4*r[ind]**3 * pdict['eps']**3 * (35 - 26*pdict['sig']**2 + 3*pdict['sig']**4)
    term3 = r[ind]**4 * pdict['eps']**2 * (-35 + 47*pdict['sig']**2 - 13*pdict['sig']**4 + pdict['sig']**6)
    term4 = -12 * r[ind]**2 * pdict['eps']**4 * (35 - 12*pdict['sig']**2 + pdict['sig']**4)
    D2p[ind] = term0 + np.exp(-r[ind]/pdict['eps'])*(term1+term2+term3+term4) / (224*np.pi*r[ind]**6 * pdict['eps']**6 * (-1+pdict['sig']**2)**4)
    return D2p

def Brinkman3DNegExpDipolesPressure(r,pdict):
    #limit not required since there is no division by r
    P = -np.exp(-r/pdict['eps'])*(r + 2*pdict['eps']) / (224*np.pi*pdict['eps']**6)
    return P

def Brinkman3DGaussianStokesletsH1(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= 1.0/(3*pdict['eps']*np.pi**1.5) - np.exp((pdict['sig']**2)/4)*pdict['alph']*(1-ss.erf(pdict['sig']/2))/(6*np.pi)     
    H1 = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 = -2*ss.erf(r[ind]/pdict['eps'])
    term1 =  np.exp((pdict['sig']**2)/4 - z)*(1+z+z**2)*(1+ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2))
    term2 = -np.exp((pdict['sig']**2)/4 + z)*(1-z+z**2)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2))
    H1[ind] = (term0 + term1 +  term2)/(8*np.pi*r[ind] * z**2) 
    return H1
    
def Brinkman3DGaussianStokesletsH2(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim= (2.0-pdict['sig']**2)/(30*(pdict['eps']**3)*(np.pi**1.5)) + np.exp((pdict['sig']**2)/4)*(pdict['alph']**3)*(1-ss.erf(pdict['sig']/2))/(60*np.pi)     
    H2 = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 = 6*ss.erf(r[ind]/pdict['eps'])
    term1 = -np.exp((pdict['sig']**2)/4 - z)*(3+3*z+z**2)*(1+ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2))
    term2 =  np.exp((pdict['sig']**2)/4 + z)*(3-3*z+z**2)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2))
    H2[ind] = (term0 + term1 +  term2)/(8*np.pi*r[ind]**3 * z**2) 
    return H2

def Brinkman3DGaussianStokesletsH1prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    H1p = lim*np.ones(r.shape)    
    term0 = (np.exp((pdict['sig']**2)/4 - z)*(3 + 3*z + 2*z**2 + z**3) / (4*np.pi*r[ind]**2*z**2) ) * (-1 + 0.5*(1-ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2)) )
    term1 = np.exp(-r[ind]**2/pdict['eps']**2)/(2*np.pi**(1.5)*r[ind]*pdict['eps'])
    term2 = 3*ss.erf(r[ind]/pdict['eps']) / (4*np.pi*r[ind]**2*z**2)
    term3 = -np.exp((pdict['sig']**2)/4 + z)*(-3 + 3*z - 2*z**2 + z**3)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2)) / (8*np.pi*r[ind]**2*z**2)
    H1p[ind] = term0 + term1 + term2 + term3 
    return H1p

def Brinkman3DGaussianStokesletsH2prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    H2p = lim*np.ones(r.shape)    
    term0 = (np.exp((pdict['sig']**2)/4 - z)*(15 + 15*z + 6*z**2 + z**3) / (4*np.pi*r[ind]**4*z**2) ) * (1 + 0.5*(1-ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2)) )
    term1 = -np.exp(-r[ind]**2/pdict['eps']**2)/(2*np.pi**(1.5)*r[ind]**3*pdict['eps'])
    term2 = -15*ss.erf(r[ind]/pdict['eps']) / (4*np.pi*r[ind]**4*z**2)
    term3 = np.exp((pdict['sig']**2)/4 + z)*(-15 + 15*z - 6*z**2 + z**3)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2)) / (8*np.pi*r[ind]**4*z**2)
    H2p[ind] = term0 + term1 + term2 + term3 
    return H2p

def Brinkman3DGaussianStokesletsPressure(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0
    P = lim*np.ones(r.shape)
    P[ind] = -2*np.exp(-r[ind]**2/pdict['eps']**2)/(4*np.pi**(1.5)*pdict['eps']*r[ind]**3) + ss.erf(r[ind]/pdict['eps']) / (4*np.pi*r[ind]**3)
    return P

def Brinkman3DGaussianDipolesD1(r,pdict): 
    ind=np.nonzero(r>=pdict['thresh'])
    lim= (-2.0+pdict['sig']**2)/(3*(pdict['eps']**3)*(np.pi**1.5)) - np.exp((pdict['sig']**2)/4)*(pdict['alph']**3)*(1-ss.erf(pdict['sig']/2))/(6*np.pi)     
    D1 = lim*np.ones(r.shape);
    z = r[ind]*pdict['alph']
    term0 = -np.exp(-r[ind]**2/pdict['eps']**2)*(2*r[ind]**2 + pdict['eps']**2) / (2*np.pi**(1.5)*r[ind]**2*pdict['eps']**3)
    term1 =  np.exp((pdict['sig']**2)/4 - z)*(1+z+z**2)*(1+ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2))/(8*np.pi*r[ind]**3)
    term2 = -np.exp((pdict['sig']**2)/4 + z)*(1-z+z**2)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2))/(8*np.pi*r[ind]**3)
    D1[ind] = term0 + term1 +  term2
    return D1

def Brinkman3DGaussianDipolesD2(r,pdict): 
    ind=np.nonzero(r>=pdict['thresh'])
    lim= (-12.0+2*pdict['sig']**2 - pdict['sig']**4)/(30*(pdict['eps']**5)*(np.pi**1.5)) + np.exp((pdict['sig']**2)/4)*(pdict['alph']**5)*(1-ss.erf(pdict['sig']/2))/(60*np.pi)     
    D2 = lim*np.ones(r.shape)
    z = r[ind]*pdict['alph']
    term0 =  np.exp(-r[ind]**2/pdict['eps']**2)*(2*r[ind]**2 + 3*pdict['eps']**2) / (2*np.pi**(1.5)*r[ind]**4*pdict['eps']**3)
    term1 = -np.exp((pdict['sig']**2)/4 - z)*(3+3*z+z**2)*(1+ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2))/(8*np.pi*r[ind]**5) 
    term2 =  np.exp((pdict['sig']**2)/4 + z)*(3-3*z+z**2)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2))/(8*np.pi*r[ind]**5) 
    D2[ind] = term0 + term1 +  term2
    return D2

def Brinkman3DGaussianDipolesD1prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    D1p = lim*np.ones(r.shape)    
    term0 = (np.exp((pdict['sig']**2)/4 - z)*(3 + 3*z + 2*z**2 + z**3) / (4*np.pi*r[ind]**4) ) * (-1 + 0.5*(1-ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2)) )
    term1 = np.exp(-r[ind]**2/pdict['eps']**2)*(4*r[ind]**4 + 2*r[ind]**2*pdict['eps']**2 + (3 + z**2)*pdict['eps']**4)/(2*np.pi**(1.5)*r[ind]**3*pdict['eps']**5)
    term2 = -np.exp((pdict['sig']**2)/4 + z)*(-3 + 3*z - 2*z**2 + z**3)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2)) / (8*np.pi*r[ind]**4)
    D1p[ind] = term0 + term1 + term2 
    return D1p

def Brinkman3DGaussianDipolesD2prime(r,pdict):
    ind=np.nonzero(r>=pdict['thresh'])
    lim=0*pdict['alph']
    z = r[ind]*pdict['alph']
    D2p = lim*np.ones(r.shape)    
    term0 = (np.exp((pdict['sig']**2)/4 - z)*(15 + 15*z + 6*z**2 + z**3) / (4*np.pi*r[ind]**6) ) * (1 - 0.5*(1-ss.erf(r[ind]/pdict['eps'] - pdict['sig']/2)) )
    term1 = np.exp(-r[ind]**2/pdict['eps']**2)*(-4*r[ind]**4 - 10*r[ind]**2*pdict['eps']**2 - (15 + z**2)*pdict['eps']**4)/(2*np.pi**(1.5)*r[ind]**5*pdict['eps']**5)
    term2 = np.exp((pdict['sig']**2)/4 + z)*(-15 + 15*z - 6*z**2 + z**3)*(1-ss.erf(r[ind]/pdict['eps'] + pdict['sig']/2)) / (8*np.pi*r[ind]**6)
    D2p[ind] = term0 + term1 + term2 
    return D2p

def Brinkman3DGaussianDipolesPressure(r,pdict):
    #limit not required since there is no division by r
    P = -2*np.exp(-r**2/pdict['eps']**2) / (np.pi**(1.5)*pdict['eps']**5)
    return P

def Brinkman3DCompactStokesletsH1(r,pdict):
    #FIXME -- precision issues
    ind=np.intersect1d(np.nonzero(r>=pdict['thresh'])[0],np.nonzero(r<=pdict['eps'])[0])
    sig = pdict['sig']
    lim0 = ( 512.0*sig**11 + (-32691859200.0*sig -3911846400.0*sig**3 -61205760.0*sig**5)*np.exp(sig) ) / (1536.0*sig**10 *pdict['eps']*np.pi*(-1 + np.exp(2*sig)))
    lim1 = (16345929600.0 - 768398400.0*sig**2 +22453200.0*sig**4 -582120.0*sig**6 + 17325.0*sig**8)*(-1 + np.exp(2*sig)) / (1536.0*sig**10 *pdict['eps']*np.pi*(-1 + np.exp(2*sig)))
    lim = (lim0+lim1)  

#    denomfactor = (-1+np.exp(2*sig))*np.pi*pdict['eps']
#    lim0 = ( 512.0/1536.0)*sig/denomfactor 
#    lim1 = ( (-32691859200.0/1536.0)*sig -(3911846400.0/1536.0)*sig**3 -(61205760.0/1536.0)*sig**5)*np.exp(sig) ) / (1536.0*sig**10 *pdict['eps']*np.pi*(-1 + np.exp(2*sig)))
#    lim1 = (16345929600.0 - 768398400.0*sig**2 +22453200.0*sig**4 -582120.0*sig**6 + 17325.0*sig**8)*(-1 + np.exp(2*sig)) / (1536.0*sig**10 *pdict['eps']*np.pi*(-1 + np.exp(2*sig)))
#    lim = (lim0+lim1)  

#    lim0 = 10**( np.log10(pdict['alph']) - np.log10( (3*(-1+np.exp(2*sig))*np.pi ) ) )
#    lim1 = - 42567525 / (4*sig**10*denomfactor) 
#    lim2 =  4002075 / (8*sig**8*denomfactor) 
#    lim3 = - 467775 / (32*sig**6*denomfactor) 
#    lim35 = 24255 / (64*sig**4*denomfactor) 
#    lim4 = - 5775 / (512*sig**2*denomfactor)
#    lim5 = -42567525*np.exp(sig)/(2*sig**9*denomfactor) 
#    lim6 = -2546775*np.exp(sig)/(sig**7*denomfactor) 
#    lim7 = -79695*np.exp(sig)/(2*sig**5*denomfactor) 
#    lim8 = 42567525*np.exp(2*sig)/(4*sig**10*denomfactor) 
#    lim9 = -4002075*np.exp(2*sig)/(8*sig**8*denomfactor) 
#    lim10 = 467775*np.exp(2*sig)/(32*sig**6*denomfactor) 
#    lim11 = -24255*np.exp(2*sig)/(64*sig**4*denomfactor) 
#    lim12 = 5775*np.exp(2*sig)/(512*sig**2*denomfactor)
#    lim = lim0 + lim1+lim2+lim3+lim4+lim5+lim6+lim7+lim8+lim9+lim10+lim11+lim12+lim35
    H1 = lim*np.ones(r.shape)  
    z = r[ind]*pdict['alph']
    term0 = ( -np.exp(z)*(1-z+z**2) + np.exp(-z)*(1+z+z**2) ) * ( -2*sig**10 + (127702575 + 15280650*sig**2 + 239085*sig**4)*np.exp(sig) )  / ( 8*sig**10*pdict['alph']**2*np.pi*(-1 + np.exp(2*sig))*r[ind]**3 )
    term1 = ( 5448643200 - 4656960*(55*sig**2 -234*z**2) + 166320*(45*sig**4 - 308*sig**2*z**2 + 351*z**4) - 27720*(7*sig**6 - 54*sig**4*z**2 +99*sig**2*z**4 - 52*z**6) + 5775*sig**8 - 38808*sig**6*z**2 +80190*sig**4*z**4 -67760*sig**2*z**6 +20475*z**8) / (512*sig**10*pdict['eps']*np.pi)
    H1[ind] = term0 + term1
    jnd=np.nonzero(r>pdict['eps'])
    zo = r[jnd]*pdict['alph']
    H1[jnd] = (-1 + np.exp(-zo)*(1+zo+zo**2) )/ (4*np.pi*r[jnd]**3 * pdict['alph']**2)
    return H1

def Brinkman3DCompactStokesletsH2(r,pdict):
    #FIXME -- precision issues
    ind=np.intersect1d(np.nonzero(r>=pdict['thresh'])[0],np.nonzero(r<=pdict['eps'])[0])
    sig = pdict['sig']
    lim = ( -64*sig**10 + (4086482400 +488980800*sig**2 +7650720*sig**4)*np.exp(sig) ) / (1920*sig**7 *pdict['eps']**3*np.pi*(-1 + np.exp(2*sig))) + (-2043241200 + 96049800*sig**2 -2806650*sig**4 +72765*sig**6) / (1920*sig**8 *pdict['eps']**3 *np.pi)
    z = r[ind]*pdict['alph']
    H2 = lim*np.ones(r.shape) 
    term0 =( np.exp(z)*(3-3*z+z**2) - np.exp(-z)*(3+3*z+z**2) ) * ( -2*sig**10 + (127702575 + 15280650*sig**2 + 239085*sig**4)*np.exp(sig) )  / ( 8*sig**10*pdict['alph']**2*np.pi*(-1 + np.exp(2*sig))*r[ind]**5 )
    term1 = 3*( -45405360 +27720*(77*sig**2 -117*z**2) -6930*(9*sig**4 - 22*sig**2*z**2 + 13*z**4) + 1617*sig**6 - 4455*sig**4*z**2 +4235*sig**2*z**4 - 1365*z**6 ) / (128*sig**8*pdict['eps']**3*np.pi)
    H2[ind] = term0 + term1
    jnd=np.nonzero(r>pdict['eps'])
    zo = r[jnd]*pdict['alph']
    H2[jnd] = (3 - np.exp(-zo)*(3+3*zo+zo**2) )/ (4*np.pi*r[jnd]**5 * pdict['alph']**2)
    return H2

def Brinkman3DCompactDipolesD1(r,pdict):
    #FIXME -- precision issues
    ind=np.intersect1d(np.nonzero(r>=pdict['thresh'])[0],np.nonzero(r<=pdict['eps'])[0])
    sig = pdict['sig']
    lim = ( -64*sig**10 + (4086482400 +488980800*sig**2 +7650720*sig**4)*np.exp(sig) ) / (-192*sig**7 *pdict['eps']**3*np.pi*(-1 + np.exp(2*sig))) + (-2043241200 + 96049800*sig**2 -2806650*sig**4 +72765*sig**6) / (-192*sig**8 *pdict['eps']**3 *np.pi)
    z = r[ind]*pdict['alph']
    D1 = lim*np.ones(r.shape)  
    term0 = ( -np.exp(z)*(1-z+z**2) + np.exp(-z)*(1+z+z**2) ) * ( -2*sig**10 + (127702575 + 15280650*sig**2 + 239085*sig**4)*np.exp(sig) )  / ( 8*sig**10*np.pi*(-1 + np.exp(2*sig))*r[ind]**3 )
    term1 = -3465*( -196560 + 168*(55*sig**2 -234*z**2) -6*(45*sig**4 - 308*sig**2*z**2 + 351*z**4) + 7*sig**6 - 54*sig**4*z**2 +99*sig**2*z**4 - 52*z**6 ) / (64*sig**8*pdict['eps']**3*np.pi)
    D1[ind] = term0 + term1
    jnd=np.nonzero(r>pdict['eps'])
    zo = r[jnd]*pdict['alph']
    D1[jnd] = np.exp(-zo)*(1+zo+zo**2) / (4*np.pi*r[jnd]**3)
    return D1

def Brinkman3DCompactDipolesD2(r,pdict):
    #FIXME -- precision issues
    ind=np.intersect1d(np.nonzero(r>=pdict['thresh'])[0],np.nonzero(r<=pdict['eps'])[0])
    sig = pdict['sig']
    lim = ( -32*sig**10 + (2043241200 +244490400*sig**2 +3825360*sig**4)*np.exp(sig) ) / (960*sig**5 *pdict['eps']**5*np.pi*(-1 + np.exp(2*sig))) + (-1021620600 + 48024900*sig**2 -1403325*sig**4) / (960*sig**6 *pdict['eps']**5 *np.pi)
    z = r[ind]*pdict['alph']
    D2 = lim*np.ones(r.shape) 
    term0 =( np.exp(z)*(3-3*z+z**2) - np.exp(-z)*(3+3*z+z**2) ) * ( -2*sig**10 + (127702575 + 15280650*sig**2 + 239085*sig**4)*np.exp(sig) )  / ( 8*sig**10*np.pi*(-1 + np.exp(2*sig))*r[ind]**5 )
    term1 = -10395*( 6552 -308*sig**2 + 468*z**2 + 9*sig**4 -22*sig**2*z**2 + 13*z**4 ) / ( 64*sig**6*pdict['eps']**5 * np.pi )
    D2[ind] = term0 + term1
    jnd=np.nonzero(r>pdict['eps'])
    zo = r[jnd]*pdict['alph']
    D2[jnd] = -np.exp(-zo)*(3+3*zo+zo**2) / (4*np.pi*r[jnd]**5)
    return D2

def Stokeslet2DCubicH1(r,pdict):
    H1 = ( 2*pdict['eps']**2/(r**2+ pdict['eps']**2) - np.log(r**2+ pdict['eps']**2) ) / (8*np.pi)
    return H1

def Stokeslet2DCubicH2(r,pdict):
    H2 = 1.0 / ( 4*np.pi*(r**2+ pdict['eps']**2) )
    return H2

def Stokeslet2DCubicPressure(r,pdict):
    P = (r**2 + 2*pdict['eps']**2)/(2*np.pi*(r**2 + pdict['eps']**2)**2)
    return P

def Dipole2DSquareD1(r,pdict):
    D1 = (r**2 - pdict['eps']**2) / (2*np.pi*(r**2 + pdict['eps']**2)**2)
    return D1

def Dipole2DSquareD2(r,pdict):
    D2 = -1.0/(np.pi*(r**2 + pdict['eps']**2)**2)
    return D2

def Dipole2DSquarePressure(r,pdict):
    P = (-4*pdict['eps']**2)/(np.pi*(r**2 + pdict['eps']**2)**3)
    return P
