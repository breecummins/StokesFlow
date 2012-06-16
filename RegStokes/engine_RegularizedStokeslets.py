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
import lib_RegularizedStokeslets as lRS

class RegularizedStokeslets(object):
    '''
    Superclass for regularized nD Stokeslets, Brinkmanlets, oscillatory Stokeslets, 
    dipoles, and summations of Stokeslets/Brinkmanlets and dipoles.  
    
    FOR CALCULATING VELOCITY:
    Subclasses define methods _H1func, _H2func (for Stokeslets, Brinkmanlets) and/or 
    _D1func, _D2func (for dipoles) that are set to zero in the superclass. They are 
    used to calculate velocity via:
    
    u = f*(H1 + cD1) + (f dot x)x*(H2 + cD2)
    
    When summing Stokeslets/Brinkmanlets and dipoles, the subclass should define all
    of _H*func and _D*func, and should be inherited by subclasses that use
    different boundary conditions to determine the relative strength of the dipoles
    through the attribute BCcst (c in the above equation), which is set to zero in 
    the superclass. 
    
    FOR CALCULATING VELOCITY GRADIENTS:
    The derivatives of the H and D functions _H1prime, _H2prime, _D1prime, and 
    _D2prime need to be defined to calculate velocity gradients:
    
    grad(u_j) = f_j*(x/r)*(H1' + cD1') + x_j*(f/r)*(H2 + cD2) 
                + (f dot x)*(x_j*(x/r)*(H2' + cD2') + e_j*(H2 + cD2))
                
    By default, the derivatives are set to zero in the superclass. 
    
    FOR CALCULATING PRESSURE:
    Subclasses define methods _HPfunc (for Stokeslets, Brinkmanlets) and/or 
    _DPfunc (for dipoles) that are set to zero in the superclass. They are used 
    in conjunction with BCcst (c in the equation below) to calculate pressure via:
    
    p = (f dot x)*(HP + cDP)
    
    '''
    def __init__(self,eps,mu=1.0,alph=0.0):
        ''' 
        Set the blob parameter eps, the dynamic fluid viscosity mu, where mu = 1.0 for nondim'l problems,
        and the Brinkman parameter alph: alph = 0 for Stokes flow, alph > 0 for porous medium flow, and
        alph complex for oscillatory flow.
        
        '''
        self.pdict = {}
        self.pdict['eps'] = eps
        self.pdict['thresh']=eps/100  #Threshold for using limits in the H and D functions.
        self.pdict['alph'] = alph
        self.pdict['sig'] = alph*eps
        self.mu = mu
        self.BCcst = 0 #Constant for combined Stokeslet + dipole methods 
        if np.iscomplex(alph):
#            print('Alpha is complex. Modeling oscillating flow...')
            self.dtype = np.complex128
        else:
            self.dtype = np.float64
#            if alph > 0:
#                print('Alpha is nonzero. Modeling Brinkman flow...')
        
    def _H1func(self,r):
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _H2func(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)

    def _H1prime(self,r):
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _H2prime(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)

    def _HPfunc(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _D1func(self,r):
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _D2func(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _D1prime(self,r):
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _D2prime(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _DPfunc(self,r):    
        '''
        Stub to be defined by subclass
        '''
        return np.zeros(r.shape,dtype=self.dtype)
    
    def _makeMatrixRows(self,pt,nodes):
        '''
        Build the velocity rows for the original Riemann sum method. 'pt' (1 x dim) is the point 
        where we want to know velocity and 'nodes' are the locations of the forces (N x dim).
        The output is a dim x N*dim array containing the rows of matrix M affecting the velocity 
        at pt.  
                
        '''
        dif = pt - nodes
        r = np.sqrt((dif**2).sum(1))
        #get H1, H2 function values (these should not depend on mu)
        H1 = self._H1func(r) + self.BCcst*self._D1func(r)
        H2 = self._H2func(r) + self.BCcst*self._D2func(r)
        #create matrix rows for point pt
        N = nodes.shape[0]
        rows = np.zeros((self.dim,self.dim*N),dtype=self.dtype)
        ind = self.dim*np.arange(N) 
        for j in range(self.dim):            
            for k in range(self.dim):
                if j == k:
                    rows[j,ind+k] = H1 + (dif[:,j]**2)*H2
                elif j < k:
                    rows[j,ind+k] = dif[:,j]*dif[:,k]*H2
                elif j > k:
                    rows[j,ind+k] = rows[k,ind+j]
        return rows/self.mu
    
    def _makeMatrix(self,obspts,nodes):
        '''
        Build the regularized Stokeslet matrix explicitly. Used for finding 
        the forces located at 'nodes' that will ensure known velocities at 'obspts'.
        'obspts' is M x dim and 'nodes' is N x dim. 
        The output matrix is M*dim x N*dim.
                
        '''
        mat = np.zeros((self.dim*obspts.shape[0],self.dim*nodes.shape[0]),dtype=self.dtype)
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            rows = self._makeMatrixRows(pt,nodes)
            for j in range(self.dim):
                mat[self.dim*k+j,:]=rows[j,:]
        return mat
        
    def calcForces(self,obspts,nodes,v):
        '''
        Enforce an approximate boundary condition of v at obspts due to 
        forces at nodes. 'v', 'obspts', and 'nodes' are all M x dim. 
        'obspts' and 'nodes' must be the same size or this method will fail.
        This method will also fail if the Stokselet matrix is singular. 
        The output is an M x dim array of forces occurring at nodes that
        enforce v.
        
        '''
        M = self._makeMatrix(obspts,nodes)
        f = np.linalg.solve(M,v.flat)
        return f.reshape(nodes.shape)

    def calcVel(self,obspts,nodes,f):
        '''
        Calculates velocity at 'obspts' due to forces 'f' located at 'nodes'. 
        The operation is simply a matrix multiplication in which the full 
        matrix is not built. 'obspts' is M x dim and 'nodes' and 'f' are N x dim. 
        The output velocity array is M x dim.        
        '''
        vel = np.zeros((obspts.shape[0],self.dim),dtype=self.dtype)
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            rows = self._makeMatrixRows(pt,nodes)
            for j in range(self.dim):
                vel[k,j] = (rows[j,:]*f.flat).sum()
        return vel
    
    def _makeVelGradRows(self,pt,nodes,i,j):
        '''
        Build the du_i/dx_j row for the original Riemann sum method. 'pt' (1 x dim) is the point 
        where we want to know the derivative and 'nodes' are the locations of the forces (N x dim).
        i and j are the indices representing the derivative du_i/dx_j.
        The output is an array of length dim*N -- the row affecting the gradient at pt.   
                        
        '''
        dif = pt - nodes
        r = np.sqrt((dif**2).sum(1))
        H1p = self._H1prime(r) + self.BCcst*self._D1prime(r)
        H2 = self._H2func(r) + self.BCcst*self._D2func(r)
        H2p = self._H2prime(r) + self.BCcst*self._D2prime(r)
        N = nodes.shape[0]
        ind = self.dim*np.arange(N) 
        row = np.zeros((self.dim*N,),dtype=self.dtype)
        for k in range(self.dim):
            row[ind+k] = (i==k)*dif[:,j]*H1p/r + ((j==k)*dif[:,i] + (i==j)*dif[:,k])*H2 + dif[:,i]*dif[:,j]*dif[:,k]*H2p/r
        return row
        
    def calcVelGradient(self,obspts,nodes,f):
        '''
        Calculates du_j/dx_i at 'obspts' due to forces 'f' located at 'nodes'.
        The operation is simply a matrix multiplication in which the full 
        matrix is not built. 'obspts' is M x dim and 'nodes' and 'f' are N x dim. 
        The output gradient array dU is M x dim x dim. So for example in 2D,
        du/dx = dU[:,0,0], du/dy = dU[:,0,1], dv/dx = dU[:,1,0], dv/dy = dU[:,1,1]. 
                 
        '''
        dU = np.zeros((obspts.shape[0],self.dim,self.dim),dtype=self.dtype)
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            for i in range(self.dim):            
                for j in range(self.dim):
                    row = self._makeVelGradRows(pt,nodes,i,j)
                    dU[k,i,j] = (row*f.flat).sum()
        return dU
    
    def calcStressTensor(self,obspts,nodes,f):
        '''
        Calculates the stress tensor sigma = -p*I + mu*(grad u + (grad u)^T)
        at each point in 'obspts' due to forces 'f' located at 'nodes'. The operation 
        is a matrix multiplication in which the full matrix is not built. 'obspts' is 
        M x dim and 'nodes' and 'f' are N x dim. The output stress tensor is 
        M x dim x dim, with sigma at point k stored as S[k,:,:].
        
        '''
        S = np.zeros((obspts.shape[0],self.dim,self.dim),dtype=self.dtype)
        dU = self.calcVelGradient(obspts,nodes,f)
        p = self.calcPressure(obspts,nodes,f)
        for i in range(self.dim):            
            for j in range(self.dim):
                S[:,i,j] = -p*(i==j) + self.mu*(dU[:,i,j]+dU[:,j,i])
        return S
            
    def _makePressureRows(self,pt,nodes):
        '''
        Build the pressure row for the original Riemann sum method. 'pt' (1 x dim) is the point 
        where we want to know the pressure and 'nodes' are the locations of the forces (N x dim).
        The output is an array of length dim*N -- the row affecting the pressure at pt.  
                
        '''
        dif = pt - nodes
        r = np.sqrt((dif**2).sum(1))
        pf = self._HPfunc(r) + self.BCcst*self._DPfunc(r)
        N = nodes.shape[0]
        P = np.zeros((self.dim*N,),dtype=self.dtype)
        ind = self.dim*np.arange(N) 
        for j in range(self.dim):            
            P[ind+j] = pf 
        row = P*dif.flat
        return row
         
    def calcPressure(self,obspts,nodes,f):
        '''
        Calculates pressure at 'obspts' due to forces 'f' located at 'nodes'. 
        The operation is a matrix multiplication in which the full 
        matrix is not built. 'obspts' is M x dim and 'nodes' and 'f' are N x dim. 
        The output pressure array is of length M.        
        '''
        p = np.zeros((obspts.shape[0],),dtype=self.dtype)
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            row = self._makePressureRows(pt,nodes)
            p[k] = (row*f.flat).sum()
        return p
    

class Brinkman3DNegExpStokeslets(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlets with the negative exponential blob.
    
    '''
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
    
    def _H1func(self,r):
        H1 = lRS.Brinkman3DNegExpStokesletsH1(r,self.pdict)
        return H1
           
    def _H2func(self,r):
        H2 = lRS.Brinkman3DNegExpStokesletsH2(r,self.pdict)
        return H2

    def _H1prime(self,r):
        H1p = lRS.Brinkman3DNegExpStokesletsH1prime(r,self.pdict)
        return H1p
           
    def _H2prime(self,r):
        H2p = lRS.Brinkman3DNegExpStokesletsH2prime(r,self.pdict)
        return H2p

    def _HPfunc(self,r):
        P = lRS.Brinkman3DNegExpStokesletsPressure(r,self.pdict)
        return P
  
    
class Brinkman3DNegExpDipoles(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlet dipoles with the negative exponential blob.
    
    '''
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
        self.BCcst = 1.0
    
    def _D1func(self,r):
        D1 = lRS.Brinkman3DNegExpDipolesD1(r,self.pdict)
        return D1
           
    def _D2func(self,r):
        D2 = lRS.Brinkman3DNegExpDipolesD2(r,self.pdict)
        return D2
 
    def _D1prime(self,r):
        D1p = lRS.Brinkman3DNegExpDipolesD1prime(r,self.pdict)
        return D1p
           
    def _D2prime(self,r):
        D2p = lRS.Brinkman3DNegExpDipolesD2prime(r,self.pdict)
        return D2p
 
    def _DPfunc(self,r):
        P = lRS.Brinkman3DNegExpDipolesPressure(r,self.pdict)
        return P

    
class Brinkman3DNegExpStokesletsAndDipoles(RegularizedStokeslets):
    '''
    Superclass for summing 3D regularized Brinkmanlets and dipoles with the 
    negative exponential blob.
    **Boundary conditions are required and set by a subclass.** 
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
            
    def _H1func(self,r):
        H1 = lRS.Brinkman3DNegExpStokesletsH1(r,self.pdict)
        return H1
        
    def _H2func(self,r):
        H2 = lRS.Brinkman3DNegExpStokesletsH2(r,self.pdict)
        return H2
        
    def _H1prime(self,r):
        H1p = lRS.Brinkman3DNegExpStokesletsH1prime(r,self.pdict)
        return H1p
           
    def _H2prime(self,r):
        H2p = lRS.Brinkman3DNegExpStokesletsH2prime(r,self.pdict)
        return H2p

    def _HPfunc(self,r):
        P = lRS.Brinkman3DNegExpStokesletsPressure(r,self.pdict)
        return P
  
    def _D1func(self,r): 
        D1 = lRS.Brinkman3DNegExpDipolesD1(r,self.pdict)
        return D1
    
    def _D2func(self,r): 
        D2 = lRS.Brinkman3DNegExpDipolesD2(r,self.pdict)
        return D2

    def _D1prime(self,r):
        D1p = lRS.Brinkman3DNegExpDipolesD1prime(r,self.pdict)
        return D1p
           
    def _D2prime(self,r):
        D2p = lRS.Brinkman3DNegExpDipolesD2prime(r,self.pdict)
        return D2p
 
    def _DPfunc(self,r):
        P = lRS.Brinkman3DNegExpDipolesPressure(r,self.pdict)
        return P


class Brinkman3DNegExpStokesletsAndDipolesSphericalBCs(Brinkman3DNegExpStokesletsAndDipoles):
    '''
    Summation of 3D regularized Brinkmanlets and dipoles with the negative exponential blob.
    BCs are steady oscillations of a sphere of radius sphererad.
    Assigned constants:
    self.BCcst is the relative dipole strength for spherical boundary conditions.
    self.fcst is the coefficient that multiplies the boundary condition velocity
    to get the Stokeslet force: f = self.fcst*v, where v is a three element vector
    representing the velocity at the sphere surface.
    
    '''
    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j),sphererad=0):
        Brinkman3DNegExpStokesletsAndDipoles.__init__(self, eps, mu, alph)
        self.dim = 3
        if sphererad <= 0:
            raise ValueError('Need a positive value for the radius of the sphere.')
        else:
#            print(self.pdict)
#            print("Value of D2:")
#            print(self._D2func(np.array([sphererad])))
            cst1 = -self._H2func(np.array([sphererad])) / self._D2func(np.array([sphererad]))
            self.BCcst = cst1[0]
            cst2 = self._H1func(np.array([sphererad])) + self.BCcst*self._D1func(np.array([sphererad]))
            self.fcst = self.mu/cst2[0]
 

class Brinkman3DGaussianStokeslets(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlets with the Gaussian blob.
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3

    def _H1func(self,r):
        H1 = lRS.Brinkman3DGaussianStokesletsH1(r,self.pdict)
        return H1
           
    def _H2func(self,r):
        H2 = lRS.Brinkman3DGaussianStokesletsH2(r,self.pdict)
        return H2
    
    def _H1prime(self,r):
        H1p = lRS.Brinkman3DGaussianStokesletsH1prime(r,self.pdict)
        return H1p
           
    def _H2prime(self,r):
        H2p = lRS.Brinkman3DGaussianStokesletsH2prime(r,self.pdict)
        return H2p
    
    def _HPfunc(self,r):
        p = lRS.Brinkman3DGaussianStokesletsPressure(r,self.pdict)
        return p
        

class Brinkman3DGaussianDipoles(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlet dipoles with the Gaussian blob.
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
        self.BCcst = 1.0

    def _D1func(self,r): 
        D1 = lRS.Brinkman3DGaussianDipolesD1(r,self.pdict)
        return D1
    
    def _D2func(self,r): 
        D2 = lRS.Brinkman3DGaussianDipolesD2(r,self.pdict)
        return D2

    def _D1prime(self,r):
        D1p = lRS.Brinkman3DGaussianDipolesD1prime(r,self.pdict)
        return D1p

    def _D2prime(self,r):
        D2p = lRS.Brinkman3DGaussianDipolesD2prime(r,self.pdict)
        return D2p

    def _DPfunc(self,r): 
        P = lRS.Brinkman3DGaussianDipolesPressure(r,self.pdict)
        return P


class Brinkman3DGaussianStokesletsAndDipoles(RegularizedStokeslets):
    '''
    Superclass for summing 3D regularized Brinkmanlets and dipoles with the Gaussian blob.
    **Boundary conditions are required and set by a subclass.** 
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
            
    def _H1func(self,r):
        H1 = lRS.Brinkman3DGaussianStokesletsH1(r,self.pdict)
        return H1
        
    def _H2func(self,r):
        H2 = lRS.Brinkman3DGaussianStokesletsH2(r,self.pdict)
        return H2
        
    def _H1prime(self,r):
        H1p = lRS.Brinkman3DGaussianStokesletsH1prime(r,self.pdict)
        return H1p
           
    def _H2prime(self,r):
        H2p = lRS.Brinkman3DGaussianStokesletsH2prime(r,self.pdict)
        return H2p
    
    def _HPfunc(self,r):
        p = lRS.Brinkman3DGaussianStokesletsPressure(r,self.pdict)
        return p
        
    def _D1func(self,r): 
        D1 = lRS.Brinkman3DGaussianDipolesD1(r,self.pdict)
        return D1
    
    def _D2func(self,r): 
        D2 = lRS.Brinkman3DGaussianDipolesD2(r,self.pdict)
        return D2

    def _D1prime(self,r):
        D1p = lRS.Brinkman3DGaussianDipolesD1prime(r,self.pdict)
        return D1p

    def _D2prime(self,r):
        D2p = lRS.Brinkman3DGaussianDipolesD2prime(r,self.pdict)
        return D2p

    def _DPfunc(self,r): 
        P = lRS.Brinkman3DGaussianDipolesPressure(r,self.pdict)
        return P


class Brinkman3DGaussianStokesletsAndDipolesSphericalBCs(Brinkman3DGaussianStokesletsAndDipoles):
    '''
    Summation of 3D regularized Brinkmanlets and dipoles with the Gaussian blob.
    BCs are steady oscillations of a sphere of radius sphererad.
    Assigned constants:
    self.BCcst is the relative dipole strength for spherical boundary conditions.
    self.fcst is the coefficient that multiplies the boundary condition velocity
    to get the Stokeslet force: f = self.fcst*v, where v is a three element vector
    representing the velocity at the sphere surface.
    
    '''
    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j),sphererad=0):
        Brinkman3DGaussianStokesletsAndDipoles.__init__(self, eps, mu, alph)
        self.dim = 3
        if sphererad <= 0:
            raise ValueError('Need a positive value for the radius of the sphere.')
        else:
#            print(self.pdict)
#            print("Value of D2:")
#            print(self._D2func(np.array([sphererad])))
            cst1 = -self._H2func(np.array([sphererad])) / self._D2func(np.array([sphererad]))
            self.BCcst = cst1[0]
            cst2 = self._H1func(np.array([sphererad])) + self.BCcst*self._D1func(np.array([sphererad]))
            self.fcst = self.mu/cst2[0]
 
            
class Brinkman3DGaussianStokesletsAndDipolesCylinderBCs(Brinkman3DGaussianStokesletsAndDipoles):
    '''
    Summation of 3D regularized Brinkmanlets and dipoles with the Gaussian blob.
    BCs are steady oscillations of a cylinder of radius cylrad.
    
    **These BCs don't operate as desired.**
    
    '''
    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j),cylrad=0):
        Brinkman3DGaussianStokesletsAndDipoles.__init__(self, eps, mu, alph)
        self.dim = 3
        if cylrad <= 0:
            raise ValueError('Need a positive value for the radius of the cylinder.')
        else:
            a = cylrad
            # constant arising from s^2 truncation
            term1 = (2*a**2 - 5*eps**2)/20.0
            term2 = 8*a**5 + 20*a**3*eps**2 + 30*a*eps**4 - 15*np.sqrt(np.pi)*eps**5*np.exp(a**2/eps**2)*ss.erf(a/eps)
            self.BCcst = term1 - (4./5.)*a**7/term2
#            # constant arising from s^4 truncation
#            term1 = (2*a**2 - 7*eps**2)/28.0
#            term2 = 16*a**7 + 56*a**5*eps**2 + 140*a**3*eps**4 + 210*a*eps**6 - 105*np.sqrt(np.pi)*eps**7*np.exp(a**2/eps**2)*ss.erf(a/eps)
#            self.BCcst = term1 - (8./7.)*a**9/term2
                
class Brinkman3DCompactStokeslets(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlets with a compact blob (smooth, second moment zero).
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3

    def _H1func(self,r):
        H1 = lRS.Brinkman3DGaussianStokesletsH1(r,self.pdict)
        return H1
           
    def _H2func(self,r):
        H2 = lRS.Brinkman3DGaussianStokesletsH2(r,self.pdict)
        return H2
            

class Brinkman3DCompactDipoles(RegularizedStokeslets):
    '''
    3D regularized Brinkmanlets with a compact blob (smooth, second moment zero).
    
    '''    
    def __init__(self,eps,mu=1.0,alph=100*np.sqrt(1j)):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 3
        self.BCcst = 1.0

    def _D1func(self,r): 
        D1 = lRS.Brinkman3DCompactDipolesD1(r,self.pdict)
        return D1
    
    def _D2func(self,r): 
        D2 = lRS.Brinkman3DCompactDipolesD2(r,self.pdict)
        return D2


class Stokeslet2DCubic(RegularizedStokeslets):
    '''
    2D regularized Stokeslets with the blob that has 
    a cube in the denominator.
    
    '''    
    def __init__(self,eps,mu=1.0,alph=0.0):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 2
        print('Derivatives of H functions are not implemented. Surface traction will be wrong.')

    def _H1func(self,r):
        H1 = lRS.Stokeslet2DCubicH1(r,self.pdict)
        return H1
        
    def _H2func(self,r):
        H2 = lRS.Stokeslet2DCubicH2(r,self.pdict)
        return H2
    
    def _HPfunc(self,r):  
        P = lRS.Stokeslet2DCubicPressure(r,self.pdict)  
        return P


class Dipole2DSquare(RegularizedStokeslets):
    '''
    2D regularized dipoles with the blob that has a square 
    in the denominator.
    
    '''    
    def __init__(self,eps,mu=1.0,alph=0.0):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 2
        self.BCcst = 1.0
        print('Derivatives of D functions are not implemented. Surface traction will be wrong.')

    def _D1func(self,r):
        D1 = lRS.Dipole2DSquareD1(r,self.pdict)
        return D1
        
    def _D2func(self,r):
        D2 = lRS.Dipole2DSquareD2(r,self.pdict)
        return D2
 
    def _DPfunc(self,r):  
        P = lRS.Dipole2DSquarePressure(r,self.pdict)  
        return P
 
    
class StokesletsAndDipoles2DMixedSquareCubic(RegularizedStokeslets):
    '''
    Superclass for summing 2D regularized Stokeslets (cubic blob) and 
    dipoles (square blob).
 
    **Boundary conditions are required and set by a subclass.** 
    
    '''    
    def __init__(self,eps,mu=1.0,alph=0.0):
        RegularizedStokeslets.__init__(self, eps, mu, alph)
        self.dim = 2
        print('Derivatives of H and D functions are not implemented. Surface traction will be wrong.')

    def _H1func(self,r):
        H1 = lRS.Stokeslet2DCubicH1(r,self.pdict)
        return H1
        
    def _H2func(self,r):
        H2 = lRS.Stokeslet2DCubicH2(r,self.pdict)
        return H2

    def _HPfunc(self,r):  
        P = lRS.Stokeslet2DCubicPressure(r,self.pdict)  
        return P

    def _D1func(self,r):
        D1 = lRS.Dipole2DSquareD1(r,self.pdict)
        return D1
        
    def _D2func(self,r):
        D2 = lRS.Dipole2DSquareD2(r,self.pdict)
        return D2

    def _DPfunc(self,r):  
        P = lRS.Dipole2DSquarePressure(r,self.pdict)  
        return P
 

class StokesletsAndDipoles2DMixedSquareCubicCircleBCs(StokesletsAndDipoles2DMixedSquareCubic):
    '''
    Summation of 2D regularized Stokeslets (cubic blob) and 
    dipoles (square blob).
    BCs are constant velocity over a circle of radius circrad.
    Assigned constants:
    self.BCcst is the relative dipole strength for circular boundary conditions.
    self.fcst is the coefficient that multiplies the boundary condition velocity
    to get the Stokeslet force: f = self.fcst*v, where v is a two element vector
    representing the velocity at the circle surface.
        
    '''
    
    def __init__(self,eps,mu=1.0,alph=0.0,circrad=0):
        StokesletsAndDipoles2DMixedSquareCubic.__init__(self, eps, mu, alph)
        self.dim = 2
        if circrad <= 0:
            raise ValueError('Need a positive value for the radius of the circle.')
        else:
            self.circrad = circrad
            cst1 = -self._H2func(np.array([circrad])) / self._D2func(np.array([circrad]))
            self.BCcst = cst1[0]
            cst2 = self._H1func(np.array([circrad])) + self.BCcst*self._D1func(np.array([circrad]))
            self.fcst = self.mu/cst2[0]

       
if __name__ == '__main__':
    pass
































