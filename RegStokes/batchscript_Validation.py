#Created by Breschine Cummins on June 16, 2012.

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

# import python modules
import numpy as np
import matplotlib.pyplot as plt
#import streamplot as sp
import matplotlib.cm as cm
import cPickle
import os

# import my home-rolled modules
import lib_ExactSolns as lES
import engine_RegularizedStokeslets as eRS
import viz_RegularizedStokeslets as vRS

#Set params, forces, and point locations
mu=1.0
alph = np.sqrt(7*1j)
Nnodes = 7
nodes = np.zeros((Nnodes,3))
zh = 0.001
nodes[:,2] = np.arange(-np.floor(Nnodes/2)*zh,np.floor(Nnodes/2)*zh+zh/2,zh)
f = np.zeros(nodes.shape)
f[:,0] = 1.0
Nobs = 200
obspts = np.zeros((Nobs,3))
obspts[:,0] = np.linspace(zh/5,Nnodes*zh*0.75,Nobs)

#Exact solution
exactvel = lES.BrinkmanletsExact(obspts,nodes,f,alph,mu)

#Regularized solutions
eps = zh/2.
rbg = eRS.Brinkman3DGaussianStokeslets(eps,mu,alph)
regvelg = rbg.calcVel(obspts,nodes,f)
eps2 = (np.sqrt(np.pi)/56)**(1./3.) * eps #get the same limit as r->0
rbn = eRS.Brinkman3DNegExpStokeslets(eps2,mu,alph)
regveln = rbn.calcVel(obspts,nodes,f)
uvel = np.column_stack([exactvel[:,0],regvelg[:,0],regveln[:,0]])
vvel = np.column_stack([exactvel[:,1],regvelg[:,1],regveln[:,1]])
wvel = np.column_stack([exactvel[:,2],regvelg[:,2],regveln[:,2]])
vRS.plainPlots(obspts[:,0],np.abs(uvel),'Magnitude','x','speed',['Exact |u|','Gaussian |u|','Neg exp |u|'],os.path.expanduser('~/scratch/validation_umag.pdf'))
vRS.plainPlots(obspts[:,0],np.angle(uvel),'Phase','x','phase (rad)',['Exact ang(u)','Gaussian ang(u)','Neg exp ang(u)'],os.path.expanduser('~/scratch/validation_uang.pdf'))
vRS.plainPlots(obspts[:,0],np.abs(vvel),'Magnitude','x','speed',['Exact |v|','Gaussian |v|','Neg exp |v|'],os.path.expanduser('~/scratch/validation_vmag.pdf'))
vRS.plainPlots(obspts[:,0],np.angle(vvel),'Phase','x','phase (rad)',['Exact ang(v)','Gaussian ang(v)','Neg exp ang(v)'],os.path.expanduser('~/scratch/validation_vang.pdf'))
vRS.plainPlots(obspts[:,0],np.abs(wvel),'Magnitude','x','speed',['Exact |w|','Gaussian |w|','Neg exp |w|'],os.path.expanduser('~/scratch/validation_wmag.pdf'))
vRS.plainPlots(obspts[:,0],np.angle(wvel),'Phase','x','phase (rad)',['Exact ang(w)','Gaussian ang(w)','Neg exp ang(w)'],os.path.expanduser('~/scratch/validation_wang.pdf'))

#compare blobs
def gaussianblob(r,eps):
    return np.exp(-r**2/eps**2) / (np.pi**(3./2.)*eps**3)

def negexpblob(r,eps):
    return np.exp(-r/eps)*(r + 2*eps)**2 / (224*np.pi*eps**5)

def compactblob(r,eps):
    ind = np.nonzero(r<=eps)
    b = np.zeros(r.shape)
    b[ind] = 17325./(1024*np.pi)*(1 - 13/5*r[ind]**2/eps**2)*(1 - r[ind]**2/eps**2)**3/eps**3  
    return b

eps3 = (17325*np.sqrt(np.pi)/1024)**(1./3.) * eps #get the same limit as r->0
obspts2 = np.zeros((Nobs,3))
obspts2[:,0] = np.linspace(0,Nnodes*zh/2,Nobs)
gblob = gaussianblob(obspts2[:,0],eps)
nblob = negexpblob(obspts2[:,0],eps2)
cblob = compactblob(obspts2[:,0],eps3)
blobs = np.column_stack([gblob,nblob,cblob])
vRS.plainPlots(obspts2[:,0],blobs,'Compare blobs','r','blob val',['Gaussian','Neg exp','Compact'],os.path.expanduser('~/scratch/validation_blobsgnc.pdf'))

