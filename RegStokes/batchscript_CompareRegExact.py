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
import lib_ExactSolns as ES
import engine_RegularizedStokeslets as RS
import viz_RegularizedStokeslets as vRS

def myfile(basedir,basename):
    try:
        os.mkdir(basedir+basename)
    except:
        pass
    F = open(basedir+basename+'.pickle', 'r')
    mydict = cPickle.Unpickler(F).load()
    F.close()
    return mydict

def makeGridCenter(size,h,origin=(0,0)):
    '''
    Constructs a uniformly-spaced, rectangular cell-centered grid in 2 dimensions.
    size is a tuple of 2 numbers giving the number of points in each direction. 
    h is the uniform point separation in all directions. Optional argument origin 
    gives the lower left corner of the domain. The output is N by 2, where N is the
    product of size.
    
    '''
    X = np.mgrid[(origin[0]+h/2.):(origin[0]+h*size[0]):h,(origin[1]+h/2.):(origin[1]+h*size[1]):h]
    x = X[0,:,:].flatten()
    y = X[1,:,:].flatten()
    l0 = np.zeros((2,len(x)))
    l0[0,:] = x
    l0[1,:] = y
    return l0.transpose()

def regSolnChainofSpheresGaussian(nodes,eps,mu,alph,circrad,vh):
    rb = RS.Brinkman3DGaussianStokesletsAndDipolesSphericalBCs(eps,mu,alph,circrad)
    v = np.zeros(nodes.shape,dtype='complex128')
    #scale the velocity at the center to get the right boundary condition
    v[:,0] = rb.fcst*(vh/mu)*( rb._H1func(np.array([0])) + rb.BCcst*rb._D1func(np.array([0])) )
#    print("velocity at center")
#    print(v)
    f = rb.calcForces(nodes,nodes,v)
    return rb, f

def regSolnChainofSpheresNegExp(nodes,eps,mu,alph,circrad,vh):
    rb = RS.Brinkman3DNegExpStokesletsAndDipolesSphericalBCs(eps,mu,alph,circrad)
    v = np.zeros(nodes.shape,dtype='complex128')
    #scale the velocity at the center to get the right boundary condition
    v[:,0] = rb.fcst*(vh/mu)*( rb._H1func(np.array([0])) + rb.BCcst*rb._D1func(np.array([0])) )
#    print("velocity at center")
#    print(v)
    f = rb.calcForces(nodes,nodes,v)
    return rb, f

def regSolnGaussianStokesletsOnly(nodes,eps,mu,alph,circrad,vh):
    rb = RS.Brinkman3DGaussianStokeslets(eps,mu,alph)
    v = np.zeros(nodes.shape,dtype='complex128')
#    #scale the velocity at the center to get the right boundary condition
#    v[:,0] = vh*( rb._H1func(np.array([0])) ) / (rb._H1func(np.array([circrad])) + circrad**2*rb._H2func(np.array([circrad])))
    # match the velocity at the center of the cylinder
    v[:,0] = vh
    f = rb.calcForces(nodes,nodes,v)
    return rb, f

def regSolnNegExpStokesletsOnly(nodes,eps,mu,alph,circrad,vh):
    rb = RS.Brinkman3DNegExpStokeslets(eps,mu,alph)
    v = np.zeros(nodes.shape,dtype='complex128')
#    #scale the velocity at the center to get the right boundary condition
#    v[:,0] = vh*( rb._H1func(np.array([0])) ) / (rb._H1func(np.array([circrad])) + circrad**2*rb._H2func(np.array([circrad])))
    # match the velocity at the center of the cylinder
    v[:,0] = vh
    f = rb.calcForces(nodes,nodes,v)
    return rb, f

def graphSolns(X,Vexact,Vreg,circrad,basedir,basename):
    print('Graphing results...')
    N = int(np.sqrt(len(X[:,0])))
    xg = np.reshape(X[:,0],(N,N))
    yg = np.reshape(X[:,1],(N,N))
    ind = np.nonzero(xg**2 + yg**2 > circrad**2)
    phi = np.linspace(0, 2*np.pi, 50)
    ues_mag, ues_ang = transformCplxSoln(Vexact[:,0],N)
    ves_mag, ves_ang = transformCplxSoln(Vexact[:,1],N)
    urb_mag, urb_ang = transformCplxSoln(Vreg[:,0],N)
    vrb_mag, vrb_ang = transformCplxSoln(Vreg[:,1],N)
    wmag, wang = transformCplxSoln(Vreg[:,2],N)
    umlevels = getULevels(ues_mag[ind],urb_mag[ind])
    vmlevels = getULevels(ves_mag[ind],vrb_mag[ind])
    ualevels = getULevels(ues_ang[ind],urb_ang[ind])
    valevels = getULevels(ves_ang[ind],vrb_ang[ind])
    wmlevels = getULevels(wmag,wmag)
    walevels = getULevels(wang,wang)
    plt.figure(1)
    plotSoln(xg,yg,ues_mag,umlevels,circrad,phi,basedir+basename+'/umag_solnexact.pdf')
    plotSoln(xg,yg,ves_mag,vmlevels,circrad,phi,basedir+basename+'/vmag_solnexact.pdf')
    plotSoln(xg,yg,ues_ang,ualevels,circrad,phi,basedir+basename+'/uang_solnexact.pdf')
    plotSoln(xg,yg,ves_ang,valevels,circrad,phi,basedir+basename+'/vang_solnexact.pdf')
    plotSoln(xg,yg,urb_mag,umlevels,circrad,phi,basedir+basename+'/umag_reg.pdf') 
    plotSoln(xg,yg,vrb_mag,vmlevels,circrad,phi,basedir+basename+'/vmag_reg.pdf') 
    plotSoln(xg,yg,urb_ang,ualevels,circrad,phi,basedir+basename+'/uang_reg.pdf') 
    plotSoln(xg,yg,vrb_ang,valevels,circrad,phi,basedir+basename+'/vang_reg.pdf') 
    plotSoln(xg,yg,wmag,wmlevels,circrad,phi,basedir+basename+'/wmag_reg.pdf')
    plotSoln(xg,yg,wang,walevels,circrad,phi,basedir+basename+'/wang_reg.pdf') 
   
def transformCplxSoln(u,N): 
    umag = np.reshape(np.abs(u),(N,N))
    uang = np.reshape(np.angle(u),(N,N))
    return umag, uang
    
def getULevels(ue,ur):
    umin = np.min([np.min(ue),np.min(ur)])
    umax = np.max([np.max(ue),np.max(ur)])
    ulevels = np.linspace(umin,umax,25)
    return ulevels
    
def plotSoln(xg,yg,u,ulevels,circrad,phi,fname):
    plt.clf()
    ph2=plt.contourf(xg,yg,u,ulevels,cmap=cm.RdGy)
    plt.colorbar(ph2)
    plt.plot(circrad*np.cos(phi),circrad*np.sin(phi),'k',linewidth=1.0)
    plt.savefig(fname,format='pdf')

def constructSolns(circrad,eps,zh,halfzpts,regfunc):
    #spatial parameters
    origin = (-0.05,-0.05) #millimeters
    size = (100,100)
    h = 1.e-3
    X = makeGridCenter(size,h,origin)
    #get exact solution
    time = [0]
    freq=10 #Hz
    vh=1.0 #mm/s
    vinf=0.0
    mu=1.85e-8 #kg/(mm s)
    nu=15.7 #mm^2/s
    print('Calculating exact solution...')
    uexact,vexact = ES.circularCylinder2DOscillating(X[:,0],X[:,1],circrad,nu,mu,freq,vh,vinf,time)
    Vexact = np.column_stack([uexact,vexact])
    print('Calculating reg velocity...')
    #get regularized solution
    nodes = np.zeros((2*halfzpts+1,3))
    nodes[:,2] = np.arange(-halfzpts*zh,(halfzpts+1)*zh,zh)
    alph = np.sqrt(1j*2*np.pi*freq/nu)
    rb, f = regfunc(nodes,eps,mu,alph,circrad,vh)
#    print("forces")
#    print(f)
    X = np.column_stack([X,np.zeros(X[:,0].shape)])
    Vreg = rb.calcVel(X,nodes,f)
    zline = np.arange(-halfzpts*zh,(halfzpts+1/10)*zh,zh/10)
    zlinearr = np.zeros((zline.shape[0],3))
    zlinearr[:,2] = zline
    zlinearr[:,0] = circrad
    Vzline = rb.calcVel(zlinearr,nodes,f)
    return X, Vexact, Vreg, Vzline, zline

def sims3D():
    circrad = 0.01 #millimeters
    eps = 0.003
    zh = 2*circrad
    halfzpts = 40
    X, Ves, Vrb, Vzline,zline = constructSolns(circrad,eps,zh,halfzpts,regSolnChainofSpheresNegExp)
    mydict = {'X':X,'Ves':Ves,'Vrb':Vrb,'circrad':circrad,'eps':eps,'zh':zh,'halfzpts':halfzpts}
    basedir = os.path.expanduser('~/CricketProject/CompReg2Exact/')
    basename = 'negexpspheres_BCsonaxis_zhdiameter_largereps40x2pts'
    F = open( basedir+basename+'.pickle', 'w' )
    cPickle.Pickler(F).dump(mydict)
    F.close()
    myfile(basedir,basename)    
    graphSolns(mydict['X'],mydict['Ves'],mydict['Vrb'],mydict['circrad'],basedir,basename)
    vRS.plainPlots(zline,np.abs(Vzline[:,0]),"|u|","z","velocity",None,basedir+basename+'/zline_umag.pdf')
    vRS.plainPlots(zline,np.abs(Vzline[:,1]),"|v|","z","velocity",None,basedir+basename+'/zline_vmag.pdf')
    vRS.plainPlots(zline,np.abs(Vzline[:,2]),"|w|","z","velocity",None,basedir+basename+'/zline_wmag.pdf')


if __name__ == '__main__':
    sims3D()





















