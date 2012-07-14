#Created by Breschine Cummins on June 20, 2012.

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

#import python modules
import numpy as np
import os, sys
from cPickle import Pickler
#import my modules
import engine_Viscoelasticity as eVE
import forces_Viscoelasticity 
import bgvels_Viscoelasticity 
import lib_Gridding as mygrids

def mySwimmer_TeranFauciShelley():

    ####################################
    #set fixed parameter values first
    ####################################
    mu = 1.0
    xorigin = 0.0
    xextent = 1.3
    yorigin = 0.05
    yextent = 0.9
    origin = (xorigin,yorigin)
    #set time parameters, save data every time step
    #note that I do not control the time step in the solver!! 
    # my time step only tells me where I will save data
    t0 = 0; totalTime = 10.0; dt = 5.e-2; 
    initTime=1.0 #need this to be a full swimmer cycle -- see curvature forces in code
    #make swimmer
    a = 0.16
    w = -2*np.pi #swimmer period is 1
    lam = 2.5*np.pi
    L = 0.6
    myForces = forces_Viscoelasticity.calcForcesSwimmerTFS
    forcedocstring = 'Swimmer curvature forces according to Teran, Fauci, and Shelley: forces_Viscoelasticity.calcForcesSwimmerTFS'
    forcedict = dict(a=a, w=w, t=0, lam=lam, L=L)
    # solver options for viscoelastic flow
    stressflag=1
    regridding=1
    regriddict = dict(timecrit=0.4,edgecrit=None,detcrit=None,scalefactor=2,addpts=0)
    vfname = 'visco_PtinC_changingNp_fixedregrid004_scalefactor2_addpts0_'

    ####################################
    #Set up dictionaries for parameters...
    ####################################
    pdict = dict( mu=mu, forcedocstring=forcedocstring)
    wdict = dict(pdict=pdict,myForces=myForces)
    pdict = None #to avoid heisenbugs during refactor

    ####################################
    #choose a directory for saving files
    ####################################
    basedir = os.path.expanduser('~/VEsims/SwimmerRefactored/TFS_FinalParams/')    
    if not os.path.exists(basedir):
        basedir = '/Volumes/LCD/VEsims/SwimmerRefactored/TFS_FinalParams/'   
        if not os.path.exists(basedir):        
            basedir = '/scratch03/bcummins/mydata/ve/SwimmerRefactored/TFS_FinalParams/'    
            if not os.path.exists(basedir):
                print('Choose a different directory for saving files')
                raise(SystemExit)
                       
    ####################################
    #loop over parameters
    ####################################
    Wilist = [1.0]#[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]#, 2.0, 2.5, 3.0]#[0.1,0.08,0.06,0.04,0.02]
    Nlist = [54,108, 216]#[20,40,80,160]
    Nplist = [26]#[26, 50, 98]
    Np = Nplist[0]
    h = L/(Np-1)
    Kcurvlist = [0.2]#[0.2, 0.2/3, 0.2/4] 
    Kcurv = Kcurvlist[0]
    Klist = [40.] #[40.,20.,10.]
    K = Klist[0]
    xr = np.array([h])
    forcedict['h'] = h
    forcedict['xr'] = xr
    forcedict['Np'] = Np
    forcedict['Kcurv'] = Kcurv
    forcedict['K'] = K
    eps_obj = 1.5*h
    wdict['pdict']['forcedict'] = forcedict
    wdict['pdict']['eps_obj'] = eps_obj

    ####################################
    #Stokes flow reference run...
    ####################################
    #Set up initial conditions for flat swimmer
    y00 = 0.5*np.ones((2*Np,))
    v = np.arange(0.3,0.3+L+h/2,h)
    y00[:-1:2] = v         
 
    #Initialize flat swimmer in Stokes flow (ramp up to emergent swimming shape)
    print('Initialize swimmer in Stokes flow...')
    StateSave = eVE.mySolver(eVE.stokesFlowUpdater,y00,t0,dt,initTime,wdict)
    inits = np.asarray(StateSave['fpts'])[-1,:]                 
     
    #run the ode solver for Stokes flow
    print('Swimmer in Stokes flow...')
    StateSave = eVE.mySolver(eVE.stokesFlowUpdater,inits,initTime,dt,totalTime+initTime,wdict)
    #save the output
    StateSave['pdict']=wdict['pdict']
    F = open( basedir+'stokes_Kcurv%03d_K%03d_epsobj%03d_Time%02d.pickle' % (int(round(forcedict['Kcurv']*100)),int(round(forcedict['K'])),int(round(eps_obj*1000)),int(totalTime+initTime)), 'w' )
    Pickler(F).dump(StateSave)
    F.close()
    for k in range(len(Nlist)):
        print(h,xextent/Nlist[k])
        
        
        ####################################
        #Oldroyd-B flow reference run...
        ####################################
        #assign and record parameters
        N = Nlist[k]
        gridspc = xextent/N
        eps_grid = eps_obj#2*gridspc
        M = int(np.ceil(yextent/gridspc))
        wdict['pdict']['N'] = N
        wdict['pdict']['M'] = M
        wdict['pdict']['gridspc'] = gridspc
        wdict['pdict']['origin'] = origin
        wdict['pdict']['eps_grid'] = eps_grid
        #make the initial grid
        l0 = mygrids.makeGridCenter(N,M,gridspc,origin)
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        # set initial conditions for VE simulations
        y0 = np.append(np.append(inits, l0.flatten()),P0.flatten())
        
        for Wi in Wilist:
            #assign and record params
            beta = 1./(2*Wi)
            wdict['pdict']['Wi'] = Wi
            wdict['pdict']['beta'] = beta
        
            #Viscoelastic run
            print('Swimmer in VE flow, N = %02d, Wi = %f' % (N,Wi))
            StateSave = eVE.mySolver(eVE.viscoElasticUpdater_force,y0,initTime,dt,totalTime+initTime,wdict,stressflag,regridding,regriddict)
            #save the output
            StateSave['pdict']=wdict['pdict']
            StateSave['regriddict']=regriddict
            F = open(basedir+vfname+'Kcurv%03d_K%03d_epsobj%03d_epsgrid%03d_N%03d_Wi%04d_Time%02d.pickle' % (int(round(forcedict['Kcurv']*100)),int(round(forcedict['K'])),int(round(eps_obj*1000)),int(round(eps_grid*1000)),N,int(round(Wi*100)),int(totalTime+initTime)), 'w' )
            Pickler(F).dump(StateSave)
            F.close()
            
def mySwimmer_sine():

    ####################################
    #set fixed parameter values first
    ####################################
    mu = 1.0
    xorigin = 0.0
    xextent = 1.6
    yorigin = 0.1
    yextent = 0.8
    origin = (xorigin,yorigin)
    #set time parameters, save data every time step
    #note that I do not control the time step in the solver!! 
    # my time step only tells me where I will save data
    t0 = 0; totalTime = 5.0; dt = 5.e-2; 
    initTime=1.0 #need this to be a full swimmer cycle -- see curvature forces in code
    #make swimmer
    a = 0.16
    w = 2*np.pi #swimmer period is 1
    lam = 2.5*np.pi
    Np = 20
    L = 0.78
    h = L/(Np-1)
    K =40.0
    xr = np.array([h])
    forcedict = dict(a=a, w=w, t=0, Kcurv=0.01, lam=lam, Np=Np, h=h, L=L, K=K, xr=xr)
    eps_obj = 2*h
    myForces = forces_Viscoelasticity.calcForcesSwimmer
    forcedocstring = 'Swimmer curvature forces according to linearly increasing sine wave: forces_Viscoelasticity.calcForcesSwimmer'
    # solver options for viscoelastic flow
    stressflag=1
    regridding=1
    regriddict = dict(timecrit=1.0,edgecrit=None,detcrit=None,scalefactor=2,addpts=0)
    vfname = 'visco_PtinC_fixedregrid010_scalefactor2_addpts0_'


    ####################################
    #Put parameters into dictionaries...
    ####################################
    pdict = dict( mu=mu, forcedict=forcedict, eps_obj=eps_obj, forcedocstring=forcedocstring)
    wdict = dict(pdict=pdict,myForces=myForces)
    pdict = None #to avoid heisenbugs during refactor

    ####################################
    #choose a directory for saving files
    ####################################
    basedir = os.path.expanduser('~/VEsims/SwimmerRefactored/SineWave/')    
    if not os.path.exists(basedir):
        basedir = '/Volumes/LCD/VEsims/SwimmerRefactored/SineWave/'   
        if not os.path.exists(basedir):        
            basedir = '/scratch03/bcummins/mydata/ve/SwimmerRefactored/SineWave/'    
            if not os.path.exists(basedir):
                print('Choose a different directory for saving files')
                raise(SystemExit)
            
    ####################################
    #Stokes flow reference run...
    ####################################
    #Set up initial conditions for flat swimmer
    y00 = 0.5*np.ones((2*Np,))
    v = np.arange(0.3,0.3+L+h/2,h)
    y00[:-1:2] = v         
 
    #Initialize flat swimmer in Stokes flow (ramp up to emergent swimming shape)
    print('Initialize swimmer in Stokes flow...')
    StateSave = eVE.mySolver(eVE.stokesFlowUpdater,y00,t0,dt,initTime,wdict)
    inits = np.asarray(StateSave['fpts'])[-1,:]                 
     
    #run the ode solver for Stokes flow
    print('Swimmer in Stokes flow...')
    StateSave = eVE.mySolver(eVE.stokesFlowUpdater,inits,initTime,dt,totalTime+initTime,wdict)
    #save the output
    StateSave['pdict']=wdict['pdict']
    F = open( basedir+'stokes_epsobj%03d_Time%02d.pickle' % (int(round(eps_obj*1000)),int(totalTime+initTime)), 'w' )
    Pickler(F).dump(StateSave)
    F.close()
           
    ####################################
    #now for viscoelastic simulations....
    ####################################
    Wilist = [0.25]#, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]#, 2.0, 2.5, 3.0]#[0.1,0.08,0.06,0.04,0.02]
    Nlist = [41] #[20,40,80,160]
    for N in Nlist:
        gridspc = xextent/N
        eps_grid = eps_obj#2*gridspc
        M = int(np.ceil(yextent/gridspc))
        #make the grid
        l0 = mygrids.makeGridCenter(N,M,gridspc,origin)
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        # set initial conditions for VE simulations
        y0 = np.append(np.append(inits, l0.flatten()),P0.flatten())
        
        for Wi in Wilist:
            #assign and record VE params
            beta = 1./(2*Wi)
            wdict['pdict']['N'] = N
            wdict['pdict']['M'] = M
            wdict['pdict']['gridspc'] = gridspc
            wdict['pdict']['origin'] = origin
            wdict['pdict']['Wi'] = Wi
            wdict['pdict']['beta'] = beta
            wdict['pdict']['eps_grid'] = eps_grid
        
            #Viscoelastic run
            print('Swimmer in VE flow, N = %02d, Wi = %f' % (N,Wi))
            StateSave = eVE.mySolver(eVE.viscoElasticUpdater_force,y0,initTime,dt,totalTime+initTime,wdict,stressflag,regridding,regriddict)
            #save the output
            StateSave['pdict']=wdict['pdict']
            StateSave['regriddict']=regriddict
            F = open(basedir+vfname+'epsobj%03d_epsgrid%03d_N%03d_Wi%04d_Time%02d.pickle' % (int(round(eps_obj*1000)),int(round(eps_grid*1000)),N,int(round(Wi*100)),int(totalTime+initTime)), 'w' )
            Pickler(F).dump(StateSave)
            F.close()
            
def myExtension_initrest():
    ####################################
    #set fixed parameter values first
    ####################################
    mu = 1.0
    Wi = 1.2
    beta = 1. / (2*Wi) 
    xorigin = -0.25
    xextent = 0.5
    yorigin = -0.25
    yextent = 0.25
    origin = (xorigin,yorigin)
    U = 0.1
    #set time parameters, save data every numskip time steps (default is every time step)
    #note that I do not control the time step in the solver!! 
    # my time step only determines the maximum time step allowed
    t0 = 0; totalTime = 3.0; dt = 5.e-2    
    myVelocity = bgvels_Viscoelasticity.Extension     
    veldocstring = 'Purely extensional flow near hyperbolic stagnation point: bgvels_Viscoelasticity.Extension'
    # solver options for viscoelastic flow
    stressflag=1
    regridding=0
    ####################################
    #choose a directory for saving files
    ####################################
    basedir = os.path.expanduser('~/VEsims/ExactExtensionalFlow/')
    if not os.path.exists(basedir):
        basedir = '/Volumes/LCD/VEsims/ExactExtensionalFlow/'   
        if not os.path.exists(basedir):        
            basedir = '/scratch03/bcummins/mydata/ve/ExactExtensionalFlow/'    
            if not os.path.exists(basedir):
                print('Choose a different directory for saving files')
                raise(SystemExit)

    for rtol in [1.e-3,1.e-4,1.e-5,1.e-6]:
        fnamestart = 'ext_initrest_PtinC_noregrid_rtol%02d' % int(np.abs(np.log10(rtol)))
    
        Nlist = [20]#,40,80]
        for N in Nlist:
            gridspc = xextent/N
            eps_grid = 2*gridspc
            M = int(np.ceil(yextent/gridspc))
            ####################################
            #Put parameters into dictionaries...
            ####################################
            pdict = dict( N = N, M = M, U=U, gridspc = gridspc, origin = origin, mu = mu, Wi = Wi, beta=beta, eps_grid=eps_grid, veldocstring=veldocstring)
            wdict = dict(pdict=pdict,myVelocity=myVelocity)
            pdict = None #to avoid heisenbugs during refactor
    
            #make the grid
            l0 = mygrids.makeGridCenter(N,M,gridspc,origin)
            P0 = np.zeros((N,M,2,2))
            P0[:,:,0,0] = 1.0
            P0[:,:,1,1] = 1.0
            print(P0.shape)
            y0 = np.append(l0.flatten(),P0.flatten())    
            #Viscoelastic run
            print('Extensional flow, initially at rest, N = %02d' % N)
            StateSave = eVE.mySolver(eVE.viscoElasticUpdater_bgvel,y0,t0,dt,totalTime,wdict,stressflag,regridding,rtol=rtol)        
            #save the output
            StateSave['pdict']=wdict['pdict']
            StateSave['dt']=dt
            fname = basedir + fnamestart
            F = open( fname+'_eps%03d_N%03d_Wi%02d_Time%02d.pickle' % (int(round(eps_grid*1000)),N,int(round(Wi)),int(round(totalTime))), 'w' )
            Pickler(F).dump(StateSave)
            F.close()
        
def myExtension_initygrad():
    ####################################
    #set fixed parameter values first
    ####################################
    mu = 1.0
    Wi = 1.2
    beta = 1. / (2*Wi) 
    xorigin = -1.0
    xextent = 2.0
    yorigin = -4.0
    yextent = 8.0
    origin = (xorigin,yorigin)
    U = 2.0/Wi   
    #set time parameters, save data every numskip time steps (default is every time step)
    #note that I do not control the time step in the solver!! 
    # my time step only determines the maximum time step allowed
    t0 = 0; totalTime = 3.0; dt = 5.e-2    
    myVelocity = bgvels_Viscoelasticity.Extension     
    veldocstring = 'Purely extensional flow near hyperbolic stagnation point: bgvels_Viscoelasticity.Extension'
    # solver options for viscoelastic flow
    stressflag=1
    regridding=0
    ####################################
    #choose a directory for saving files
    ####################################
    basedir = os.path.expanduser('~/VEsims/ExactExtensionalFlow/')
    if not os.path.exists(basedir):
        basedir = '/Volumes/LCD/VEsims/ExactExtensionalFlow/'   
        if not os.path.exists(basedir):        
            basedir = '/scratch03/bcummins/mydata/ve/ExactExtensionalFlow/'    
            if not os.path.exists(basedir):
                print('Choose a different directory for saving files')
                raise(SystemExit)

    for rtol in [1.e-3,1.e-4,1.e-5,1.e-6]:
        fnamestart = 'ext_initygrad_PtinC_noregrid_rtol%02d' % int(np.abs(np.log10(rtol)))
    
        Nlist = [20]#,40,80]
        for N in Nlist:
            gridspc = xextent/N
            eps_grid = 2*gridspc
            M = int(np.ceil(yextent/gridspc))
            ####################################
            #Put parameters into dictionaries...
            ####################################
            pdict = dict( N = N, M = M, U=U, gridspc = gridspc, origin = origin, mu = mu, Wi = Wi, beta=beta, eps_grid=eps_grid, veldocstring=veldocstring)
            wdict = dict(pdict=pdict,myVelocity=myVelocity)
            pdict = None #to avoid heisenbugs during refactor
    
            #make the grid
            l0 = mygrids.makeGridCenter(N,M,gridspc,origin)
            P0 = np.zeros((N,M,2,2))
            P0[:,:,0,0] = (1.0 +l0[:,:,1]**2)**(1.0/(2*Wi*U) - 1.0) + 1.0/(1.0-2*Wi*U)
            P0[:,:,1,1] = 1.0
            print(P0.shape)
            y0 = np.append(l0.flatten(),P0.flatten())    
            #Viscoelastic run
            print('Extensional flow, initial y gradient, N = %02d' % N)
            StateSave = eVE.mySolver(eVE.viscoElasticUpdater_bgvel,y0,t0,dt,totalTime,wdict,stressflag,regridding,rtol=rtol)        
            #save the output
            StateSave['pdict']=wdict['pdict']
            StateSave['dt']=dt
            fname = basedir + fnamestart
            F = open( fname+'_eps%03d_N%03d_Wi%02d_Time%02d.pickle' % (int(round(eps_grid*1000)),N,int(round(Wi)),int(round(totalTime))), 'w' )
            Pickler(F).dump(StateSave)
            F.close()


if __name__ == '__main__':
    mySwimmer_TeranFauciShelley()
#    mySwimmer_sine()
#    myExtension_initrest()
#    myExtension_initygrad()
#    
    
    
    
    
    
