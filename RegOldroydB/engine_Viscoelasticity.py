#import python modules
import numpy as np
from scipy.integrate import ode
import os, sys
from cPickle import Pickler
#import my modules
import SpatialDerivs2D as SD2D
import Gridding as mygrids
import forces_Viscoelasticity 
import bgvels_Viscoelasticity 
#import swimmervisualizer as sv
try:
    import pythoncode.cext.CubicStokeslet2D as CM
except:
    print('Please compile the C extension CubicStokeslet2D.c and put the .so in the cext folder.')
    raise(SystemExit)

def isIntDivisible(x,y,p=1.e-8):
    m = x/y
    return abs(round(m)-m) < p
    
def stokesFlowUpdater(t,y,wdict):
    '''
    t = current time, y = [fpts.flatten(), l.flatten(), P.flatten()], pdict 
    contains: K is spring constant, xr is resting position, blob is regularized 
    Stokeslet object, myForces is a function handle, forcedict is a dictionary
    containing optional parameters for calculating forces.
    
    '''
    pdict = wdict['pdict']
    pdict['forcedict']['t'] = t
    Q=len(y)/2
    fpts = np.reshape(y,(Q,2))
    f = wdict['myForces'](fpts,**pdict['forcedict'])
    yt = CM.matmult(pdict['eps_obj'],pdict['mu'],fpts,fpts,f)
    return yt

def stokesFlowUpdaterWithMarkers(t,y,wdict):
    pdict = wdict['pdict']
    pdict['forcedict']['t'] = t
    N = pdict['N']
    M = pdict['M']
    Q = len(y)/2 - N*M
    fpts = np.reshape(y[:2*Q],(Q,2))
    ap = np.reshape(y,(Q+N*M,2)) 
    #calculate spring forces
    f = pdict['myForces'](fpts,**pdict['forcedict'])
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    lt = CM.matmult(pdict['eps_obj'],pdict['mu'],ap,fpts,f)
    return lt

            
def viscoElasticUpdater_force(t,y,wdict):
    #interior function for force pts only
    #split up long vector into individual sections (force pts, Lagrangian pts, stress components)
    pdict = wdict['pdict']
    pdict['forcedict']['t'] = t
    N = pdict['N']
    M = pdict['M']
    Q = len(y)/2 - N*M - 2*N*M
    fpts = np.reshape(y[:2*Q],(Q,2))
    l2 = np.reshape(y[range(2*Q,2*Q+2*N*M)],(N*M,2))
    l3 = np.reshape(l2,(N,M,2))
    allpts = np.reshape(y[:2*Q+2*N*M],(Q+N*M,2)) # both force points and Lagrangian points
    P2 = np.reshape(y[(2*Q+2*N*M):],(N*M,2,2))
    P3 = np.reshape(P2,(N,M,2,2))
    #calculate tensor derivative
    Pd = pdict['beta']*SD2D.tensorDiv(P3,pdict['gridspc'],N,M)
    Pd = np.reshape(Pd,(N*M,2))
    #calculate spring forces
    f = wdict['myForces'](fpts,**pdict['forcedict'])
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    lt = pdict['gridspc']**2*CM.matmult(pdict['eps_grid'],pdict['mu'],allpts,l2,Pd) + CM.matmult(pdict['eps_obj'],pdict['mu'],allpts,fpts,f)
    # reshape the velocities on the grid
    lt3 = np.reshape(lt[2*Q:],(N,M,2)) 
    #calculate deformation matrix and its inverse
    F = SD2D.vectorGrad(l3,pdict['gridspc'],N,M)
    F = np.reshape(F,(N*M,2,2))
    #calculate new stress time derivatives
#    gradlt = pdict['gridspc']**2*CM.derivop(pdict['eps_grid'],pdict['mu'],l2,l2,Pd,F) + CM.derivop(pdict['eps_obj'],pdict['mu'],l2,fpts,f,F)   #grad(Stokeslet) method
    gradlt = SD2D.vectorGrad(lt3,pdict['gridspc'],N,M)   
    gradlt = np.reshape(gradlt,(N*M,2,2))
#    Finv = CM.matinv2x2(F)   # first grad(u) method
#    Pt = np.zeros((N*M,2,2))
#    for j in range(N*M):
#        Pt[j,:,:] = np.dot(np.dot(gradlt[j,:,:],Finv[j,:,:]),P2[j,:,:]) - (1./pdict['Wi'])*(P2[j,:,:] - Finv[j,:,:].transpose())        
    Pt = CM.stressDeriv(pdict['Wi'],gradlt,F,P2)
    return np.append(lt,Pt.flatten())

def viscoElasticUpdater_bgvel(t,y,wdict):
    #interior function for incompressible background flow only
    #split up long vector into individual sections (force pts, Lagrangian pts, stress components)
    pdict = wdict['pdict']
    N = pdict['N']
    M = pdict['M']
    l2 = np.reshape(y[range(2*N*M)],(N*M,2))
    l3 = np.reshape(l2,(N,M,2))
    P2 = np.reshape(y[(2*N*M):],(N*M,2,2))
    P3 = np.reshape(P2,(N,M,2,2))
    #calculate tensor derivative
    Pd = pdict['beta']*SD2D.tensorDiv(P3,pdict['gridspc'],N,M)
    Pd = np.reshape(Pd,(N*M,2))
    #calculate deformation matrix and its inverse
    F = SD2D.vectorGrad(l3,pdict['gridspc'],N,M)
    F = np.reshape(F,(N*M,2,2))
    #calculate new velocities at all points of interest
    ub, gradub = wdict['myVelocity'](pdict,l2)
    lt0 = pdict['gridspc']**2*CM.matmult(pdict['eps_grid'],pdict['mu'],l2,l2,Pd) 
    # reshape the velocities on the grid
    lt3 = np.reshape(lt0,(N,M,2)) 
    lt = ub + lt0
#    calculate new stress time derivatives
    gradlt = SD2D.vectorGrad(lt3,pdict['gridspc'],N,M)   
    gradlt = np.reshape(gradlt,(N*M,2,2))
    Pt = CM.stressDeriv(pdict['Wi'],gradub,gradlt,F,P2)
#    gradlt = pdict['gridspc']**2*CM.derivop(pdict['eps_grid'],pdict['mu'],l2,l2,Pd,F) #grad(Stokeslet) method
#    Finv = CM.matinv2x2(F) #first grad(u) method
#    Pt = np.zeros((N*M,2,2)) #first grad(u) method
#    for j in range(N*M):
#        Pt[j,:,:] = np.dot(gradub[j,:,:],P2[j,:,:]) + np.dot(np.dot(gradlt[j,:,:],Finv[j,:,:]),P2[j,:,:]) - (1./pdict['Wi'])*(P2[j,:,:] - Finv[j,:,:].transpose())        
    return np.append(lt,Pt.flatten())


def regridFlagAdaptiveDistortedDet(adarray,detcrit,addpts):
    '''
    Regrid when determinants are distorted. May be used with addpts = 0 or 1.
    
    '''
    regridflag = False
    addptsflag = bool(addpts)
    if np.any(np.abs(adarray-1.0)) > detcrit:
        regridflag = True
    return regridflag, addptsflag


def regridFlagFixedTime(t,t0,timecrit,addpts):
    '''
    Regrid every timecrit units. May be used with addpts = 0 or 1.
    
    '''
    regridflag = False
    addptsflag = bool(addpts)
    if t > t0 and isIntDivisible((t-t0),timecrit): 
        regridflag = True
    return regridflag, addptsflag


def regridFlagAdaptiveCloseToEdge(trace,N,M,edgecrit,addpts):
    '''
    Regrid when stress trace is large near domain edge. Must be used with addpts = 1.
    
    '''
    regridflag = False
    if np.any(np.abs(trace[:8,:]-2) > edgecrit) or np.any(np.abs(trace[N-8:,:]-2) > edgecrit) or np.any(np.abs(trace[:,:8]-2) > edgecrit) or np.any(np.abs(trace[:,M-8:]-2) > edgecrit): 
        regridflag = True
        addptsflag = True
    if not addpts:
        print('Warning: Incompatible options. Adding points at every regrid even though addpts == 0.')
    return regridflag, addptsflag


def calcState_nodets(r,t0,regridding,regriddict,alldetflag=0):
    regridflag = False
    addptsflag = False
    StateNow = {}
    StateNow['t'] = r.t
    wdict = r.f_params[0]
    pdict=wdict['pdict']
    try:
        N = pdict['N']
        M = pdict['M']
    except:
        N = 0
        M = 0
    Q = len(r.y) - 2*N*M - 4*N*M
    if N == 0: #stokes flow without markers
        StateNow['fpts']=r.y
    elif Q < 0: #stokes flow with markers
        StateNow['fpts']=r.y[:2*N*M]
        l3D = np.reshape(r.y[-2*N*M:],(N,M,2))
        StateNow['l']=l3D
    elif Q >= 0: #viscoelastic flow
        StateNow['fpts']=r.y[:Q]
        l3D = np.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2))
        StateNow['l']=l3D
        F = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
        Ptemp = np.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
        Stemp = np.zeros((N,M,2,2))
        tr = np.zeros((N,M))
        for j in range(N):
            for k in range(M):
                Fs = F[j,k,:,:]
                Ps = Ptemp[j,k,:,:]
                Stemp[j,k,:,:]=np.dot(Ps,Fs.transpose())
                tr[j,k] = Stemp[j,k,0,0] + Stemp[j,k,1,1]
        StateNow['S']=Stemp
        StateNow['Strace']=tr
        if regridding:
            if regriddict['timecrit'] != None:
                regridflag, addptsflag = regridFlagFixedTime(r.t,t0,regriddict['timecrit'],regriddict['addpts'])
            elif regriddict['edgecrit'] != None:
                regridflag, addptsflag = regridFlagAdaptiveCloseToEdge(tr,N,M,regriddict['edgecrit'],regriddict['addpts'])
            else:
                print('Warning: No regridding algorithm specified. Regridding ignored.')
    return StateNow, regridflag, addptsflag

def calcState_dets(r,t0,regridding,regriddict,alldetflag=1):
    regridflag = False
    addptsflag = False
    StateNow = {}
    StateNow['t'] = r.t
    wdict = r.f_params[0]
    pdict=wdict['pdict']
    try:
        N = pdict['N']
        M = pdict['M']
    except:
        N = 0
        M = 0
    Q = len(r.y) - 2*N*M - 4*N*M
    adarray=np.zeros((N,M))
    if N == 0: #stokes flow without markers
        StateNow['fpts']=r.y
    elif Q < 0: #stokes flow with markers
        StateNow['fpts']=r.y[:2*N*M]
        l3D = np.reshape(r.y[-2*N*M:],(N,M,2))
        StateNow['l']=l3D
        if alldetflag:
            F = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
            for j in range(N):
                for k in range(M):
                    Fs = F[j,k,:,:]
                    gdet = np.linalg.det(Fs)
                    adarray[j,k] = gdet
            StateNow['alldets']=adarray
    elif Q >= 0: #viscoelastic flow
        StateNow['fpts']=r.y[:Q]
        l3D = np.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2))
        StateNow['l']=l3D
        F = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
        Ptemp = np.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
        Stemp = np.zeros((N,M,2,2))
        tr = np.zeros((N,M))
        for j in range(N):
            for k in range(M):
                Fs = F[j,k,:,:]
                if alldetflag or (regriddict['detcrit'] != None):
                    gdet = np.linalg.det(Fs)
                    adarray[j,k] = gdet
                Ps = Ptemp[j,k,:,:]
                Stemp[j,k,:,:]=np.dot(Ps,Fs.transpose())
                tr[j,k] = Stemp[j,k,0,0] + Stemp[j,k,1,1]
        StateNow['S']=Stemp
        StateNow['Strace']=tr
        if alldetflag:
            StateNow['alldets']=adarray 
        if regridding:
            if regriddict['timecrit'] != None:
                regridflag, addptsflag = regridFlagFixedTime(r.t,t0,regriddict['timecrit'],regriddict['addpts'])
            elif regriddict['edgecrit'] != None:
                regridflag, addptsflag = regridFlagAdaptiveCloseToEdge(tr,N,M,regriddict['edgecrit'],regriddict['addpts'])
            elif regriddict['detcrit'] != None:
                regridflag, addptsflag = regridFlagAdaptiveDistortedDet(adarray,regriddict['detcrit'],regriddict['addpts'])
            else:
                print('Warning: No regridding algorithm specified. Regridding ignored.')
    return StateNow, regridflag, addptsflag


def logState(StateNow,StateSave):
    StateSave['t'].append(StateNow['t'])
    StateSave['fpts'].append(StateNow['fpts'])
    if StateNow.has_key('l'):
        StateSave['l'].append(StateNow['l'])
    if StateNow.has_key('S'):
        StateSave['S'].append(StateNow['S'])
        StateSave['Strace'].append(StateNow['Strace'])
    if StateNow.has_key('alldets'):
        StateSave['alldets'].append(StateNow['alldets'])
    return StateSave

def mySolver(myodefunc,y0,t0,dt,totalTime,wdict,stressflag=0,regridding=0,regriddict=dict(timecrit=None,detcrit=None,edgecrit=None,scalefactor=2.0,addpts=0),alldetflag=0,rtol=1.e-3,method='bdf'):
    '''
    myodefunc is f in y' = f(y); y0, t0 are initial conditions; 
    dt is time step for saving data (integrator time step is chosen automatically); 
    totalTime is the stopping time for the integration; 
    pdict is a dictionary containing the parameters for the update function f;
    method is a string specifying which ode solver to use.
    
    '''
    #initialize integrator
    r = ode(myodefunc).set_integrator('vode',method=method,rtol=rtol).set_initial_value(y0,t0).set_f_params(wdict) 
    # initialize list of saved variables
    StateSave={}
    StateSave['t']=[]
    StateSave['fpts']=[]
    if myodefunc != stokesFlowUpdater:
        StateSave['l']=[]
    if stressflag:
        StateSave['S']=[]
        StateSave['Strace']=[]
    if alldetflag:
        StateSave['alldets']=[]
    # construct variables
    pdict=wdict['pdict']
    if alldetflag or (regriddict['detcrit'] != None):
        mycalcState = calcState_dets
    else:
        mycalcState = calcState_nodets
    StateNow, regridflag, addptsflag = mycalcState(r,t0,regridding,regriddict,alldetflag)
    StateSave = logState(StateNow,StateSave)
    # integrate in time until regridding is required
    try:
        numskip = pdict['numskip']
    except:
        numskip = 0
    print(t0) #let the user know the simulation has begun
    c=1
    while r.successful() and r.t < totalTime:  
        r.integrate(t0+c*dt)
        if numskip ==0 or np.mod(c,numskip) == 0:
            print(r.t)
            StateNow, regridflag, addptsflag = mycalcState(r,t0,regridding,regriddict,alldetflag)
            StateSave = logState(StateNow,StateSave)
        c+=1
        # regrid when needed, then reset integrator
        if regridding and regridflag and r.t < totalTime:
            print('time to regrid...')
            lnew,Pnew,Nnew,Mnew = mygrids.interp2NewGrid(StateNow['l'],StateNow['S'],pdict['gridspc'],stressflag,regriddict['scalefactor'],addptsflag) 
#            plotforRegrid(StateNow['l'],StateNow['S'],lnew,Pnew,r.t)
            y0 = np.append(np.append(StateNow['fpts'],lnew.flatten()),Pnew.flatten())
            t0 = r.t
            pdict['N'] = Nnew
            pdict['M'] = Mnew
            wdict['pdict'] = pdict
            print(StateNow['l'].shape)
            print(lnew.shape)
            # integrate starting at new grid
            r = ode(myodefunc).set_integrator('vode',method=method,rtol=rtol).set_initial_value(y0,t0).set_f_params(wdict) 
            regridflag = False
            addptsflag = False
            StateNow={}
            c=1
    return StateSave

def plotforRegrid(lold,Sold,lnew,Snew,time):
    xmin = np.min(lnew[:,:,0]); xmax = np.max(lnew[:,:,0])
    ymin = np.min(lnew[:,:,1]); ymax = np.max(lnew[:,:,1])
    xdif = xmax-xmin; ydif = ymax-ymin
    limits = [xmin-0.1*xdif,xmax+0.1*xdif,ymin-0.1*ydif,ymax+0.1*ydif]
    S11min = np.min([np.min(Sold[:,:,0,0]),np.min(Snew[:,:,0,0])])
    S11max = np.max([np.max(Sold[:,:,0,0]),np.max(Snew[:,:,0,0])])
    S22min = np.min([np.min(Sold[:,:,1,1]),np.min(Snew[:,:,1,1])])
    S22max = np.max([np.max(Sold[:,:,1,1]),np.max(Snew[:,:,1,1])])
    vlims = [0.9*S11min,1.1*S11max,0.9*S22min,1.1*S22max]
    lvls = [np.linspace(np.min(Sold[:,:,0,0]),np.max(Sold[:,:,0,0]),10),np.linspace(np.min(Sold[:,:,1,1]),np.max(Sold[:,:,1,1]),10)]
    sv.contourRegrid(lold,Sold,limits,time,1,lvls,vlims)
    sv.contourRegrid(lnew,Snew,limits,time,0,lvls,vlims)
    
if __name__ == '__main__':
    pass
    
    
    
    
    
    
