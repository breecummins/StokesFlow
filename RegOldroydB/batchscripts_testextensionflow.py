#import python modules
import numpy as np
import os, re
#import home-rolled python modules 
import SpatialDerivs2D as SD2D
import Gridding as mygrids
import mainC
import viz_testextensionflow as vizTEF
from utilities import loadPickle
try:
    import pythoncode.cext.CubicStokeslet2D as CM
except:
    print('Please compile the C extension CubicStokeslet2D.c and put the .so in the cext folder.')
    raise(SystemExit)


def knownsolutioninitialrest(U, Wi, l, times):
    '''Calculate exact solution.'''
    lcalc = []
    Pcalc = []
    Scalc = []
    Fcalc = []
    Finvcalc = []
    initgrid = l[0]
    for k in range(len(times)):
        t = times[k]
        # make matrices of the correct size
        pos = np.zeros(initgrid.shape)
        stress = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        stressS = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        # position solution 
        pos[:,:,0] = initgrid[:,:,0]*np.exp(U*t)
        pos[:,:,1] = initgrid[:,:,1]*np.exp(-U*t)
        lcalc.append(pos.copy())
        # Lagrangian stress solution
        stress[:,:,0,0] = ( np.exp(-U*t) - 2*Wi*U*np.exp((U - 1./Wi)*t) ) / (1 - 2*Wi*U)
        stress[:,:,1,1] = ( np.exp(U*t) + 2*Wi*U*np.exp(-(U + 1./Wi)*t) ) / (1 + 2*Wi*U)
        Pcalc.append(stress.copy())
        # Eulerian stress solution
        stressS[:,:,0,0] = stress[:,:,0,0]*np.exp(U*t)
        stressS[:,:,1,1] = stress[:,:,1,1]*np.exp(-U*t)
        Scalc.append(stressS.copy())
        # deformation matrix and inverse
        Fcalc.append(np.array([[np.exp(U*t), 0],[0,np.exp(-U*t)]]))
        Finvcalc.append(np.array([[np.exp(-U*t), 0],[0,np.exp(U*t)]]))
    return lcalc, Pcalc, Scalc, Fcalc, Finvcalc

def knownsolutioninitialygrad(U, Wi, l, times):
    '''Calculate exact solution.'''
    lcalc = []
    Pcalc = []
    Scalc = []
    Fcalc = []
    Finvcalc = []
    initgrid = l[0]
    for k in range(len(times)):
        t = times[k]
        # make matrices of the correct size
        pos = np.zeros(initgrid.shape)
        stress = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        stressS = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        # position solution 
        pos[:,:,0] = initgrid[:,:,0]*np.exp(U*t)
        pos[:,:,1] = initgrid[:,:,1]*np.exp(-U*t)
        lcalc.append(pos.copy())
        # Eulerian stress solution
        stressS[:,:,0,0] = np.exp((2*U - 1./Wi)*t)*(1.0 + initgrid[:,:,1]**2)**(1.0/(2*Wi*U) - 1.0) + 1.0/(1 - 2*Wi*U)
        stressS[:,:,1,1] = ( 1 + 2*Wi*U*np.exp(-(2*U + 1./Wi)*t) ) / (1 + 2*Wi*U)
        Scalc.append(stressS.copy())
        # Lagrangian stress solution
        stress[:,:,0,0] = np.exp(-U*t)*stressS[:,:,0,0]
        stress[:,:,1,1] = np.exp(U*t)*stressS[:,:,1,1]
        Pcalc.append(stress.copy())
        # deformation matrix and inverse
        Fcalc.append(np.array([[np.exp(U*t), 0],[0,np.exp(-U*t)]]))
        Finvcalc.append(np.array([[np.exp(-U*t), 0],[0,np.exp(U*t)]]))
    return lcalc, Pcalc, Scalc, Fcalc, Finvcalc


def simresults(basename, basedir):
    '''Retrieve approximate solution from saved output'''
    mydict = loadPickle(basename, basedir)
    l = mydict['l']
    S=mydict['S']
    F=[]
    Finv=[]
    P=[]
    N = l[0].shape[0]
    M = l[0].shape[1]
    for k in range(len(mydict['t'])):
        Ft = SD2D.vectorGrad(l[k],mydict['pdict']['gridspc'],N,M)
        Ftemp = np.reshape(Ft,(N*M,2,2))
        Ftinv = CM.matinv2x2(Ftemp)
        Ftinv = np.reshape(Ftinv,(N,M,2,2))
        F.append(Ft.copy())
        Finv.append(Ftinv.copy())
        stress = np.zeros((N,M,2,2))
        for j in range(N):
            for m in range(M):
                stress[j,m,:,:] = S[k][j,m,:,:]*Ftinv[j,m,:,:].transpose()
        P.append(stress.copy())
    return l, P, S, F, Finv, mydict

def calcerrs(l, P, S, F, Finv, lcalc, Pcalc, Scalc, Fcalc, Finvcalc, mydict):
    '''Calculate Linf and L2 errors between the exact solution and the numerical approximation'''
    errorsLinf = [[] for k in range(10)]
    errorsLtwo = [[] for k in range(10)]
    N = mydict['pdict']['N']
    M = mydict['pdict']['M']
    fN = np.floor(N/2.)
    startN = 0#fN-2#np.floor(fN/2.)
    endN = N#fN+2#startN+fN
    fM = np.floor(M/2.)
    startM = 0#fM-2#np.floor(fM/2.)
    endM = M#fM+2#startM+fM
    for k in range(len(mydict['t'])):
        errorsLinf[0].append( np.max(np.abs( (l[k][startN:endN,startM:endM,0] - lcalc[k][startN:endN,startM:endM,0]) / lcalc[k][startN:endN,startM:endM,0]) ) )
#        print(np.max(np.abs( (l[k][startN:endN,startM:endM,0] - lcalc[k][startN:endN,startM:endM,0]) / lcalc[k][startN:endN,startM:endM,0]) ))
#        print(np.mean(np.abs( (l[k][startN:endN,startM:endM,0] - lcalc[k][startN:endN,startM:endM,0]) / lcalc[k][startN:endN,startM:endM,0]) ))
#        print(np.median(np.abs( (l[k][startN:endN,startM:endM,0] - lcalc[k][startN:endN,startM:endM,0]) / lcalc[k][startN:endN,startM:endM,0]) )) 
        errorsLinf[1].append( np.max(np.abs( (l[k][startN:endN,startM:endM,1] - lcalc[k][startN:endN,startM:endM,1]) / lcalc[k][startN:endN,startM:endM,1]) ) ) 
        errorsLtwo[0].append( np.sqrt( np.sum( (l[k][startN:endN,startM:endM,0] - lcalc[k][startN:endN,startM:endM,0])**2 ) ) / np.sqrt( np.sum( (lcalc[k][startN:endN,startM:endM,0])**2 ) ) )
        errorsLtwo[1].append( np.sqrt( np.sum( (l[k][startN:endN,startM:endM,1] - lcalc[k][startN:endN,startM:endM,1])**2 ) ) / np.sqrt( np.sum( (lcalc[k][startN:endN,startM:endM,1])**2 ) ) )
        errorsLinf[2].append( np.max(np.abs( (P[k][startN:endN,startM:endM,0,0] - Pcalc[k][startN:endN,startM:endM,0,0]) / Pcalc[k][startN:endN,startM:endM,0,0]) ) ) 
        errorsLinf[3].append( np.max(np.abs(P[k][startN:endN,startM:endM,1,0] - Pcalc[k][startN:endN,startM:endM,1,0])))
        errorsLinf[4].append( np.max(np.abs(P[k][startN:endN,startM:endM,0,1] - Pcalc[k][startN:endN,startM:endM,0,1])))
        errorsLinf[5].append( np.max(np.abs( (P[k][startN:endN,startM:endM,1,1] - Pcalc[k][startN:endN,startM:endM,1,1]) / Pcalc[k][startN:endN,startM:endM,1,1]) ) )
        errorsLtwo[2].append( np.sqrt( np.sum( (P[k][startN:endN,startM:endM,0,0] - Pcalc[k][startN:endN,startM:endM,0,0])**2 ) ) / np.sqrt( np.sum( (Pcalc[k][startN:endN,startM:endM,0,0])**2 ) ) )
        errorsLtwo[3].append( np.sqrt( np.sum( (P[k][startN:endN,startM:endM,1,0] - Pcalc[k][startN:endN,startM:endM,1,0])**2 ) )*mydict['pdict']['gridspc'] )
        errorsLtwo[4].append( np.sqrt( np.sum( (P[k][startN:endN,startM:endM,0,1] - Pcalc[k][startN:endN,startM:endM,0,1])**2 ) )*mydict['pdict']['gridspc'] )
        errorsLtwo[5].append( np.sqrt( np.sum( (P[k][startN:endN,startM:endM,1,1] - Pcalc[k][startN:endN,startM:endM,1,1])**2 ) ) / np.sqrt( np.sum( (Pcalc[k][startN:endN,startM:endM,1,1])**2 ) ) )
        errorsLinf[6].append( np.max(np.abs( (S[k][startN:endN,startM:endM,0,0] - Scalc[k][startN:endN,startM:endM,0,0]) / Scalc[k][startN:endN,startM:endM,0,0]) ) )
        errorsLinf[7].append( np.max(np.abs(S[k][startN:endN,startM:endM,1,0] - Scalc[k][startN:endN,startM:endM,1,0])) )
        errorsLinf[8].append( np.max(np.abs(S[k][startN:endN,startM:endM,0,1] - Scalc[k][startN:endN,startM:endM,0,1])) )
        errorsLinf[9].append( np.max(np.abs( (S[k][startN:endN,startM:endM,1,1] - Scalc[k][startN:endN,startM:endM,1,1]) / Scalc[k][startN:endN,startM:endM,1,1]) ) )
        errorsLtwo[6].append( np.sqrt( np.sum( (S[k][startN:endN,startM:endM,0,0] - Scalc[k][startN:endN,startM:endM,0,0])**2 ) ) / np.sqrt( np.sum( (Scalc[k][startN:endN,startM:endM,0,0])**2 ) ) )
        errorsLtwo[7].append( np.sqrt( np.sum( (S[k][startN:endN,startM:endM,1,0] - Scalc[k][startN:endN,startM:endM,1,0])**2 ) ) ) #The Eulerian domain is changing in time, so the grid spacing is too. Need to figure out what it is.
        errorsLtwo[8].append( np.sqrt( np.sum( (S[k][startN:endN,startM:endM,0,1] - Scalc[k][startN:endN,startM:endM,0,1])**2 ) ) ) #The Eulerian domain is changing in time, so the grid spacing is too. Need to figure out what it is.
        errorsLtwo[9].append( np.sqrt( np.sum( (S[k][startN:endN,startM:endM,1,1] - Scalc[k][startN:endN,startM:endM,1,1])**2 ) ) / np.sqrt( np.sum( (Scalc[k][startN:endN,startM:endM,1,1])**2 ) ) )
#        N = F[k].shape[0]
#        M = F[k].shape[1]
#        temp1=np.zeros((N,M,2,2))
#        temp2=np.zeros((N,M,2,2))
#        for j in range(N):
#            for m in range(M):
#                temp1[j,m,:,:] = F[k][j,m,:,:] - Fcalc[k]
#                temp2[j,m,:,:] = Finv[k][j,m,:,:] - Finvcalc[k]
#        errorsLinf[10].append( np.max(np.abs(temp1[:,:,0,0])) / np.abs(Fcalc[k][0,0]) )
#        errorsLinf[11].append( np.max(np.abs(temp1[:,:,1,0])))
#        errorsLinf[12].append( np.max(np.abs(temp1[:,:,0,1])))
#        errorsLinf[13].append( np.max(np.abs(temp1[:,:,1,1])) / np.abs(Fcalc[k][1,1]) )
#        errorsLtwo[10].append( np.sqrt( np.sum( (temp1[:,:,0,0])**2 ) ) / np.sqrt( N*M*(Fcalc[k][0,0])**2  ) )
#        errorsLtwo[11].append( np.sqrt( np.sum( (temp1[:,:,1,0])**2 ) )*mydict['pdict']['gridspc'])
#        errorsLtwo[12].append( np.sqrt( np.sum( (temp1[:,:,0,1])**2 ) )*mydict['pdict']['gridspc'])
#        errorsLtwo[13].append( np.sqrt( np.sum( (temp1[:,:,1,1])**2 ) ) / np.sqrt( N*M*(Fcalc[k][1,1])**2  ) )
#        errorsLinf[14].append( np.max(np.abs(temp2[:,:,0,0])) / np.abs(Finvcalc[k][0,0]) )
#        errorsLinf[15].append( np.max(np.abs(temp2[:,:,1,0])))
#        errorsLinf[16].append( np.max(np.abs(temp2[:,:,0,1])))
#        errorsLinf[17].append( np.max(np.abs(temp2[:,:,1,1])) / np.abs(Finvcalc[k][1,1]) )
#        errorsLtwo[14].append( np.sqrt( np.sum( (temp2[:,:,0,0])**2 ) ) / np.sqrt( N*M*(Finvcalc[k][0,0])**2  ) )
#        errorsLtwo[15].append( np.sqrt( np.sum( (temp2[:,:,1,0])**2 ) )*mydict['pdict']['gridspc'])
#        errorsLtwo[16].append( np.sqrt( np.sum( (temp2[:,:,0,1])**2 ) )*mydict['pdict']['gridspc'])
#        errorsLtwo[17].append( np.sqrt( np.sum( (temp2[:,:,1,1])**2 ) ) / np.sqrt( N*M*(Finvcalc[k][1,1])**2  ) )        
    return errorsLinf,errorsLtwo,mydict 

def calcerrsonepoint(l,P,S,lcalc, Pcalc, Scalc,mydict):
    '''Calculate the error between exact and approximate solutions at a single point'''
    pind = [0,0]
    errors = [[] for k in range(6)]
    for k in range(len(mydict['t'])):
        errors[0].append( np.abs(l[k][pind[0],pind[1],0] -   lcalc[k][pind[0],pind[1],0])   / (np.abs(lcalc[k][pind[0],pind[1],0]  )) )
        errors[1].append( np.abs(l[k][pind[0],pind[1],1] -   lcalc[k][pind[0],pind[1],1])   / (np.abs(lcalc[k][pind[0],pind[1],1]  )) )
        errors[2].append( np.abs(P[k][pind[0],pind[1],0,0] - Pcalc[k][pind[0],pind[1],0,0]) / (np.abs(Pcalc[k][pind[0],pind[1],0,0])) )
        errors[3].append( np.abs(P[k][pind[0],pind[1],1,1] - Pcalc[k][pind[0],pind[1],1,1]) / (np.abs(Pcalc[k][pind[0],pind[1],1,1])) )
        errors[4].append( np.abs(S[k][pind[0],pind[1],0,0] - Scalc[k][pind[0],pind[1],0,0]) / (np.abs(Scalc[k][pind[0],pind[1],0,0])) )
        errors[5].append( np.abs(S[k][pind[0],pind[1],1,1] - Scalc[k][pind[0],pind[1],1,1]) / (np.abs(Scalc[k][pind[0],pind[1],1,1])) )        
    return errors,mydict 


def getErrors(basedir,basenamelist,soln='ygrad',xvals=None,xlab=None,leglist=['x','y','$P_{11}$','$P_{12}$','$P_{21}$','$P_{22}$','$S_{11}$','$S_{12}$','$S_{21}$','$S_{22}$']):
    errorsLinf=[]
    Pnum = []
    Snum = []
    if soln == 'rest':
        solnhandle = knownsolutioninitialrest
    elif soln == 'ygrad':
        solnhandle = knownsolutioninitialygrad
    else:
        print('Solution type not recognized.')
        raise(SystemExit)
#    errorsLtwo=[]
#    errorsonept=[]
    for basename in basenamelist:
        print(basename)
        l, P, S, F, Finv, mydict = simresults(basename, basedir)
        Pnum.append(P)
        Snum.append(S)
        lcalc, Pcalc, Scalc, Fcalc, Finvcalc = solnhandle(mydict['pdict']['U'], mydict['pdict']['Wi'], l, mydict['t'])
        eLinf, eLtwo, mydict = calcerrs(l, P, S, F, Finv, lcalc, Pcalc, Scalc, Fcalc, Finvcalc, mydict)
        vizTEF.stressComponents(P, S, mydict, basename,basedir)
        errtitlestr = '$L_\infty$ error'
        vizTEF.plotErrs(eLinf, mydict, basename, basedir,'relerrors_Linf',errtitlestr)
        errorsLinf.append(eLinf)
#        errorsLtwo.append(eLtwo)
#        err, mydict = calcerrsonepoint(l, P, S, lcalc, Pcalc, Scalc, mydict)
#        errorsonept.append(err)
    errsinf=np.asarray(errorsLinf)
    if xvals != None:
        timeind = -1
        timestr = '%03d' % int(mydict['t'][timeind])
        errfname = 'maxerrs_time' + timestr
        errtitlestr = errtitlestr + ' at time = ' + timestr
        vizTEF.compareErrs(errsinf,xvals,xlab,basedir,errfname,errtitlestr,timeind,timestr,leglist)
    return errsinf,Pnum,Snum


if __name__ == '__main__':
    basedir = '/Volumes/ExtMacBree/VEsims/ExactExtensionalFlow/'
    basenamelist = ['ext_initrest_PtinC_noregrid_rtol%02d_eps200_N020_Wi01_Time03' % k for k in [6,5,4,3] ]
    soln='rest'
    xvals = [1.e-6,1.e-5,1.e-4,1.e-3]
    xlab = 'relative tolerance'
    errsinf,Pnum,Snum=getErrors(basedir,basenamelist,soln,xvals,xlab)
    
    
    basedir = '/Volumes/ExtMacBree/VEsims/ExactExtensionalFlow/'
    basenamelist = ['ext_initygrad_PtinC_noregrid_rtol%02d_eps200_N020_Wi01_Time03' % k for k in [6,5,4,3] ]
    soln='ygrad'
    xvals = [1.e-6,1.e-5,1.e-4,1.e-3]
    xlab = 'relative tolerance'
    errsinf,Pnum,Snum=getErrors(basedir,basenamelist,soln,xvals,xlab)
    
    
    
    
    
    
    
    
    
    
    
    
    
        