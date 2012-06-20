#import python modules
import numpy as np

def FourRollMill(pdict,l2):
    x = l2[:,0]
    y = l2[:,1]
    u = np.zeros((len(x)*2))
    u[::2] = pdict['U']*np.sin(x)*np.cos(y)
    u[1::2] = -pdict['U']*np.cos(x)*np.sin(y)
    ugrad = np.zeros((len(x),2,2))
    ugrad[:,0,0] = pdict['U']*np.cos(x)*np.cos(y)
    ugrad[:,0,1] = -pdict['U']*np.sin(x)*np.sin(y)
    ugrad[:,1,0] = pdict['U']*np.sin(x)*np.sin(y)
    ugrad[:,1,1] = -pdict['U']*np.cos(x)*np.cos(y)
    return u, ugrad

def Extension(pdict,l2):
    u = np.zeros((l2.shape[0]*2,))
    u[::2] = pdict['U']*l2[:,0]
    u[1::2] = -pdict['U']*l2[:,1]
    ugrad = np.zeros((l2.shape[0],2,2))
    ugrad[:,0,0] = pdict['U']
    ugrad[:,1,1] = -pdict['U']
    return u, ugrad

def ParabolicShear(pdict,l2):
    u = np.zeros(l2.shape)
    y = l2[:,1].copy()
    inds = np.nonzero(y < 0)
    u[:,0] = pdict['U']*y*(1-y) # y >= 0
    u[inds,0] = pdict['U']*y[inds]*(1+y[inds]) # y < 0
    u = u.flatten()
    ugrad = np.zeros((l2.shape[0],2,2))
    ugrad[:,0,1] = 1 - 2*y # y >= 0
    ugrad[inds,0,1] = 1 + 2*y[inds] # y < 0
    return u, ugrad

def regDipole(pdict,l2):
    ''' 
    dipolef is the strength and dipolex0 the location of the dipole.
    
    '''
    u = np.zeros(l2.shape)
    dx = l2[:,0] - pdict['dipolex0'][0] 
    dy = l2[:,1] - pdict['dipolex0'][1] 
    eps = 5*pdict['eps']
    r2 = dx**2 + dy**2 + eps**2
    D1 = (2*eps**2 - r2) / r2**2
    D2 = 2/r2**2
    f1 = pdict['dipolef'][0] 
    f2 = pdict['dipolef'][1]
    fdotx = f1*dx + f2*dy
    u[:,0] = ( f1*D1 + fdotx*dx*D2 ) / (2*np.pi*pdict['mu'])
    u[:,1] = ( f2*D1 + fdotx*dy*D2 ) / (2*np.pi*pdict['mu'])
    u = u.flatten()
    ugrad = np.zeros((l2.shape[0],2,2))
    denom = np.pi*pdict['mu']*r2**3
    ugrad[:,0,0] = ( f1*dx*(4*dy**2 -   r2) + f2*dy*(-4*dx**2 + r2) ) / denom
    ugrad[:,0,1] = ( f1*dy*(4*dy**2 - 3*r2) + f2*dx*(-4*dy**2 + r2) ) / denom
    ugrad[:,1,0] = ( f2*dx*(4*dx**2 - 3*r2) + f1*dy*(-4*dx**2 + r2) ) / denom
    ugrad[:,1,1] = ( f2*dy*(4*dx**2 -   r2) + f1*dx*(-4*dy**2 + r2) ) / denom
    return u, ugrad

def circleFlow(pdict,l2):
    x = l2[:,0]
    y = l2[:,1]
    u = np.zeros((l2.shape[0]*2,))
    r2 = x**2 + y**2
    u[::2] = -pdict['U']*y*np.exp(-r2)
    u[1::2] = pdict['U']*x*np.exp(-r2)
    ugrad = np.zeros((l2.shape[0],2,2))
    ugrad[:,0,0] = pdict['U']*np.exp(-r2)*2*x*y
    ugrad[:,0,1] = pdict['U']*np.exp(-r2)*(-1+2*y**2)
    ugrad[:,1,0] = pdict['U']*np.exp(-r2)*(1-2*x**2)
    ugrad[:,1,1] = pdict['U']*np.exp(-r2)*(-2*x*y)
    return u, ugrad
       
