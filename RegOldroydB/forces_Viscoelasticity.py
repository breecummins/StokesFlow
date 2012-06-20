#import python modules
import numpy as np

def calcForcesSimpleSpring(fpts,**kwargs):
    '''
    current position 'fpts' should be a length 2 vector. 
    dict kwargs has 'xr' (the rest length) and 'K' (scalar 
    spring constant acting along the vector connecting
    fpts to the origin). 
    
    '''
    return -kwargs['K']*(fpts - kwargs['xr'])       
    
def calcForcesTwoHeadSpring(fpts,**kwargs):
    '''
    current position 'fpts' should be a 2 x 2 matrix (one row vector containing 
    x and y components for each bead). dict kwargs has scalar 'xr' (the rest 
    length between beads) and 'K' (scalar spring constant acting along the vector 
    connecting the beads). 
    
    '''
    distvec = fpts[0,:] - fpts[1,:]
    dist = np.sqrt(np.sum(distvec**2))
    g = -kwargs['K']*(1 - kwargs['xr']/dist)*distvec
    return np.row_stack([g,-g])
    
def calcForcesGravity(fpts,**kwargs):
    '''
    current position fpts should be an n x 2 matrix (one row vector containing 
    x and y components for each bead). dict kwargs has 'K' (scalar gravity
    force) and 'n' (the number of the force points that are 
    actually carrying gravity (the rest are marker points)).
    
    '''
    g = np.zeros(fpts.shape)
    g[:kwargs['n'],1] = -kwargs['K']
    return g
    
def calcForcesQuadraticSpring(fpts,**kwargs):
    '''
    current position fpts should be a 2 x 2 matrix (one row vector containing 
    x and y components for each bead). dict kwargs has scalar 'xr' (the rest 
    length between beads) and 'K' (scalar spring constant acting along the vector 
    connecting the beads). 
    
    '''
    distvec = fpts[0,:] - fpts[1,:]
    dist = np.sqrt(np.sum(distvec**2))
    g = -kwargs['K']*(dist/kwargs['xr'] - 1)*distvec
    return np.row_stack([g,-g])

def calcForcesConstant(fpts,**kwargs):
    '''
    current position fpts should be an N x 2 matrix (one row vector containing 
    x and y components for each bead). dict kwargs has 'c' (constant force
    for every bead, length 2 vector or list or tuple).
    
    '''
    return np.column_stack([kwargs['c'][0]*np.ones(fpts[:,0].shape),kwargs['c'][1]*np.ones(fpts[:,0].shape)])


def calcForcesSwimmer(fpts,**kwargs):
    '''
    Calculate spring and curvature forces due to a moving swimmer:
    x = s; y = a*s*sin(lam*s - w*t) 
    Current position fpts should be an n x 2 matrix (one row vector containing 
    x and y components for each point on the swimmer). dict kwargs has resting 
    length 'xr' (also the discretization along the swimmer) and spring constant 
    'K', describing springs between adjacent points.
    'a', 'w', 'lam', 'Kcurv', 't' are curvature force parameters. a is the maximum
    amplitude of the sine wave at the tail; w is the angular frequency of the 
    sine wave; lam is the wave number; Kcurv is the curvature stiffness; and t 
    is the current time. 
    
    '''

    dx = fpts[1:,0]-fpts[:-1,0]
    dy = fpts[1:,1]-fpts[:-1,1]

    #spring forces first
    #note that this is linear force density
    sep = np.sqrt(dx**2 + dy**2)
    fx = -kwargs['K']*(sep/kwargs['xr'] - 1)*dx/sep
    fy = -kwargs['K']*(sep/kwargs['xr'] - 1)*dy/sep
    F1 = np.zeros(fpts.shape)
    F1[1:,0] = fx
    F1[1:,1] = fy
    F1[:-1,0] = F1[:-1,0] - fx
    F1[:-1,1] = F1[:-1,1] - fy
    
    #now curvature forces
    #first calculate desired curvature for yt = kwargs['a']*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #a=kwargs['a']
    a = min([kwargs['a'],kwargs['t']*kwargs['a']]) #ramp up to full amplitude    
    Np = fpts.shape[0]
    xt = np.arange(kwargs['xr'],(Np-1.5)*kwargs['xr'],kwargs['xr'])
    curv = -a*kwargs['lam']**2*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t']) + 2*a*kwargs['lam']*np.cos(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #now calculate approximate actual curvature
    numcurv = -( dx[1:]*dy[:-1] - dx[:-1]*dy[1:] )/kwargs['xr']**3
    coeff = kwargs['Kcurv']*(numcurv-curv)/kwargs['xr']**2
    F2 = np.zeros(fpts.shape)
    F2[2:,0] = coeff*dy[:-1]
    F2[2:,1] = -coeff*dx[:-1]
    F2[1:-1,0] = F2[1:-1,0] - coeff*(dy[:-1]+dy[1:])
    F2[1:-1,1] = F2[1:-1,1] + coeff*(dx[:-1]+dx[1:])
    F2[:-2,0] = F2[:-2,0]   + coeff*dy[1:]
    F2[:-2,1] = F2[:-2,1]   - coeff*dx[1:]   
    return F1+F2

def calcForcesSwimmerTFS(fpts,**kwargs):
    '''
    Calculate spring and curvature forces due to a moving swimmer
    with target curvature as in Teran, Fauci, and Shelley.
    Current position fpts should be an n x 2 matrix (one row vector containing 
    x and y components for each point on the swimmer). dict kwargs has resting 
    length 'xr' (also the discretization along the swimmer) and spring constant 
    'K', describing springs between adjacent points.
    'a', 'w', 'lam', 'Kcurv', 't' are curvature force parameters. a is the maximum
    amplitude of the sine wave at the tail; w is the angular frequency of the 
    sine wave; lam is the wave number; Kcurv is the curvature stiffness; and t 
    is the current time. 
    
    '''

    dx = fpts[1:,0]-fpts[:-1,0]
    dy = fpts[1:,1]-fpts[:-1,1]

    #spring forces first
    #note that this is linear force density
    sep = np.sqrt(dx**2 + dy**2)
    fx = -kwargs['K']*(sep/kwargs['xr'] - 1)*dx/sep
    fy = -kwargs['K']*(sep/kwargs['xr'] - 1)*dy/sep
    F1 = np.zeros(fpts.shape)
    F1[1:,0] = fx
    F1[1:,1] = fy
    F1[:-1,0] = F1[:-1,0] - fx
    F1[:-1,1] = F1[:-1,1] - fy
    
    #now curvature forces
    #first calculate desired curvature for yt = kwargs['a']*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #a=kwargs['a']
    a = min([kwargs['a'],kwargs['t']*kwargs['a']]) #ramp up to full amplitude    
    Np = fpts.shape[0]
    xt = np.arange(kwargs['xr'],(Np-1.5)*kwargs['xr'],kwargs['xr'])
    curv = -a*kwargs['lam']**2*(xt-1)*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #now calculate approximate actual curvature
    numcurv = -( dx[1:]*dy[:-1] - dx[:-1]*dy[1:] )/kwargs['xr']**3
    coeff = kwargs['Kcurv']*(numcurv-curv)/kwargs['xr']**2
    F2 = np.zeros(fpts.shape)
    F2[2:,0] = coeff*dy[:-1]
    F2[2:,1] = -coeff*dx[:-1]
    F2[1:-1,0] = F2[1:-1,0] - coeff*(dy[:-1]+dy[1:])
    F2[1:-1,1] = F2[1:-1,1] + coeff*(dx[:-1]+dx[1:])
    F2[:-2,0] = F2[:-2,0]   + coeff*dy[1:]
    F2[:-2,1] = F2[:-2,1]   - coeff*dx[1:]   
    return F1+F2

