import numpy as nm
import sys

def vectorGrad(vec,gridspc,N,M):
    '''
    vec is a 3D numpy array with first index = i (x-coord), second index = j (y-coord), 
    third index = vector component. gridspc is the grid spacing in both x and y. N is 
    the number of points in the x direction, M is the number in the y direction. The 
    output is a 4D numpy array (shape = (N,M,2,2)) containing a discrete center difference 
    approximation to the vector gradient, with one-sided derivatives at the boundaries.
    
    '''
    if N != vec.shape[0] or M != vec.shape[1] or vec.shape[2] != 2:
        print('Shape mismatch. Aborting')
    out = nm.zeros((N,M,2,2))
    #domain center
    dxm1 = vec[:N-2,:,:] 
    dxp1 = vec[2:,:,:]
    dym1 = vec[:,:M-2,:] 
    dyp1 = vec[:,2:,:]

    out[1:N-1,:,0,0] = dxp1[:,:,0] - dxm1[:,:,0]
    out[1:N-1,:,1,0] = dxp1[:,:,1] - dxm1[:,:,1]
    out[:,1:M-1,0,1] = dyp1[:,:,0] - dym1[:,:,0]
    out[:,1:M-1,1,1] = dyp1[:,:,1] - dym1[:,:,1]
    #domain edges, order h^2
    out[0,:,0,0] += (-3*vec[0,:,0] + 4*vec[1,:,0] - vec[2,:,0])
    out[N-1,:,0,0] += (3*vec[N-1,:,0] - 4*vec[N-2,:,0] + vec[N-3,:,0]) 
    out[0,:,1,0] += (-3*vec[0,:,1] + 4*vec[1,:,1] - vec[2,:,1]) 
    out[N-1,:,1,0] += (3*vec[N-1,:,1] - 4*vec[N-2,:,1] + vec[N-3,:,1]) 
    out[:,0,0,1] += (-3*vec[:,0,0] + 4*vec[:,1,0] - vec[:,2,0]) 
    out[:,M-1,0,1] += (3*vec[:,M-1,0] - 4*vec[:,M-2,0] + vec[:,M-3,0]) 
    out[:,0,1,1] += (-3*vec[:,0,1] + 4*vec[:,1,1] - vec[:,2,1]) 
    out[:,M-1,1,1] += (3*vec[:,M-1,1] - 4*vec[:,M-2,1] + vec[:,M-3,1])
    #scale by grid spacing
    out = out/(2*gridspc)
    return out
    
def tensorDiv(tensor,gridspc,N,M):
    '''tensor is a 4D numpy array with first index = i (x-coord), second index = j (y-coord), third index = tensor row component, fourth index = tensor column component. gridspc is the grid spacing in both x and y. N is the number of points in the x direction, M is the number in the y direction. The output is a 3D numpy array containing a discrete center difference approximation to the divergence, with one-sided derivatives at the boundaries.'''
    if N != tensor.shape[0] or M != tensor.shape[1] or tensor.shape[2] != 2 or tensor.shape[3] != 2:
        print('Shape mismatch. Aborting')
    out = nm.zeros((N,M,2))
    #domain center, center difference derivatives
    dxm1 = tensor[:N-2,:,:,:] 
    dxp1 = tensor[2:,:,:,:]
    dym1 = tensor[:,:M-2,:,:] 
    dyp1 = tensor[:,2:,:,:]
    out[1:N-1,:,0] = dxp1[:,:,0,0] - dxm1[:,:,0,0] 
    out[:,1:M-1,0] = out[:,1:M-1,0] + dyp1[:,:,0,1] - dym1[:,:,0,1]
    out[1:N-1,:,1] = dxp1[:,:,1,0] - dxm1[:,:,1,0] 
    out[:,1:M-1,1] = out[:,1:M-1,1] + dyp1[:,:,1,1] - dym1[:,:,1,1]
    #domain edges, second order one-sided
    out[0,:,0] += (-3*tensor[0,:,0,0] + 4*tensor[1,:,0,0] - tensor[2,:,0,0])
    out[N-1,:,0] += (3*tensor[N-1,:,0,0] - 4*tensor[N-2,:,0,0] + tensor[N-3,:,0,0])
    out[0,:,1] += (-3*tensor[0,:,1,0] + 4*tensor[1,:,1,0] - tensor[2,:,1,0])
    out[N-1,:,1] += (3*tensor[N-1,:,1,0] - 4*tensor[N-2,:,1,0] + tensor[N-3,:,1,0])
    out[:,0,0] += (-3*tensor[:,0,0,1] + 4*tensor[:,1,0,1] - tensor[:,2,0,1])
    out[:,M-1,0] += (3*tensor[:,M-1,0,1] - 4*tensor[:,M-2,0,1] + tensor[:,M-3,0,1])
    out[:,0,1] += (-3*tensor[:,0,1,1] + 4*tensor[:,1,1,1] - tensor[:,2,1,1])
    out[:,M-1,1] += (3*tensor[:,M-1,1,1] - 4*tensor[:,M-2,1,1] + tensor[:,M-3,1,1])    
    #scale by grid spacing
    out = out/(2*gridspc)
    return out
    
def profileme():
    gridspc = 0.1
    N = 1000
    M = 1000
    l = nm.zeros((N,M,2))
    x = nm.zeros((N,1))
    x[:,0] = nm.linspace(0+gridspc/2,0+N*gridspc-gridspc/2,N)
    l[:,:,0] = nm.tile(x,(1,M))
    y = nm.linspace(1+gridspc/2,1+M*gridspc-gridspc/2,M)
    l[:,:,1] = nm.tile(y,(N,1))
        
if __name__ == '__main__':
    import Gridding
    gridspc = 0.2
    N = 6
    M = 11
    l = Gridding.makeGrid(N,M,gridspc)
    
    #test vectorGrad
    f = nm.zeros(l.shape)
    f[:,:,0] = l[:,:,0]**2 + l[:,:,1]**2
    f[:,:,1] = 2*l[:,:,0]
    gfa = vectorGrad(f,gridspc,N,M)
    gf = nm.zeros(gfa.shape)
    gf[:,:,0,0] = 2*l[:,:,0]
    gf[:,:,0,1] = 2*l[:,:,1]
    gf[:,:,1,0] = 2*nm.ones((N,M))
    gf[:,:,1,1] = nm.zeros((N,M))
    print('gfa[N-1,0,:,:]')
    print(gfa[ N-1,0,:,:])
    print('gf[ N-1,0,:,:]')
    print(gf[  N-1,0,:,:])
    print(nm.max(nm.abs(gfa-gf)))
    
    # #test tensorDiv
    # F = nm.zeros((N,M,2,2))
    # F[:,:,0,0] = l[:,:,0]**2 + l[:,:,1]
    # F[:,:,1,0] = 2*l[:,:,0]
    # F[:,:,0,1] = l[:,:,1]
    # F[:,:,1,1] = l[:,:,0] + l[:,:,1]**2
    # dFa = tensorDiv(F,gridspc,N,M)
    # dF = nm.zeros(dFa.shape)
    # dF[:,:,0] = 2*l[:,:,0] + 1
    # dF[:,:,1] = 2*l[:,:,1] + 2
    # print('dFa[0,1,:]')
    # print(dFa[ 0,1,:])
    # print('dF[ 0,1,:]')
    # print(dF[  0,1,:])
    # print(nm.max(nm.abs(dFa-dF)))
    # print(nm.abs(dFa-dF))
    
    
    
