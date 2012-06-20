import numpy as np

def makeGridCenter(N,M,h,origin=(0,0)):
    '''
    Constructs a rectangular cell-centered grid.
    N is the number of points in the x direction, M is the number of points in y direction, 
    h is the uniform point separation in both x and y directions. Optional argument origin 
    gives the lower left corner of the domain. Output is three dimensional array, with
    x = l0[:,:,0] and y = l0[:,:,1]. So x_ij = l0[i,j,0], y_ij = l0[i,j,1]. 
    
    '''
    l0 = np.zeros((N,M,2))
    x = np.zeros((N,1))
    x[:,0] = np.arange(N,dtype=np.float64)
    x = x*h + origin[0]+h/2.
    l0[:,:,0] = np.tile(x,(1,M))
    y = np.arange(M,dtype=np.float64)
    y = y*h + origin[1]+h/2.
    l0[:,:,1] = np.tile(y,(N,1))
    return l0
    

def makeGridNode(N,M,h,origin=(0,0)):
    '''
    Constructs a rectangular node-centered grid.
    N is the number of points in the x direction, M is the number of points in y direction, 
    h is the uniform point separation in both x and y directions. Optional argument origin 
    gives the lower left corner of the domain. Output is three dimensional array, with
    x = l0[:,:,0] and y = l0[:,:,1]. So x_ij = l0[i,j,0], y_ij = l0[i,j,1]. 
    
    '''
    l0 = np.zeros((N,M,2))
    x = np.zeros((N,1))
    x[:,0] = np.arange(N,dtype=np.float64)
    x = x*h + origin[0]
    l0[:,:,0] = np.tile(x,(1,M))
    y = np.arange(M,dtype=np.float64)
    y = y*h + origin[1]
    l0[:,:,1] = np.tile(y,(N,1))
    return l0


def makeNewGrid(lold,h,scalefactor):
    '''
    Make a new cell-centered grid with the same spacing as the old (undistorted) grid. 
    Beginning of the regridding process.
    '''
    xold = lold[:,:,0]
    yold = lold[:,:,1]
    pad = 2*scalefactor
    xmin = np.min(xold)-pad*h
    xmax = np.max(xold)+pad*h
    ymin = np.min(yold)-pad*h
    ymax = np.max(yold)+pad*h
    Nn = np.ceil((xmax-xmin)/h)
    Mn = np.ceil((ymax-ymin)/h)
    l = makeGridCenter(Nn,Mn,h,(xmin,ymin))
    return l

def findDists(lold,l,N,M,h,scalefactor):
    '''
    Locate the patch of points on the new grid associated with each point on the 
    old grid. Find distance in units of 'h' (grid spacing) to each of these new 
    points in the x and y directions independently.
    '''
    xvals = l[:,0,0]
    yvals = l[0,:,1]
    xmin=xvals[0]
    ymin=yvals[0]
    rx = np.zeros((N,M,4*scalefactor))
    ry = np.zeros((N,M,4*scalefactor))
    xind = np.zeros((N,M,4*scalefactor),dtype=np.int32)
    yind = np.zeros((N,M,4*scalefactor),dtype=np.int32)
    for i in range(N):
        for j in range(M):
            i0 = np.fix((lold[i,j,0]-xmin)/h)
            j0 = np.fix((lold[i,j,1]-ymin)/h)
            xind[i,j,:] = i0 + range(-2*scalefactor+1,2*scalefactor+1)
            yind[i,j,:] = j0 + range(-2*scalefactor+1,2*scalefactor+1)
            rx[i,j,:] = np.abs(xvals[xind[i,j,:]]-lold[i,j,0])/h
            ry[i,j,:] = np.abs(yvals[yind[i,j,:]]-lold[i,j,1])/h
    return rx,ry,xind,yind

def makeWeights(r,scalefactor):
    '''
    Create the weights required for each nearby point.
	r is a vector containing the distances between points.
	scalefactor is an integer scaling factor that spreads the value 
	to 2*scalefactor grid points on all sides.
    '''
    rk=r/scalefactor
    beta=1./scalefactor
    wd=np.zeros((len(rk),2))
    wd[:,0] = beta*(1. - 5./2.*rk**2 + 3./2.*rk**3)
    wd[:,1] = beta*(1./2.*(2.-rk)**2 * (1.-rk))
    w = wd[:,0]*np.int_((r>=0.)*(r<scalefactor)) + wd[:,1]*np.int_((r>=scalefactor)*(r<2*scalefactor))
    return w

def interp2NewGrid(lold,Fold,h,stress,scalefactor=1,addpts=1):
    '''
    Interpolate a matrix valued function Fold (4D array) from an old grid 
    lold with original (undistorted) spacing of h to a new regular grid with the same 
    spacing. The number of points in the grid can change, but the spacing won't.
    The flag stress = 1 means to use the identity matrix as the default outside the
    domain, instead of zeros. scalefactor (integer) tells by what factor to stretch 
    the initial 4 point interpolation (e.g., scalefactor = 2 gives an 8 point interp).
    addpts = 1 means add 2*scalefactor points along all the edges. addpts = 0 means
    use the extra points along the edges in the calculation, but don't return them
    as part of the interpolation. This means that the matrices will stay the same 
    size.
    
    ''' 
    l = makeNewGrid(lold,h,scalefactor)
    Nnew=l.shape[0]
    Mnew=l.shape[1]
    N=lold.shape[0]
    M=lold.shape[1]
    F = np.zeros((Nnew,Mnew,Fold.shape[2],Fold.shape[3]))
    d = Fold.copy()
    if stress: # the stress matrix should be the identity when relaxed; only distribute excess/deficit stress
        F[:,:,0,0] = 1
        F[:,:,1,1] = 1
        d[:,:,0,0] = Fold[:,:,0,0]-1
        d[:,:,1,1] = Fold[:,:,1,1]-1
    rx,ry,xind,yind = findDists(lold,l,N,M,h,scalefactor)
    for i in range(N):
        for j in range(M):
            dX = makeWeights(rx[i,j,:],scalefactor)
            dY = makeWeights(ry[i,j,:],scalefactor)
            mypatch=np.outer(dX,dY)
            for k in range(Fold.shape[2]):
                for m in range(Fold.shape[3]):
                    F[xind[i,j,0]:xind[i,j,-1]+1,yind[i,j,0]:yind[i,j,-1]+1,k,m] += d[i,j,k,m]*mypatch  
    if not addpts: 
        Nd = Nnew-N
        Md = Mnew-M
        Nnew = N
        Mnew = M
        if stress:
            F,l = changeSize(1,0,0,1,Nd,Md,F,l)
        else:
            F,l = changeSize(0,0,0,0,Nd,Md,F,l)
    #test that mass is conserved
    dtot=[]
    Ftot=[]
    for k in range(Fold.shape[2]):
        for m in range(Fold.shape[3]):
            dtot.append(np.sum(d[:,:,k,m]))
            Ftot.append(np.sum(F[:,:,k,m])-(k==m)*F.shape[0]*F.shape[1])  #subtract off the diagonal because the rest state is the identity
    dF = [np.abs((dtot[k]-Ftot[k])/dtot[k]) for k in range(len(dtot))]
    if max(dF) > 1.e-8:  
        print('Interpolation not conserving mass in [S11, S12, S21, S22]')   
        print('Relative difference in total summation over all points:', dF)
        print('Original total:', dtot)                  
        print('New total:', Ftot)                  
    return l,F,Nnew,Mnew

def changeSize(a,b,c,d,Nd,Md,F,l):
    F,l = droprows(a,b,c,d,Nd,F,l)
    F,l = dropcolumns(a,b,c,d,Md,F,l)
    return F, l 

def droprows(a,b,c,d,Nd,F,l):
    if Nd > 0:
        boollist = []
        boollist.append(np.max(np.abs(F[0,:,0,0]-a)) > np.max(np.abs(F[-1,:,0,0]-a)))
        boollist.append(np.max(np.abs(F[0,:,0,1]-b)) > np.max(np.abs(F[-1,:,0,1]-b)))
        boollist.append(np.max(np.abs(F[0,:,1,0]-c)) > np.max(np.abs(F[-1,:,1,0]-c)))
        boollist.append(np.max(np.abs(F[0,:,1,1]-d)) > np.max(np.abs(F[-1,:,1,1]-d)))
        if sum(boollist) < 2:
            F = F[1:,:,:,:]
            l = l[1:,:,:]
        else:
            F = F[:-1,:,:,:]
            l = l[:-1,:,:]
        Nd -= 1
        return droprows(a,b,c,d,Nd,F,l)
    else:
        return F,l

def dropcolumns(a,b,c,d,Md,F,l):
    if Md > 0:
        boollist = []
        boollist.append(np.max(np.abs(F[:,0,0,0]-a)) > np.max(np.abs(F[:,-1,0,0]-a)))
        boollist.append(np.max(np.abs(F[:,0,0,1]-b)) > np.max(np.abs(F[:,-1,0,1]-b)))
        boollist.append(np.max(np.abs(F[:,0,1,0]-c)) > np.max(np.abs(F[:,-1,1,0]-c)))
        boollist.append(np.max(np.abs(F[:,0,1,1]-d)) > np.max(np.abs(F[:,-1,1,1]-d)))
        if sum(boollist) < 2:
            F = F[:,1:,:,:]
            l = l[:,1:,:]
        else:
            F = F[:,:-1,:,:]
            l = l[:,:-1,:]
        Md -= 1
        return dropcolumns(a,b,c,d,Md,F,l)
    else:
        return F,l
        
def testRegrid():
    endpt = 2.0
    N=40
    h=endpt/40
    ltemp=makeGridCenter(N,N,h,origin=(0,0))
    lold = np.zeros(ltemp.shape)
#    # do random perturbations
#    eps = h*0.5
#    r = -eps+2*eps*np.random.rand(*lold.shape)
#    lold=ltemp+r

    # rotate grid
    th = 0.533  
    lold[:,:,0] = ltemp[:,:,0]*np.cos(th) -  ltemp[:,:,1]*np.sin(th)
    lold[:,:,1] = ltemp[:,:,0]*np.sin(th) +  ltemp[:,:,1]*np.cos(th)

    import matplotlib.pyplot as plt
    import os
    
    Fold1 = np.exp(-((lold[:,:,0]-1.)**2+(lold[:,:,1]-1)**2)/2)*np.cos(lold[:,:,0]-1)*np.cos(lold[:,:,1]-1) 
    Fold = np.zeros((lold.shape[0],lold.shape[1],2,2))
    Fold[:,:,0,0] = Fold1
    Fold[:,:,1,1] = lold[:,:,1]
    Gold = np.zeros((lold.shape[0],lold.shape[1],2,2))
    Gold[:,:,0,0] = 1.1
    Gold[:,:,1,1] = 0.9
    plt.close()
    plt.pcolor(lold[:,:,0],lold[:,:,1],Fold[:,:,0,0])
    plt.savefig(os.path.expanduser('~/scratch/gridfunctionrotate0'))
    plt.close()
    plt.pcolor(lold[:,:,0],lold[:,:,1],Fold[:,:,1,1])
    plt.savefig(os.path.expanduser('~/scratch/gridfunctionrotate1'))
    plt.close()
    plt.pcolor(lold[:,:,0],lold[:,:,1],Gold[:,:,0,0])
    plt.savefig(os.path.expanduser('~/scratch/gridfunctionconstant0'))
    plt.close()
    plt.pcolor(lold[:,:,0],lold[:,:,1],Gold[:,:,1,1])
    plt.savefig(os.path.expanduser('~/scratch/gridfunctionconstant1'))
    
    for scalefactor in [1,2,4]:
        l,F,N,M = interp2NewGrid(lold,Fold,h,0,scalefactor)
        plt.close()
        ph=plt.pcolor(l[:,:,0],l[:,:,1],F[:,:,0,0])
        plt.colorbar(ph)
        plt.savefig(os.path.expanduser('~/scratch/gridfuncrotate0_regrid%d' % scalefactor))
        plt.close()
        ph=plt.pcolor(l[:,:,0],l[:,:,1],F[:,:,1,1])
        plt.colorbar(ph)
        plt.savefig(os.path.expanduser('~/scratch/gridfuncrotate1_regrid%d' % scalefactor))

        l,G,N,M = interp2NewGrid(lold,Gold,h,1,scalefactor)
        plt.close()
        ph=plt.pcolor(l[:,:,0],l[:,:,1],G[:,:,0,0])
        plt.colorbar(ph)
        plt.savefig(os.path.expanduser('~/scratch/gridfuncconstant0_regrid%d' % scalefactor))
        plt.close()
        ph=plt.pcolor(l[:,:,0],l[:,:,1],G[:,:,1,1])
        plt.colorbar(ph)
        plt.savefig(os.path.expanduser('~/scratch/gridfuncconstant1_regrid%d' % scalefactor))
        
#        tot=np.sum(F)
#        totregrid=np.sum(Fold)
#        print(tot)
#        print(totregrid)
#        
#        plt.close()
#        plt.plot(lold[:,:,0],lold[:,:,1],'k.')
#        plt.plot(l[:,:,0],l[:,:,1],'r.')
#        plt.savefig(os.path.expanduser('~/scratch/gridrotate%d' % scalefactor))

    
    
    
def CCSArtShowPic():
    endpt = 2.0
    N=40
    h=endpt/40
    lold=makeGridCenter(N,N,h,origin=(0,0))
    eps = h*2.5
    r = -eps+2*eps*np.random.rand(*lold.shape)
    lold=lold+r
    import matplotlib.pyplot as plt
    import os
    Fold = np.exp(-((lold[:,:,0]-1.)**2+(lold[:,:,1]-1)**2)/2)*np.cos(lold[:,:,0]-1)*np.cos(lold[:,:,1]-1) 
    Fold = Fold[:,:,np.newaxis]
    plt.pcolor(lold[:,:,0],lold[:,:,1],Fold[:,:,0],cmap='gray') #add edgecolor='k' to get black edges
    plt.xlim([-0.5,2.5])
    plt.ylim([-0.5,2.5])
    plt.axis('off')
    plt.savefig(os.path.expanduser('~/scratch/gridfunctionArtShow.pdf'),transparent='True')

   
    
if __name__ == '__main__':
    testRegrid()
    # CCSArtShowPic()
