import CubicStokeslet2D as cm

print cm.__file__
print dir(cm)

#insert unit tests here
import numpy as np
eps=1.0
mu=1.0

##################################
# test stress derivative
##################################
N=5
Wi = 0.25
gradub=-1 + 2*np.random.random((N,2,2))
gradlt=-10 + 20*np.random.random((N,2,2))
F=-2 + 4*np.random.random((N,2,2))
P=-5 + 10*np.random.random((N,2,2))
out =cm.stressDeriv(Wi,gradub,gradlt,F,P)
print(out)

#python answer
Finv = cm.matinv2x2(F)
Pt = np.zeros((N,2,2))
for j in range(N):
    Pt[j,:,:] = np.dot(gradub[j,:,:],P[j,:,:]) + np.dot(np.dot(gradlt[j,:,:],Finv[j,:,:]),P[j,:,:]) - (1./Wi)*(P[j,:,:] - Finv[j,:,:].transpose())        
print(Pt)

#error
print('Error is ....')
print(np.max(np.abs(Pt-out)))


##################################
## test regular kernel
##################################
#obspts=np.array([[1.,3.],[34.,8.]])
#f=np.random.rand(*obspts.shape)
##pt=pt[np.newaxis,:]
#nodes=np.arange(10,dtype='double').reshape((5,2))
#out =cm.matmult(eps,mu,obspts,nodes,f)
#print(out)
#
##python answer
#output = np.zeros((2*obspts.shape[0],))
#for k in range(obspts.shape[0]):
#    pt = obspts[k,:]
#    pt = obspts[k,:]
#    dif = pt - nodes
#    r2 = (dif**2).sum(1) + eps**2
#    xdiff = dif[:,0]
#    ydiff = dif[:,1]
#    H1 = (2*eps**2/r2 - np.log(r2))/(8*np.pi*mu)
#    H2 = (2/r2)/(8*np.pi*mu)
#    N = nodes.shape[0]
#    row1 = np.zeros((2*N,))
#    row2 = np.zeros((2*N,))
#    ind = 2*np.arange(N) 
#    row1[ind] = (H1 + (xdiff**2)*H2)
#    row1[ind+1] = ((xdiff*ydiff)*H2)
#    row2[ind+1]= (H1 + (ydiff**2)*H2)
#    row2[ind] = row1[ind+1]
#    output[2*k] = (row1*f.flat).sum()
#    output[2*k+1] = (row2*f.flat).sum()
#
#
##error
#print('Error is ....')
#print(np.max(np.abs(output-out)))


# ##############################
# # test derivative kernel
# ##############################
# obspt = -5 + 10*np.random.rand(2)
# print(obspt)
# nodes=-10 + 20*np.random.rand(5,2)
# print(nodes)
# f=np.random.rand(*nodes.shape)
# print(f)
# F = np.array([[1.1,0.2],[0.15,0.95]])
# print(F)
# out =cm.derivop(eps,mu,obspt,nodes,f,F)
# print(out)
# 
# #python output
# output = np.zeros((2,2))
# dif = obspt - nodes
# re2 = (dif**2).sum(1) + eps**2
# h2 = 1/re2
# dh2 = -2/re2**2 #derivative over r
# dh1 = -1/re2 + dh2*eps**2 #derivative over r
# fdotl = (f*dif).sum(1) #dot product for all points
# #transpose matrix products with Jacobian
# Fdx = F[0,0]*dif[:,0] + F[1,0]*dif[:,1]  
# Fdy = F[0,1]*dif[:,0] + F[1,1]*dif[:,1] 
# Fdif = [Fdx,Fdy] 
# Ffx = F[0,0]*f[:,0] + F[1,0]*f[:,1]
# Ffy = F[0,1]*f[:,0] + F[1,1]*f[:,1]
# Ff = [Ffx,Ffy] 
# output = np.zeros((2,2))
# for i in range(2):
#     for k in range(2):
#         val = dh1*f[:,i]*Fdif[k] + dh2*dif[:,i]*fdotl*Fdif[k] + h2*fdotl*F[i,k] + h2*dif[:,i]*Ff[k]
#         output[i,k] = val.sum() #summing over j (l1, l2 coordinates), and summing over all nodes (integration)
# output=output/(4*mu*np.pi)
# #error
# print('Error is ....')
# print(np.max(np.abs(output-out)))
#
###############################
## test derivative kernel
###############################
#obspts = -5 + 10*np.array([[0.5, 7.1], [4.2, 4.6], [3.1, 8.2]])
##print(obspts)
#nodes=-10 + 20*np.array([[1, 14.2], [8.4, 9.2], [6.2, 16.4], [19.8,2.3], [12.0,14.9]])
##print(nodes)
#f=np.array([[1, 14.2], [8.4, 9.2], [6.2, 16.4], [19.8,2.3], [12.0,14.9]])/20.0
##print(f)
#F = np.array([[[1.1,0.2],[0.15,0.95]],[[1.05,0.01],[0.1,0.92]],[[0.9,0.15],[0.2,1.01]]])
##print(F)
#out =cm.derivop(eps,mu,obspts,nodes,f,F)
#print(out)
#
##python output
#output = np.zeros((obspts.shape[0],2,2))
#for k in range(obspts.shape[0]):
#    pt = obspts[k,:]
#    Fh = F[k,:,:]
#    dif = pt - nodes
#    re2 = (dif**2).sum(1) + eps**2
#    h2 = 1/re2
#    dh2 = -2/re2**2 #derivative over r
#    dh1 = -1/re2 + dh2*eps**2 #derivative over r
#    fdotl = (f*dif).sum(1) #dot product for all points
#    #transpose matrix products with Jacobian
#    Fdx = Fh[0,0]*dif[:,0] + Fh[1,0]*dif[:,1]  
#    Fdy = Fh[0,1]*dif[:,0] + Fh[1,1]*dif[:,1] 
#    Fdif = [Fdx,Fdy] 
#    Ffx = Fh[0,0]*f[:,0] + Fh[1,0]*f[:,1]
#    Ffy = Fh[0,1]*f[:,0] + Fh[1,1]*f[:,1]
#    Ff = [Ffx,Ffy] 
#    delv = np.zeros((2,2))
#    for i in range(2):
#        for j in range(2):
#            val = dh1*f[:,i]*Fdif[j] + dh2*dif[:,i]*fdotl*Fdif[j] + h2*fdotl*Fh[i,j] + h2*dif[:,i]*Ff[j]
#            delv[i,j] = val.sum() #summing over j (l1, l2 coordinates), and summing over all nodes (integration)
#    output[k,:,:] = delv/(4*mu*np.pi)
##error
#print(output)
#
#print('Error is ....')
#print(np.max(np.abs(output-out)))
#print('Relative error is ....')
#print(np.max(np.abs(output-out))/np.max(np.abs(output)))
#
#
#print('Exact solution test...')
##set variables
#print('Variable values...')
#eps=1.0
#mu=1.0
#obspts=np.array([[2.,2.],[-2.,2.]])
#print(obspts)
#nodes=np.array([0.,0.])
#nodes=nodes[np.newaxis,:]
#print(nodes)
#f=np.array([0.,-0.5])
#f=f[np.newaxis,:]
#print(f)
#F = np.array([ [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]] ])
#print(F)
#
#
##########################
## test matrix mult
##########################
#print('Testing matrix multiplication.')
#out =cm.matmult(eps,mu,obspts,nodes,f)
#print(out)
#
##exact solution
#exact = np.array([-1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.), 1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.)])
#print(exact)
#
##error
#print('Error between exact and C')
#print(exact-out)
#print(np.max(np.abs(out-exact)))
#
###############################
## test derivative kernel
###############################
#print('Testing derivative kernel.')
#out =cm.derivop(eps,mu,obspts,nodes,f,F)
#print(out)
#
##exact solution
#exact = np.array([ [[2./(81*np.pi)-1./(36*np.pi), 2./(81*np.pi)-1./(36*np.pi)],
#[19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]],[[2./(81*np.pi)-1./(36*np.pi),
#-2./(81*np.pi)+1./(36*np.pi)], [-19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]] ])
#print(exact)
#
##error
#print('Error between exact and C')
#print(exact-out)
#print(np.max(np.abs(out-exact)))

