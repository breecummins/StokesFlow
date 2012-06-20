import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import SpatialDerivs2D as SD2D
import numpy as np
from utilities import loadPickle
import os, sys
import mainC

mpl.rcParams.update({'font.size': 20})

def swimmerOnly(basename,basedir):
    plt.close()
    mydict = loadPickle(basename,basedir)
    myvars = swimmerVars(mydict)
    fpts = mydict['fpts']
    fig=plt.figure()
    for k in range(len(mydict['t'])):
        plt.plot([fpts[0][-2],fpts[0][-2]],[myvars[4],myvars[5]],'b',linewidth=4.0)
        plt.plot(fpts[k][:-1:2],fpts[k][1::2],'r',linewidth=4.0)
        plt.axis('equal')
        plt.xlim(myvars[:2])
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(basedir+basename+'/frame%03d' % k)
        plt.clf()
    plt.close()
   
def swimmerOnlyComp(vebasename,vebasedir,stbasename,stbasedir,swimdir):
    plt.close()
    mydictve = loadPickle(vebasename,vebasedir)
    mydictst = loadPickle(stbasename,stbasedir)
    fname = os.path.expanduser(vebasedir +vebasename +'/comp2stokesframe')
    myvars = swimmerVars(mydictve)
    vefpts = mydictve['fpts']
    stfpts = mydictst['fpts']
    if swimdir == 'right':
        barind = -2
    elif swimdir == 'left':
        barind = 0
    else:
        print('Swimming direction not recognized. Choose "left" or "right".')
        sys.exit()
    fig=plt.figure()
    for k in range(len(mydictve['t'])):
        plt.plot([vefpts[0][barind],vefpts[0][barind]],[myvars[4],myvars[5]],'b',linewidth=4.0)
        plt.plot(stfpts[k][:-1:2],stfpts[k][1::2],'r',linewidth=4.0,label='Stokes')
        plt.plot(vefpts[k][:-1:2],vefpts[k][1::2],'k',linewidth=4.0,label='OB')
        plt.axis(myvars[:4])
        plt.legend(loc='upper right')
        plt.title('Time = '+str(mydictve['t'][k]))
        plt.savefig(os.path.expanduser(fname+'%03d' % k))
        plt.clf()
    plt.close()
 
def trajvort():
    plt.close()
    mydict = mat2py.read(os.path.expanduser('~/scratch/swimmerC_visco_initialize.mat'))
    fname = '0trajvort'
    x = mydict['l'][:,:,:,0]
    y = mydict['l'][:,:,:,1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    dt = mydict['dt'].flatten()
    h = mydict['pdict']['gridspc'].flatten()
    u = (x[len(mydict['t'].flatten())/2,:,:]-x[len(mydict['t'].flatten())/2-2,:,:])/(2*dt)
    v = (y[len(mydict['t'].flatten())/2,:,:]-y[len(mydict['t'].flatten())/2-2,:,:])/(2*dt)
    vx = (v[2:,:] - v[:-2,:])/(2*h)
    uy = (u[:,2:] - u[:,:-2])/(2*h)
    w = vx[:,1:-1] - uy[1:-1,:]
    plt.figure(figsize=(10,5))
    #cool, Paired, Set1, Dark2, Accent, hsv, bone,
    plt.imshow(np.flipud(w.transpose()),cmap='copper',interpolation='nearest')
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    xl=w.shape[0]
    yl=w.shape[1]
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            plt.plot((xl/1.5)*x[range(len(mydict['t'].flatten())/2),i,j]-(xl/1.5/32),(xl/1.5)*y[range(len(mydict['t'].flatten())/2),i,j]-(xl/1.5/24),'k')
    plt.plot((xl/1.5)*mydict['fpts'][len(mydict['t'].flatten())/2-1,:-1:2]-(xl/1.5/32),(xl/1.5)*mydict['fpts'][len(mydict['t'].flatten())/2-1,1::2]-(xl/1.5/24),'k',linewidth=5.0)
    plt.axis([0,xl-1,0,yl-1])
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.gca().xaxis.set_ticks( [] )
    plt.gca().yaxis.set_ticks( [] )
#    plt.savefig(os.path.expanduser('~/scratch/'+fname+'.pdf'))
    plt.axis([7,xl-1,0,yl-1])
    plt.savefig(os.path.expanduser('~/scratch/'+fname+'_short.pdf'))
 
def vorticity():
    plt.close()
    mydict = mat2py.read(os.path.expanduser('~/scratch/swimmerC_visco_artshow.mat'))
    fname = 'artshow_'
    x = mydict['l'][:,:,:,0]
    y = mydict['l'][:,:,:,1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    dt = mydict['dt'].flatten()
    h = mydict['pdict']['gridspc'].flatten()
    for k in range(1,len(mydict['t'].flatten())/2):
        u = (x[k+1,:,:]-x[k-1,:,:])/(2*dt)
        v = (y[k+1,:,:]-y[k-1,:,:])/(2*dt)
        vx = (v[2:,:] - v[:-2,:])/(2*h)
        uy = (u[:,2:] - u[:,:-2])/(2*h)
        w = vx[:,1:-1] - uy[1:-1,:]
#        vmin = np.min([-6,np.min(w)])
#        vmax=np.max([6,np.max(w)])
        plt.imshow(np.flipud(w.transpose()),cmap='gray',interpolation='nearest')
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)
        plt.gca().xaxis.set_ticks( [] )
        plt.gca().yaxis.set_ticks( [] )
        plt.savefig(os.path.expanduser('~/scratch/'+fname+'%03d' % k))
        plt.clf()
          
def swimmerVars(mydict):
    fpts = mydict['fpts']
#    meany = np.mean(fpts[0][1::2])
#    stdy = np.std(fpts[0][1::2])
#    barmin = meany - 2*stdy
#    barmax = meany + 2*stdy
    xmax = fpts[0][0]
    xmin = fpts[0][0]
    ymax = fpts[0][1]
    ymin = fpts[0][1]
    for k in range(len(mydict['t'])):
        xmax = np.max([xmax,np.max(fpts[k][:-1:2])])
        xmin = np.min([xmin,np.min(fpts[k][:-1:2])])
        ymax = np.max([ymax,np.max(fpts[k][1::2])])
        ymin = np.min([ymin,np.min(fpts[k][1::2])])
    barmin = ymin
    barmax = ymax
    xdif = xmax-xmin
    ydif = ymax-ymin
    xmin = xmin - 0.1*xdif
    xmax = xmax + 0.05*xdif
    ymin = ymin - 0.45*ydif
    ymax = ymax + 0.45*ydif
    return [xmin,xmax,ymin,ymax,barmin,barmax]

def stressVars(mydict):
    xmin = np.min(mydict['l'][0][:,:,0])
    xmax = np.max(mydict['l'][0][:,:,0])
    ymin = np.min(mydict['l'][0][:,:,1])
    ymax = np.max(mydict['l'][0][:,:,1])
    symin = (ymax+ymin)/2 + (ymax-ymin)/16
    symax = (ymax+ymin)/2 - (ymax-ymin)/16
    for k in range(len(mydict['l'])):
        lx = mydict['l'][k][:,:,0]
        ly = mydict['l'][k][:,:,1]
        xmin = np.min([xmin,np.min(lx)])
        xmax = np.max([xmax,np.max(lx)])
        ymin = np.min([ymin,np.min(ly)])
        ymax = np.max([ymax,np.max(ly)])
    if xmax-xmin >= ymax-ymin:
        xmin = xmin - 0.01*(xmax-xmin)
        xmax = xmax + 0.01*(xmax-xmin)
        dif = (xmax-xmin - ymax + ymin)/2
        ymin = ymin - dif
        ymax = ymax + dif
    else:
        ymin = ymin - 0.01*(ymax-ymin)
        ymax = ymax + 0.01*(ymax-ymin)
        dif = (ymax-ymin - xmax + xmin)/2
        xmin = xmin - dif
        xmax = xmax + dif      
    vmin = 2.0
    vmax = 2.0
    for k in range(len(mydict['t'])):
        vmi = np.min(mydict['Strace'][k][:,:])
        vma = np.max(mydict['Strace'][k][:,:])
        vmin=np.min([vmin,vmi])
        vmax=np.max([vmax,vma])
    return [xmin,xmax,ymin,ymax,vmin,vmax,symin,symax]
    

def stressTrace(basename,basedir,swimdir):
    '''
    FIXME: Rewrite so that swimmer can be plotted over any of the other plots. Maybe pass function handle.
    '''
    plt.close()
    mydict = loadPickle(basename,basedir)
    fname = os.path.expanduser(basedir +basename +'/traceframe')
    myvars = stressVars(mydict)
    myswimvars = swimmerVars(mydict)
    if swimdir == 'right':
        barind = -2
    elif swimdir == 'left':
        barind = 0
    else:
        print('Swimming direction not recognized. Choose "left" or "right".')
        sys.exit()
    fig=plt.figure()
    for k in range(len(mydict['t'])):
        ph = plt.pcolor(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['Strace'][k][:,:],vmin=myvars[4],vmax=myvars[5],cmap=cm.RdGy)
        fig.colorbar(ph)
        plt.plot([mydict['fpts'][0][barind],mydict['fpts'][0][barind]],[myswimvars[4],myswimvars[5]],'w',linewidth=4.0)
        plt.plot(mydict['fpts'][k][:-1:2],mydict['fpts'][k][1::2],'k',linewidth=4.0)
        plt.axis(myvars[:4])
#        plt.axis([-0.2,1.8,-0.4,1.0])
#        plt.axis('equal')
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(os.path.expanduser(fname+'%03d' % k))
        plt.clf()
        
def stressTraceContour(basename,basedir):
    plt.close()
    mydict = loadPickle(basename,basedir)
    myvars = stressVars(mydict)
    fname = os.path.expanduser(basedir+basename+'/tracecontourframe')
    for k in range(1,len(mydict['t'])):
#        ph=plt.contour(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['Strace'][k][:,:],30,cmap=cm.cool)
        ph2=plt.contourf(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['Strace'][k][:,:],np.arange(myvars[4],myvars[5]+0.1,(myvars[5]-myvars[4])/50),cmap=cm.cool)
#        ph2.set_clim(myvars[4],myvars[5])
        ph2.set_clim(2,3)
        plt.colorbar(ph2)
        plt.plot([mydict['fpts'][0][0],mydict['fpts'][0][0]],[myvars[6],myvars[7]],'w',linewidth=2.0)
        plt.plot(mydict['fpts'][k][:-1:2],mydict['fpts'][k][1::2],'k',linewidth=2.0)
#        plt.axis('equal')
        plt.axis(myvars[:4])
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(os.path.expanduser(basedir+fname+'%03d' % k))
        plt.clf()


def stressTraceExtension(basename,basedir):
    plt.close()
    mydict = loadPickle(basename,basedir)
    fname = os.path.expanduser(basedir+basename +'/traceframe')
    myvars = stressVars(mydict)
    for k in range(0,len(mydict['t'])):
        ph = plt.pcolor(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['Strace'][k][:,:],vmin=myvars[4],vmax=myvars[5],cmap=cm.RdGy)
        plt.colorbar(ph)
#        plt.axis('equal')
        plt.axis(myvars[:4])
#        plt.axis('off')
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(fname+'%03d' % k + '.pdf')
        plt.clf()
        
def stressComponentsPColor(basename,basedir,i,j):
    '''
    S[i,j] = indices of stress component, S_{i+1,j+1} in matrix indices
    '''
    plt.clf()
    mydict = loadPickle(basename,basedir)
    fname=os.path.expanduser(basedir + basename + '/S%d%d_' % (i+1,j+1))
    myvars = stressVars(mydict)
    for k in range(0,len(mydict['t'])/2):
        vmin=np.min(mydict['S'][k][:,:,i,j])
        vmax=np.max(mydict['S'][k][:,:,i,j])
        ph = plt.pcolor(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['S'][k][:,:,i,j],vmin=vmin,vmax=vmax,cmap=cm.RdGy)
        plt.colorbar(ph)
#        plt.axis('equal')
#        plt.axis(myvars[:4])
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(fname+'%03d' % k)
        plt.clf()

        
def stressComponentsMaxMin(basename,basedir):
    plt.close()
    mydict = loadPickle(basename,basedir)
    fname = os.path.expanduser(basedir + basename + '/4RMCompsOverTime')
    S11 = np.zeros((len(mydict['t']),2))
    S12 = np.zeros((len(mydict['t']),2))
    S22 = np.zeros((len(mydict['t']),2))
    for k in range(0,len(mydict['t'])):
        S11[k,0]=np.max(mydict['S'][k][:,:,0,0])
        S11[k,1]=np.min(mydict['S'][k][:,:,0,0])
        S12[k,0]=np.max(mydict['S'][k][:,:,0,1])
        S12[k,1]=np.min(mydict['S'][k][:,:,0,1])
        S22[k,0]=np.max(mydict['S'][k][:,:,1,1])
        S22[k,1]=np.min(mydict['S'][k][:,:,1,1])
    plt.plot(mydict['t'],S11[:,0],'k-',label='S11 max')
    plt.plot(mydict['t'],S11[:,1],'k--',label='S11 min')
    plt.plot(mydict['t'],S22[:,0],'b-', label='S22 max')
    plt.plot(mydict['t'],S22[:,1],'b--', label='S22 min')
    plt.plot(mydict['t'],S12[:,0],'r-', label='S12 max')
    plt.plot(mydict['t'],S12[:,1],'r--', label='S12 min')
    xmin,xmax, ymin, ymax = plt.axis()
    plt.ylim(ymin - (ymax-ymin)/10.,ymax)
    plt.legend( bbox_to_anchor=(0., 0., 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0. )
    plt.title('Stress components over time')
    plt.savefig(fname)
    plt.clf()

      
def stressTrace4RMContour(basename,basedir):
    plt.close()
    mydict = loadPickle(basename,basedir)
    fname=os.path.expanduser(basedir + basename + '/tracecontourframe')
#    vmax=np.max(mydict['Strace'])
#    x = mydict['l'][:,:,:,0]
#    y = mydict['l'][:,:,:,1]
    xmin = np.min(mydict['l'][-1][:,:,0])+0.5
    xmax = np.max(mydict['l'][-1][:,:,0])+0.5
    ymin = np.min(mydict['l'][-1][:,:,1])+0.5
    ymax = np.max(mydict['l'][-1][:,:,1])+0.5
    for k in range(0,len(mydict['t']),100):
        lvls = np.linspace(np.min(mydict['Strace'][k][:,:]),np.max(mydict['Strace'][k][:,:]),21)
        ph=plt.contour(mydict['l'][k][:,:,0],mydict['l'][k][:,:,1],mydict['Strace'][k][:,:],levels=lvls)
#        plt.clabel(ph, inline=1, fontsize=10)        
#        plt.axis('equal')
#        plt.axis([xmin,xmax,ymin,ymax])
        plt.title('Time = '+str(mydict['t'][k]))
        plt.savefig(fname+'%03d' % k)
        plt.clf()
        
def contourRegrid(l,S,limits,time,before,lvls=[],vlims=[]):
    plt.close()
    def makeplot():
#        plt.clabel(ph, inline=1, fontsize=10)        
#        plt.axis('equal')
        plt.axis(limits)
        if before:
            plt.title(str+' before regridding, time = %03f' % time)
            fname = os.path.expanduser('~/VEsims/') + str +'Regridding%02dBefore.pdf' % int(round(time))
        else:
            plt.title(str+' after regridding, time = %03f' % time)
            fname = os.path.expanduser('~/VEsims/') + str +'Regridding%02dRegrid.pdf' % int(round(time))
        plt.savefig(fname)
    ph=plt.contour(l[:,:,0],l[:,:,1],S[:,:,0,0],levels=lvls[0])
#    ph = plt.pcolor(l[:,:,0],l[:,:,1],S[:,:,0,0],vmin=vlims[0],vmax=vlims[1],cmap=cm.cool)
    plt.colorbar(ph)
    str='S11'
    makeplot()
    plt.clf()
    ph=plt.contour(l[:,:,0],l[:,:,1],S[:,:,1,1],levels=lvls[1])
#    ph = plt.pcolor(l[:,:,0],l[:,:,1],S[:,:,1,1],vmin=vlims[2],vmax=vlims[3],cmap=cm.cool)
    plt.colorbar(ph)
    str='S22'
    makeplot()
    plt.close()
            
def pointTraj(basename,basedir):
    '''
    Don't use with regridded data sets.    
    '''
    plt.close()
    mydict = loadPickle(basename,basedir)
    myvars = stressVars(mydict)
    l = mydict['l']
    ksave = [0]
    k = -1
    while k < len(mydict['t'])-1:
        x = np.zeros((len(mydict['t']),l[ksave[-1]].shape[0],l[ksave[-1]].shape[1]))
        y = np.zeros((len(mydict['t']),l[ksave[-1]].shape[0],l[ksave[-1]].shape[1]))
        for k in range(ksave[-1],len(mydict['t'])):
            print(k)
            if k == ksave[-1] or l[k-1].shape == l[k].shape:
                x[k,:,:] = l[k][:,:,0]
                y[k,:,:] = l[k][:,:,1]
            else:
                ksave.append(k) 
                print('regridding changed size of domain')               
                break
        fname = os.path.expanduser(basedir+basename +'/0traj_%03d' % (k-1,) )
        if k == len(mydict['t'])-1:
            ksave.append(len(mydict['t']))
        for i in range(0,l[0].shape[0],2):
                for j in range(0,l[0].shape[1],2):
                    plt.plot(x[ksave[-2]:ksave[-1],i,j],y[ksave[-2]:ksave[-1],i,j],'k')
        plt.axis(myvars[:4])
#        plt.setp(plt.gca().get_xticklabels(), visible=False)
#        plt.setp(plt.gca().get_yticklabels(), visible=False)
#        plt.gca().xaxis.set_ticks( [] )
#        plt.gca().yaxis.set_ticks( [] )
        print(ksave)
        plt.title('Time %03f to time %03f' % (mydict['t'][ksave[-2]],mydict['t'][ksave[-1]-1]))
        plt.savefig(fname+'.pdf')
        plt.clf()
        
def specificPointTraj(basename,basedir):
    '''
    Don't use with regridded data sets.    
    '''
    plt.close()
    mydict = loadPickle(basename,basedir)
    myvars = stressVars(mydict)
    l = mydict['l']
#    pts = [(0,0),(5,0),(0,5),(-5,0),(0,-5),(5,10),(10,5),(-5,10),(10,-5),
#           (5,-10),(-10,5),(-5,-10),(-10,-5)]
    pts = [(0,0),(-5,5),(5,-5),(-10,10),(10,-10),(-15,15),(15,-15)]
#    pts=[(15,20)]
    x = np.zeros((len(pts),len(mydict['t'])))
    y = np.zeros((len(pts),len(mydict['t'])))
    for k in range(len(mydict['t'])):
        ind = l[k].shape[0]/2
        for p in range(len(pts)):
            x[p,k] = l[k][ind+pts[p][0],ind+pts[p][1],0]
            y[p,k] = l[k][ind+pts[p][0],ind+pts[p][1],1]
    for p in range(len(pts)):
        plt.plot(x[p,:],y[p,:],'k')
        plt.plot(x[p,0],y[p,0],'ro')
    plt.title('Time %03f to time %03f' % (mydict['t'][0],mydict['t'][-1]))
    plt.savefig(os.path.expanduser(basedir+basename +'/0trajlinenoregrid.pdf'))
      
def plotQuiverDiff(basename,basedir,velfunc):
    '''
    Don't use with regridded data sets.    
    '''
    plt.close()
    mydict = loadPickle(basename,basedir)
    fname = os.path.expanduser(basedir+basename +'/quivframe')
    myvars = stressVars(mydict)
    for k in range(1,len(mydict['t'])-1):
        lk = mydict['l'][k]
        lkm1 = mydict['l'][k-1]
        lkp1 = mydict['l'][k+1]
        if lkm1.shape == lkp1.shape:
            l2col = np.reshape(lk,(lk.shape[0]*lk.shape[1],2))
            u, junk = velfunc(mydict['pdict'],lk.flatten(),l2col)
            unum = (lkp1 - lkm1)/(2*mydict['dt']) 
            unum = np.reshape(unum, (unum.shape[0]*unum.shape[1],2))
            plt.quiver(l2col[::4,0],l2col[1::4,1],u[::8],u[1::8],color='r')#, units='x', linewidths=(2,), edgecolors=('k'), headaxislength=5)
            plt.quiver(l2col[::4,0],l2col[1::4,1],unum[::4,0],unum[::4,1],color='b')#, units='x',linewidths=(2,), edgecolors=('k'), headaxislength=5)
            plt.axis(myvars[:4])
            plt.title('Time = %03f' % mydict['t'][k])
            plt.savefig(os.path.expanduser(basedir+basename +fname+'%03d.pdf') % k)
            plt.clf()
 
def plotFinalPosition(basedir,bnamelist,xvals,xlab,fnameend): 
    xf =[]
    for k in range(len(bnamelist)):
        print(xvals[k])
        basename = bnamelist[k]
        mydict = loadPickle(basename,basedir)
        xf.append(np.abs(np.max(mydict['fpts'][0][:-1:2])-np.max(mydict['fpts'][-1][:-1:2])))
    plt.close()
    plt.plot(xvals,[x/xf[0] for x in xf],linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel('dist')
    plt.title('Distance traveled in x at time %.01f' % mydict['t'][-1])
    plt.savefig(basedir+'finaldistance'+fnameend+'.pdf')
    return xf
 
def plotFinalPositionUnnormalized(basedir,bnamelist,xvals,xlab,fnameend): 
    xf =[]
    for k in range(len(bnamelist)):
        print(xvals[k])
        basename = bnamelist[k]
        mydict = loadPickle(basename,basedir)
        xf.append(np.abs(np.max(mydict['fpts'][0][:-1:2])-np.max(mydict['fpts'][-1][:-1:2])))
    plt.close()
    plt.plot(xvals,xf,linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel('dist')
    plt.title('Distance traveled in x at time %.01f' % mydict['t'][-1])
    plt.savefig(basedir+'finaldistance'+fnameend+'.pdf')
    return xf

def checkSwimmerLength(basename,basedir):   
    plt.close()
    mydict = loadPickle(basename,basedir)
    fpts = mydict['fpts']
    plt.figure()
    L = mydict['pdict']['forcedict']['L']
    h = mydict['pdict']['forcedict']['h']
    critterlen=[]
    hmax = []
    hmin = []
    for k in range(len(mydict['t'])):
        clen = np.sqrt((fpts[k][2:-1:2] - fpts[k][:-3:2])**2 + (fpts[k][3::2] - fpts[k][1:-2:2])**2)
        critterlen.append( clen.sum() / L )
        hmax.append(np.max(clen) / h)
        hmin.append(np.min(clen) / h)
    plt.plot(mydict['t'],critterlen)
    plt.xlabel('Time')
    plt.ylabel('Normalized length')
    plt.title('Swimmer length vs time')
    plt.savefig(basedir+basename+'/SwimmerLengthvsTime.pdf')
    plt.clf()
    plt.plot(mydict['t'],hmax)
    plt.xlabel('Time')
    plt.ylabel('Normalized max segment')
    plt.title('Max segment length vs time')
    plt.savefig(basedir+basename+'/MaxSegLengthvsTime.pdf')
    plt.clf()
    plt.plot(mydict['t'],hmin)
    plt.xlabel('Time')
    plt.ylabel('Normalized min segment')
    plt.title('Min segment length vs time')
    plt.savefig(basedir+basename+'/MinSegLengthvsTime.pdf')
    
def makeEllipses(basedir,basename):
    plt.close()
    fname = os.path.expanduser(basedir +basename +'/ellipseframe')
    mydict = loadPickle(basename,basedir)
    myvars = stressVars(mydict)
    fpts = mydict['fpts']
    gridspc = mydict['pdict']['gridspc']
    Nt = len(mydict['t'])
    N = mydict['S'][0].shape[0]
    M = mydict['S'][0].shape[1]
    fig = plt.figure()
    for i in range(Nt):
        S = mydict['S'][i]
        l = mydict['l'][i]
        ax = fig.add_subplot(111) #,aspect='equal'
        ax.set_xlim(myvars[:2])
        ax.set_ylim(myvars[2:4])
        ells=[]
        for j in range(N):
            for k in range(M):
#                S = 1/np.sqrt(2)*np.array([[1,2],[-1,2]])
#                w,V = np.linalg.eigh(S)
                w, V = np.linalg.eigh(S[j,k,:,:]) 
#                print('Eigenvalues')
#                print(w)
#                print('Eigenvector dot prod')
#                print(np.dot(V[:,0],V[:,1]))
                center = l[j,k,:]
                ind = np.nonzero(w == np.max(w))
                ind=ind[0][0]
                horzdist = w[ind] * gridspc #put major eigval on x-axis and scale by the grid spacing (to fit in graph)
                vertdist = w[np.mod(ind+1,2)] * gridspc
                ang = np.arccos(V[0,ind]*1 + V[1,ind]*0) #calculate angle of rotation from eigenvector (using orthogonality of eigvecs here)
                ells.append(mpl.patches.Ellipse(xy=center, width=horzdist, height=vertdist, angle=ang*180/np.pi))
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('w')
        plt.plot(fpts[i][:-1:2],fpts[i][1::2],'k',linewidth=4.0)
        plt.title('Time = '+str(mydict['t'][i]))
        plt.savefig(os.path.expanduser(fname+'%03d' % i))
        plt.clf()

def makeEllipses_Deviation(basedir,basename):
    plt.close()
    fname = os.path.expanduser(basedir +basename +'/devellipseframe')
    mydict = loadPickle(basename,basedir)
    myvars = stressVars(mydict)
    fpts = mydict['fpts']
    gridspc = mydict['pdict']['gridspc']
    Nt = len(mydict['t'])
    N = mydict['S'][0].shape[0]
    M = mydict['S'][0].shape[1]
    fig = plt.figure()
    for i in range(Nt):
        S = mydict['S'][i]
        l = mydict['l'][i]
        ax = fig.add_subplot(111) #,aspect='equal'
        ax.set_xlim(myvars[:2])
        ax.set_ylim(myvars[2:4])
        ells=[]
        for j in range(N):
            for k in range(M):
                w, V = np.linalg.eigh(S[j,k,:,:]-np.eye(2,2)) 
                center = l[j,k,:]
                ind = np.nonzero(w == np.max(w))
                ind=ind[0][0]
                horzdist = w[ind] * gridspc
                vertdist = w[np.mod(ind+1,2)] * gridspc
                ang = np.arccos(V[0,ind]*1 + V[1,ind]*0) #calculate angle of rotation from eigenvector (using orthogonality of eigvecs here)
                ells.append(mpl.patches.Ellipse(xy=center, width=horzdist, height=vertdist, angle=ang*180/np.pi))
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('w')
        plt.plot(fpts[i][:-1:2],fpts[i][1::2],'k',linewidth=4.0)
        plt.title('Time = '+str(mydict['t'][i]))
        plt.savefig(os.path.expanduser(fname+'%03d.pdf' % i))
        plt.clf()
        

        
if __name__ == '__main__':
##    basename = 'stokes_TFS_K40_Kcurv01_lam25pi_amp016_L060_N050_Time11'        
##    basedir = os.path.expanduser('~/VEsims/Swimmer/')             
##    checkSwimmerLength(basename,basedir)
##    swimmerOnly(basename,basedir)
##    basename = 'dipole_noregrid_N050_Wi01_Time05'
##    basedir = os.path.expanduser('~/VEsims/DipoleFlow/')
##    stressTraceExtension(basename,basedir)
##    pointTraj(basename,basedir)
##    specificPointTraj(basename,basedir)
##    plotQuiverDiff(basename,basedir)
    basename = 'visco_fixedregrid004_scalefactor2_addpts0_epsobj036_epsgrid036_N054_Wi0100_Time11'
    basenamestokes = 'stokes_epsobj036_Time11'
    basedir = os.path.expanduser('~/VEsims/SwimmerRefactored/TFS_FinalParams/')
    swimdir = 'right'
#    checkSwimmerLength(basename,basedir)
#    swimmerOnlyComp(basename,basedir,basenamestokes,basedir,swimdir)
#    stressTrace(basename,basedir,swimdir)
    makeEllipses_Deviation(basedir,basename)
#    makeEllipses(basedir,basename)
#    Wilist = [0.0,0.25, 0.5, 0.75]#, 1.0, 1.25, 1.5, 1.75]#[1.6]#[0.1,0.08,0.06,0.04,0.02]
#    epslist = ['036']
#    basedir = os.path.expanduser('~/VEsims/SwimmerRefactored/TFS_FinalParams/')
#    for estr in epslist:
#        bnamelist = ['visco_fixedregrid004_scalefactor2_addpts0_epsobj'+estr+'_epsgrid'+estr+'_N054_Wi%04d_Time11' % int(Wi*100) for Wi in Wilist]
#        bnamelist[0]='stokes_epsobj'+estr+'_Time11'
#        xf = plotFinalPosition(basedir,bnamelist,Wilist,'Wi','_vsWi_eps'+estr)
#        print(xf)
#    epslist = [0.024,0.036,0.048,0.060,0.072,0.096]
#    basedir = os.path.expanduser('~/VEsims/Swimmer/TFS_FinalParams_SmallerDomain/')
#    bnamelist = ['stokes_eps%03d_N054_Time11' % int(e*1000) for e in epslist]
#    xf = plotFinalPositionUnnormalized(basedir,bnamelist,epslist,'$\epsilon$','_stokesdistvseps')
#    print(xf)

    