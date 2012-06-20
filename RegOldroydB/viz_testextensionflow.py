#import python modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os, re
#import home-rolled python modules 
import testextensionflow as tef
from utilities import loadPickle

mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'axes.formatter.limits': (-4,4)})

def plotErrsOnePoint(errors, mydict, basename, basedir, fname, titlestr):
    '''Plot the errors or error differences at a single point'''
    fnamepic=basedir + basename + '/' + fname 
    plt.close()
    plt.figure()
    plt.plot(mydict['t'], errors[0],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errors[1],'r',linewidth=2.0)  
    plt.title(titlestr + ' in position')
    plt.legend(('X','Y'), loc=2)
    plt.xlabel('time')
    plt.savefig(fnamepic+'X')
    plt.figure()
    plt.plot(mydict['t'], errors[2],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errors[3],'r',linewidth=2.0)  
    plt.title(titlestr + ' in P')
    plt.legend(('$P_{11}$','$P_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fnamepic+'P')
    plt.figure()
    plt.plot(mydict['t'], errors[4],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errors[5],'r',linewidth=2.0)  
    plt.title(titlestr + ' in S')
    plt.legend(('$S_{11}$','$S_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fnamepic+'S')
    plt.close()

def plotErrs(errorsLinf, mydict, basename, basedir, fname, titlestr):
    fname=basedir + basename + '/' + fname
    plt.close()
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[0],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[1],'r',linewidth=2.0)  
    plt.title(titlestr + ' in position')
    plt.legend(('X','Y'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'X')
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[2],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[3],'r',linewidth=2.0)  
    plt.plot(mydict['t'], errorsLinf[4],'b',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[5],'g',linewidth=2.0)  
    plt.title(titlestr + ' in P')
    plt.legend(('$P_{11}$','$P_{12}$','$P_{21}$','$P_{22}$'), loc=2)
#    plt.legend(('$P_{11}$','$P_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'P.pdf')
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[6],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[7],'r',linewidth=2.0)  
    plt.plot(mydict['t'], errorsLinf[8],'b',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[9],'g',linewidth=2.0)  
    plt.title(titlestr + ' in S')
    plt.legend(('$S_{11}$','$S_{12}$','$S_{21}$','$S_{22}$'), loc=2)
#    plt.legend(('$S_{11}$','$S_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'S.pdf')
#    plt.figure()
#    plt.plot(mydict['t'], errorsLinf[10],'k',linewidth=2.0)         
#    plt.plot(mydict['t'], errorsLinf[11],'r',linewidth=2.0)         
#    plt.plot(mydict['t'], errorsLinf[12],'b',linewidth=2.0)             
#    plt.plot(mydict['t'], errorsLinf[13],'g',linewidth=2.0)      
#    plt.title(titlestr + ' in F')
#    plt.legend(('$F_{11}$','$F_{12}$','$F_{21}$','$F_{22}$'), loc=2)
#    plt.xlabel('time')
#    plt.savefig(fnameinf+'F')
#    plt.figure()
#    plt.plot(mydict['t'], errorsLinf[14],'k',linewidth=2.0)         
#    plt.plot(mydict['t'], errorsLinf[15],'r',linewidth=2.0)         
#    plt.plot(mydict['t'], errorsLinf[16],'b',linewidth=2.0)             
#    plt.plot(mydict['t'], errorsLinf[17],'g',linewidth=2.0)      
#    plt.title(titlestr + ' in $F^{-1}$')
#    plt.legend(('$F^{-1}_{11}$','$F^{-1}_{12}$','$F^{-1}_{21}$','$F^{-1}_{22}$'), loc=2)
#    plt.xlabel('time')
#    plt.savefig(fnameinf+'Finv')
    plt.close()

def stressComponents(P, S, mydict, basename,basedir,fname='NumSoln'):
    plt.close()
    S11 = np.zeros((len(mydict['t']),))
    S12 = np.zeros((len(mydict['t']),))
    S22 = np.zeros((len(mydict['t']),))
    P11 = np.zeros((len(mydict['t']),))
    P12 = np.zeros((len(mydict['t']),))
    P22 = np.zeros((len(mydict['t']),))
    for k in range(0,len(mydict['t'])):
        S11[k]=np.max(S[k][:,:,0,0])
        S12[k]=np.max(S[k][:,:,0,1])
        S22[k]=np.max(S[k][:,:,1,1])
        P11[k]=np.max(P[k][:,:,0,0])
        P12[k]=np.max(P[k][:,:,0,1])
        P22[k]=np.max(P[k][:,:,1,1])
#        ml = S[k].shape[0]/2;  ml2 = S[k].shape[1]/2 
#        S11[k]=S[k][ml,ml2,0,0]
#        S12[k]=S[k][ml,ml2,0,1]
#        S22[k]=S[k][ml,ml2,1,1]
#        P11[k]=P[k][ml,ml2,0,0]
#        P12[k]=P[k][ml,ml2,0,1]
#        P22[k]=P[k][ml,ml2,1,1]
    plt.figure()
    plt.plot(mydict['t'],S11,'k')
    plt.plot(mydict['t'],S22,'b')
    plt.plot(mydict['t'],S12,'r')
    plt.legend( ('S11', 'S22', 'S12, S21'), loc=6 )
    plt.title('Stress components over time')
    fnameS = basedir + basename + '/' + fname + '_S.pdf'
    plt.savefig(fnameS)
    plt.figure()
    plt.plot(mydict['t'],P11,'k')
    plt.plot(mydict['t'],P22,'b')
    plt.plot(mydict['t'],P12,'r')
    plt.legend( ('P11', 'P22',  'P12, P21'), loc=2 )
    plt.title('Stress components over time')
    fnameP = basedir + basename + '/' + fname + '_P.pdf'
    plt.savefig(fnameP)
    plt.close()
    
def compareErrs(errs,xvals,xlab,basedir,errfname,errtitlestr,timeind,timestr,leglist=None):
    fixedtimelist = []    
    print(errs.shape)
    for k in range(errs.shape[0]):
        print( np.max(errs[k,:10,:]) )
        fixedtimelist.append( errs[k,:10,timeind] )
    ftl = np.asarray(fixedtimelist)
    print(ftl)
    plt.clf()
    plt.close()
    plt.plot(xvals,ftl,linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel('Error')
    if leglist != None:
        plt.legend(leglist,bbox_to_anchor=(1.15, 1.15))
    plt.title(errtitlestr)
    fnameP = basedir + errfname + '_compareErrs.pdf'
    plt.savefig(fnameP)


if __name__ == '__main__':
    pass    
    
    
    
    
    
    
    
    
    
    
    
    
    
        