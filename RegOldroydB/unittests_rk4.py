#Created by Breschine Cummins on December 16, 2012.

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
import matplotlib.pyplot as plt
import numpy as np

#import my modules
import rk4

#test 1: dy/dt = y**2, y(0) = y0
#soln 1: y = -1 / (t - 1/y0)
def exact1(y0, h, totaltime):
    y=[y0]
    for t in np.arange(0,totaltime-h,h):
        y.append(-1.0 / ( (t+h) - (1.0/y0) ) )
    return y

def num1(y0, h, totaltime):
    y=[y0]
    for t in np.arange(0,totaltime-h,h):
        y.append(rk4.solver(t,y[-1],h,lambda t,y: y**2))
    return y

def check1():
    timesteps=[0.01,0.005,0.0025]
    totaltime=0.95
    y0=1.0
    mycheck(timesteps,totaltime,y0,exact1,num1)
    return None

#test 2: dy/dt = t / sin(y), y(0) = y0
#soln 2: y = arccos( -(t**2 / 2) + cos(y0) ) 
def exact2(y0, h, totaltime):
    y=[y0]
    for t in np.arange(0,totaltime-h,h):
        y.append(np.arccos( -((t+h)**2)/2.0 + np.cos(y0) ) )
    return y

def num2(y0, h, totaltime):
    y=[y0]
    for t in np.arange(0,totaltime-h,h):
        y.append(rk4.solver(t,y[-1],h,lambda t,y: t/np.sin(y)))
    return y

def check2():
    timesteps=[0.01,0.005,0.0025]
    totaltime=1.0
    y0=np.pi/3.0
    mycheck(timesteps,totaltime,y0,exact2,num2)
    return None
    
def mycheck(timesteps,totaltime,y0,extfunc,numfunc):
    maxdif=[]
    for h in timesteps:
        yexact = np.asarray(extfunc(y0,h,totaltime))
        ynum = np.asarray(numfunc(y0,h,totaltime))
        maxdif.append(np.max(np.abs((yexact - ynum)/yexact)) )
        plotme(np.arange(0,totaltime,h),yexact,ynum)
    print(maxdif)
    return None
    
def plotme(times,yexact,ynum):
    plt.figure()
    plt.plot(times,yexact,'k-',label='exact')
    plt.hold(True)
    plt.plot(times,ynum,'r-',label='num')
    plt.legend(loc=2)
    plt.xlabel('time')
    plt.show()
    
    
if __name__ == '__main__':
    print('test 1')
    check1()
    print('test 2')
    check2()
        
        
    
