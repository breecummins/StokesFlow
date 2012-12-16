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
import numpy as np

def solver(ti,yi,h,func):
    # 4th order Runge-Kutta solver; func takes args ti,yi; h is time step
    k1 = h*func(ti,yi)
    k2 = h*func(ti+0.5*h, yi+0.5*k1)
    k3 = h*func(ti+0.5*h, yi+0.5*k2)
    k4 = h*func(ti+h, yi+k3)
    return yi + (k1 + 2*k2 + 2*k3 + k4)/6.0

def solverp(ti,yi,h,func,**params):
    # 4th order Runge-Kutta solver; func takes args ti,yi,**params; h is time step
    k1 = h*func(ti,yi,**params)
    k2 = h*func(ti+0.5*h, yi+0.5*k1,**params)
    k3 = h*func(ti+0.5*h, yi+0.5*k2,**params)
    k4 = h*func(ti+h, yi+k3,**params)
    return yi + (k1 + 2*k2 + 2*k3 + k4)/6.0
    