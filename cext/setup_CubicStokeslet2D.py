#Created by Breschine Cummins on June 20, 2012.

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

'''python setup_foo.py build will build the extension module 
foo.so in ./build/lib.arch-id/'''

from distutils.core import setup, Extension
import sys, os, numpy

includen=[numpy.get_include()]

module1 = Extension('CubicStokeslet2D',
					include_dirs=includen,
                    sources = ['CubicStokeslet2D.c'])

setup (name = '2D Cubic Stokeslet module',
       version = '1.0',
       description = 'Functions implementing regularized Stokeslets in 2D using a/( )^3 blob.',
       ext_modules = [module1])
	   
