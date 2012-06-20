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
	   
